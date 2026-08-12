
from abc import ABC, abstractmethod
from functools import partial
import os
import uuid

import datasets

from utils import clean_conversation, convert_role_conversation


Conversation = list[dict]
Latent = str | int | None
SingleMapResult = tuple[Conversation | None, Latent, bool]
MultipleMapResult = tuple[
    list[Conversation | None],
    list[Latent],
    list[bool],
]


class BaseHandler(ABC):

    # basic dataset information
    url = None
    subset = None
    split = None

    # the kind of data (e.g. chat, math, code, etc.)
    kind = None

    # to fix loading on some datasets
    verification_mode = None

    # Some handlers load source tables large enough that every map worker's
    # Arrow scan buffers materially increase system memory usage. Leave the
    # caller's requested parallelism unchanged unless a handler opts into a
    # safe cap.
    max_num_proc = None

    # Controls passed to ``trajectify`` by compile.py. Handlers with ordered
    # sequences can opt out of within-latent shuffling and padding.
    shuffle_episodes = True
    drop_incomplete_trajectories = False

    # Optional fixed schema for handlers whose valid output types cannot be
    # inferred from every independent map-worker shard.
    output_features = None


    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


    def _load_dataset_part(self, subset, split, max_count=None):
        ds = datasets.load_dataset(
            self.url,
            subset,
            split=split,
            verification_mode=self.verification_mode,
            streaming=max_count is not None,
        )
        if max_count is None:
            return ds

        rows = list(ds.take(max_count))
        return datasets.Dataset.from_list(rows, features=ds.features)


    def load_dataset(self, max_count=None):
        if self.subset is not None and self.split is not None:
            assert (not isinstance(self.subset, list)) or (not isinstance(self.split, list)), "Cannot have both subset and split as lists."

        # load multiple subsets (single split)
        if self.subset is not None and isinstance(self.subset, list):

            # load every subset
            subs = []
            loaded_subsets = []
            remaining = max_count
            for subset in self.subset:
                if remaining == 0:
                    break
                ds = self._load_dataset_part(subset, self.split, remaining)
                subs.append(ds)
                loaded_subsets.append(subset)
                if remaining is not None:
                    remaining -= len(ds)

            # add a column to each subset indicating which subset it is
            subs = [
                ds.add_column("subset", [s] * len(ds))
                for ds, s in zip(subs, loaded_subsets)
            ]

            # only keep columns that are common to all subsets
            common_columns = set.intersection(*[set(ds.column_names) for ds in subs])
            subs = [
                ds.remove_columns(
                    [col for col in ds.column_names if col not in common_columns]
                )
                for ds in subs
            ]

            return datasets.concatenate_datasets(subs)

        # load multiple splits (single subset)
        if self.split is not None and isinstance(self.split, list):

            # load every split
            splits = []
            loaded_splits = []
            remaining = max_count
            for split in self.split:
                if remaining == 0:
                    break
                ds = self._load_dataset_part(self.subset, split, remaining)
                splits.append(ds)
                loaded_splits.append(split)
                if remaining is not None:
                    remaining -= len(ds)

            # add a column to each split indicating which split it is
            splits = [
                ds.add_column("split", [s] * len(ds))
                for ds, s in zip(splits, loaded_splits)
            ]

            # only keep columns that are common to all splits
            common_columns = set.intersection(*[set(ds.column_names) for ds in splits])
            splits = [
                ds.remove_columns(
                    [col for col in ds.column_names if col not in common_columns]
                )
                for ds in splits
            ]

            return datasets.concatenate_datasets(splits)

        # simple
        return self._load_dataset_part(self.subset, self.split, max_count)


    def process(
        self,
        tokenizer=None,
        max_count=None,
        num_proc=1,
        batch_size=1000,
        intermediate_cache_dir=None,
    ):
        if self.max_num_proc is not None:
            num_proc = min(num_proc, self.max_num_proc)

        self.intermediate_cache_dir = intermediate_cache_dir
        ds = self.load_dataset(max_count=max_count)
        if max_count is not None:
            ds = ds.select(range(min(max_count, len(ds))))

        return self.process_dataset(
            ds,
            tokenizer=tokenizer,
            num_proc=num_proc,
            batch_size=batch_size,
            intermediate_cache_dir=intermediate_cache_dir,
        )


    def generator_cache_dir(self):
        """Return an ephemeral cache root for Dataset.from_generator."""
        if self.intermediate_cache_dir is None:
            return None
        path = os.path.join(self.intermediate_cache_dir, "generator")
        os.makedirs(path, exist_ok=True)
        return path


    def process_dataset(
        self,
        ds,
        tokenizer=None,
        num_proc=1,
        batch_size=1000,
        load_from_cache_file=False,
        intermediate_cache_dir=None,
    ):
        """Map and filter one already-loaded source dataset."""

        def cache_file_name(stage):
            if intermediate_cache_dir is None:
                return None
            os.makedirs(intermediate_cache_dir, exist_ok=True)
            return os.path.join(
                intermediate_cache_dir,
                f"{stage}-{uuid.uuid4().hex}.arrow",
            )

        ds = ds.map(
            self.full_map_batch_fn,
            num_proc=num_proc,
            batched=True,
            batch_size=batch_size,
            remove_columns=ds.column_names,
            # A source may already use an output name with an incompatible
            # physical type (Toucan SFT stores ``messages`` as JSON text).
            # Those columns are removed, so infer the mapped output afresh.
            try_original_type=False,
            features=self.output_features,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name("map"),
        )
        ds = ds.filter(
            partial(self.filter_fn, tokenizer=tokenizer),
            num_proc=num_proc,
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name("filter"),
        )

        # ``filter`` returns an indices mapping. ``add_column`` otherwise
        # materializes that mapping with its single-process default, which is
        # prohibitively slow for full-sized datasets.
        ds = ds.flatten_indices(
            num_proc=num_proc,
            cache_file_name=cache_file_name("flatten"),
        )

        ds = ds.add_column("source", [self.source()] * len(ds))
        ds = ds.add_column("kind", [self.kind] * len(ds))
        ds = ds.remove_columns(["keep"])

        return ds


    def source(self):
        return self.url


    @staticmethod
    def _expand_map_result(result):
        if not isinstance(result, (list, tuple)) or len(result) != 3:
            raise TypeError(
                "map_fn must return (conversation, latent, keep) or parallel "
                "(conversations, latents, keeps) sequences."
            )

        conversations, latents, keeps = result
        is_multiple = isinstance(conversations, (list, tuple)) and (
            not conversations or not isinstance(conversations[0], dict)
        )
        if not is_multiple:
            return [(conversations, latents, keeps)]

        if not isinstance(latents, (list, tuple)) or not isinstance(keeps, (list, tuple)):
            raise TypeError(
                "Multiple conversations require corresponding latent and keep sequences."
            )
        if len(conversations) != len(latents) or len(conversations) != len(keeps):
            raise ValueError(
                "Multiple conversation, latent, and keep sequences must have equal lengths."
            )

        return zip(conversations, latents, keeps)


    @staticmethod
    def _normalize_map_result(conversation, latent, keep):

        if conversation is None:
            keep = False
        else:
            conversation = convert_role_conversation(conversation)
            conversation = clean_conversation(conversation)

            count = 0
            for message in conversation:
                if message["role"] == "assistant" and len(message["content"]) > 0:
                    count += 1
            if count == 0:
                keep = False
            
        if not keep:
            conversation = None
            latent = None
        else:
            latent = latent.strip() if isinstance(latent, str) else latent
        
        return {
            "messages": conversation,
            "latent": latent,
            "keep": keep
        }


    def full_map_fn(self, example):
        return [
            self._normalize_map_result(conversation, latent, keep)
            for conversation, latent, keep in self._expand_map_result(
                self.map_fn(example)
            )
        ]


    def full_map_batch_fn(self, examples):
        output = {"messages": [], "latent": [], "keep": []}
        if not examples:
            return output

        keys = tuple(examples)
        for values in zip(*(examples[key] for key in keys)):
            example = dict(zip(keys, values))
            for mapped in self.full_map_fn(example):
                for key in output:
                    output[key].append(mapped[key])

        return output


    @abstractmethod
    def map_fn(self, example) -> SingleMapResult | MultipleMapResult:
        """Convert one source example into one or more mapped outputs.

        A handler may return one ``(conversation, latent, keep)`` triple or
        parallel ``(conversations, latents, keeps)`` lists of equal length.
        """
    

    def filter_fn(self, examples, tokenizer=None):
        keeps = examples["keep"].copy()

        if tokenizer is None:
            return keeps

        m = []
        inds = []
        for i, keep in enumerate(keeps):
            if keep:
                m.append(examples["messages"][i])
                inds.append(i)

        if len(m) != 0:
            mask = tokenizer(m)["assistant_mask"].any(dim=-1)
            for i, m in zip(inds, mask):
                if not m:
                    keeps[i] = False

        return keeps
