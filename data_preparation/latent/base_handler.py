
from collections import defaultdict
import datasets

from utils import clean_conversation, convert_role_conversation


class BaseHandler:

    # basic dataset information
    url = None
    subset = None
    split = None

    # the kind of data (e.g. chat, math, code, etc.)
    kind = None

    # to fix loading on some datasets
    verification_mode = None


    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._latent_counts = None
    

    def load_dataset(self):
        if self.subset is not None and self.split is not None:
            assert (not isinstance(self.subset, list)) or (not isinstance(self.split, list)), "Cannot have both subset and split as lists."

        # load multiple subsets (single split)
        if self.subset is not None and isinstance(self.subset, list):

            # load every subset
            subs = [
                datasets.load_dataset(self.url, s, split=self.split, verification_mode=self.verification_mode)
                for s in self.subset
            ]

            # add a column to each subset indicating which subset it is
            subs = [
                ds.add_column("subset", [s] * len(ds))
                for ds, s in zip(subs, self.subset)
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
            splits = [
                datasets.load_dataset(self.url, self.subset, split=s, verification_mode=self.verification_mode)
                for s in self.split
            ]

            # add a column to each split indicating which split it is
            splits = [
                ds.add_column("split", [s] * len(ds))
                for ds, s in zip(splits, self.split)
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
        return datasets.load_dataset(self.url, self.subset, split=self.split, verification_mode=self.verification_mode)


    def process(self, max_count=None, num_proc=1, batch_size=1000):
        self._latent_counts = defaultdict(int)

        ds = self.load_dataset()
        if max_count is not None:
            ds = ds.select(range(min(max_count, len(ds))))

        ds = ds.map(
            self.full_map_fn,
            num_proc=None,
            batched=False,
            remove_columns=[
                n for n in ds.column_names
                if n not in ["messages", "latent", "keep"]
            ],
            load_from_cache_file=False
        )
        ds = ds.filter(
            self.filter_fn,
            input_columns="keep",
            num_proc=num_proc,
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=False
        )

        ds = ds.add_column("source", [self.source()] * len(ds))
        ds = ds.add_column("kind", [self.kind] * len(ds))
        ds = ds.remove_columns(["keep"])

        counts = self._latent_counts
        self._latent_counts = None
        
        return ds, counts


    def source(self):
        return self.url


    def full_map_fn(self, example):
        assert self._latent_counts is not None, "full_map_fn should only be called during processing."
        
        conversation, latent, keep = self.map_fn(example)

        if not keep:
            conversation = None
            latent = None
        else:
            conversation = convert_role_conversation(conversation)
            conversation = clean_conversation(conversation)
            latent = latent.strip() if isinstance(latent, str) else latent
            self._latent_counts[latent] += 1
        
        return {
            "messages": conversation,
            "latent": latent,
            "keep": keep
        }

    def map_fn(self, example) -> tuple[list[dict]|None, str|int|None, bool]:
        raise NotImplementedError("Subclasses must implement this method.")
    

    def filter_fn(self, keeps):
        return keeps
    