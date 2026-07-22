
import datasets


def single_turn(conversation):
    out = []
    for turn in conversation:
        out.append(turn)
        if turn["role"] == "assistant":
            break
    return out


def remove_think(content):
    if "</think>" in content:
        return content.split("</think>")[-1].strip()
    return content


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
    

    def load_dataset(self):
        if self.subset is not None and self.split is not None:
            assert not (isinstance(self.subset, list) and isinstance(self.split, list)), "Cannot have both subset and split as lists."

        # load multiple subsets (single split)
        if self.subset is not None and isinstance(self.subset, list):

            subs = [
                datasets.load_dataset(self.url, sub, split=self.split, verification_mode=self.verification_mode)
                for sub in self.subset
            ]

            splits = [
                ds.add_column("subset", [s] * len(ds))
                for ds, s in zip(subs, self.subset)
            ]
            
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

            splits = [
                datasets.load_dataset(self.url, self.subset, split=s, verification_mode=self.verification_mode)
                for s in self.split
            ]

            splits = [
                ds.add_column("split", [s] * len(ds))
                for ds, s in zip(splits, self.split)
            ]

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


    def name(self, example=None):

        name = self.url

        if self.subset is not None and not isinstance(self.subset, list):
            name += f"/{self.subset}"
        if self.split is not None and not isinstance(self.split, list):
            name += f"/{self.split}"

        if example is not None:
            if "subset" in example:
                name += f"/{example['subset']}"
            if "split" in example:
                name += f"/{example['split']}"

        return name


    def full_map(self, example):
        
        conversation = self.map(example)

        keep = True
        if conversation is not None:
            
            conversation = single_turn(conversation)
            conversation[-1]["content"] = remove_think(conversation[-1]["content"])

            if conversation[0]["role"] == "system" and (conversation[0]["content"] is None or conversation[0]["content"].strip() == ""):
                conversation = conversation[1:]

            for turn in conversation:
                if turn["content"] is None or turn["content"].strip() == "":
                    keep = False
                    break
                turn["content"] = turn["content"].strip()
                turn.pop("reasoning_content", None)

        if not keep:
            conversation = None
            
        return {
            "source": self.name(example),
            "kind": self.kind,
            "messages": conversation
        }


    def map(self, example):
        raise NotImplementedError("Subclasses must implement this method.")
    

    def filter(self, examples):
        return [
            (m is not None and len(m) > 0)
            for m in examples["messages"]
        ]
    