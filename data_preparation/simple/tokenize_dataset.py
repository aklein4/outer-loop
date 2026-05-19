
import numpy as np
from functools import partial

import datasets

from transformers import AutoTokenizer


TOKENIZER_URL = "meta-llama/Llama-3.2-1B"

DATASET = ("Lyun0912/LongABC", None)

BS = 1024

LENGTH = 1024 * 32

SAVE_URL = "aklein4/longattn-Llama3-32K"


def tokenize_example(
    example,
    tokenizer: AutoTokenizer=None,
    length: int = None,
):

    input_tokens = tokenizer(
        example["content"],
        add_special_tokens=True,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=length,
    ).input_ids.astype(np.uint32)

    keep = input_tokens[:, -1] != tokenizer.pad_token_id

    out = {
        "input_ids": [x for x in input_tokens],
        "keep": [x for x in keep],
    }

    return out


def tokenize_dataset(
    data_url: tuple[str, str],
    tokenizer: AutoTokenizer,
    length: int,
):
    
    data = datasets.load_dataset(*data_url, split="train", streaming=False)

    data = data.map(
        partial(tokenize_example, tokenizer=tokenizer, length=length),
        remove_columns=data.column_names,
        batched=True,
        batch_size=BS,
    )
    data = data.filter(
        lambda example: example["keep"],
    )
    data = data.remove_columns("keep")

    return data


def main():

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_URL)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    data = tokenize_dataset(DATASET, tokenizer, LENGTH)
    
    data = data.shuffle(seed=42)

    data.push_to_hub(
        SAVE_URL, 
        private=False,
    )


if __name__ == "__main__":
    main()
    