import datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase


INPUT_DATASET = "aklein4/single-turn-compilation"
OUTPUT_DATASET = "aklein4/single-turn-compilation-SmolLM2-1024"

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

MESSAGES_COLUMN = "messages"
NUM_TOKENS_COLUMN = "num_tokens"

MAX_TOKENS = 1024
BATCH_SIZE = 1000


def add_token_lengths(batch, tokenizer: PreTrainedTokenizerBase):
    
    messages = batch[MESSAGES_COLUMN]
    for m in messages:
        for turn in m:
            turn["content"] = turn["content"][:MAX_TOKENS * 10]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=MAX_TOKENS+1,
    )

    return {
        NUM_TOKENS_COLUMN: [len(ids) for ids in input_ids]
    }


def length_filter(batch):
    return [
        (n <= MAX_TOKENS)
        for n in batch[NUM_TOKENS_COLUMN]
    ]


def main():

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    configs = datasets.get_dataset_config_names(INPUT_DATASET)

    for i, config in enumerate(configs):

        print("")
        print(f"[{i+1}/{len(configs)}] Processing subset: {config}")
        print("")

        ds = datasets.load_dataset(INPUT_DATASET, config, split="train")

        ds = ds.map(
            add_token_lengths,
            batched=True,
            batch_size=BATCH_SIZE,
            fn_kwargs={"tokenizer": tokenizer},
            desc=f"Tokenizing {config}",
        )
        ds = ds.filter(
            length_filter,
            batched=True,
            batch_size=BATCH_SIZE,
            desc=f"Filtering {config}",
        )

        print("")
        print(f"{config}: {len(ds):_} examples")
        print("")

        ds.push_to_hub(
            OUTPUT_DATASET,
            config_name=config,
            split="train",
            private=False,
        )


if __name__ == "__main__":
    main()
