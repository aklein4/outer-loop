import argparse
import re
import time

import datasets
from huggingface_hub import HfApi
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DATASET_AUTHOR = "bitext"
OUTPUT_DATASET = "aklein4/Bitext-SmolLM2-1024-natural-instructions-format"

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

TRAIN_SPLIT = "train"
INSTRUCTION_COLUMN = "instruction"
RESPONSE_COLUMN = "response"

NUM_TRAIN_COLUMN = "num_train"
NUM_TEST_COLUMN = "num_test"
TRAIN_DATA_COLUMN = "train_data"
TEST_DATA_OUT_COLUMN = "test_data"
MESSAGES_COLUMN = "messages"

EXCLUDED_DATASETS = {
    "bitext/Bitext-combined-banking-wealth_management-mortgage_loans",
}

TRAIN_EXAMPLES_PER_ROW = 1024
TEST_EXAMPLES_PER_ROW = 100
EXAMPLES_PER_ROW = TRAIN_EXAMPLES_PER_ROW + TEST_EXAMPLES_PER_ROW
ROWS_PER_DATASET = 3

MAX_TOKENS = 1024
BATCH_SIZE = 1000
SEED = 42
PRIVATE = False
PUSH_RETRIES = 5

OUTPUT_FEATURES = datasets.Features({
    NUM_TRAIN_COLUMN: datasets.Value("int64"),
    NUM_TEST_COLUMN: datasets.Value("int64"),
    TRAIN_DATA_COLUMN: [[{
        "role": datasets.Value("string"),
        "content": datasets.Value("string"),
    }]],
    TEST_DATA_OUT_COLUMN: [[{
        "role": datasets.Value("string"),
        "content": datasets.Value("string"),
    }]],
})


def user_turn(content):
    return {"role": "user", "content": content}


def assistant_turn(content):
    return {"role": "assistant", "content": content}


def format_messages(instruction, response):
    return [
        user_turn(instruction),
        assistant_turn(response),
    ]


def normalize_whitespace(text):
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize_and_filter(batch, tokenizer: PreTrainedTokenizerBase):
    messages = batch[MESSAGES_COLUMN]
    for m in messages:
        for turn in m:
            turn["content"] = turn["content"][:MAX_TOKENS * 10]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=MAX_TOKENS + 1,
    )

    return [
        (len(ids) <= MAX_TOKENS)
        for ids in input_ids
    ]


def should_skip_dataset(dataset_id, features, splits):
    if dataset_id in EXCLUDED_DATASETS:
        return True

    required_columns = [INSTRUCTION_COLUMN, RESPONSE_COLUMN]
    if any(column not in features for column in required_columns):
        return True

    if splits is None or TRAIN_SPLIT not in splits:
        return True

    return False


def config_name_for(dataset_id):
    return dataset_id.replace("/", "--")


def bitext_dataset_ids(dataset_ids=None):
    if dataset_ids is not None:
        return [
            dataset_id
            for dataset_id in dataset_ids
            if dataset_id not in EXCLUDED_DATASETS
        ]

    api = HfApi()
    return sorted(
        dataset_info.id
        for dataset_info in api.list_datasets(author=DATASET_AUTHOR)
        if dataset_info.id not in EXCLUDED_DATASETS
    )


def examples_from_dataset(ds):
    messages = []
    seen = set()

    for row in ds:
        instruction = normalize_whitespace(row[INSTRUCTION_COLUMN])
        response = normalize_whitespace(row[RESPONSE_COLUMN])

        if not instruction or not response:
            continue

        key = (instruction, response)
        if key in seen:
            continue
        seen.add(key)

        messages.append(format_messages(instruction, response))

    return messages


def filter_examples(messages, tokenizer, desc):
    if len(messages) == 0:
        return []

    ds = datasets.Dataset.from_dict({
        MESSAGES_COLUMN: messages,
    })

    ds = ds.filter(
        tokenize_and_filter,
        batched=True,
        batch_size=BATCH_SIZE,
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=False,
        desc=desc,
    )

    return list(ds[MESSAGES_COLUMN])


def pack_rows(messages):
    rows = []
    num_rows = min(ROWS_PER_DATASET, len(messages) // EXAMPLES_PER_ROW)

    for row_idx in range(num_rows):
        start = row_idx * EXAMPLES_PER_ROW
        train_end = start + TRAIN_EXAMPLES_PER_ROW
        test_end = train_end + TEST_EXAMPLES_PER_ROW

        train_data = messages[start:train_end]
        test_data = messages[train_end:test_end]

        rows.append({
            NUM_TRAIN_COLUMN: len(train_data),
            NUM_TEST_COLUMN: len(test_data),
            TRAIN_DATA_COLUMN: train_data,
            TEST_DATA_OUT_COLUMN: test_data,
        })

    if len(rows) == 0:
        return datasets.Dataset.from_dict({
            NUM_TRAIN_COLUMN: [],
            NUM_TEST_COLUMN: [],
            TRAIN_DATA_COLUMN: [],
            TEST_DATA_OUT_COLUMN: [],
        }, features=OUTPUT_FEATURES)

    return datasets.Dataset.from_list(rows, features=OUTPUT_FEATURES)


def convert_dataset(dataset_id, tokenizer, seed):
    ds = datasets.load_dataset(dataset_id, split=TRAIN_SPLIT)
    ds = ds.shuffle(seed=seed)

    messages = examples_from_dataset(ds)
    messages = filter_examples(
        messages,
        tokenizer,
        desc=f"{dataset_id}: tokenizing and filtering",
    )

    return pack_rows(messages), len(messages)


def push_to_hub_with_retries(ds, output_dataset, config, private):
    for attempt in range(1, PUSH_RETRIES + 1):
        try:
            ds.push_to_hub(
                output_dataset,
                config_name=config,
                split=TRAIN_SPLIT,
                private=private,
            )
            return
        except Exception as e:
            if attempt == PUSH_RETRIES:
                raise
            wait_seconds = min(60, 2 ** attempt)
            print(
                f"Push failed for {config} on attempt {attempt}/{PUSH_RETRIES}: {e}. "
                f"Retrying in {wait_seconds}s."
            )
            time.sleep(wait_seconds)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    dataset_ids = bitext_dataset_ids(args.dataset)

    if args.dataset_limit is not None:
        dataset_ids = dataset_ids[:args.dataset_limit]

    for i, dataset_id in enumerate(dataset_ids):
        print("")
        print(f"[{i + 1}/{len(dataset_ids)}] Processing dataset: {dataset_id}")
        print("")

        try:
            builder = datasets.load_dataset_builder(dataset_id)
        except Exception as e:
            print(f"Skipping {dataset_id}: could not load dataset builder: {e}")
            continue

        if should_skip_dataset(dataset_id, builder.info.features, builder.info.splits):
            print(f"Skipping {dataset_id}")
            continue

        try:
            ds, num_examples = convert_dataset(dataset_id, tokenizer, args.seed)
        except Exception as e:
            print(f"Skipping {dataset_id}: conversion failed: {e}")
            continue

        config = config_name_for(dataset_id)

        print("")
        print(
            f"{config}: {len(ds):_} rows from {num_examples:_} filtered examples "
            f"({TRAIN_EXAMPLES_PER_ROW:_} train / {TEST_EXAMPLES_PER_ROW:_} test each)"
        )
        print("")

        if not args.dry_run:
            push_to_hub_with_retries(ds, args.output_dataset, config, args.private)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dataset", default=OUTPUT_DATASET)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--private", action="store_true", default=PRIVATE)
    args = parser.parse_args()

    main(args)
