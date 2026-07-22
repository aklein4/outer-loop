import re
import time

import datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase


INPUT_DATASET = "launch/ManyICLBench"
OUTPUT_DATASET = "aklein4/ManyICLBench-SmolLM2-1024-natural-instructions-format"

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

TRAIN_COLUMN = "128k"
TEST_DATA_COLUMN = "Test Data"
TEST_TARGET_COLUMN = "Test Target"

NUM_TRAIN_COLUMN = "num_train"
NUM_TEST_COLUMN = "num_test"
TRAIN_DATA_COLUMN = "train_data"
TEST_DATA_OUT_COLUMN = "test_data"
MESSAGES_COLUMN = "messages"

SKIP_SUBSTRINGS = [
    "MT_",
    "_cot",
    "GSM8K",
    "XLSUM",
]

TASK_DESCRIPTIONS = {
    "banking77": "Classify the banking customer query into the correct intent.",
    "dialogRE": (
        "Given a dialogue and a list of entity pairs, predict the relation label "
        "for each pair in the same order."
    ),
    "trec_50": "Classify the question into its fine-grained TREC answer type.",
    "clinc150": "Classify the user request into the correct intent.",
    "GPQA": "Answer the multiple-choice science question. Respond with the letter of the correct option.",
    "ARC-Challenge": "Answer the multiple-choice science question. Respond with the letter of the correct option.",
    "ARC-Easy": "Answer the multiple-choice science question. Respond with the letter of the correct option.",
    "BBH-geometric_shapes": (
        "Identify which shape is drawn by the SVG path. Respond with the option "
        "letter in parentheses."
    ),
    "BBH-salient_translation_error_detection": (
        "Identify the type of error in the translation. Respond with the option "
        "letter in parentheses."
    ),
    "BBH-word_sorting": "Sort the words alphabetically.",
    "BBH-dyck_languages": "Complete the bracket sequence so all brackets are closed properly.",
    "goEmotions": "Classify the emotional category expressed by the comment.",
}

MAX_TOKENS = 1024
BATCH_SIZE = 1000
PRIVATE = False
PUSH_RETRIES = 5


def user_turn(content):
    return {"role": "user", "content": content}


def assistant_turn(content):
    return {"role": "assistant", "content": content}


def format_messages(task_description, input_text, output_text):
    input_text = str(input_text).strip()
    output_text = str(output_text).strip()

    if task_description:
        input_text = f"{task_description}\n{input_text}"

    return [
        user_turn(input_text),
        assistant_turn(output_text),
    ]


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


def should_skip_config(config, features):
    if any(s in config for s in SKIP_SUBSTRINGS):
        return True

    required_columns = [TRAIN_COLUMN, TEST_DATA_COLUMN, TEST_TARGET_COLUMN]
    if any(column not in features for column in required_columns):
        return True

    return not isinstance(features[TRAIN_COLUMN], datasets.Value)


def normalize_whitespace(text):
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_prefix(text, prefix):
    return re.sub(r"^\s*" + re.escape(prefix) + r"\s*", "", text, count=1)


def strip_trailing_marker(text, marker):
    marker = str(marker).strip()
    if marker and text.rstrip().endswith(marker):
        text = text.rstrip()[:-len(marker)]
    return text.strip()


def task_description_for(config, parsed_task_description):
    description = TASK_DESCRIPTIONS.get(config, parsed_task_description)
    return normalize_whitespace(description)


def clean_input_text(config, input_text, marker):
    text = strip_trailing_marker(normalize_whitespace(input_text), marker)

    if config in {"banking77", "clinc150"}:
        return strip_prefix(text, "Query:")

    if config == "trec_50":
        return strip_prefix(text, "Question:")

    if config == "goEmotions":
        return strip_prefix(text, "Comment:")

    if config in {"GPQA", "ARC-Challenge", "ARC-Easy"}:
        return strip_prefix(text, "Question:")

    if config == "BBH-geometric_shapes":
        return strip_prefix(text, "Input:")

    if config == "BBH-salient_translation_error_detection":
        text = strip_prefix(text, "Input:")
        text = re.sub(
            r"^The following translations from German to English contain a particular error\..*?"
            r"Please identify that error\.\s*",
            "",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(r"\nThe translation contains an error pertaining to\s*", "\n", text)
        return text.strip()

    if config == "BBH-word_sorting":
        text = strip_prefix(text, "Input:")
        return re.sub(r"^Sort the following words alphabetically:\s*List:\s*", "", text).strip()

    if config == "BBH-dyck_languages":
        text = strip_prefix(text, "Input:")
        if "Input:" in text:
            text = text.rsplit("Input:", 1)[1]
        return text.strip()

    if config == "dialogRE":
        return re.sub(
            r"\nThe list of \d+ relations are\s*",
            "\n\nEntity pairs:\n",
            text,
        ).strip()

    return text


def clean_output_text(output_text):
    return normalize_whitespace(output_text)


def response_marker(test_data):
    return str(test_data[0]).strip().splitlines()[-1].strip()


def example_start_marker(test_data):
    first_example = str(test_data[0]).lstrip()
    return first_example.split(":", 1)[0] + ":"


def split_train_text(train_text, start_marker):
    start = train_text.find(start_marker)
    if start < 0:
        raise ValueError(f"could not find example start marker: {start_marker}")

    task_description = train_text[:start].strip()
    train_text = train_text[start:].strip()
    pattern = re.compile(r"\n\s*\n(?=" + re.escape(start_marker) + r")")
    examples = [
        example.strip()
        for example in pattern.split(train_text)
        if example.strip()
    ]

    return task_description, examples


def parse_train_pairs(config, row):
    marker = response_marker(row[TEST_DATA_COLUMN])
    start_marker = example_start_marker(row[TEST_DATA_COLUMN])
    parsed_task_description, examples = split_train_text(row[TRAIN_COLUMN], start_marker)
    task_description = task_description_for(config, parsed_task_description)

    pairs = []
    for example in examples:
        split_at = example.rfind(marker)
        if split_at < 0:
            raise ValueError(f"could not find response marker: {marker}")

        input_text = example[:split_at + len(marker)]
        output_text = example[split_at + len(marker):]
        pairs.append(format_messages(
            task_description,
            clean_input_text(config, input_text, marker),
            clean_output_text(output_text),
        ))

    return pairs, task_description


def parse_test_pairs(config, row, task_description):
    marker = response_marker(row[TEST_DATA_COLUMN])
    test_data = row[TEST_DATA_COLUMN]
    test_targets = row[TEST_TARGET_COLUMN]

    if len(test_data) != len(test_targets):
        raise ValueError(f"mismatched test data and target lengths: {len(test_data)} != {len(test_targets)}")

    return [
        format_messages(
            task_description,
            clean_input_text(config, input_text, marker),
            clean_output_text(output_text),
        )
        for input_text, output_text in zip(test_data, test_targets)
    ]


def filter_pairs(pairs, tokenizer):
    if len(pairs) == 0:
        return []

    ds = datasets.Dataset.from_dict({
        MESSAGES_COLUMN: pairs
    })

    ds = ds.filter(
        tokenize_and_filter,
        batched=True,
        batch_size=BATCH_SIZE,
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=False,
        desc="Tokenizing and filtering",
    )

    return list(ds[MESSAGES_COLUMN])


def convert_seed(config, row, tokenizer):
    train_data, task_description = parse_train_pairs(config, row)
    test_data = parse_test_pairs(config, row, task_description)

    train_data = filter_pairs(train_data, tokenizer)
    test_data = filter_pairs(test_data, tokenizer)

    return {
        NUM_TRAIN_COLUMN: len(train_data),
        NUM_TEST_COLUMN: len(test_data),
        TRAIN_DATA_COLUMN: train_data,
        TEST_DATA_OUT_COLUMN: test_data,
    }


def convert_config(config, tokenizer):
    builder = datasets.load_dataset_builder(INPUT_DATASET, config)

    rows = []
    for split in builder.info.splits:
        ds = datasets.load_dataset(INPUT_DATASET, config, split=split)
        rows.append(convert_seed(config, ds[0], tokenizer))

    return datasets.Dataset.from_list(rows)


def push_to_hub_with_retries(ds, config):
    for attempt in range(1, PUSH_RETRIES + 1):
        try:
            ds.push_to_hub(
                OUTPUT_DATASET,
                config_name=config,
                split="train",
                private=PRIVATE,
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


def main():
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    configs = datasets.get_dataset_config_names(INPUT_DATASET)

    for i, config in enumerate(configs):
        print("")
        print(f"[{i + 1}/{len(configs)}] Processing subset: {config}")
        print("")

        try:
            builder = datasets.load_dataset_builder(INPUT_DATASET, config)
        except Exception as e:
            print(f"Skipping {config}: could not load dataset builder: {e}")
            continue

        if should_skip_config(config, builder.info.features):
            print(f"Skipping {config}")
            continue

        try:
            ds = convert_config(config, tokenizer)
        except Exception as e:
            print(f"Skipping {config}: conversion failed: {e}")
            continue

        print("")
        print(f"{config}: {len(ds):_} seeds")
        print("")

        push_to_hub_with_retries(ds, config)


if __name__ == "__main__":
    main()
