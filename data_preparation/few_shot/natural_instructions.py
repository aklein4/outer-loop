import argparse
import glob
import json
import os
import random
import numpy as np
from collections import defaultdict

import datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase


TASKS_DIR = os.path.expanduser("~/natural-instructions/tasks")
TEST_TASKS_FILE = os.path.expanduser("~/natural-instructions/splits/default/test_tasks.txt")

OUTPUT_DATASET = "aklein4/few-shot-TrackStar" # "aklein4/few-shot-natural-instructions"
SUBSET = "natural-instructions"

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

LANGUAGE = "English"
LANGUAGE_COLUMNS = ["Input_language", "Output_language", "Instruction_language"]

MESSAGES_COLUMN = "messages"
SOURCE_COLUMN = "source"
KIND_COLUMN = "kind"
NUM_TOKENS_COLUMN = "num_tokens"
CLUSTER_COLUMN = "cluster"
CLUSTER_MEAN_COSINE_SIMILARITY_COLUMN = "similarity_to_cluster_mean"

TASK_COLUMN = "_task"
FORMAT_GROUP_COLUMN = "_format_group"
SPLIT_COLUMN = "_split"

SOURCE_PREFIX = "natural-instructions"
KIND = "natural_instructions"

TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

MAX_TOKENS = 1024
BATCH_SIZE = 1000
CLUSTER_SIZE = 64
NUM_FORMAT_GROUPS = 3
SEED = 42
RANDOM_CLUSTER_SIMILARITY = 0.0


def system_turn(content):
    return {"role": "system", "content": content}

def user_turn(content):
    return {"role": "user", "content": content}

def assistant_turn(content):
    return {"role": "assistant", "content": content}


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n\n".join([str(v).strip() for v in value if str(v).strip()]).strip()
    return str(value).strip()


def is_english(value):
    if isinstance(value, str):
        value = [value]
    return isinstance(value, list) and set(value) == {LANGUAGE}


def is_english_english_task(task):
    return all(is_english(task.get(column)) for column in LANGUAGE_COLUMNS)


def split_evenly(values, num_groups):
    values = values.copy()
    random.shuffle(values)
    groups = []
    start = 0
    for i in range(num_groups):
        size = len(values) // num_groups
        if i < len(values) % num_groups:
            size += 1
        groups.append(values[start : start + size])
        start += size
    return groups


def format_messages(format_group, task_description, input_text, output_text):
    if format_group == 0:
        return [
            system_turn(task_description),
            user_turn(input_text),
            assistant_turn(output_text),
        ]
    if format_group == 1:
        return [
            user_turn(f"{task_description}\n{input_text}"),
            assistant_turn(output_text),
        ]
    if format_group == 2:
        return [
            user_turn(f"{task_description}\n\n{input_text}"),
            assistant_turn(output_text),
        ]
    raise ValueError(f"unknown format group: {format_group}")


def load_test_tasks(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def valid_outputs(instance):
    outputs = instance.get("output")
    if not isinstance(outputs, list):
        outputs = [outputs]

    valid = []
    for output in outputs:
        output = normalize_text(output)
        if output:
            valid.append(output)
    return valid


def load_task_rows(path, test_tasks, max_examples_per_task=None):
    task_name = os.path.splitext(os.path.basename(path))[0]

    with open(path) as f:
        task = json.load(f)

    if not is_english_english_task(task):
        return []

    if normalize_text(task.get("Definition")) == "":
        return []
    definitions = task.get("Definition")
    if isinstance(definitions, str):
        definitions = [definitions]

    instances = task.get("Instances") or []
    if max_examples_per_task is not None:
        instances = instances[:max_examples_per_task]

    split = TEST_SPLIT if task_name in test_tasks else TRAIN_SPLIT
    source = f"{SOURCE_PREFIX}/{task_name}"

    rows = []
    for format_group, group in enumerate(split_evenly(instances, NUM_FORMAT_GROUPS)):
        for instance in group:
            input_text = normalize_text(instance.get("input"))
            outputs = valid_outputs(instance)

            if input_text == "" or len(outputs) == 0:
                continue

            rows.append({
                SOURCE_COLUMN: source,
                KIND_COLUMN: KIND,
                MESSAGES_COLUMN: format_messages(
                    format_group,
                    normalize_text(random.choice(definitions)),
                    input_text,
                    random.choice(outputs),
                ),
                TASK_COLUMN: task_name,
                FORMAT_GROUP_COLUMN: format_group,
                SPLIT_COLUMN: split,
            })

    return rows


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
        max_length=MAX_TOKENS + 1,
    )

    return {
        NUM_TOKENS_COLUMN: [len(ids) for ids in input_ids]
    }


def length_filter(batch):
    return [
        (n <= MAX_TOKENS)
        for n in batch[NUM_TOKENS_COLUMN]
    ]


def load_rows(args):
    test_tasks = load_test_tasks(args.test_tasks_file)
    task_paths = sorted(glob.glob(os.path.join(args.tasks_dir, "*.json")))

    if args.task_limit is not None:
        task_paths = task_paths[:args.task_limit]

    rows = []
    for path in task_paths:
        rows.extend(load_task_rows(path, test_tasks, args.max_examples_per_task))

    return rows


def tokenize_and_filter(rows):
    ds = datasets.Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    ds = ds.map(
        add_token_lengths,
        batched=True,
        batch_size=BATCH_SIZE,
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=False,
        desc="Tokenizing",
    )
    ds = ds.filter(
        length_filter,
        batched=True,
        batch_size=BATCH_SIZE,
        load_from_cache_file=False,
        desc="Filtering",
    )

    return ds


def cluster_rows(ds):
    groups = defaultdict(list)
    for row in ds:
        groups[(row[SPLIT_COLUMN], row[TASK_COLUMN], row[FORMAT_GROUP_COLUMN])].append(row)

    clustered = defaultdict(list)
    cluster = 0
    for key in sorted(groups):
        rows = groups[key]
        random.shuffle(rows)

        keep = CLUSTER_SIZE * (len(rows) // CLUSTER_SIZE)
        rows = rows[:keep]

        for start in range(0, len(rows), CLUSTER_SIZE):
            for row in rows[start : start + CLUSTER_SIZE]:
                row[CLUSTER_COLUMN] = cluster
                row[CLUSTER_MEAN_COSINE_SIMILARITY_COLUMN] = RANDOM_CLUSTER_SIMILARITY
                clustered[row[SPLIT_COLUMN]].append(row)
            cluster += 1

    return clustered


def build_split_datasets(clustered):
    split_datasets = {}

    for split, rows in clustered.items():
        ds = datasets.Dataset.from_list(rows)
        ds = ds.remove_columns([TASK_COLUMN, FORMAT_GROUP_COLUMN, SPLIT_COLUMN])
        split_datasets[split] = ds.sort(CLUSTER_COLUMN)

    return split_datasets


def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = load_rows(args)
    print(f"Loaded {len(rows):_} examples")
    if len(rows) == 0:
        return

    ds = tokenize_and_filter(rows)
    print(f"Kept {len(ds):_} examples after length filtering")

    clustered = cluster_rows(ds)
    split_datasets = build_split_datasets(clustered)

    for split in [TRAIN_SPLIT, TEST_SPLIT]:
        if split not in split_datasets:
            continue

        ds = split_datasets[split]
        print(f"{split}: {len(ds):_} examples, {len(ds.unique(CLUSTER_COLUMN)):_} clusters")

        if not args.dry_run:
            ds.push_to_hub(
                args.output_dataset,
                config_name=args.subset,
                split=split,
                private=args.private,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", default=TASKS_DIR)
    parser.add_argument("--test-tasks-file", default=TEST_TASKS_FILE)
    parser.add_argument("--output-dataset", default=OUTPUT_DATASET)
    parser.add_argument("--subset", default=SUBSET)
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--max-examples-per-task", type=int)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    main(args)
