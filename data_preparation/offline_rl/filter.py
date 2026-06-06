import argparse
import re

import datasets
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoTokenizer


INPUT_DATASET = "aklein4/offline-RL-binary-2"
OUTPUT_DATASET = "aklein4/offline-RL-binary-2-filtered"
TOKENIZER_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HISTOGRAM_PATH = "length_histogram.png"

SOURCE_COLUMN = "source"
INPUT_COLUMN = "input"
OUTPUT_COLUMN = "output"
INPUT_LENGTH_COLUMN = "input_length"
OUTPUT_LENGTH_COLUMN = "output_length"
TOTAL_LENGTH_COLUMN = "total_length"
TRAIN_SPLIT = "train"
HISTOGRAM_CLIP = 2000


def deduplicate_by_input(ds, limit=None):
    seen = set()
    keep = []
    for i, text in enumerate(ds[INPUT_COLUMN]):
        if text not in seen:
            seen.add(text)
            keep.append(i)
        if limit and len(keep) == limit:
            break
    return ds.select(keep)


def token_lengths(tokenizer, input_text, output_text):
    messages = [
        {"role": "user", "content": str(input_text)},
        {"role": "assistant", "content": str(output_text)},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=True,
        add_generation_prompt=True,
    )
    total_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    input_length = len(input_ids)
    total_length = len(total_ids)
    return input_length, max(total_length - input_length, 0), total_length


def add_token_lengths(batch, tokenizer):
    input_lengths = []
    output_lengths = []
    total_lengths = []

    for input_text, output_text in zip(batch[INPUT_COLUMN], batch[OUTPUT_COLUMN]):
        input_length, output_length, total_length = token_lengths(
            tokenizer,
            input_text,
            output_text,
        )
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        total_lengths.append(total_length)

    return {
        INPUT_LENGTH_COLUMN: input_lengths,
        OUTPUT_LENGTH_COLUMN: output_lengths,
        TOTAL_LENGTH_COLUMN: total_lengths,
    }


def source_filter(row, source):
    return row[SOURCE_COLUMN] == source


def safe_config_name(source):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", source.replace("/", "--"))


def plot_length_histogram(input_lengths, output_lengths, total_lengths, path, clip):
    series = [
        ("Input", input_lengths),
        ("Output", output_lengths),
        ("Total", total_lengths),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (title, lengths) in zip(axes, series):
        clipped = [min(length, clip) for length in lengths]
        ax.hist(clipped, bins=50)
        ax.set_title(title)
        ax.set_xlim(0, clip)
        ax.set_xlabel(f"Tokens (clipped at {clip})")
        ax.set_ylabel("Examples")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset", default=INPUT_DATASET)
    parser.add_argument("--output-dataset", default=OUTPUT_DATASET)
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME)
    parser.add_argument("--histogram-path", default=HISTOGRAM_PATH)
    parser.add_argument("--histogram-clip", type=int, default=HISTOGRAM_CLIP)
    parser.add_argument("--configs", nargs="*")
    parser.add_argument("--sources", nargs="*")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--limit-per-source", type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    configs = args.configs or datasets.get_dataset_config_names(args.input_dataset)

    input_lengths = []
    output_lengths = []
    total_lengths = []

    for config in tqdm(configs, desc="Configs"):
        ds = datasets.load_dataset(args.input_dataset, config, split=TRAIN_SPLIT)

        for source in sorted(ds.unique(SOURCE_COLUMN)):
            if args.sources and source not in args.sources:
                continue

            subset = ds.filter(source_filter, fn_kwargs={"source": source})
            subset = deduplicate_by_input(subset, limit=args.limit_per_source)

            length_columns = [
                column
                for column in [
                    INPUT_LENGTH_COLUMN,
                    OUTPUT_LENGTH_COLUMN,
                    TOTAL_LENGTH_COLUMN,
                ]
                if column in subset.column_names
            ]
            if length_columns:
                subset = subset.remove_columns(length_columns)

            subset = subset.map(
                add_token_lengths,
                batched=True,
                batch_size=args.batch_size,
                fn_kwargs={"tokenizer": tokenizer},
                desc=f"Tokenizing {source}",
            )

            input_lengths.extend(subset[INPUT_LENGTH_COLUMN])
            output_lengths.extend(subset[OUTPUT_LENGTH_COLUMN])
            total_lengths.extend(subset[TOTAL_LENGTH_COLUMN])

            out_config = safe_config_name(source)
            print(f"{source}: {len(subset):_} examples")

            if not args.dry_run:
                subset.push_to_hub(
                    args.output_dataset,
                    config_name=out_config,
                    split=TRAIN_SPLIT,
                    private=args.private,
                )

    plot_length_histogram(
        input_lengths,
        output_lengths,
        total_lengths,
        args.histogram_path,
        args.histogram_clip,
    )


if __name__ == "__main__":
    main()
