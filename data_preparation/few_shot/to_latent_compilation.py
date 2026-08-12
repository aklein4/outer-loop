import argparse
import random

import datasets


INPUT_DATASET = "aklein4/few-shot-TrackStar"
OUTPUT_DATASET = "aklein4/latent-compilation"
CLUSTER_SIZE = 64
SEED = 42

CONFIGS = {
    "HuggingFaceTB--smoltalk2--SFT": {
        "output_config": "HuggingFaceTB--smoltalk2",
        "base_source": "HuggingFaceTB/smoltalk2",
    },
    "natural-instructions": {
        "output_config": "natural-instructions",
        "base_source": "natural-instructions",
    },
}

EPISODE_COLUMNS = [f"episode_{i:02d}" for i in range(CLUSTER_SIZE)]
OUTPUT_COLUMNS = ["latent", "source", "kind", *EPISODE_COLUMNS]
MESSAGE_FEATURE = datasets.List(
    {
        "role": datasets.Value("string"),
        "content": datasets.Value("string"),
    }
)
OUTPUT_FEATURES = datasets.Features(
    {
        "latent": datasets.Value("string"),
        "source": datasets.Value("string"),
        "kind": datasets.Value("string"),
        **{column: MESSAGE_FEATURE for column in EPISODE_COLUMNS},
    }
)


def one_value(values, name, config):
    unique = set(values)
    if len(unique) != 1:
        raise ValueError(f"{config} batch has multiple {name} values: {sorted(unique)}")
    return values[0]


def split_source(current_source, base_source, config):
    if current_source == base_source:
        raise ValueError(f"{config} source has no latent suffix: {current_source!r}")

    prefix = f"{base_source}/"
    if not current_source.startswith(prefix):
        raise ValueError(
            f"{config} source {current_source!r} does not start with {prefix!r}"
        )

    latent = current_source[len(prefix) :].strip("/")
    if not latent:
        raise ValueError(f"{config} source has an empty latent: {current_source!r}")
    return base_source.rstrip("/"), latent


def convert_batch(batch, indices, *, config, base_source, seed):
    if len(indices) != CLUSTER_SIZE:
        raise ValueError(
            f"{config} batch starting at {indices[0]} has {len(indices)} rows, "
            f"expected {CLUSTER_SIZE}"
        )

    cluster = one_value(batch["cluster"], "cluster", config)
    source = one_value(batch["source"], "source", config)
    kind = one_value(batch["kind"], "kind", config)
    output_source, latent = split_source(source, base_source, config)

    episodes = list(batch["messages"])
    # A per-cluster seed makes results reproducible even if processing is resumed.
    random.Random(f"{seed}:{config}:{cluster}").shuffle(episodes)

    output = {
        "latent": [latent],
        "source": [output_source],
        "kind": [kind],
    }
    output.update(
        {column: [episode] for column, episode in zip(EPISODE_COLUMNS, episodes)}
    )
    return output


def convert_config(config, *, output_dataset, private):
    settings = CONFIGS[config]
    print(f"Downloading {INPUT_DATASET}/{config}...", flush=True)
    source_ds = datasets.load_dataset(INPUT_DATASET, config, split="train")

    if len(source_ds) % CLUSTER_SIZE:
        raise ValueError(
            f"{config} has {len(source_ds)} rows, which is not divisible by {CLUSTER_SIZE}"
        )

    print(f"Converting {len(source_ds):_} episodes...", flush=True)
    output_ds = source_ds.map(
        convert_batch,
        batched=True,
        batch_size=CLUSTER_SIZE,
        with_indices=True,
        fn_kwargs={
            "config": config,
            "base_source": settings["base_source"],
            "seed": SEED,
        },
        remove_columns=source_ds.column_names,
        features=OUTPUT_FEATURES,
        load_from_cache_file=False,
        writer_batch_size=128,
        desc=f"Converting {config}",
    )

    expected_rows = len(source_ds) // CLUSTER_SIZE
    if len(output_ds) != expected_rows:
        raise ValueError(f"created {len(output_ds)} rows, expected {expected_rows}")
    if output_ds.column_names != OUTPUT_COLUMNS:
        raise ValueError(
            f"incorrect columns: {output_ds.column_names}; expected {OUTPUT_COLUMNS}"
        )

    print(
        f"Uploading {len(output_ds):_} rows to "
        f"{output_dataset}/{settings['output_config']}...",
        flush=True,
    )
    output_ds.push_to_hub(
        output_dataset,
        config_name=settings["output_config"],
        split="train",
        private=private,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--output-dataset", default=OUTPUT_DATASET)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = args.configs or CONFIGS
    unknown = set(configs) - CONFIGS.keys()
    if unknown:
        raise ValueError(f"unknown configs: {sorted(unknown)}")
    for config in configs:
        convert_config(
            config,
            output_dataset=args.output_dataset,
            private=args.private,
        )


if __name__ == "__main__":
    main()
