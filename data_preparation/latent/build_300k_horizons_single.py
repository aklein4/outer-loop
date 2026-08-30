"""Expand, globally shuffle, and upload the 300k horizons dataset."""

import tempfile

import datasets


SOURCE_DATASET = "aklein4/300k-horizons"
DESTINATION_DATASET = "aklein4/300k-horizons-single"
SPLIT = "train"
NUM_HORIZONS = 64
SOURCE_ROWS = 300_000
SHUFFLE_SEED = 42

TOOL_CALL = {
    "type": datasets.Value("string"),
    "function": {
        "name": datasets.Value("string"),
        "arguments": datasets.Value("string"),
    },
}
MESSAGE = {
    "role": datasets.Value("string"),
    "content": datasets.Value("string"),
    "tool_calls": datasets.List(TOOL_CALL),
}
FEATURES = datasets.Features(
    {
        "source": datasets.Value("string"),
        "latent": datasets.Value("string"),
        "messages": datasets.List(MESSAGE),
    }
)


def split_horizons(batch):
    """Turn every source row into one row per horizon."""
    output = {"source": [], "latent": [], "messages": []}
    for row_index, (source, latent) in enumerate(
        zip(batch["source"], batch["latent"], strict=True)
    ):
        for horizon_index in range(NUM_HORIZONS):
            output["source"].append(source)
            output["latent"].append(latent)
            output["messages"].append(
                batch[f"episode_{horizon_index:02d}"][row_index]
            )
    return output


def main():
    dataset = datasets.load_dataset(SOURCE_DATASET, split=SPLIT)
    if len(dataset) != SOURCE_ROWS:
        raise RuntimeError(
            f"Source contains {len(dataset):_} rows; expected {SOURCE_ROWS:_}"
        )

    expected_rows = SOURCE_ROWS * NUM_HORIZONS
    with tempfile.TemporaryDirectory(prefix="build-300k-horizons-single-") as cache_dir:
        print(f"Expanding {len(dataset):_} rows to {expected_rows:_} rows...", flush=True)
        dataset = dataset.map(
            split_horizons,
            batched=True,
            batch_size=100,
            remove_columns=dataset.column_names,
            features=FEATURES,
            load_from_cache_file=False,
            cache_file_name=f"{cache_dir}/expanded.arrow",
            desc="Splitting horizons",
        )
        if len(dataset) != expected_rows:
            raise RuntimeError(
                f"Expansion produced {len(dataset):_} rows; expected {expected_rows:_}"
            )

        print(f"Globally shuffling {len(dataset):_} rows...", flush=True)
        dataset = dataset.shuffle(
            seed=SHUFFLE_SEED,
            load_from_cache_file=False,
            indices_cache_file_name=f"{cache_dir}/shuffle-indices.arrow",
        )

        print(f"Uploading to {DESTINATION_DATASET}...", flush=True)
        result = dataset.push_to_hub(
            DESTINATION_DATASET,
            split=SPLIT,
            private=False,
            max_shard_size="1GB",
        )
        print(result, flush=True)


if __name__ == "__main__":
    main()
