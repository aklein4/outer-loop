import random

import datasets
from tqdm.auto import tqdm


INPUT_DATASET = "aklein4/few-shot-TrackStar"
OUTPUT_DATASET = "aklein4/horizons-10B"

MESSAGES_COLUMN = "messages"
SOURCE_COLUMN = "source"
CLUSTER_COLUMN = "cluster"
CLUSTER_MEAN_COSINE_SIMILARITY_COLUMN = "similarity_to_cluster_mean"
AVERAGE_SIMILARITY_TO_MEAN_COLUMN = "average_similarity_to_mean"
NUM_TOKENS_COLUMN = "num_tokens"

CLUSTER_SIZE = 64
SEED = 42


EPISODE_COLUMNS = [
    f"episode_{i + 1}"
    for i in range(CLUSTER_SIZE)
]


def check_columns(ds, config):
    required = [
        MESSAGES_COLUMN,
        SOURCE_COLUMN,
        CLUSTER_COLUMN,
        CLUSTER_MEAN_COSINE_SIMILARITY_COLUMN,
        NUM_TOKENS_COLUMN,
    ]
    missing = [
        column
        for column in required
        if column not in ds.column_names
    ]
    if missing:
        raise ValueError(f"{config} is missing required columns: {missing}")


def cluster_row(rows, rng):
    if len(rows) != CLUSTER_SIZE:
        raise ValueError(f"expected cluster size {CLUSTER_SIZE}, got {len(rows)}")

    sources = set(row[SOURCE_COLUMN] for row in rows)
    if len(sources) != 1:
        raise ValueError(f"expected one source per cluster, got {sorted(sources)}")

    episodes = [
        row[MESSAGES_COLUMN]
        for row in rows
    ]
    rng.shuffle(episodes)

    out = {
        SOURCE_COLUMN: rows[0][SOURCE_COLUMN],
        NUM_TOKENS_COLUMN: sum(row[NUM_TOKENS_COLUMN] for row in rows),
        AVERAGE_SIMILARITY_TO_MEAN_COLUMN: (
            sum(row[CLUSTER_MEAN_COSINE_SIMILARITY_COLUMN] for row in rows) / len(rows)
        ),
    }
    out.update(dict(zip(EPISODE_COLUMNS, episodes)))

    return out


def cluster_rows(ds, config, rng):
    check_columns(ds, config)

    rows = []
    current_cluster = None
    current_rows = []
    seen_clusters = set()

    for row in tqdm(ds, desc=f"Reformatting {config}"):
        cluster = row[CLUSTER_COLUMN]

        if current_cluster is None:
            current_cluster = cluster

        if cluster != current_cluster:

            if len(current_rows) != CLUSTER_SIZE:
                raise ValueError(f"{config} has cluster {current_cluster} with {len(current_rows)} rows, expected {CLUSTER_SIZE}")
            rows.append(cluster_row(current_rows, rng))

            seen_clusters.add(current_cluster)
            if cluster in seen_clusters:
                raise ValueError(f"{config} has non-contiguous rows for cluster {cluster}")
            
            current_cluster = cluster
            current_rows = []

        current_rows.append(row)

    if current_rows:
        if len(current_rows) != CLUSTER_SIZE:
            raise ValueError(f"{config} has trailing cluster {current_cluster} with {len(current_rows)} rows, expected {CLUSTER_SIZE}")
        rows.append(cluster_row(current_rows, rng))

    return datasets.Dataset.from_list(rows)


def main():
    rng = random.Random(SEED)
    configs = datasets.get_dataset_config_names(INPUT_DATASET)

    parts = []
    for i, config in enumerate(configs):
        print("")
        print(f"[{i + 1}/{len(configs)}] Processing subset: {config}")
        print("")

        ds = datasets.load_dataset(INPUT_DATASET, config, split="train")
        parts.append(cluster_rows(ds, config, rng))

    ds = datasets.concatenate_datasets(parts)
    ds = ds.shuffle(seed=SEED)

    total_tokens = sum(ds[NUM_TOKENS_COLUMN])
    print(f"Total tokens: {total_tokens:_}")

    ds.push_to_hub(
        OUTPUT_DATASET,
        split="train",
        private=False,
    )


if __name__ == "__main__":
    main()
