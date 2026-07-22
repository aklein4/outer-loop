import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import datasets


DEFAULT_DATASET = "aklein4/few-shot-TrackStar-8192-test"
DEFAULT_CONFIG = "HuggingFaceTB--smoltalk--train"
DEFAULT_SPLIT = "train"
DEFAULT_CLUSTER_SIZE = 64
DEFAULT_FIRST_CLUSTERS = 100
DEFAULT_NUM_CLUSTERS = 10
DEFAULT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save flattened examples for sampled clusters from a clustered few-shot dataset."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--cluster-size", type=int, default=DEFAULT_CLUSTER_SIZE)
    parser.add_argument("--first-clusters", type=int, default=DEFAULT_FIRST_CLUSTERS)
    parser.add_argument("--num-clusters", type=int, default=DEFAULT_NUM_CLUSTERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="*",
        help="Explicit cluster IDs to save. If omitted, clusters are sampled from range(first_clusters).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for cluster_###.json files. Defaults to cluster_samples/<config> next to this script.",
    )
    return parser.parse_args()


def choose_clusters(args):
    if args.clusters:
        clusters = sorted(set(args.clusters))
    else:
        clusters = sorted(random.Random(args.seed).sample(range(args.first_clusters), args.num_clusters))

    if any(cluster < 0 for cluster in clusters):
        raise ValueError(f"cluster IDs must be non-negative: {clusters}")
    if any(cluster >= args.first_clusters for cluster in clusters):
        raise ValueError(f"clusters must be within the first {args.first_clusters} clusters: {clusters}")
    return clusters


def flatten_messages(messages):
    system = None
    user = None
    response = None

    for turn in messages:
        role = turn.get("role")
        content = turn.get("content")
        if role == "system" and system is None:
            if content is not None and content.strip():
                system = content
        elif role == "user" and user is None:
            user = content
        elif role == "assistant" and response is None:
            response = content

    if user is None or response is None:
        raise ValueError(f"expected at least one user and one assistant message, got: {messages!r}")

    return {
        "system": system,
        "user": user,
        "response": response,
    }


def output_dir_for(args):
    if args.output_dir is not None:
        return args.output_dir
    return Path(__file__).resolve().parent / "cluster_samples_8192" / args.config


def main():
    args = parse_args()
    clusters = choose_clusters(args)
    output_dir = output_dir_for(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    row_count = args.cluster_size * args.first_clusters
    split = f"{args.split}[:{row_count}]"
    ds = datasets.load_dataset(args.dataset, args.config, split=split)

    counts = Counter(ds["cluster"])
    missing = [cluster for cluster in clusters if counts[cluster] == 0]
    if missing:
        raise RuntimeError(f"missing sampled clusters in first {row_count} rows: {missing}")

    examples_by_cluster = defaultdict(list)
    for row in ds:
        cluster = row["cluster"]
        if cluster in clusters:
            examples_by_cluster[cluster].append(flatten_messages(row["messages"]))

    for cluster in clusters:
        path = output_dir / f"cluster_{cluster:03d}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(examples_by_cluster[cluster], f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"{path}: {len(examples_by_cluster[cluster])} examples")


if __name__ == "__main__":
    main()
