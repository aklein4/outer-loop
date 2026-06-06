import argparse
import math
import re

import datasets
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

import balanced_assignment


INPUT_DATASET = "aklein4/offline-RL-binary-2"
OUTPUT_DATASET = "aklein4/offline-RL-binary-clustered"
MODEL_NAME = "google/embeddinggemma-300m"

MAX_LENGTH = 2048

SOURCE_COLUMN = "source"
INPUT_COLUMN = "input"
CLUSTER_KEY_COLUMN = "cluster_key"
CLUSTER_COLUMN = "cluster"
TRAIN_SPLIT = "train"


def balanced_kmeans(x, k, steps=25, assignment_steps=100, seed=42):
    x = F.normalize(x.float().cuda(), dim=1)
    k = min(k, len(x))

    g = torch.Generator(device=x.device).manual_seed(seed)
    centers = x[torch.randperm(len(x), generator=g, device=x.device)[:k]].clone()

    for _ in tqdm(range(steps), desc="Balanced k-means", leave=False):
        scores = x @ centers.T

        # The assignment extension requires an equal number of points per cluster,
        # so add dummy rows when len(x) is not divisible by k.
        per_cluster = math.ceil(len(x) / k)
        padded = per_cluster * k
        if padded != len(x):
            dummy = scores.new_full((padded - len(x), k), -2.0)
            dummy[torch.arange(len(dummy), device=x.device), torch.arange(len(dummy), device=x.device)] = 2.0
            scores = torch.cat([scores, dummy])

        order, _ = balanced_assignment.balanced_assignment(scores.contiguous(), assignment_steps)
        labels = torch.empty(padded, dtype=torch.long, device=x.device)
        labels[order.long()] = torch.arange(k, device=x.device).repeat_interleave(per_cluster)
        labels = labels[: len(x)]

        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k).clamp_min(1).to(new_centers.dtype)
        new_centers = F.normalize(new_centers / counts[:, None], dim=1)
        if torch.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset", default=INPUT_DATASET)
    parser.add_argument("--output-dataset", default=OUTPUT_DATASET)
    parser.add_argument("--configs", nargs="*")
    parser.add_argument("--sources", nargs="*")
    parser.add_argument("--cluster-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--kmeans-steps", type=int, default=25)
    parser.add_argument("--assignment-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--limit-per-source", type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    configs = args.configs or datasets.get_dataset_config_names(args.input_dataset)

    for config in tqdm(configs, desc="Configs"):
        ds = datasets.load_dataset(args.input_dataset, config, split=TRAIN_SPLIT)

        for source in sorted(ds.unique(SOURCE_COLUMN)):
            if args.sources and source not in args.sources:
                continue

            subset = ds.filter(lambda row: row[SOURCE_COLUMN] == source)

            seen = set()
            keep = []
            for i, text in enumerate(subset[INPUT_COLUMN]):
                if text not in seen:
                    seen.add(text)
                    keep.append(i)
                if args.limit_per_source and len(keep) == args.limit_per_source:
                    break
            keep = keep[: args.cluster_size * (len(keep) // args.cluster_size)]
            subset = subset.select(keep)

            print(f"{source}: {len(subset):_} examples")

            embeddings = model.encode(
                [str(x) for x in subset[CLUSTER_KEY_COLUMN]],
                prompt_name="Clustering",
                batch_size=args.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=True,
                processing_kwargs={"text": {"max_length": MAX_LENGTH, "truncation": True}}
            )
            labels = balanced_kmeans(
                embeddings,
                k=len(embeddings)//args.cluster_size,
                steps=args.kmeans_steps,
                assignment_steps=args.assignment_steps,
                seed=args.seed,
            )

            if CLUSTER_COLUMN in subset.column_names:
                subset = subset.remove_columns(CLUSTER_COLUMN)
            subset = subset.add_column(CLUSTER_COLUMN, labels)
            subset = subset.sort(CLUSTER_COLUMN)

            counts = torch.bincount(torch.tensor(labels))
            out_config = re.sub(r"[^A-Za-z0-9._-]+", "_", source.replace("/", "--"))
            print(f"{source}: {len(counts)} clusters, min={counts.min().item()}, max={counts.max().item()}")

            if not args.dry_run:
                subset.push_to_hub(
                    args.output_dataset,
                    config_name=out_config,
                    split=TRAIN_SPLIT,
                    private=args.private,
                )


if __name__ == "__main__":
    main()
