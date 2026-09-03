"""Evaluate online adaptation on trajectories made from distinct PolicyBench tasks.

Each source row is independently shuffled and divided into ``n_tasks`` equal
chunks.  Chunk position i is then independently permuted across trajectories,
subject to no trajectory containing two chunks from the same source row.  This
uses every chunk exactly once.  At every requested adaptation step, all test
examples for every task position are evaluated and logged as ``task_1``, ...,
``task_n``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import datasets
import torch
from tqdm import tqdm

import evaluate_persona as persona
import evaluate_policy as policy
import utils.constants as constants


def system_prompt(institution: str) -> str:
    if not institution:
        raise ValueError("PolicyBench rows must specify an institution")
    return f"Answer the question by following the rules in the {institution} handbook."


def format_adaptation_messages(item: dict, institution: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(institution)},
        *policy.format_adaptation_messages(item),
    ]


def format_evaluation_messages(item: dict, institution: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(institution)},
        *policy.format_evaluation_messages(item),
    ]


def parse_args():
    parser = policy.make_parser()
    parser.description = __doc__
    parser.add_argument(
        "--n-tasks",
        type=int,
        required=True,
        help="Number of equal training chunks and distinct tasks per trajectory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for within-row shuffling and cross-row chunk assignment",
    )
    return parser.parse_args()


def _distinct_column_permutations(
    num_rows: int, n_tasks: int, rng: random.Random
) -> list[list[int]]:
    """Return one row permutation per task with no within-trajectory repeats."""
    columns: list[list[int]] = []
    for _ in range(n_tasks):
        for _attempt in range(10_000):
            candidate = list(range(num_rows))
            rng.shuffle(candidate)
            if all(candidate[i] not in {column[i] for column in columns} for i in range(num_rows)):
                columns.append(candidate)
                break
        else:
            # A randomized cyclic Latin rectangle is a guaranteed fallback.
            base = list(range(num_rows))
            rng.shuffle(base)
            offsets = list(range(num_rows))
            rng.shuffle(offsets)
            return [
                [base[(trajectory_idx + offsets[task_idx]) % num_rows] for trajectory_idx in range(num_rows)]
                for task_idx in range(n_tasks)
            ]
    return columns


def make_trajectories(rows: list[dict], n_tasks: int, trajectory_length: int, seed: int):
    if n_tasks < 1:
        raise ValueError("--n-tasks must be positive")
    if n_tasks > len(rows):
        raise ValueError(
            f"--n-tasks={n_tasks} requires at least {n_tasks} eligible rows, got {len(rows)}"
        )
    if trajectory_length % n_tasks:
        raise ValueError(
            f"training length {trajectory_length} must be divisible by --n-tasks={n_tasks}"
        )

    chunk_length = trajectory_length // n_tasks
    rng = random.Random(seed)
    chunks: list[list[list[dict]]] = []
    for row in rows:
        examples = list(row["train_data"])
        row_rng = random.Random(rng.getrandbits(64))
        row_rng.shuffle(examples)
        examples = examples[:trajectory_length]
        chunks.append(
            [examples[start : start + chunk_length] for start in range(0, trajectory_length, chunk_length)]
        )

    columns = _distinct_column_permutations(len(rows), n_tasks, rng)
    trajectories = []
    for trajectory_idx in range(len(rows)):
        tasks = []
        for task_idx, column in enumerate(columns):
            source_idx = column[trajectory_idx]
            source = rows[source_idx]
            tasks.append(
                {
                    "subset": f"task_{task_idx + 1}",
                    "source_index": source_idx,
                    "source_subset": source["subset"],
                    "institution": source["institution"],
                    "train_data": chunks[source_idx][task_idx],
                    "test_data": source["test_data"],
                }
            )
        trajectories.append(
            {
                "tasks": tasks,
                "train_data": [
                    {"item": example, "institution": task["institution"]}
                    for task in tasks
                    for example in task["train_data"]
                ],
            }
        )
    return trajectories


def adapt(train_fn, tokenizer, trajectories, example_idx, args, device, lr_scale):
    examples = [trajectory["train_data"][example_idx] for trajectory in trajectories]
    input_ids, assistant_mask, attention_mask = persona.encode(
        tokenizer,
        [
            format_adaptation_messages(example["item"], example["institution"])
            for example in examples
        ],
        args.max_length,
        device,
    )
    lr_scale = torch.as_tensor(
        [lr_scale], device=device, dtype=torch.float32
    ).reshape(1, 1, 1)
    with torch.enable_grad():
        train_fn(input_ids, assistant_mask, attention_mask, lr_scale)


def encode_task_prompts(tokenizer, tasks, test_idx: int, max_length: int, device):
    messages = [
        format_evaluation_messages(task["test_data"][test_idx], task["institution"])
        for task in tasks
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_dict=True,
    )
    attention_mask = encoded["attention_mask"].bool()
    lengths = attention_mask.sum(dim=1)
    if (lengths == max_length).any():
        print(
            "Warning: A continual PolicyBench evaluation prompt reached "
            f"--max-length on {(lengths == max_length).sum().item()} examples."
        )
    return (
        encoded["input_ids"].to(device),
        attention_mask.to(device),
        (lengths - 1).to(device),
    )


@torch.no_grad()
def evaluate_all_tasks(
    model, choice_logits_fn, tokenizer, trajectories, args, device, token_ids
):
    model.eval()
    scores = torch.zeros((len(trajectories), args.n_tasks), dtype=torch.float64)
    for task_idx in range(args.n_tasks):
        for test_idx in tqdm(
            range(args.num_eval),
            desc=f"evaluating task_{task_idx + 1}",
            leave=False,
        ):
            tasks = [trajectory["tasks"][task_idx] for trajectory in trajectories]
            items = [task["test_data"][test_idx] for task in tasks]
            input_ids, attention_mask, prompt_indices = encode_task_prompts(
                tokenizer, tasks, test_idx, args.max_length, device
            )
            logits = choice_logits_fn(input_ids, attention_mask, prompt_indices, token_ids)
            labels = torch.tensor(
                [int(item["correct_option"]) for item in items],
                device=device,
                dtype=torch.long,
            )
            scores[:, task_idx] += logits.argmax(dim=-1).eq(labels).double().cpu()
    model.train()
    return scores / args.num_eval


def add_scores(totals, counts, n, scores):
    for task_idx in range(scores.shape[1]):
        subset = f"task_{task_idx + 1}"
        totals[n][subset] = totals[n].get(subset, 0.0) + scores[:, task_idx].sum().item()
        counts[n][subset] = counts[n].get(subset, 0) + scores.shape[0]


def evaluate_trajectories(
    model, train_fn, choice_logits_fn, tokenizer, trajectories, args, device
):
    totals = {n: {} for n in args.num_examples}
    counts = {n: {} for n in args.num_examples}
    token_ids = policy.choice_token_ids(tokenizer, device)

    for start in tqdm(
        range(0, len(trajectories), args.batch_size), desc="batches", leave=True
    ):
        batch = trajectories[start : start + args.batch_size]
        model.init_state(len(batch), device)
        model.empty_state()

        if 0 in totals:
            scores = evaluate_all_tasks(
                model, choice_logits_fn, tokenizer, batch, args, device, token_ids
            )
            add_scores(totals, counts, 0, scores)

        total_adaptation_steps = max(args.num_examples)
        for example_idx in tqdm(
            range(total_adaptation_steps), desc="adapting", leave=False
        ):
            adapt(
                train_fn,
                tokenizer,
                batch,
                example_idx,
                args,
                device,
                lr_scale=persona.adaptation_lr_scale(
                    args, example_idx, total_adaptation_steps
                ),
            )
            n = example_idx + 1
            if n in totals:
                scores = evaluate_all_tasks(
                    model, choice_logits_fn, tokenizer, batch, args, device, token_ids
                )
                add_scores(totals, counts, n, scores)

        model.zero_grad(set_to_none=True)
        model.empty_state()

    results = []
    for n in args.num_examples:
        benchmarks = {
            subset: totals[n][subset] / counts[n][subset]
            for subset in sorted(totals[n])
        }
        results.append(
            {
                "num_examples": n,
                "benchmarks": benchmarks,
                "average": sum(benchmarks.values()) / len(benchmarks),
            }
        )
    return results


def save_results(args, label: int | str, results):
    root = Path(constants.LOCAL_DATA_PATH) / f"policy_{args.n_tasks}_results"
    if args.fresh_config is None:
        directory = (
            args.save_name + "/" + args.checkpoint.replace("/", "--")
            if args.save_name is not None
            else args.checkpoint.replace("/", "--")
        )
        path = root / directory / f"{label:012d}.json"
    else:
        directory = args.save_name if args.save_name is not None else "fresh"
        path = root / directory / Path(args.fresh_config).stem / f"{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {path}")


def main():
    args = parse_args()
    if args.fresh_config is None and args.checkpoint_steps is None:
        raise ValueError("--checkpoint-steps is required unless --fresh-config is set")

    args.num_examples = sorted(set(args.num_examples))
    if not args.num_examples or args.num_examples[0] < 0:
        raise ValueError("--num-examples must contain non-negative evaluation points")
    trajectory_length = max(args.num_examples)
    if trajectory_length < 1:
        raise ValueError("the largest --num-examples value must be positive")
    if args.lr_scale_decay:
        print(
            f"Using linear lr-scale decay from {args.lr_scale_start:g} "
            f"to {args.lr_scale_end:g} over {trajectory_length} adaptation steps"
        )

    device = torch.device(args.device)
    subsets = args.subsets or datasets.get_dataset_config_names(args.dataset)
    rows = policy.load_all_rows(args, subsets)
    if not rows:
        raise ValueError("No PolicyBench rows satisfy the requested train/evaluation sizes")
    trajectories = make_trajectories(rows, args.n_tasks, trajectory_length, args.seed)
    chunk_length = trajectory_length // args.n_tasks
    print(
        f"Built {len(trajectories)} trajectories from {len(rows)} rows: "
        f"{args.n_tasks} tasks x {chunk_length} examples (seed {args.seed})"
    )

    if args.fresh_config is not None:
        for base_lr in args.base_lrs or [None]:
            for step in args.checkpoint_steps or [None]:
                model = persona.load_fresh_model(args.fresh_config, base_lr, step, device)
                tokenizer = persona.load_tokenizer(args.tokenizer)
                train_fn, choice_logits_fn = policy.make_fns(model, args, device)
                results = evaluate_trajectories(
                    model,
                    train_fn,
                    choice_logits_fn,
                    tokenizer,
                    trajectories,
                    args,
                    device,
                )
                save_results(args, persona.lr_label(base_lr, model, step), results)
        return

    for step in args.checkpoint_steps:
        model = persona.load_model(args.checkpoint, step, device)
        tokenizer = persona.load_tokenizer(args.tokenizer)
        train_fn, choice_logits_fn = policy.make_fns(model, args, device)
        results = evaluate_trajectories(
            model, train_fn, choice_logits_fn, tokenizer, trajectories, args, device
        )
        save_results(args, step, results)


if __name__ == "__main__":
    main()
