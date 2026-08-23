"""Teacher-forced ARC-AGI-1/2 evaluation with per-task test-time learning.

This evaluator never generates.  Every public test pair becomes an independent
row, the model adapts to augmented leave-one-out versions of that row's
demonstrations, and the gold response is scored with teacher forcing.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import evaluate_icl as icl
from models.piano import PianoMode, PianoModel
import utils.constants as constants


MAX_SEQUENCE_LENGTH = 1024
DEFAULT_DATA_ROOTS = [
    Path(constants.LOCAL_DATA_PATH) / "ARC-AGI" / "data",
    Path(constants.LOCAL_DATA_PATH) / "ARC-AGI-2" / "data",
]
PROMPT_PREAMBLE = (
    "These grids follow a hidden input-output rule. Infer the output of the "
    "last input grid."
)

Grid = list[list[int]]
Pair = dict[str, Grid]


@dataclass(frozen=True)
class ArcRow:
    split: str
    task_id: str
    test_index: int
    train: list[Pair]
    test: Pair
    dataset: str = "arc-agi-2"


@dataclass(frozen=True)
class Transform:
    name: str
    function: Callable[[np.ndarray, random.Random], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-config", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        nargs="+",
        default=DEFAULT_DATA_ROOTS,
        help="One or more ARC data directories (defaults to ARC-AGI-1 and ARC-AGI-2).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["public-test"],
        choices=["public-test", "evaluation", "validation", "training"],
        help=(
            "ARC split(s). public-test, evaluation, and validation are aliases "
            "for each dataset's official evaluation set."
        ),
    )
    parser.add_argument(
        "--task-count", type=int, default=None,
        help=(
            "Use only the first N eligible rows after tasks with multiple test "
            "pairs have been duplicated into independent rows."
        ),
    )
    parser.add_argument(
        "--ttt-steps", type=int, default=64,
        help="Maximum augmented leave-one-out adaptation sequences per row.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help=(
            "Number of independent ARC rows adapted and scored in parallel. "
            "Each row still receives one sequence per test-time step."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default=icl.DEFAULT_TOKENIZER)
    parser.add_argument("--base-lr", type=float, default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument(
        "--model-kwargs",
        type=json.loads,
        default={},
        metavar="JSON",
        help=(
            "JSON object of model-config overrides; dotted keys address nested "
            "fields (for example '{\"base_lr\":0.0001}')."
        ),
    )
    parser.add_argument("--aux-weight", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(constants.REPO_PATH) / "local_data" / "arc_agi_results",
        help="Root directory for checkpoint/config-organized result files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output file override.",
    )
    return parser.parse_args()


def grid_text(grid: Grid) -> str:
    """Use compact digit rows; ARC colors are always single digits (0--9)."""
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def make_prompt(examples: list[Pair], question: Grid) -> str:
    sections = [PROMPT_PREAMBLE]
    for index, pair in enumerate(examples, start=1):
        sections.append(
            f"# Example {index}\n\n"
            f"## Input\n\n{grid_text(pair['input'])}\n\n"
            f"## Output\n\n{grid_text(pair['output'])}"
        )
    sections.append(f"# Question\n\n## Input\n\n{grid_text(question)}\n\n## Output")
    return "\n\n".join(sections)


def conversation(examples: list[Pair], question: Pair):
    return [
        {"role": "user", "content": make_prompt(examples, question["input"])},
        {"role": "assistant", "content": grid_text(question["output"])},
    ]


def canonical_split(split: str) -> str:
    return "evaluation" if split in {"public-test", "validation"} else split


def load_rows(
    data_root: Path,
    splits: Iterable[str],
    dataset: str | None = None,
) -> list[ArcRow]:
    if dataset is None:
        dataset = (
            "arc-agi-2"
            if data_root.parent.name.lower().endswith("-2")
            else "arc-agi-1"
        )
    rows: list[ArcRow] = []
    seen_splits: set[str] = set()
    for requested_split in splits:
        split = canonical_split(requested_split)
        if split in seen_splits:
            continue
        seen_splits.add(split)
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"ARC split directory not found: {split_dir}. Pass the ARC-AGI-1 "
                "or ARC-AGI-2 data directory with --data-root."
            )
        for path in sorted(split_dir.glob("*.json")):
            task = json.loads(path.read_text())
            for test_index, test_pair in enumerate(task["test"]):
                if "output" not in test_pair:
                    continue
                rows.append(
                    ArcRow(
                        split="validation" if split == "evaluation" else split,
                        task_id=path.stem,
                        test_index=test_index,
                        train=task["train"],
                        test=test_pair,
                        dataset=dataset,
                    )
                )
    return rows


def _apply_grid(grid: Grid, transform: Transform, rng: random.Random) -> Grid:
    array = transform.function(np.asarray(grid, dtype=np.int64), rng)
    return np.asarray(array, dtype=np.int64).tolist()


def _identity(x: np.ndarray, _: random.Random) -> np.ndarray:
    return x.copy()


def _rotate(k: int):
    return lambda x, _: np.rot90(x, k=k)


def _flip(axis: int):
    return lambda x, _: np.flip(x, axis=axis)


def _reflect(axis: int, reverse: bool):
    def apply(x: np.ndarray, _: random.Random) -> np.ndarray:
        reflected = np.flip(x, axis=axis)
        pieces = (reflected, x) if reverse else (x, reflected)
        return np.concatenate(pieces, axis=axis)
    return apply


def _translate(x: np.ndarray, rng: random.Random) -> np.ndarray:
    shift_x = rng.randrange(min(4, x.shape[0]))
    shift_y = rng.randrange(min(4, x.shape[1]))
    return np.roll(x, (shift_x, shift_y), axis=(0, 1))


def _scale(axis: int | None):
    def apply(x: np.ndarray, _: random.Random) -> np.ndarray:
        if axis is None:
            return np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)
        return np.repeat(x, 2, axis=axis)
    return apply


def _repeat(axis: int | None):
    def apply(x: np.ndarray, _: random.Random) -> np.ndarray:
        if axis is None:
            return np.tile(x, (2, 2))
        return np.concatenate((x, x), axis=axis)
    return apply


def _chain(first, second):
    return lambda x, rng: second(first(x, rng), rng)


def paper_transforms() -> list[Transform]:
    """The transformation set used by the paper's released TTT pipeline."""
    upscale = _scale(None)
    transforms = [
        Transform("identity", _identity),
        Transform("rotate-90", _rotate(1)),
        Transform("rotate-270", _rotate(3)),
        Transform("rotate-180", _rotate(2)),
        Transform("flip-vertical", _flip(0)),
        Transform("flip-horizontal", _flip(1)),
        Transform("reflect-vertical-reverse", _reflect(0, True)),
        Transform("reflect-horizontal-reverse", _reflect(1, True)),
        Transform("reflect-vertical", _reflect(0, False)),
        Transform("reflect-horizontal", _reflect(1, False)),
        Transform("translate-xy", _translate),
        Transform("transpose", lambda x, _: x.T),
        Transform("increase-resolution-2", upscale),
        Transform("increase-height-2", _scale(0)),
        Transform("increase-width-2", _scale(1)),
    ]
    for name, transform in (
        ("rotate-90", _rotate(1)),
        ("rotate-270", _rotate(3)),
        ("rotate-180", _rotate(2)),
        ("flip-vertical", _flip(0)),
        ("flip-horizontal", _flip(1)),
        ("transpose", lambda x, _: x.T),
    ):
        transforms.append(Transform(f"{name}+increase-resolution-2", _chain(transform, upscale)))
    transforms.extend(
        [
            Transform("repeat-height-2", _repeat(0)),
            Transform("repeat-width-2", _repeat(1)),
            Transform("repeat-both-2", _repeat(None)),
        ]
    )
    return transforms


def _transformed_pair(pair: Pair, transform: Transform, seed: int) -> Pair:
    # Reinitializing the RNG gives every grid in a task the same stochastic
    # transform, matching the shared-RNG behavior of the paper's implementation.
    return {
        "input": _apply_grid(pair["input"], transform, random.Random(seed)),
        "output": _apply_grid(pair["output"], transform, random.Random(seed)),
    }


def _fits_arc_shape(pairs: Iterable[Pair]) -> bool:
    for pair in pairs:
        for key in ("input", "output"):
            grid = pair[key]
            if not grid or not grid[0] or len(grid) > 30 or len(grid[0]) > 30:
                return False
    return True


def _color_permutation(pairs: list[Pair], rng: random.Random) -> list[Pair]:
    colors = list(range(10))
    shuffled = colors.copy()
    rng.shuffle(shuffled)
    mapping = dict(zip(colors, shuffled))
    return [
        {
            key: [[mapping[value] for value in row] for row in pair[key]]
            for key in ("input", "output")
        }
        for pair in pairs
    ]


def _sequence_key(examples: list[Pair], question: Pair) -> str:
    return json.dumps([examples, question], separators=(",", ":"), sort_keys=True)


def adaptation_conversations(row: ArcRow, seed: int):
    """Build leave-one-out data, transformations, colors, and example shuffles."""
    values: list[tuple[str, list[dict[str, str]]]] = []
    seen: set[str] = set()
    for held_index, held_pair in enumerate(row.train):
        examples = [pair for i, pair in enumerate(row.train) if i != held_index]
        for transform_index, transform in enumerate(paper_transforms()):
            transform_seed = seed + held_index * 10_000 + transform_index
            transformed_examples = [
                _transformed_pair(pair, transform, transform_seed) for pair in examples
            ]
            transformed_question = _transformed_pair(held_pair, transform, transform_seed)
            all_pairs = transformed_examples + [transformed_question]
            if not _fits_arc_shape(all_pairs):
                continue
            variants = [(transform.name, transformed_examples, transformed_question)]
            permutation_rng = random.Random(transform_seed + 1_000_003)
            colored = _color_permutation(copy.deepcopy(all_pairs), permutation_rng)
            colored_examples, colored_question = colored[:-1], colored[-1]
            permutation_rng.shuffle(colored_examples)
            variants.append((f"{transform.name}+colors+example-order", colored_examples, colored_question))
            for name, variant_examples, variant_question in variants:
                key = _sequence_key(variant_examples, variant_question)
                if key not in seen:
                    seen.add(key)
                    values.append((name, conversation(variant_examples, variant_question)))
    return values


def tokenized_length(tokenizer, messages) -> int:
    return len(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            padding=False,
            truncation=False,
        )
    )


def encode_untruncated(tokenizer, message_batch, device: torch.device):
    lengths = [tokenized_length(tokenizer, messages) for messages in message_batch]
    if any(length > MAX_SEQUENCE_LENGTH for length in lengths):
        raise ValueError(
            f"sequence has {max(lengths)} tokens, limit is {MAX_SEQUENCE_LENGTH}"
        )
    # A fixed shape avoids recompilation as transformed grid sizes vary.
    return icl.encode(
        tokenizer,
        message_batch,
        MAX_SEQUENCE_LENGTH,
        device,
        padding="max_length",
    )


def _restore_inactive_states(model, frozen_states, active_rows: torch.Tensor) -> None:
    if frozen_states is None:
        return
    for frozen, current in zip(frozen_states, model.state_containers()):
        current.data.copy_(
            torch.where(active_rows[:, None, None], current, frozen)
        )


def make_batched_fns(model, args, device: torch.device):
    """Create task-parallel TTT and scoring functions.

    Losses are summed over batch rows so each row's fast-weight gradient has
    the same scale as batch-size-one TTT. Fast-weight tensors retain a separate
    leading batch row, so there is no cross-task adaptation.
    """
    is_piano = isinstance(model, PianoModel)

    def train_core(input_ids, assistant_mask, attention_mask, active_rows):
        with icl.autocast(device, args.dtype), torch.enable_grad():
            if is_piano:
                logits = model(
                    input_ids,
                    valid_mask=attention_mask,
                    mode=PianoMode.TRAIN_FIRST,
                    logits_to_keep=slice(0, -1),
                )[0]
            else:
                logits = model(input_ids, logits_to_keep=slice(0, -1))[0]
            _, output_losses, auxiliary_losses = icl.adaptation_loss(
                input_ids,
                assistant_mask,
                attention_mask,
                logits,
                args.aux_weight,
                return_aux=True,
            )
            row_losses = output_losses + args.aux_weight * auxiliary_losses
            loss = (row_losses * active_rows.float()).sum()
        loss.backward()

    if args.compile:
        train_core = torch.compile(train_core, fullgraph=False)

    def train_fn(input_ids, assistant_mask, attention_mask, active_rows):
        train_core(input_ids, assistant_mask, attention_mask, active_rows)

        # Shorter task horizons must stop changing while longer tasks finish.
        all_active = bool(active_rows.all().item())
        frozen_states = None if all_active else [
            state.detach().clone() for state in model.state_containers()
        ]
        if is_piano:
            model.update_state(PianoMode.TRAIN_FIRST)
        else:
            model.update_state()
        _restore_inactive_states(model, frozen_states, active_rows)

    def logits_fn(input_ids, attention_mask):
        with icl.autocast(device, args.dtype):
            if is_piano:
                return model(
                    input_ids,
                    valid_mask=attention_mask,
                    mode=PianoMode.INFERENCE,
                    logits_to_keep=slice(0, -1),
                )[0]
            return model(input_ids, logits_to_keep=slice(0, -1))[0]

    if args.compile:
        logits_fn = torch.compile(logits_fn, fullgraph=True)
    return train_fn, logits_fn


@torch.no_grad()
def score_batch(model, logits_fn, tokenizer, message_batch, device: torch.device):
    model.eval()
    input_ids, assistant_mask, attention_mask = encode_untruncated(
        tokenizer, message_batch, device
    )
    logits = logits_fn(input_ids, attention_mask)
    labels = input_ids[:, 1:]
    mask = assistant_mask[:, 1:] & attention_mask[:, 1:]
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).view_as(labels)
    predictions = logits.argmax(-1)
    token_counts = mask.sum(1)
    correct_counts = ((predictions == labels) & mask).sum(1)
    negative_log_likelihoods = (losses * mask).sum(1)
    exact = ((predictions == labels) | ~mask).all(1) & mask.any(1)
    model.train()
    metrics = []
    for batch_index in range(len(message_batch)):
        token_count = int(token_counts[batch_index].item())
        correct = int(correct_counts[batch_index].item())
        nll = float(negative_log_likelihoods[batch_index].item())
        metrics.append(
            {
                "loss": nll / max(token_count, 1),
                "next_token_accuracy": correct / max(token_count, 1),
                "exact_accuracy": float(exact[batch_index].item()),
                "output_tokens": token_count,
                "correct_tokens": correct,
                "negative_log_likelihood": nll,
                "sequence_tokens": tokenized_length(
                    tokenizer, message_batch[batch_index]
                ),
            }
        )
    return metrics


def write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(payload.items())
    if items and items[0][0] == "model_config_overrides":
        override_key, overrides = items[0]
        compact_overrides = json.dumps(
            overrides, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        remaining = json.dumps(dict(items[1:]), indent=2, ensure_ascii=False)
        body = remaining[2:-2]
        text = (
            "{\n  " + json.dumps(override_key) + ": " + compact_overrides
            + (",\n" + body if body else "") + "\n}\n"
        )
    else:
        text = json.dumps(payload, indent=4, ensure_ascii=False) + "\n"
    path.write_text(text)


def _safe_path_component(value: str) -> str:
    raw = str(value)
    safe = re.sub(r"[^A-Za-z0-9._=-]+", "-", raw).strip(".-")
    if not safe:
        safe = "value"
    if safe != raw or safe in {".", ".."} or len(safe) > 160:
        digest = hashlib.sha256(raw.encode()).hexdigest()[:10]
        safe = f"{safe[:145]}--{digest}"
    return safe


def model_kwargs_filename(model_kwargs: dict) -> str:
    if not model_kwargs:
        return "default.json"
    parts = []
    changed = False
    for key, value in sorted(model_kwargs.items()):
        raw_value = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        raw_part = f"{key}={raw_value}"
        safe_part = re.sub(r"[^A-Za-z0-9._=-]+", "-", raw_part).strip(".-")
        changed = changed or safe_part != raw_part
        parts.append(safe_part or "value")
    label = "__".join(parts)
    if changed or len(label) > 180:
        canonical = json.dumps(
            model_kwargs, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:10]
        label = f"{label[:165]}--{digest}"
    return f"{label}.json"


def default_result_path(
    output_root: Path,
    checkpoint_url: str,
    checkpoint_step: int,
    model_kwargs: dict,
) -> Path:
    if checkpoint_step is None:
        raise ValueError("checkpoint step is required for the result path")
    checkpoint_name = _safe_path_component(checkpoint_url.replace("/", "--"))
    return (
        output_root
        / checkpoint_name
        / f"{int(checkpoint_step):012d}"
        / model_kwargs_filename(model_kwargs)
    )


def summarize(results: list[dict]) -> dict:
    output_tokens = sum(row["output_tokens"] for row in results)
    return {
        "rows": len(results),
        "test_loss": sum(row["negative_log_likelihood"] for row in results) / max(output_tokens, 1),
        "next_token_accuracy": sum(row["correct_tokens"] for row in results) / max(output_tokens, 1),
        "exact_accuracy": sum(row["exact_accuracy"] for row in results) / max(len(results), 1),
        "output_tokens": output_tokens,
    }


def summarize_by_dataset(results: list[dict]) -> dict[str, dict]:
    datasets = sorted({row["dataset"] for row in results})
    return {
        dataset: summarize([row for row in results if row["dataset"] == dataset])
        for dataset in datasets
    }


def main() -> None:
    args = parse_args()
    if args.task_count is not None and args.task_count < 1:
        raise ValueError("--task-count must be positive")
    if args.ttt_steps < 0:
        raise ValueError("--ttt-steps must be non-negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not isinstance(args.model_kwargs, dict):
        raise ValueError("--model-kwargs must decode to a JSON object")
    invalid_override_keys = [
        key for key in args.model_kwargs
        if not isinstance(key, str) or not key or key.startswith("_")
    ]
    if invalid_override_keys:
        raise ValueError(f"invalid model config override keys: {invalid_override_keys!r}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = icl.load_tokenizer(args.tokenizer)

    expanded_rows = []
    for data_root in args.data_root:
        expanded_rows.extend(load_rows(data_root, args.splits))
    eligible_rows = []
    skipped_test_sequences = 0
    for row in expanded_rows:
        messages = conversation(row.train, row.test)
        if tokenized_length(tokenizer, messages) <= MAX_SEQUENCE_LENGTH:
            eligible_rows.append(row)
        else:
            skipped_test_sequences += 1
    length_eligible_rows = eligible_rows.copy()
    if args.task_count is not None:
        eligible_rows = eligible_rows[: args.task_count]
    print(
        f"Loaded {len(expanded_rows)} ARC-AGI-1/2 post-duplication rows; "
        f"{len(eligible_rows)} selected with <= {MAX_SEQUENCE_LENGTH} tokens "
        f"({skipped_test_sequences} over-length rows filtered)."
    )

    load_args = argparse.Namespace(
        dtype=args.dtype,
        aux_weight=args.aux_weight,
        compile=args.compile,
    )
    model, checkpoint_url, checkpoint_step = icl.load_fresh_model(
        args.fresh_config,
        args.base_lr,
        args.checkpoint_step,
        device,
        return_info=True,
        config_overrides=args.model_kwargs,
    )
    output_path = args.output or default_result_path(
        args.output_root, checkpoint_url, checkpoint_step, args.model_kwargs
    )
    # load_fresh_model enables activation/gradient checkpointing before TTT.
    train_fn, logits_fn = make_batched_fns(model, load_args, device)

    results = []
    filtered_adaptation_sequences = 0
    filtered_adaptation_by_dataset: dict[str, int] = {}
    started = time.time()
    payload = None
    for batch_start in tqdm(
        range(0, len(eligible_rows), args.batch_size), desc="ARC batches"
    ):
        batch = eligible_rows[batch_start:batch_start + args.batch_size]
        model.init_state(len(batch), device)
        model.empty_state()

        batch_started = time.time()
        all_candidates = []
        all_selected = []
        for row in batch:
            row_seed = args.seed + int(
                hashlib.sha256(
                    f"{row.dataset}-{row.task_id}-{row.test_index}".encode()
                ).hexdigest()[:8],
                16,
            )
            candidates = adaptation_conversations(row, row_seed)
            candidates_under_limit = []
            for transform_name, messages in candidates:
                if tokenized_length(tokenizer, messages) <= MAX_SEQUENCE_LENGTH:
                    candidates_under_limit.append((transform_name, messages))
                else:
                    filtered_adaptation_sequences += 1
                    filtered_adaptation_by_dataset[row.dataset] = (
                        filtered_adaptation_by_dataset.get(row.dataset, 0) + 1
                    )
            random.Random(row_seed).shuffle(candidates_under_limit)
            all_candidates.append(candidates_under_limit)
            all_selected.append(candidates_under_limit[: args.ttt_steps])

        max_steps = max((len(selected) for selected in all_selected), default=0)
        fallback_messages = [conversation(row.train, row.test) for row in batch]
        for step in range(max_steps):
            active = [step < len(selected) for selected in all_selected]
            message_batch = [
                selected[step][1] if is_active else fallback
                for selected, is_active, fallback in zip(
                    all_selected, active, fallback_messages
                )
            ]
            input_ids, assistant_mask, attention_mask = encode_untruncated(
                tokenizer, message_batch, device
            )
            with torch.enable_grad():
                train_fn(
                    input_ids,
                    assistant_mask,
                    attention_mask,
                    torch.tensor(active, device=device, dtype=torch.bool),
                )

        batch_metrics = score_batch(
            model, logits_fn, tokenizer, fallback_messages, device
        )
        batch_elapsed = time.time() - batch_started
        for row, candidates, selected, metrics in zip(
            batch, all_candidates, all_selected, batch_metrics
        ):
            results.append(
                {
                    "dataset": row.dataset,
                    "split": row.split,
                    "task_id": row.task_id,
                    "test_index": row.test_index,
                    "available_ttt_sequences": len(candidates),
                    "ttt_steps": len(selected),
                    "ttt_transforms": [name for name, _ in selected],
                    "batch_size": len(batch),
                    "batch_elapsed_seconds": batch_elapsed,
                    **metrics,
                }
            )
        payload = {
            "model_config_overrides": args.model_kwargs,
            "config": {
                "model_config": args.fresh_config,
                "checkpoint_url": checkpoint_url,
                "checkpoint_step": checkpoint_step,
                "splits": args.splits,
                "data_roots": [str(path) for path in args.data_root],
                "task_count": args.task_count,
                "ttt_steps": args.ttt_steps,
                "batch_size": args.batch_size,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "gradient_checkpointing": True,
                "aux_weight": args.aux_weight,
                "seed": args.seed,
                "grid_format": "single-digit cells, newline-delimited rows",
            },
            "data": {
                "post_duplication_rows": len(expanded_rows),
                "selected_rows": len(eligible_rows),
                "filtered_test_sequences": skipped_test_sequences,
                "filtered_adaptation_sequences": filtered_adaptation_sequences,
                "by_dataset": {
                    dataset: {
                        "post_duplication_rows": sum(
                            row.dataset == dataset for row in expanded_rows
                        ),
                        "selected_rows": sum(
                            row.dataset == dataset for row in eligible_rows
                        ),
                        "filtered_test_sequences": sum(
                            row.dataset == dataset for row in expanded_rows
                        ) - sum(
                            row.dataset == dataset for row in length_eligible_rows
                        ),
                        "filtered_adaptation_sequences": (
                            filtered_adaptation_by_dataset.get(dataset, 0)
                        ),
                    }
                    for dataset in sorted({row.dataset for row in expanded_rows})
                },
            },
            "summary": summarize(results),
            "summary_by_dataset": summarize_by_dataset(results),
            "results": results,
            "elapsed_seconds": time.time() - started,
        }
        write_results(output_path, payload)
        model.zero_grad(set_to_none=True)
        model.empty_state()

    if payload is None:
        payload = {
            "model_config_overrides": args.model_kwargs,
            "config": {
                "model_config": args.fresh_config,
                "checkpoint_url": checkpoint_url,
                "checkpoint_step": checkpoint_step,
                "splits": args.splits,
                "data_roots": [str(path) for path in args.data_root],
                "task_count": args.task_count,
                "ttt_steps": args.ttt_steps,
                "batch_size": args.batch_size,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "gradient_checkpointing": True,
                "aux_weight": args.aux_weight,
                "seed": args.seed,
                "grid_format": "single-digit cells, newline-delimited rows",
            },
            "data": {
                "post_duplication_rows": len(expanded_rows),
                "selected_rows": 0,
                "filtered_test_sequences": skipped_test_sequences,
                "filtered_adaptation_sequences": 0,
                "by_dataset": {
                    dataset: {
                        "post_duplication_rows": sum(
                            row.dataset == dataset for row in expanded_rows
                        ),
                        "selected_rows": 0,
                        "filtered_test_sequences": sum(
                            row.dataset == dataset for row in expanded_rows
                        ) - sum(
                            row.dataset == dataset for row in length_eligible_rows
                        ),
                        "filtered_adaptation_sequences": 0,
                    }
                    for dataset in sorted({row.dataset for row in expanded_rows})
                },
            },
            "summary": summarize([]),
            "summary_by_dataset": {},
            "results": [],
            "elapsed_seconds": time.time() - started,
        }
        write_results(output_path, payload)
    print(json.dumps(payload["summary_by_dataset"], indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
