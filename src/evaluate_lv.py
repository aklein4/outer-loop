"""Evaluate stateful OLoop and Piano models on the English LV-Eval tasks.

LV-Eval contexts are adapted as the same ordered, single-turn chunk-pair
horizons used by :mod:`evaluate_standard_long_context`. Terminal answers are
scored token-by-token from teacher-forced logits, matching the accuracy-oriented
RULER evaluation in the standard long-context evaluator.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import evaluate_icl as icl
from evaluate_standard_long_context import (
    Chunker,
    adapt_documents_padded,
    format_horizon_losses,
    make_compiled_fns,
)
import utils.constants as constants


DEVICE = constants.DEVICE

LV_EVAL_REPO = "Infinigence/LVEval"
# Pin the released data so cached and fresh runs use identical examples.
LV_EVAL_REVISION = "86a3b0e6f2266281d481bcb46a0bda5b511cffb0"

ENGLISH_DATASETS = (
    "hotpotwikiqa_mixup",
    "loogle_SD_mixup",
    "loogle_CR_mixup",
    "loogle_MIR_mixup",
    "multifieldqa_en_mixup",
    "factrecall_en",
)
LENGTH_LEVELS = ("16k", "32k", "64k", "128k", "256k")

# Number of rows in every length level.  LV-Eval reuses the same QA pairs at
# all five lengths, so these counts are constant across levels.
DATASET_COUNTS = {
    "hotpotwikiqa_mixup": 124,
    "loogle_SD_mixup": 160,
    "loogle_CR_mixup": 99,
    "loogle_MIR_mixup": 139,
    "multifieldqa_en_mixup": 101,
    "factrecall_en": 200,
}

# These are the terminal portions of LV-Eval's released prompts.  The article
# itself is omitted here because it has already been consumed by adaptation.
PASSAGE_QUESTION = (
    "Please answer the following question based on the above passages. "
    "Questions and answers are only relevant to one passage. Only give me the "
    "answer and do not output any other explanation and evidence.\n\n"
    "Question: {input}\nAnswer:"
)
DATASET_QUESTION_PROMPTS = {
    "hotpotwikiqa_mixup": (
        "Please answer the following question based on the above passages. "
        "Questions and answers are only relevant to some passages. Only give "
        "me the answer and do not output any other explanation and evidence.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "loogle_SD_mixup": PASSAGE_QUESTION,
    "loogle_CR_mixup": PASSAGE_QUESTION,
    "loogle_MIR_mixup": PASSAGE_QUESTION,
    "multifieldqa_en_mixup": PASSAGE_QUESTION,
    "factrecall_en": (
        "Please answer the following question based on the above article.\n\n"
        "Question: {input}\nAnswer:"
    ),
}

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--fresh-config", required=True)
    p.add_argument("--aux-weight", type=float, required=True)
    p.add_argument("--tokenizer", default=icl.DEFAULT_TOKENIZER)
    p.add_argument("--save-name", default=None)
    p.add_argument(
        "--datasets", nargs="+", choices=ENGLISH_DATASETS,
        default=list(ENGLISH_DATASETS),
    )
    p.add_argument(
        "--length-levels", nargs="+", choices=LENGTH_LEVELS,
        default=list(LENGTH_LEVELS),
    )
    p.add_argument(
        "--max-samples", "--max-samples-per-dataset",
        dest="max_samples", type=int, default=None,
        help="Maximum examples per dataset and length level (for smoke tests).",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument("--compile", action="store_true")
    args = p.parse_args(argv)

    if args.batch_size <= 0:
        p.error("--batch-size must be positive")
    if args.max_length <= 1:
        p.error("--max-length must be greater than 1")
    if args.max_samples is not None and args.max_samples <= 0:
        p.error("--max-samples must be positive")
    return args


def _archive_member(dataset: str, length_level: str) -> str:
    return f"{dataset}/{dataset}_{length_level}.jsonl"


def load_lv_rows(dataset: str, length_level: str, max_samples=None, seed=42):
    """Load one LV-Eval split directly from its pinned official ZIP archive."""
    archive = Path(hf_hub_download(
        repo_id=LV_EVAL_REPO,
        filename=f"{dataset}.zip",
        repo_type="dataset",
        revision=LV_EVAL_REVISION,
    ))
    member = _archive_member(dataset, length_level)
    with zipfile.ZipFile(archive) as zf:
        try:
            with zf.open(member) as stream:
                rows = [json.loads(line) for line in stream]
        except KeyError as exc:
            raise FileNotFoundError(
                f"{member} not found in LV-Eval archive {archive}"
            ) from exc

    if max_samples is not None and len(rows) > max_samples:
        # A string seed remains stable across Python processes, unlike hash().
        rng = random.Random(f"{seed}:{dataset}:{length_level}")
        indices = sorted(rng.sample(range(len(rows)), max_samples))
        rows = [rows[index] for index in indices]
    return rows


def _stop_token_ids(tokenizer: PreTrainedTokenizer) -> set[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(token_id) for token_id in eos}


def _prompt_tokens(tokenizer: PreTrainedTokenizer, prompt: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        padding=False,
        truncation=False,
        return_dict=False,
    )


def _completion_tokens(
    tokenizer: PreTrainedTokenizer, prompt: str, answer: str
) -> tuple[list[int], list[int]]:
    """Tokenize an answer as the exact assistant continuation of a prompt."""
    prompt_tokens = _prompt_tokens(tokenizer, prompt)
    full_tokens = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        tokenize=True,
        add_generation_prompt=False,
        padding=False,
        truncation=False,
        return_dict=False,
    )
    if full_tokens[:len(prompt_tokens)] == prompt_tokens:
        answer_tokens = full_tokens[len(prompt_tokens):]
        stop_tokens = _stop_token_ids(tokenizer)
        while answer_tokens and answer_tokens[-1] in stop_tokens:
            answer_tokens.pop()
    else:
        # This fallback covers chat templates whose generation prefix differs
        # from their serialized assistant prefix.
        answer_tokens = tokenizer(answer, add_special_tokens=False).input_ids
    return prompt_tokens, answer_tokens


def score_gold_answer(
    logits_fn,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    answer: str,
    max_length: int,
):
    """Teacher-force the first official answer for comparable accuracy stats."""
    prompt_tokens, answer_tokens = _completion_tokens(tokenizer, prompt, answer)
    if not answer_tokens:
        raise ValueError("LV-Eval answer tokenized to an empty sequence")
    if len(prompt_tokens) + len(answer_tokens) > max_length:
        raise ValueError(
            "LV-Eval question and answer exceed --max-length: "
            f"{len(prompt_tokens)} + {len(answer_tokens)} > {max_length}"
        )

    input_ids = torch.tensor(
        [prompt_tokens + answer_tokens], device=DEVICE, dtype=torch.long
    )
    logits = logits_fn(input_ids, slice(len(prompt_tokens) - 1, -1)).float()
    targets = input_ids[:, len(prompt_tokens):]
    losses = F.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), reduction="none"
    )
    correct = logits.argmax(-1).eq(targets).flatten()
    return {
        "exact_acc": float(correct.all().item()),
        "correct_tokens": int(correct.sum().item()),
        "answer_tokens": int(correct.numel()),
        "token_loss_sum": float(losses.sum().item()),
    }


def _aggregate(records):
    count = len(records)
    answer_tokens = sum(record["answer_tokens"] for record in records)
    if count == 0:
        return {
            "exact_acc": 0.0,
            "token_acc": 0.0,
            "token_loss": 0.0,
            "count": 0,
            "answer_tokens": 0,
        }
    return {
        "exact_acc": sum(r["exact_acc"] for r in records) / count,
        "token_acc": (
            sum(r["correct_tokens"] for r in records) / answer_tokens
            if answer_tokens else 0.0
        ),
        "token_loss": (
            sum(r["token_loss_sum"] for r in records) / answer_tokens
            if answer_tokens else 0.0
        ),
        "count": count,
        "answer_tokens": answer_tokens,
    }


def _aggregate_views(records, datasets, length_levels):
    by_dataset = defaultdict(dict)
    for dataset in datasets:
        for length_level in length_levels:
            selected = [
                record for record in records
                if record["dataset"] == dataset
                and record["length_level"] == length_level
            ]
            by_dataset[dataset][length_level] = _aggregate(selected)

    by_length = {
        length_level: _aggregate([
            record for record in records
            if record["length_level"] == length_level
        ])
        for length_level in length_levels
    }
    table = {
        dataset: {
            length_level: by_dataset[dataset][length_level]["exact_acc"]
            for length_level in length_levels
        }
        for dataset in datasets
    }
    return by_dataset, by_length, table


def _result_path(args, info) -> Path:
    save_path = constants.LOCAL_DATA_PATH / "lv_eval_results"
    if args.save_name is not None:
        save_path /= args.save_name
    save_path /= info["checkpoint_url"].replace("/", "--")
    save_path.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"aux={str(round(args.aux_weight, 2)).replace('.', 'p')}_"
        f"{info['checkpoint_step']:012d}.json"
    )
    return save_path / file_name


def _build_results(
    model, args, records, assistant_losses, aux_losses, started
):
    by_dataset, by_length, table = _aggregate_views(
        records, args.datasets, args.length_levels
    )
    return {
        "evaluation_protocol": {
            "benchmark": "LV-Eval English",
            "data": f"{LV_EVAL_REPO}@{LV_EVAL_REVISION}",
            "context_handling": "ordered_chunk_pair_test_time_training",
            "decoding": "teacher_forced_gold_prefix_top1",
            "sequential_sampling_rollouts": False,
            "primary_metric": "exact_acc",
            "score_scale": "0_to_1",
            "comparison_note": (
                "exact_acc requires every gold answer token to be top-1 under "
                "its gold prefix; official LV-Eval reports metrics on decoded strings"
            ),
        },
        "total_evaluation_time": time.time() - started,
        "overall": _aggregate(records),
        "by_dataset": by_dataset,
        "by_length": by_length,
        "lv_eval_table": table,
        "horizon": format_horizon_losses(assistant_losses, aux_losses),
        "args": vars(args),
        "model_config": omegaconf.OmegaConf.to_container(
            model.config, resolve=True
        ),
    }


def evaluate_lv(model, train_fn, logits_fn, tokenizer, chunker, args, info):
    assistant_loss_accumulator = defaultdict(list)
    aux_loss_accumulator = defaultdict(list)
    records = []
    started = time.time()

    total = sum(
        min(DATASET_COUNTS[dataset], args.max_samples or DATASET_COUNTS[dataset])
        for dataset in args.datasets
        for _ in args.length_levels
    )
    progress = tqdm(total=total, desc="LV-Eval English")

    for dataset in args.datasets:
        prompt_template = DATASET_QUESTION_PROMPTS[dataset]
        for length_level in args.length_levels:
            rows = load_lv_rows(
                dataset, length_level,
                max_samples=args.max_samples,
                seed=args.seed,
            )
            for index_start in range(0, len(rows), args.batch_size):
                batch_rows = rows[index_start:index_start + args.batch_size]
                adapt_documents_padded(
                    train_fn, model, tokenizer, chunker,
                    [row["context"] for row in batch_rows],
                    args,
                    assistant_loss_accumulator,
                    aux_loss_accumulator,
                )
                frozen_states = [
                    state.detach().clone() for state in model.state_containers()
                ]

                for batch_index, row in enumerate(batch_rows):
                    for module_index, module in enumerate(model.fast_modules()):
                        module.state = frozen_states[module_index][batch_index][None]

                    answers = row.get("answers") or []
                    if not answers:
                        raise ValueError(
                            f"LV-Eval row has no answers: {dataset}/{length_level}"
                        )
                    answer = str(answers[0])
                    prompt = prompt_template.format(input=row["input"])
                    gold_score = score_gold_answer(
                        logits_fn, tokenizer, prompt, answer, args.max_length
                    )
                    records.append({
                        "dataset": dataset,
                        "length_level": length_level,
                        **gold_score,
                    })

                results = _build_results(
                    model, args, records,
                    assistant_loss_accumulator, aux_loss_accumulator,
                    started,
                )
                output_path = _result_path(args, info)
                with open(output_path, "w") as stream:
                    json.dump(results, stream, indent=4)
                print(f"\nresults written to: {output_path}\n")
                progress.update(len(batch_rows))

    progress.close()


@torch.no_grad()
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, model_url, model_step = icl.load_fresh_model(
        args.fresh_config, None, None, DEVICE, True
    )
    info = {
        "checkpoint_url": model_url,
        "checkpoint_step": model_step,
    }
    tokenizer = icl.load_tokenizer(args.tokenizer)
    chunker = Chunker(args.tokenizer)
    train_fn, logits_fn = make_compiled_fns(model, args, dynamic=True)
    evaluate_lv(
        model, train_fn, logits_fn,
        tokenizer, chunker, args, info,
    )


if __name__ == "__main__":
    main()
