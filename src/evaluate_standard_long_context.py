"""Comparable QuALITY accuracy and RULER NIAH retrieval evaluation.

Long documents are adapted as ordered, single-turn chunk-pair horizons.  The
terminal QuALITY question is scored as a one-token 1/2/3/4 classification.
RULER answer strings are scored token-by-token from teacher-forced logits.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import argparse
import json
import runpy
import subprocess
import sys
import time
from collections import defaultdict
import os
from pathlib import Path
from tqdm import tqdm
import hashlib
import numpy as np
import omegaconf

import datasets
import semchunk
from transformers import PreTrainedTokenizer

import evaluate_icl as icl
from models.piano import PianoMode, PianoModel
import utils.constants as constants


DEVICE = constants.DEVICE

QUALITY_DS = {
    "path": "emozilla/quality",
    "split": "validation"
}

RULER_REPO = "https://github.com/NVIDIA/RULER.git"
# RULERv1 generator from before RULERv2, at the commit that removed its
# incidental NeMo dependency. Pinning it makes cached sample sets reproducible.
RULER_REVISION = "6c1e0a0b5c0c046ffd8f0f4766701bde2e96507e"

RULER_NIAH_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
)
RULER_ESSAY_TASKS = {
    "niah_single_2", "niah_single_3", "niah_multikey_1",
    "niah_multivalue", "niah_multiquery",
}

E2E_RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
)
E2E_RULER_CONTEXT_LENGTHS = (8192, 16384, 32768, 65536, 131072)
E2E_RULER_LABELS = {
    "niah_single_1": "S-NIAH-1",
    "niah_single_2": "S-NIAH-2",
    "niah_single_3": "S-NIAH-3",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fresh-config", required=True)
    p.add_argument("--benchmark", choices=["quality", "ruler"], required=True)
    p.add_argument("--aux-weight", type=float, required=True)
    p.add_argument("--tokenizer", default=icl.DEFAULT_TOKENIZER)
    p.add_argument("--save-name", default=None)
    p.add_argument("--max-articles", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--ruler-context-lengths", type=int, nargs="+",
        default=list(E2E_RULER_CONTEXT_LENGTHS),
    )
    p.add_argument(
        "--ruler-tasks", nargs="+", choices=RULER_NIAH_TASKS,
        default=list(E2E_RULER_TASKS)
    )
    p.add_argument(
        "--ruler-num-samples", type=int, default=500,
        help="Samples per RULER task and context length (official default: 500).",
    )
    return p.parse_args()


def masked_update(model, mask, *args, **kwargs):
    frozen = [x.detach().clone() for x in model.state_containers()]
    model.update_state(*args, **kwargs)
    for frozen_tensor, current_tensor in zip(frozen, model.state_containers()):
        current_tensor.data.copy_(
            torch.where(mask[:, None, None], current_tensor, frozen_tensor)
        )


def make_compiled_fns(model, args, dynamic=False):
    is_piano = isinstance(model, PianoModel)

    def train_fn(input_ids, assistant_mask, attention_mask, active_rows):

        with icl.autocast(DEVICE, args.dtype), torch.enable_grad():
            if is_piano:
                logits = model(
                    input_ids,
                    valid_mask=attention_mask,
                    mode=PianoMode.TRAIN_FIRST,
                    logits_to_keep=slice(0, -1),
                )[0]
            else:
                logits = model(input_ids, logits_to_keep=slice(0, -1))[0]

            loss, output_loss, aux_loss = icl.adaptation_loss(input_ids, assistant_mask, attention_mask, logits, args.aux_weight, return_aux=True)

        loss.backward()

        if is_piano:
            masked_update(model, active_rows, PianoMode.TRAIN_FIRST)
        else:
            masked_update(model, active_rows)

        return output_loss, aux_loss

        
    def logits_fn(input_ids, logits_to_keep=None):
        with icl.autocast(DEVICE, args.dtype):
            if is_piano:
                return model(
                    input_ids,
                    valid_mask=torch.ones_like(input_ids, dtype=torch.bool),
                    mode=PianoMode.INFERENCE,
                    logits_to_keep=logits_to_keep,
                )[0]

            return model(
                input_ids,
                logits_to_keep=logits_to_keep
            )[0]

    if args.compile:
        return (
            torch.compile(train_fn, fullgraph=False, dynamic=dynamic),
            torch.compile(logits_fn, fullgraph=True, dynamic=dynamic),
        )
    return (
        train_fn,
        logits_fn,
    )


def format_horizon_losses(
    assistant_loss_accumulator, aux_loss_accumulator
):
    n = len(assistant_loss_accumulator)
    assert set(assistant_loss_accumulator.keys()) == set(aux_loss_accumulator.keys())
    assert set(assistant_loss_accumulator.keys()) == set(range(n))
    for i in range(n):
        assert len(assistant_loss_accumulator[i]) > 0, f"Assistant loss accumulator for index {i} is empty"
        assert len(assistant_loss_accumulator[i]) == len(aux_loss_accumulator[i]), f"Mismatch in lengths for index {i}"

    assistant_losses = [
        torch.stack(assistant_loss_accumulator[i]).mean().item() for i in range(n)
    ]
    aux_losses = [
        torch.stack(aux_loss_accumulator[i]).mean().item() for i in range(n)
    ]

    return {
        "assistant_losses": assistant_losses,
        "aux_losses": aux_losses,
        "counts": [len(assistant_loss_accumulator[i]) for i in range(n)]
    }


def encode_horizons(tokenizer, conversations, max_length):
    bs = len(conversations)
    horizon_length = max(len(c) for c in conversations)

    horizons = [
        [
            tuple(x[0] for x in icl.encode(tokenizer, c, max_length, device=DEVICE, padding="max_length"))
            for c in conv
        ] for conv in conversations
    ]

    mask = torch.zeros(horizon_length, bs, device=DEVICE, dtype=torch.bool)
    for i, h in enumerate(horizons):
        mask[:len(h), i] = True

    default_episode = tuple(
        torch.zeros_like(x) for x in horizons[0][0]
    )
    for h in horizons:
        if len(h) < horizon_length:
            h.extend([default_episode] * (horizon_length - len(h)))

    # reshape to (horizon_length, batch_size, max_length)
    batch = []
    for i in range(len(default_episode)):
        batch.append(
            torch.stack(
                [
                    torch.stack(
                        [horizons[b][t][i] for b in range(bs)]
                    )
                    for t in range(horizon_length)
                ]
            )
        )

    for b in batch:
        assert b.shape == (horizon_length, bs, max_length), f"Expected {(horizon_length, bs, max_length)}, got {b.shape}"
    assert mask.shape == (horizon_length, bs), f"Expected {(horizon_length, bs)}, got {mask.shape}"

    return tuple(batch), mask


def adapt_documents_padded(
    train_fn, model: PianoModel, tokenizer, chunker,
    documents,
    args,
    assistant_loss_accumulator, aux_loss_accumulator,
):
    bs = len(documents)
    model.init_state(bs, DEVICE)

    conversations = chunker(documents)
    horizon_length = max(len(c) for c in conversations)

    packed, mask = encode_horizons(tokenizer, conversations, max_length=args.max_length)
    cpu_mask = mask.cpu()

    for t in tqdm(range(horizon_length), "adapting", leave=False):

        assistant_loss, aux_loss = train_fn(
            packed[0][t], packed[1][t], packed[2][t],
            mask[t],
        )

        for b in range(bs):
            if cpu_mask[t, b]:
                assistant_loss_accumulator[t].append(assistant_loss[b].detach())
                aux_loss_accumulator[t].append(aux_loss[b].detach())


def single_turn(user: str, assistant: str):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


class Chunker:

    _chunk_size = 448 # from latent data handlers

    def __init__(self, tokenizer_url):
        self._chunker = semchunk.chunkerify(
            tokenizer_url,
            chunk_size=self._chunk_size,
        )


    def __call__(self, text):
        if not isinstance(text, str):
            return [self(t) for t in text]

        chunks = self._chunker(text.strip())

        turns = [
            single_turn(chunks[i-1].strip(), chunks[i].strip())
            for i in range(1, len(chunks), 2)
        ]

        if len(chunks) % 2 == 0:
            return turns

        last_chunk = chunks[-1]
        last_turn = single_turn(
            last_chunk[:len(last_chunk)//2].strip(),
            last_chunk[len(last_chunk)//2:].strip()
        )

        return turns + [last_turn]


def quality_prompt(row):
    options = "\n".join(f"{i+1}. {option.strip()}" for i, option in enumerate(row["options"]))
    return (
        f"Question: {row['question']}\n\n{options}\n\n"
        "Start your response with the number corresponding to your answer: 1, 2, 3, or 4."
    )

def quality_options(row):
    n = len(row["options"])
    return (
        tuple(str(i) for i in range(1,n+1)),
        tuple(f"{i+1}. {option.strip()}" for i, option in enumerate(row["options"]))
    )


def get_quality_results(model, logits_fn, tokenizer: PreTrainedTokenizer, row, args):
    prompt = quality_prompt(row)
    single_options, full_options = quality_options(row)

    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        padding=False,
        truncation=False,
        return_dict=False
    )
    n_prompt = len(prompt_tokens)

    single_tokens = tokenizer(
        single_options, add_special_tokens=False
    ).input_ids
    assert all(len(t)==1 for t in single_tokens)

    full_tokens = tokenizer(
        full_options, add_special_tokens=False
    ).input_ids
    assert all(len(t)>=1 for t in full_tokens)

    full_input_ids = [
        torch.tensor(prompt_tokens + t, device=DEVICE, dtype=torch.long)
        for t in full_tokens
    ]

    for ids in full_input_ids:
        if len(ids) > args.max_length:
            return None

    input_ids = torch.nn.utils.rnn.pad_sequence(
        full_input_ids,
        batch_first=True,
        padding_value=-1,
    )
    mask = (input_ids != -1)
    input_ids = torch.where(mask, input_ids, 0)
    mask = mask.float()

    logits = logits_fn(input_ids, slice(n_prompt-1,-1))

    single_logits = logits[0, 0]
    single_acc = float(single_logits[[t[0] for t in single_tokens]].argmax(-1).item() == row["answer"])

    logp = -F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        input_ids[:, n_prompt:].reshape(-1),
        reduction='none'
    ).reshape(input_ids[:, n_prompt:].shape)

    logp = (logp * mask[:, n_prompt:]).sum(-1)
    logp_norm = logp / mask[:, n_prompt:].sum(-1).clamp_min(1)

    full_acc = float((logp.argmax(-1).item() == row["answer"]))
    full_acc_norm = float((logp_norm.argmax(-1).item() == row["answer"]))

    return {
        "single_acc": single_acc,
        "full_acc": full_acc,
        "full_acc_norm": full_acc_norm
    }


def hash(text):
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def evaluate_quality(
    model, train_fn, logits_fn, tokenizer, chunker, args, info
):
    ds = datasets.load_dataset(**QUALITY_DS)
    ds = [row for row in ds]

    hashes = set()
    rows = defaultdict(list)
    articles = defaultdict(list)
    for row in ds:
        k = hash(row["article"])
        hashes.add(k)
        rows[k].append(row)
        articles[k] = row["article"].strip()

    hashes = list(hashes)
    if args.max_articles is not None:
        hashes = np.random.choice(
            hashes,
            size=min(args.max_articles, len(hashes)),
            replace=False
        )
    hashes = sorted(hashes, key=lambda x: len(articles[x]), reverse=True)
    rows = {k: rows[k] for k in hashes}
    articles = {k: articles[k] for k in hashes}

    assistant_loss_accumulator = defaultdict(list)
    aux_loss_accumulator = defaultdict(list)

    easy_results = defaultdict(list)
    easy_count = 0
    hard_results = defaultdict(list)
    hard_count = 0

    started = time.time()
    progress = tqdm(total=len(hashes), desc="QuALITY articles")
    for index_start in range(0, len(hashes), args.batch_size):

        curr_hashes = hashes[index_start:index_start + args.batch_size]
        curr_articles = [articles[k] for k in curr_hashes]

        adapt_documents_padded(
            train_fn, model, tokenizer, chunker,
            curr_articles,
            args,
            assistant_loss_accumulator,
            aux_loss_accumulator,
        )

        frozen_state_containers = [
            s.detach().clone() for s in model.state_containers()
        ]

        for i in tqdm(range(len(curr_hashes)), "answering", leave=False):

            for l, m in enumerate(model.fast_modules()):
                m.state = frozen_state_containers[l][i][None]

            for row in rows[curr_hashes[i]]:
                row_result = get_quality_results(model, logits_fn, tokenizer, row, args)

                if row_result is not None:
                    hard = row["hard"]

                    for k in row_result:
                        if hard:
                            hard_results[k].append(row_result[k])
                        else:
                            easy_results[k].append(row_result[k])

                    if hard:
                        hard_count += 1
                    else:
                        easy_count += 1

        progress.update(len(curr_hashes))
                
    horizon_losses = format_horizon_losses(
        assistant_loss_accumulator,
        aux_loss_accumulator,
    )

    all_results = {}
    if set(easy_results.keys()) == set(hard_results.keys()):
        all_results = {k: np.mean(easy_results[k] + hard_results[k]) for k in easy_results.keys()}

    easy_results = {k: np.mean(v) for k, v in easy_results.items()}
    hard_results = {k: np.mean(v) for k, v in hard_results.items()}

    results = {
        "total_evaluation_time": time.time() - started,
        "all": all_results | {"count": hard_count + easy_count if len(all_results) > 0 else 0},
        "hard": hard_results | {"count": hard_count},
        "easy": easy_results | {"count": easy_count},
        "horizon": horizon_losses,
        "args": vars(args),
        "model_config": omegaconf.OmegaConf.to_container(model.config, resolve=True),
    }

    save_path = constants.LOCAL_DATA_PATH / "quality_results" 
    if args.save_name is not None:
        save_path = save_path / args.save_name
    save_path = save_path / f"{info['checkpoint_url'].replace('/', '--')}"
    os.makedirs(save_path, exist_ok=True)

    file_name = f"aux={str(round(args.aux_weight, 2)).replace('.', 'p')}_{info['checkpoint_step']:012d}.json"
    with open(save_path / file_name, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nresults written to: {save_path / file_name}\n")



def split_ruler_prompt(row):
    """Split a RULER sample into the state-filling text and terminal prompt."""
    marker = "\nWhat "
    split = row["input"].rfind(marker)
    if split < 0:
        raise ValueError("Could not locate the terminal RULER question")
    return (
        row["input"][:split].strip(),
        row["input"][split + 1:].strip()
    )


def _ruler_checkout():

    repo = Path(constants.LOCAL_DATA_PATH) / "ruler_cache" / RULER_REVISION[:12] / "repo"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", RULER_REPO, str(repo)], check=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", RULER_REVISION], cwd=repo, check=True,
        )

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()

    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()

    # if revision != RULER_REVISION or dirty:
    #     raise ValueError(
    #         f"Cached NVIDIA/RULER checkout is not the clean pinned revision: {repo}"
    #     )

    return repo


def _load_ruler_rows(args, task, context_length):
    """Generate with pinned NVIDIA/RULER code, caching only matching JSONL."""
    repo = _ruler_checkout()
    data_root = repo / "scripts/data"

    if task in RULER_ESSAY_TASKS:
        essay_dir = data_root / "synthetic/json"
        essay_file = essay_dir / "PaulGrahamEssays.json"

        if not essay_file.exists():
            subprocess.run(
                [sys.executable, "download_paulgraham_essay.py"],
                cwd=essay_dir, check=True,
            )

        subprocess.run(
            [
                sys.executable, "-c",
                "import nltk; nltk.download('punkt', quiet=True); "
                "nltk.download('punkt_tab', quiet=True)",
            ],
            check=True,
        )

    tokenizer_key = hashlib.sha256(args.tokenizer.encode()).hexdigest()[:12]
    generated_root = (
        Path(constants.LOCAL_DATA_PATH) / "ruler_cache" / RULER_REVISION[:12]
        / "generated" / tokenizer_key / f"seed={args.seed}"
        / f"length={context_length}" / f"samples={args.ruler_num_samples}"
    )

    manifest = generated_root / task / "validation.jsonl"
    if not manifest.exists():
        task_config = omegaconf.OmegaConf.to_container(
            omegaconf.OmegaConf.load(repo / "scripts/synthetic.yaml")[task],
            resolve=True,
        )
        niah_config = runpy.run_path(
            str(data_root / "synthetic/constants.py")
        )["TASKS"]["niah"]
        template = niah_config["template"] + niah_config["answer_prefix"]
        command = [
            sys.executable, str(data_root / "synthetic/niah.py"),
            "--save_dir", str(generated_root),
            "--save_name", task,
            "--subset", "validation",
            "--tokenizer_path", args.tokenizer,
            "--tokenizer_type", "hf",
            "--max_seq_length", str(context_length),
            "--tokens_to_generate", str(niah_config["tokens_to_generate"]),
            "--num_samples", str(args.ruler_num_samples),
            "--random_seed", str(args.seed),
            "--template", template,
        ]
        for key, value in task_config["args"].items():
            command.extend([f"--{key}", str(value)])
        subprocess.run(
            command,
            cwd=data_root, check=True,
        )

    if not manifest.is_file():
        raise RuntimeError(f"RULER generator did not create its manifest: {manifest}")

    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(rows) != args.ruler_num_samples:
        raise ValueError(
            f"Cached RULER data has {len(rows)} samples, expected "
            f"{args.ruler_num_samples}: {manifest}"
        )

    return rows


def get_ruler_results(logits_fn, tokenizer: PreTrainedTokenizer, prompt, answer_prefix, answer: str, args):
    """Score the gold needle tokens with teacher-forced next-token logits."""

    if answer_prefix is None:
        prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            padding=False,
            truncation=False,
            return_dict=False,
        )
    else:
        prompt_tokens = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer_prefix},
            ],
            tokenize=True,
            add_generation_prompt=False,
            padding=False,
            truncation=False,
            return_dict=False,
        )
        if prompt_tokens[-1] == tokenizer.eos_token_id:
            prompt_tokens = prompt_tokens[:-1]

        space_tokens = tokenizer(
            " ", add_special_tokens=False
        ).input_ids
        bare_answer_tokens = tokenizer(
            answer, add_special_tokens=False
        ).input_ids
        spaced_answer_tokens = tokenizer(
            " " + answer, add_special_tokens=False
        ).input_ids

        if spaced_answer_tokens == space_tokens + bare_answer_tokens:
            prompt_tokens += space_tokens
            answer_tokens = bare_answer_tokens
        else:
            answer_tokens = spaced_answer_tokens
            
    answer_tokens = tokenizer(answer, add_special_tokens=False).input_ids

    if not answer_tokens:
        raise ValueError("RULER answer tokenized to an empty sequence")
    if len(prompt_tokens) + len(answer_tokens) > args.max_length:
        raise ValueError(
            "RULER question and answer exceed --max-length: "
            f"{len(prompt_tokens)} + {len(answer_tokens)} > {args.max_length}"
        )

    input_ids = torch.tensor(
        [prompt_tokens + answer_tokens], device=DEVICE, dtype=torch.long
    )

    logits = logits_fn(input_ids, slice(len(prompt_tokens) - 1, -1)).float()

    targets = input_ids[:, len(prompt_tokens):]
    token_losses = F.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), reduction="none"
    )
    correct = logits.argmax(-1).eq(targets).flatten()

    return {
        "exact_acc": float(correct.all().item()),
        "correct_tokens": int(correct.sum().item()),
        "answer_tokens": int(correct.numel()),
        "token_loss_sum": float(token_losses.sum().item()),
    }


def _aggregate_ruler_results(records):
    count = len(records)
    answer_tokens = sum(row["answer_tokens"] for row in records)

    if count == 0 or answer_tokens == 0:
        return {
            "exact_acc": 0.0,
            "token_acc": 0.0,
            "token_loss": 0.0,
            "count": count,
            "answer_tokens": answer_tokens,
        }

    return {
        "exact_acc": sum(row["exact_acc"] for row in records) / count,
        "token_acc": sum(row["correct_tokens"] for row in records) / answer_tokens,
        "token_loss": sum(row["token_loss_sum"] for row in records) / answer_tokens,
        "count": count,
        "answer_tokens": answer_tokens,
    }


def _format_e2e_ruler_table(by_task, tasks, context_lengths):
    table = {}
    for task in tasks:
        if task not in E2E_RULER_LABELS:
            continue
        table[E2E_RULER_LABELS[task]] = {
            f"{context_length // 1024}K": by_task[task][str(context_length)]["exact_acc"]
            for context_length in context_lengths
        }
    return table


def evaluate_ruler(
    model, train_fn, logits_fn, tokenizer, chunker, args, info
):

    assistant_loss_accumulator = defaultdict(list)
    aux_loss_accumulator = defaultdict(list)
    records = []
    started = time.time()

    total = len(args.ruler_context_lengths) * len(args.ruler_tasks) * args.ruler_num_samples

    progress = tqdm(total=total, desc="RULER")
    for context_length in args.ruler_context_lengths:

        for task in args.ruler_tasks:
            rows = _load_ruler_rows(args, task, context_length)

            for index_start in range(0, len(rows), args.batch_size):
                batch_rows = rows[index_start:index_start + args.batch_size]
                parsed = [split_ruler_prompt(row) for row in batch_rows]

                adapt_documents_padded(
                    train_fn, model, tokenizer, chunker,
                    [document for document, _ in parsed],
                    args,
                    assistant_loss_accumulator,
                    aux_loss_accumulator,
                )
                frozen_states = [
                    state.detach().clone() for state in model.state_containers()
                ]

                for batch_index, (row, (_, question)) in enumerate(zip(batch_rows, parsed)):
                    for module_index, module in enumerate(model.fast_modules()):
                        module.state = frozen_states[module_index][batch_index][None]

                    references = [str(value) for value in row["outputs"]]
                    answer = ", ".join(references)
                    score = get_ruler_results(
                        logits_fn, tokenizer,
                        question, row.get("answer_prefix", None),
                        answer, args
                    )

                    records.append({
                        "task": task,
                        "context_length": context_length,
                        **score,
                    })

                by_task = defaultdict(dict)
                for task in args.ruler_tasks:
                    for context_length in args.ruler_context_lengths:

                        selected = [
                            row for row in records
                            if row["task"] == task and row["context_length"] == context_length
                        ]

                        by_task[task][str(context_length)] = _aggregate_ruler_results(selected)

                results = {
                    "evaluation_protocol": {
                        "paper_table": "E2E Table 2",
                        "data_generator": f"NVIDIA/RULER@{RULER_REVISION}",
                        "decoding": "teacher_forced_gold_prefix_top1",
                        "sequential_sampling_rollouts": False,
                        "primary_metric": "exact_acc",
                        "comparison_note": (
                            "exact_acc requires every gold answer token to be top-1 under its "
                            "gold prefix; the paper reports RULER accuracy from decoded strings"
                        ),
                    },
                    "total_evaluation_time": time.time() - started,
                    "overall": _aggregate_ruler_results(records),
                    "by_task": by_task,
                    "e2e_table": _format_e2e_ruler_table(
                        by_task, args.ruler_tasks, args.ruler_context_lengths
                    ),
                    "horizon": format_horizon_losses(
                        assistant_loss_accumulator, aux_loss_accumulator
                    ),
                    "args": vars(args),
                    "model_config": omegaconf.OmegaConf.to_container(model.config, resolve=True),
                }

                save_path = constants.LOCAL_DATA_PATH / "ruler_results"
                if args.save_name is not None:
                    save_path = save_path / args.save_name
                save_path = save_path / info["checkpoint_url"].replace("/", "--")
                os.makedirs(save_path, exist_ok=True)
                file_name = (
                    f"aux={str(round(args.aux_weight, 2)).replace('.', 'p')}_"
                    f"{info['checkpoint_step']:012d}.json"
                )
                with open(save_path / file_name, "w") as f:
                    json.dump(results, f, indent=4)

                print(f"\nresults written to: {save_path / file_name}\n")

                progress.update(len(batch_rows))
    progress.close()


@torch.no_grad()
def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, model_url, model_step = icl.load_fresh_model(args.fresh_config, None, None, DEVICE, True)
    info = {
        "checkpoint_url": model_url,
        "checkpoint_step": model_step,
    }

    tokenizer = icl.load_tokenizer(args.tokenizer)
    chunker = Chunker(args.tokenizer)

    train_fn, logits_fn = make_compiled_fns(
        model, args
    )

    if args.benchmark == "quality":
        evaluate_quality(
            model, train_fn, logits_fn,
            tokenizer, chunker, args, info
        )

    else:
        evaluate_ruler(
            model, train_fn, logits_fn,
            tokenizer, chunker, args, info
        )


if __name__ == "__main__":
    main()
