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
import time
from collections import defaultdict
import os
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fresh-config", required=True)
    p.add_argument("--tokenizer", default=icl.DEFAULT_TOKENIZER)
    p.add_argument("--save-name", default=None)
    p.add_argument("--benchmark", choices=["quality", "ruler"], required=True)
    p.add_argument("--aux-weight", type=float, required=True)
    p.add_argument("--max-articles", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--ruler-manifests", nargs="*", default=[])
    p.add_argument(
        "--ruler-num-samples", type=int, default=500,
        help="Samples per RULER task/length manifest (official default: 500).",
    )
    return p.parse_args()


def masked_update(model, mask, *args, **kwargs):
    frozen = [x.detach().clone() for x in model.state_containers()]
    model.update_state(*args, **kwargs)
    for frozen_tensor, current_tensor in zip(frozen, model.state_containers()):
        current_tensor.data.copy_(
            torch.where(mask, current_tensor, frozen_tensor)
        )


def make_compiled_fns(model, args):
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
            torch.compile(train_fn, fullgraph=False),
            torch.compile(logits_fn, fullgraph=True),
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
            icl.encode(tokenizer, c, max_length, device=DEVICE)
            for c in conv
        ] for conv in conversations
    ]

    mask = torch.zeros(horizon_length, bs, device=DEVICE, dtype=torch.bool)
    for i, h in enumerate(horizons):
        mask[:len(h), i] = True

    default_episode = (
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

    if len(next(model.state_containers())) != bs:
        model.init_state(bs, DEVICE)
    else:
        model.empty_state()

    conversations = chunker(documents)
    horizon_length = max(len(c) for c in conversations)

    packed, mask = encode_horizons(tokenizer, conversations, max_length=2 * args.chunk_tokens + 64)
    cpu_mask = mask.cpu()

    for t in range(horizon_length):

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
            single_turn(chunks[i].strip(), chunks[i + 1].strip())
            for i in range(0, len(chunks), 2)
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
        tuple(range(1,n+1)),
        tuple(f"{i+1}. {option.strip()}" for i, option in enumerate(row["options"]))
    )


def get_quality_results(model, logits_fn, tokenizer: PreTrainedTokenizer, row, args):
    prompt = quality_prompt(row)
    single_options, full_options = quality_options(row)

    prompt_tokens = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        padding=False,
        truncation=False,
        return_dict=False
    )
    n_prompt = len(prompt_tokens)

    single_tokens = tokenizer(
        single_options, add_special_tokens=False
    )
    assert all(len(t)==1 for t in single_tokens)

    full_tokens = tokenizer(
        full_options, add_special_tokens=False
    )
    assert all(len(t)>=1 for t in full_tokens)

    full_input_ids = [
        torch.tensor(prompt_tokens + [t], device=DEVICE, dtype=torch.long())
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
    mask = (input_ids != -1).float()
    input_ids = torch.where(mask, input_ids, 0)

    logits = logits_fn(model, input_ids, slice(n_prompt-1,-1))

    single_logits = logits[0, 0]
    single_acc = float(single_logits[single_tokens].argmax(-1).item() == row["answer"])

    logp = -F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        full_input_ids[n_prompt:].reshape(-1),
        reduction='none'
    ).reshape(full_input_ids[n_prompt:].shape)

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

    hashes = []
    rows = defaultdict(list)
    articles = defaultdict(list)
    for row in ds:
        k = hash(row["article"])
        hashes.append(k)
        rows[k].append(row)
        articles[k] = row["article"]

    if args.max_articles is not None:
        hashes = np.random.choice(
            hashes,
            size=min(args.max_articles, len(hashes)),
            replace=False
        )
    rows = {k: [k] for k in hashes}
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

        for i in tqdm(len(curr_hashes), "Questions", leave=False):

            for l, m in enumerate(model.fast_modules()):
                m.state = frozen_state_containers[l][i]

            for row in rows[curr_hashes[i]]:
                row_result = get_quality_results(model, logits_fn, tokenizer, row, args)

                if row_result is not None:
                    hard = row_result["hard"]

                    for k in row_result:
                        if hard:
                            hard_results[k].append(row_result[k])
                        else:
                            easy_results[k].append(row_result[k])

                    if hard:
                        hard_count += 1
                    else:
                        easy_count += 1

        progress.update(1)
                
    horizon_losses = format_horizon_losses(hard_results, easy_results)

    all_results = {}
    if set(easy_results.keys()) == set(hard_results.keys()):
        all_results = {k: torch.stack(easy_results[k] + hard_results[k]).mean() for k in easy_results.keys()}

    easy_results = {k: torch.stack(v).mean() for k, v in easy_results.items()}
    hard_results = {k: torch.stack(v).mean() for k, v in hard_results.items()}

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
    chunker = Chunker(args.tokenizer_url)

    train_fn, logits_fn = make_compiled_fns(
        model, args
    )

    if args.benchmark == "quality":
        evaluate_quality(
            model, train_fn, logits_fn,
            tokenizer, chunker, args, info
        )

    else:
        raise NotImplementedError(f"RULER is not implemented yet!")
        if not args.ruler_manifests:
            raise ValueError("--ruler-manifests is required for RULER")
        result = evaluate_ruler(
            model, train_fn, answer_states_fn, tokenizer, args
        )


if __name__ == "__main__":
    main()
