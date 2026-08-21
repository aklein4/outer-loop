"""Evaluate stateful models on horizonized long-context and NIAH tasks.

Documents are split into consecutive pairs of <= ``chunk_tokens`` token chunks,
matching ``DhsaLongDataCollectionsHandler``.  Each pair is a single-turn
conversation used for one adaptation step.  The terminal question and gold
answer are also represented as one single-turn conversation.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import zipfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

import evaluate_icl as icl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fresh-config", required=True)
    p.add_argument("--aux-weight", type=float, required=True)
    p.add_argument("--context-lengths", type=int, nargs="+", required=True)
    p.add_argument("--benchmarks", nargs="+", choices=["longbench", "niah"],
                   default=["longbench", "niah"])
    p.add_argument("--longbench-subsets", nargs="+", default=["qasper", "multifieldqa_en"])
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--chunk-tokens", type=int, default=448)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer", default=icl.DEFAULT_TOKENIZER)
    p.add_argument("--base-lr", type=float, default=None)
    p.add_argument("--checkpoint-step", type=int, default=None,
                   help="Override pretrained_step without editing the model config")
    p.add_argument("--output", required=True)
    return p.parse_args()


def single_turn(user: str, assistant: str):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def horizonize(tokenizer, text: str, token_budget: int, chunk_tokens: int):
    """Return ordered single-turn chunk pairs within a source-token budget."""
    ids = tokenizer.encode(text, add_special_tokens=False)[:token_budget]
    # DHSA pairs chunks [0, 1], [2, 3], ... and drops an unmatched tail.
    chunks = [
        tokenizer.decode(ids[i:i + chunk_tokens], skip_special_tokens=True).strip()
        for i in range(0, len(ids), chunk_tokens)
    ]
    return [
        single_turn(a, b) for a, b in zip(chunks[::2], chunks[1::2]) if a and b
    ], len(ids)


def load_longbench(subsets, samples: int, seed: int):
    archive = Path(hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset"))
    rng = random.Random(seed)
    rows = []
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for subset in subsets:
            candidates = [f"{subset}.jsonl", f"data/{subset}.jsonl"]
            name = next((x for x in candidates if x in names), None)
            if name is None:
                raise FileNotFoundError(f"{subset}.jsonl not found in LongBench archive")
            values = [json.loads(line) for line in zf.read(name).decode().splitlines()]
            rng.shuffle(values)
            for value in values[:samples]:
                answers = value.get("answers") or []
                if answers:
                    rows.append({
                        "benchmark": f"longbench/{subset}",
                        "document": value["context"],
                        "question": value["input"],
                        "answer": str(answers[0]),
                    })
    return rows


HAY = (
    "The archive contains routine status reports about roads, rainfall, crops, "
    "library inventories, and municipal meetings. No action is required. "
)


def make_niah(tokenizer, token_budget: int, sample: int, seed: int):
    rng = random.Random(seed + token_budget * 17 + sample)
    key = "K" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(15))
    needle = f"\nIMPORTANT RECORD: The pass key is {key}. Remember this exact pass key.\n"
    hay_ids = tokenizer.encode(HAY, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    # Cycle a neutral paragraph without constructing a huge intermediate string.
    ids = (hay_ids * ((token_budget // len(hay_ids)) + 2))[:max(0, token_budget - len(needle_ids))]
    depth = (sample + 1) / (sample + 2)
    pos = int(len(ids) * depth)
    ids[pos:pos] = needle_ids
    document = tokenizer.decode(ids[:token_budget], skip_special_tokens=True)
    # Decoding and re-tokenizing repetitive text can merge across boundaries.
    # Pad the serialized document until horizonize() can consume the requested
    # number of source tokens, so reported million-token cases are truly 1M.
    actual = len(tokenizer.encode(document, add_special_tokens=False))
    while actual < token_budget:
        missing = token_budget - actual
        repeats = max(1, (missing // max(1, len(hay_ids))) + 2)
        document += HAY * repeats
        actual = len(tokenizer.encode(document, add_special_tokens=False))
    return {
        "benchmark": "niah",
        "document": document,
        "question": "What is the exact pass key in the important record? Reply with only the pass key.",
        "answer": key,
        "needle_depth": depth,
    }


def score_answer(model, logits_fn, tokenizer, question, answer, device, dtype):
    args = argparse.Namespace(max_length=2048, dtype=dtype, eval_fn="output_loss")
    ids, assistant_mask, _ = icl.encode(tokenizer, [single_turn(question, answer)], 2048, device)
    with torch.no_grad():
        logits = logits_fn(ids)
        loss = icl.output_loss(ids, assistant_mask, logits)[0]
        labels = ids[:, 1:]
        mask = assistant_mask[:, 1:]
        pred = logits.argmax(-1)
        correct = ((pred == labels) & mask).sum().item()
        count = mask.sum().item()
        exact = bool((((pred == labels) | ~mask).all(1) & mask.any(1))[0].item())
    return {"loss": float(loss), "token_accuracy": correct / max(count, 1), "exact_match": exact,
            "answer_tokens": count}


def main():
    args = parse_args()
    random.seed(args.seed)
    device = torch.device(args.device)
    load_args = argparse.Namespace(dtype=args.dtype, aux_weight=args.aux_weight, compile=False)
    model = icl.load_fresh_model(args.fresh_config, args.base_lr, args.checkpoint_step, device)
    tokenizer = icl.load_tokenizer(args.tokenizer)
    train_fn, logits_fn = icl.make_fns(model, load_args, device)

    long_rows = load_longbench(args.longbench_subsets, args.samples, args.seed) \
        if "longbench" in args.benchmarks else []
    results = []
    started = time.time()
    for length in args.context_lengths:
        rows = list(long_rows)
        if "niah" in args.benchmarks:
            rows += [make_niah(tokenizer, length, i, args.seed) for i in range(args.samples)]
        for row_index, row in enumerate(tqdm(rows, desc=f"{length:,} tokens")):
            conversations, used = horizonize(
                tokenizer, row["document"], length, args.chunk_tokens
            )
            model.init_state(1, device)
            model.empty_state()
            adapt_args = argparse.Namespace(max_length=2 * args.chunk_tokens + 64,
                                            dtype=args.dtype, aux_weight=args.aux_weight)
            row_started = time.time()
            for step, conversation in enumerate(conversations):
                ids, assistant_mask, attention_mask = icl.encode(
                    tokenizer, [conversation], adapt_args.max_length, device
                )
                with torch.enable_grad():
                    train_fn(ids, assistant_mask, attention_mask,
                             torch.ones((1, 1, 1), device=device))
            model.eval()
            score = score_answer(model, logits_fn, tokenizer, row["question"],
                                 row["answer"], device, args.dtype)
            model.train()
            result = {
                "model_config": args.fresh_config,
                "aux_weight": args.aux_weight,
                "target_context_tokens": length,
                "used_context_tokens": used,
                "horizon_steps": len(conversations),
                "benchmark": row["benchmark"],
                "sample": row_index,
                "elapsed_seconds": time.time() - row_started,
                **score,
            }
            if "needle_depth" in row:
                result["needle_depth"] = row["needle_depth"]
            results.append(result)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
            model.zero_grad(set_to_none=True)
            model.empty_state()
    print(json.dumps({"rows": len(results), "elapsed_seconds": time.time() - started,
                      "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
