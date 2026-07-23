from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from collators.horizon import ASSISTANT_MASK_CHAT_TEMPLATE
from models import load_checkpoint, load_checkpoint_state
from utils.import_utils import import_model
import utils.constants as constants


DEFAULT_CHECKPOINT = "aklein4/Horizon-TPU_alpha"
DEFAULT_TOKENIZER = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_DATASET = "aklein4/Bitext-SmolLM2-1024-natural-instructions-format"

DEFAULT_NUM_EXAMPLES = [0] + list(2 ** i for i in range(0, 11))  # 0, 1, 2, 4, 8, ..., 1024
DEFAULT_NUM_TEST = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=None)
    parser.add_argument("--fresh-config", default=None)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--base-lrs", type=float, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-examples", type=int, nargs="+", default=DEFAULT_NUM_EXAMPLES)
    parser.add_argument("--num-eval", type=int, default=DEFAULT_NUM_TEST)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subsets", nargs="*", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--lr-scale", type=float, default=1.0, help="Scale the learning rate for adaptation")
    parser.add_argument("--aux-weight", type=float, default=0.0, help="Weight for adaptation loss on non-assistant attended tokens")
    parser.add_argument("--eval-fn", default="output_loss", choices=["exact_match", "output_loss"])
    parser.add_argument("--save-name", default=None, help="Name for saving results (default: checkpoint name)")
    return parser.parse_args()


def load_model(checkpoint: str, step: int, device: torch.device):
    print(f"Loading {checkpoint} at step {step}")
    model = load_checkpoint(
        checkpoint,
        step,
        attention_kernel="gpu_flash_attention" if device.type == "cuda" else None,
    )

    model.to(device=device, dtype=torch.float32)
    model.train()
    model.gradient_checkpointing_enable()
    for param in model.parameters():
        param.requires_grad_(False)
    model.model.embed_tokens.requires_grad_(True)
    return model


def load_fresh_model(config_path: str, base_lr: float | None, device: torch.device):
    config = OmegaConf.load(config_path)
    if base_lr is not None:
        config.base_lr = base_lr
    config.attention_kernel = "gpu_flash_attention" if device.type == "cuda" else None

    model_class = import_model(config.type)

    print(f"Loading fresh {config.type} from {config_path}")
    if base_lr is not None:
        print(f"Using base_lr={base_lr:g}")
    model = model_class(config)

    if config.pretrained_url is not None:
        print(f"Loading {config.pretrained_url} at step {config.pretrained_step} with strict={config.pretrained_strict}")
        model = load_checkpoint_state(
            model,
            config.pretrained_url,
            config.pretrained_step,
            strict=config.pretrained_strict,
        )

    model.to(device=device, dtype=torch.float32)
    model.train()
    model.gradient_checkpointing_enable()
    for param in model.parameters():
        param.requires_grad_(False)
    model.model.embed_tokens.requires_grad_(True)
    return model


def load_tokenizer(tokenizer_url: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)
    tokenizer.padding_side = "right"
    tokenizer.chat_template = ASSISTANT_MASK_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode(tokenizer, messages, max_length: int, device: torch.device):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    assistant_mask = encoded.get("assistant_masks")
    if assistant_mask is None:
        assistant_mask = encoded["assistant_tokens_mask"]
    return (
        encoded["input_ids"].to(device),
        assistant_mask.to(device).bool(),
        encoded["attention_mask"].to(device).bool(),
    )


def adaptation_loss(input_ids, assistant_mask, attention_mask, logits, aux_weight: float):
    labels = input_ids[:, 1:]
    mask = assistant_mask[:, 1:].float()
    attn = attention_mask[:, 1:].float()
    aux_mask = (1.0 - mask) * attn
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    output_loss = (losses * mask).sum(1).div(mask.sum(1).clamp(min=1)).mean()
    aux_loss = (losses * aux_mask).sum(1).div(aux_mask.sum(1).clamp(min=1)).mean()
    return output_loss + aux_weight * aux_loss


def exact_match(input_ids, assistant_mask, logits):
    labels = input_ids[:, 1:]
    mask = assistant_mask[:, 1:]
    preds = logits.argmax(dim=-1)
    return ((preds == labels) | ~mask).all(dim=1) & mask.any(dim=1)


def output_loss(input_ids, assistant_mask, logits):
    labels = input_ids[:, 1:]
    mask = assistant_mask[:, 1:].float()
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    return (losses * mask).sum(1).div(mask.sum(1).clamp(min=1))


def autocast(device: torch.device, dtype: str):
    if dtype == "float32" or device.type == "cpu":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type=device.type, dtype=getattr(torch, dtype))


def make_fns(model, args, device):
    def train_fn(input_ids, assistant_mask, attention_mask, lr_scale):
        with autocast(device, args.dtype):
            logits = model(input_ids, logits_to_keep=slice(0, -1))[0]
            loss = adaptation_loss(input_ids, assistant_mask, attention_mask, logits, args.aux_weight)
        loss.backward()
        model.update_state()

    def logits_fn(input_ids):
        with autocast(device, args.dtype):
            return model(input_ids, logits_to_keep=slice(0, -1))[0]

    if args.compile:
        train_fn = torch.compile(train_fn, fullgraph=False)
        logits_fn = torch.compile(logits_fn, fullgraph=False)

    return train_fn, logits_fn


def adapt(train_fn, tokenizer, rows, example_idx, args, device, lr_scale):
    input_ids, assistant_mask, attention_mask = encode(
        tokenizer,
        [row["train_data"][example_idx] for row in rows],
        args.max_length,
        device,
    )
    with torch.enable_grad():
        train_fn(input_ids, assistant_mask, attention_mask, lr_scale)



@torch.no_grad()
def evaluate(model, logits_fn, tokenizer, rows, args, device):
    model.eval()
    scores = torch.zeros(len(rows), dtype=torch.float64)
    for test_idx in tqdm(range(args.num_eval), desc="evaluating", leave=False):
        input_ids, assistant_mask, _ = encode(
            tokenizer,
            [row["test_data"][test_idx] for row in rows],
            args.max_length,
            device,
        )
        logits = logits_fn(input_ids)
        if args.eval_fn == "exact_match":
            score = exact_match(input_ids, assistant_mask, logits)
        elif args.eval_fn == "output_loss":
            score = output_loss(input_ids, assistant_mask, logits)
        else:
            raise ValueError(f"Unknown eval_fn: {args.eval_fn}")
        scores += score.double().cpu()
    model.train()
    return (scores / args.num_eval).tolist()


def load_rows(args, subset: str):
    rows = []
    dataset = datasets.load_dataset(args.dataset, subset, split="train")
    max_num_examples = max(args.num_examples)
    for row in dataset:
        if row.get("num_examples", max_num_examples) < max_num_examples:
            continue
        if len(row["train_data"]) >= max_num_examples and len(row["test_data"]) >= args.num_eval:
            rows.append({
                "subset": subset,
                "train_data": row["train_data"],
                "test_data": row["test_data"],
            })
        if args.max_rows is not None and len(rows) >= args.max_rows:
            break
    return rows


def load_all_rows(args, subsets):
    rows = []
    for subset in tqdm(subsets, desc="loading data"):
        rows.extend(load_rows(args, subset))
    return rows


def add_scores(totals, counts, n, batch, scores):
    for row, score in zip(batch, scores):
        subset = row["subset"]
        totals[n][subset] = totals[n].get(subset, 0.0) + score
        counts[n][subset] = counts[n].get(subset, 0) + 1


def evaluate_rows(model, train_fn, logits_fn, tokenizer, rows, args, device):
    totals = {n: {} for n in args.num_examples}
    counts = {n: {} for n in args.num_examples}

    for start in tqdm(range(0, len(rows), args.batch_size), desc="evaluating", leave=True):
        batch = rows[start:start + args.batch_size]
        model.init_state(len(batch), device)
        model.empty_state()

        if 0 in totals:
            add_scores(totals, counts, 0, batch, evaluate(model, logits_fn, tokenizer, batch, args, device))

        for example_idx in tqdm(range(max(args.num_examples)), desc="adapting", leave=False):
            adapt(train_fn, tokenizer, batch, example_idx, args, device, lr_scale=args.lr_scale)
            n = example_idx + 1
            if n in totals:
                add_scores(totals, counts, n, batch, evaluate(model, logits_fn, tokenizer, batch, args, device))

        model.zero_grad(set_to_none=True)
        model.empty_state()

    results = []
    for n in args.num_examples:
        result = {"num_examples": n, "benchmarks": {}}
        for subset, total in totals[n].items():
            result["benchmarks"][subset] = total / counts[n][subset]
        result["average"] = sum(result["benchmarks"].values()) / len(result["benchmarks"])
        results.append(result)
    return results


def save_results(args, label: int | str, results):
    if args.fresh_config is None:
        path = Path(constants.LOCAL_DATA_PATH) / "icl_results" / ((args.save_name+"/"+args.checkpoint.replace("/", "--")) if args.save_name is not None else args.checkpoint.replace("/", "--")) / f"{label:012d}.json"
    else:
        path = Path(constants.LOCAL_DATA_PATH) / "icl_results" / (args.save_name if args.save_name is not None else "fresh") / Path(args.fresh_config).stem / f"{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {path}")


def lr_label(base_lr: float | None, model) -> str:
    lr = model.config.base_lr if base_lr is None else base_lr
    return f"base_lr_{lr:.0e}".replace("+", "")


def main():
    args = parse_args()
    if args.fresh_config is None and args.checkpoint_steps is None:
        raise ValueError("--checkpoint-steps is required unless --fresh-config is set")

    args.num_examples = sorted(set(args.num_examples))
    device = torch.device(args.device)
    subsets = args.subsets or datasets.get_dataset_config_names(args.dataset)
    rows = load_all_rows(args, subsets)
    print(f"Loaded {len(rows)} rows from {len(subsets)} subsets")

    if args.fresh_config is not None:
        for base_lr in args.base_lrs or [None]:
            model = load_fresh_model(args.fresh_config, base_lr, device)
            tokenizer = load_tokenizer(args.tokenizer)
            train_fn, logits_fn = make_fns(model, args, device)
            results = evaluate_rows(model, train_fn, logits_fn, tokenizer, rows, args, device)
            save_results(args, lr_label(base_lr, model), results)
        return

    for step in args.checkpoint_steps:
        model = load_model(args.checkpoint, step, device)
        tokenizer = load_tokenizer(args.tokenizer)
        train_fn, logits_fn = make_fns(model, args, device)
        results = evaluate_rows(model, train_fn, logits_fn, tokenizer, rows, args, device)
        save_results(args, step, results)


if __name__ == "__main__":
    main()
