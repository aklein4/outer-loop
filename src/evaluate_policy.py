from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets
import torch
from tqdm import tqdm

import evaluate_persona as persona
from models.forte import ForteMode, ForteModel
from models.oloop import OLoopModel
from models.piano import PianoMode, PianoModel
import utils.constants as constants


DEFAULT_CHECKPOINT = persona.DEFAULT_CHECKPOINT
DEFAULT_TOKENIZER = persona.DEFAULT_TOKENIZER
DEFAULT_DATASET = "aklein4/PolicyBench"

DEFAULT_NUM_EXAMPLES = persona.DEFAULT_NUM_EXAMPLES
DEFAULT_NUM_TEST = 100
LETTERS = ("A", "B", "C", "D")


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
    parser.add_argument(
        "--lr-scale-decay",
        action="store_true",
        help="Linearly decay the adaptation learning-rate scale across the loop",
    )
    parser.add_argument("--lr-scale-start", type=float, default=1.0)
    parser.add_argument("--lr-scale-end", type=float, default=0.1)
    parser.add_argument(
        "--aux-loss-weight",
        "--aux-weight",
        dest="aux_weight",
        type=float,
        default=0.0,
        help="Weight for non-assistant loss in adaptation gradients only (default: 0)",
    )
    parser.add_argument("--save-name", default=None, help="Name for saving results (default: checkpoint name)")
    return parser.parse_args()


def format_user_message(item: dict) -> str:
    options = item["options"]
    if len(options) != len(LETTERS):
        raise ValueError(f"PolicyBench questions must have four options, got {len(options)}")
    return "\n".join(
        [item["question"], *(f"{letter}. {option}" for letter, option in zip(LETTERS, options))]
    )


def format_adaptation_messages(item: dict) -> list[dict[str, str]]:
    correct_option = int(item["correct_option"])
    if correct_option not in range(len(LETTERS)):
        raise ValueError(f"correct_option must be in [0, 3], got {correct_option}")
    letter = LETTERS[correct_option]
    return [
        {"role": "user", "content": format_user_message(item)},
        {"role": "assistant", "content": f"{letter}. {item['options'][correct_option]}"},
    ]


def format_evaluation_messages(item: dict) -> list[dict[str, str]]:
    return [{"role": "user", "content": format_user_message(item)}]


def encode_prompts(tokenizer, items, max_length: int, device: torch.device):
    """Encode user-only chats and retain the position that predicts the first answer token."""
    messages = [format_evaluation_messages(item) for item in items]
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
        # raise ValueError(
        #     "A PolicyBench evaluation prompt reached --max-length; increase it so "
        #     "the assistant generation suffix is not truncated"
        # )
        print(
            f"Warning: A PolicyBench evaluation prompt reached --max-length on {(lengths == max_length).sum().item()} examples."
        )
    return (
        encoded["input_ids"].to(device),
        attention_mask.to(device),
        (lengths - 1).to(device),
    )


def choice_token_ids(tokenizer, device: torch.device) -> torch.Tensor:
    token_ids = []
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Choice {letter!r} is not a single tokenizer token: {ids}")
        token_ids.append(ids[0])
    return torch.tensor(token_ids, device=device, dtype=torch.long)


def make_fns(model, args, device):
    is_forte = isinstance(model, ForteModel)
    is_oloop = isinstance(model, OLoopModel)
    is_piano = isinstance(model, PianoModel)

    def train_fn(input_ids, assistant_mask, attention_mask, lr_scale):
        with persona.autocast(device, args.dtype):
            if is_forte:
                with torch.no_grad():
                    hidden_states = model.forward_backbone(input_ids, mode=ForteMode.INFERENCE)
                    embeddings = model.forward_embeddings(hidden_states, attention_mask)
                logits = model(
                    input_ids,
                    embeddings=embeddings,
                    embedding_mask=attention_mask,
                    mode=ForteMode.TRAIN_FIRST,
                    logits_to_keep=slice(0, -1),
                )
            elif is_piano:
                logits = model(
                    input_ids,
                    valid_mask=attention_mask,
                    mode=PianoMode.TRAIN_FIRST,
                    logits_to_keep=slice(0, -1),
                )[0]
            else:
                logits = model(input_ids, logits_to_keep=slice(0, -1))[0]
            loss = persona.adaptation_loss(
                input_ids, assistant_mask, attention_mask, logits, args.aux_weight
            )
        if is_forte:
            torch.autograd.backward(loss, inputs=model.grad_containers())
            model.update_state(ForteMode.TRAIN_FIRST, lr_scale=lr_scale)
        else:
            loss.backward()
            if is_oloop:
                model.update_state(lr_scale=lr_scale)
            elif is_piano:
                model.update_state(PianoMode.TRAIN_FIRST, lr_scale=lr_scale)
            else:
                model.update_state()

    def choice_logits_fn(input_ids, attention_mask, prompt_indices, token_ids):
        """Apply the LM head only to each prompt's final hidden state."""
        with persona.autocast(device, args.dtype):
            if is_forte:
                hidden_states = model.forward_backbone(input_ids, mode=ForteMode.INFERENCE)
                lm_states = model.forward_lm_states(hidden_states, mode=ForteMode.INFERENCE)
                final_states = lm_states[torch.arange(input_ids.shape[0], device=device), prompt_indices]
                logits = model.lm_head(final_states).float()
            else:
                model_kwargs = {"compute_logits": False}
                if is_piano:
                    model_kwargs.update(valid_mask=attention_mask, mode=PianoMode.INFERENCE)
                lm_states = model(input_ids, **model_kwargs)[0]
                final_states = lm_states[torch.arange(input_ids.shape[0], device=device), prompt_indices]
                logits = model.apply_head(final_states)
        return logits.index_select(-1, token_ids)

    if args.compile:
        train_fn = torch.compile(train_fn, fullgraph=False, dynamic=True)
        choice_logits_fn = torch.compile(choice_logits_fn, fullgraph=False, dynamic=True)

    return train_fn, choice_logits_fn


def adapt(train_fn, tokenizer, rows, example_idx, args, device, lr_scale):
    input_ids, assistant_mask, attention_mask = persona.encode(
        tokenizer,
        [format_adaptation_messages(row["train_data"][example_idx]) for row in rows],
        args.max_length,
        device,
    )
    lr_scale = torch.as_tensor([lr_scale], device=device, dtype=torch.float32).reshape(1, 1, 1)
    with torch.enable_grad():
        train_fn(input_ids, assistant_mask, attention_mask, lr_scale)


@torch.no_grad()
def evaluate(model, choice_logits_fn, tokenizer, rows, args, device, token_ids):
    model.eval()
    scores = torch.zeros(len(rows), dtype=torch.float64)
    for test_idx in tqdm(range(args.num_eval), desc="evaluating", leave=False):
        items = [row["test_data"][test_idx] for row in rows]
        input_ids, attention_mask, prompt_indices = encode_prompts(
            tokenizer, items, args.max_length, device
        )
        logits = choice_logits_fn(input_ids, attention_mask, prompt_indices, token_ids)
        labels = torch.tensor(
            [int(item["correct_option"]) for item in items], device=device, dtype=torch.long
        )
        scores += logits.argmax(dim=-1).eq(labels).double().cpu()
    model.train()
    return (scores / args.num_eval).tolist()


def load_rows(args, subset: str):
    rows = []
    dataset = datasets.load_dataset(args.dataset, subset, split="train")
    max_num_examples = max(args.num_examples)
    for row in dataset:
        if row.get("num_train", max_num_examples) < max_num_examples:
            continue
        if row.get("num_test", args.num_eval) < args.num_eval:
            continue
        if len(row["train_data"]) >= max_num_examples and len(row["test_data"]) >= args.num_eval:
            rows.append(
                {
                    "subset": subset,
                    "institution": row.get("Institution"),
                    "train_data": row["train_data"],
                    "test_data": row["test_data"],
                }
            )
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


def evaluate_rows(model, train_fn, choice_logits_fn, tokenizer, rows, args, device):
    totals = {n: {} for n in args.num_examples}
    counts = {n: {} for n in args.num_examples}
    token_ids = choice_token_ids(tokenizer, device)

    for start in tqdm(range(0, len(rows), args.batch_size), desc="batches", leave=True):
        batch = rows[start : start + args.batch_size]
        model.init_state(len(batch), device)
        model.empty_state()

        if 0 in totals:
            scores = evaluate(model, choice_logits_fn, tokenizer, batch, args, device, token_ids)
            add_scores(totals, counts, 0, batch, scores)

        total_adaptation_steps = max(args.num_examples)
        for example_idx in tqdm(range(total_adaptation_steps), desc="adapting", leave=False):
            adapt(
                train_fn,
                tokenizer,
                batch,
                example_idx,
                args,
                device,
                lr_scale=persona.adaptation_lr_scale(args, example_idx, total_adaptation_steps),
            )
            n = example_idx + 1
            if n in totals:
                scores = evaluate(model, choice_logits_fn, tokenizer, batch, args, device, token_ids)
                add_scores(totals, counts, n, batch, scores)

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
        directory = (
            args.save_name + "/" + args.checkpoint.replace("/", "--")
            if args.save_name is not None
            else args.checkpoint.replace("/", "--")
        )
        path = Path(constants.LOCAL_DATA_PATH) / "policy_results" / directory / f"{label:012d}.json"
    else:
        directory = args.save_name if args.save_name is not None else "fresh"
        path = (
            Path(constants.LOCAL_DATA_PATH)
            / "policy_results"
            / directory
            / Path(args.fresh_config).stem
            / f"{label}.json"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {path}")


def main():
    args = parse_args()
    if args.fresh_config is None and args.checkpoint_steps is None:
        raise ValueError("--checkpoint-steps is required unless --fresh-config is set")

    args.num_examples = sorted(set(args.num_examples))
    if args.lr_scale_decay:
        print(
            f"Using linear lr-scale decay from {args.lr_scale_start:g} "
            f"to {args.lr_scale_end:g} over {max(args.num_examples)} adaptation steps"
        )
    device = torch.device(args.device)
    subsets = args.subsets or datasets.get_dataset_config_names(args.dataset)
    rows = load_all_rows(args, subsets)
    if not rows:
        raise ValueError("No PolicyBench rows satisfy the requested train/evaluation sizes")
    print(f"Loaded {len(rows)} rows from {len(subsets)} subsets")

    if args.fresh_config is not None:
        for base_lr in args.base_lrs or [None]:
            for step in args.checkpoint_steps or [None]:
                model = persona.load_fresh_model(args.fresh_config, base_lr, step, device)
                tokenizer = persona.load_tokenizer(args.tokenizer)
                train_fn, choice_logits_fn = make_fns(model, args, device)
                results = evaluate_rows(
                    model, train_fn, choice_logits_fn, tokenizer, rows, args, device
                )
                save_results(args, persona.lr_label(base_lr, model, step), results)
        return

    for step in args.checkpoint_steps:
        model = persona.load_model(args.checkpoint, step, device)
        tokenizer = persona.load_tokenizer(args.tokenizer)
        train_fn, choice_logits_fn = make_fns(model, args, device)
        results = evaluate_rows(model, train_fn, choice_logits_fn, tokenizer, rows, args, device)
        save_results(args, step, results)


if __name__ == "__main__":
    main()
