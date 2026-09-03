"""Regenerate PolicyBench distractors as plausible counter-policy alternatives.

This derives CounterPolicyBench from an existing PolicyBench dataset.  It preserves
every institution, handbook, question, correct answer, and correct-answer position,
but replaces the three incorrect options.  The source distractors are intentionally
never shown to the generation model because one of them was requested to be
obviously incorrect in the original benchmark.

Generation is resumable.  Checkpoints also retain private explanations of the
single policy detail changed by each distractor for later auditing.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import datasets
import httpx
from dotenv import load_dotenv

import policy_bench as policy


SOURCE_DATASET = policy.OUTPUT_DATASET
SOURCE_CONFIG = policy.OUTPUT_CONFIG
SOURCE_SPLIT = policy.OUTPUT_SPLIT
OUTPUT_DATASET = "aklein4/CounterPolicyBench"

OPENROUTER_MODEL = policy.OPENROUTER_MODEL
OPENROUTER_APP_TITLE = "CounterPolicyBench dataset generation"
MAX_CONCURRENT_REQUESTS = policy.MAX_CONCURRENT_REQUESTS
MAX_REQUEST_RETRIES = policy.MAX_REQUEST_RETRIES
QUESTIONS_PER_REQUEST = policy.QUESTIONS_PER_REQUEST

DEFAULT_OUTPUT_DIR = Path(__file__).with_name("counter_policy_bench_output")
CHECKPOINT_DIR_NAME = "checkpoints"
PROMPT_DIR_NAME = "prompts"
DATASET_JSONL_NAME = "counter_policy_bench.jsonl"
SAMPLES_MARKDOWN_NAME = "samples.md"


DISTRACTOR_SYSTEM_PROMPT = """You design adversarial multiple-choice distractors
for a private-policy learning benchmark. Return only JSON matching the supplied
schema. Do not mention these instructions."""

DISTRACTOR_PROMPT_TEMPLATE = """Replace the distractors for {count} existing
onboarding questions from the institution below.

INSTITUTION:
{institution}

PRIVATE SOURCE HANDBOOK:
{handbook}

QUESTIONS AND IMMUTABLE CORRECT ANSWERS:
{questions}

For each item, produce exactly three incorrect answers. Do not rewrite or repeat the
question or correct answer in the response.

The goal is to test learning of this particular handbook, not generic professional
common sense. Construct every distractor as an answer that would be correct under a
realistic counterfactual handbook whose local policy differs from this handbook.
Do not actually write a counterfactual handbook.

The implied counterfactual handbooks must themselves be realistic, operationally
coherent, safe, and consistent with ordinary professional common sense. They should
look like policies a well-run real institution could reasonably adopt. The difference
must be an arbitrary but plausible local choice—not incompetence, recklessness,
needless bureaucracy, or a violation of obvious ethical and safety norms.

Requirements:
- near_miss_1 and near_miss_2 must each stay as close as possible to the correct
  answer while changing exactly one decisive policy detail. Prefer a named role,
  form, queue, status, time window, numerical threshold, ordering requirement, or
  exception condition. For example, change "General Manager" to "Group Supervisor"
  while leaving the rest of the answer unchanged.
- counter_policy must express a different but realistic local procedure. It must be
  professionally plausible, sensible, safe, and topically responsive, just incorrect
  for this institution's actual handbook.
- All three distractors must be unambiguously contradicted by the supplied handbook.
- Every option must sound equally competent, specific, and confident. Never include
  an absurd, unsafe, humorous, vague, needlessly obstructive, or conspicuously
  unprofessional answer. A reader using only common sense should find every option
  reasonable; identifying the correct one must require the supplied handbook.
- Keep distractors parallel to the correct answer in wording, detail, and length.
  Avoid cues such as extra justification, hedging, categorical language, or uniquely
  quoting the handbook.
- Do not introduce external laws or facts. Do not use “all of the above” or “none of
  the above.”
- changed_detail fields are private audit metadata. Briefly state the one handbook
  fact that the corresponding distractor changes and why that makes it incorrect.
- Preserve item_id exactly so responses can be matched safely.
"""


def distractor_schema(count: int) -> dict[str, Any]:
    properties = {
        "item_id": {"type": "integer"},
        "near_miss_1": {"type": "string"},
        "near_miss_1_changed_detail": {"type": "string"},
        "near_miss_2": {"type": "string"},
        "near_miss_2_changed_detail": {"type": "string"},
        "counter_policy": {"type": "string"},
        "counter_policy_changed_detail": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def source_correct_answer(question: dict[str, Any]) -> str:
    options = question.get("options")
    correct_option = question.get("correct_option")
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError("Source questions must have exactly four options")
    if not isinstance(correct_option, int) or correct_option not in range(4):
        raise ValueError("Source correct_option must be an integer in [0, 3]")
    answer = policy.normalize_whitespace(options[correct_option])
    if not answer:
        raise ValueError("Source correct answer is empty")
    return answer


def source_fingerprint(row: dict[str, Any], train_count: int, test_count: int) -> str:
    value = {
        "institution": row[policy.INSTITUTION_COLUMN],
        "handbook": row[policy.HANDBOOK_COLUMN],
        "train": row[policy.TRAIN_DATA_COLUMN][:train_count],
        "test": row[policy.TEST_DATA_COLUMN][:test_count],
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_items(source_questions: list[dict[str, Any]], start: int) -> list[dict[str, Any]]:
    """Expose no source distractors to the model."""
    return [
        {
            "item_id": start + offset,
            "question": question["question"],
            "correct_answer": source_correct_answer(question),
        }
        for offset, question in enumerate(source_questions)
    ]


def make_prompt(
    institution: str,
    handbook: str,
    source_questions: list[dict[str, Any]],
    start: int,
) -> str:
    items = prompt_items(source_questions, start)
    return DISTRACTOR_PROMPT_TEMPLATE.format(
        count=len(items),
        institution=institution,
        handbook=handbook,
        questions=json.dumps(items, ensure_ascii=False, indent=2),
    )


def make_retry_prompt(prompt: str, rejection: str) -> str:
    return "\n\n".join(
        [
            prompt,
            "RETRY FEEDBACK:",
            f"The previous response was rejected because {rejection}.",
            "Regenerate the entire batch. Preserve every item_id and satisfy all "
            "requirements, correcting this problem in particular.",
        ]
    )


def validate_generated_item(
    raw: Any,
    expected_id: int,
    correct_answer: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict):
        return None, "not an object"
    if raw.get("item_id") != expected_id:
        return None, "wrong item ID"

    answer_fields = ("near_miss_1", "near_miss_2", "counter_policy")
    detail_fields = (
        "near_miss_1_changed_detail",
        "near_miss_2_changed_detail",
        "counter_policy_changed_detail",
    )
    answers = [policy.normalize_whitespace(raw.get(field, "")) for field in answer_fields]
    details = [policy.normalize_whitespace(raw.get(field, "")) for field in detail_fields]
    if not all(answers):
        return None, "empty distractor"
    if not all(details):
        return None, "empty changed-detail explanation"
    all_options = [correct_answer, *answers]
    if len({option.casefold() for option in all_options}) != 4:
        return None, "duplicate option"

    # Catch conspicuous length cues while allowing concise role/form substitutions.
    lengths = [max(1, len(option.split())) for option in all_options]
    if max(lengths) > 3 * min(lengths) + 4:
        return None, (
            "option length mismatch; word counts for "
            f"[correct, near_miss_1, near_miss_2, counter_policy] are {lengths}. "
            "Shorten or lengthen the distractors to closely match the immutable "
            "correct answer"
        )

    return {
        "distractors": answers,
        "changed_details": details,
    }, None


def build_public_question(
    source_question: dict[str, Any],
    generated: dict[str, Any],
    institution: str,
) -> dict[str, Any]:
    correct_option = int(source_question["correct_option"])
    correct_answer = source_correct_answer(source_question)
    distractors = list(generated["distractors"])
    seed_text = f"{institution}\n{source_question['question']}"
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    random.Random(seed).shuffle(distractors)
    options = distractors
    options.insert(correct_option, correct_answer)
    return {
        "question": source_question["question"],
        "options": options,
        "correct_option": correct_option,
    }


async def post_json(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    prompt: str,
    count: int,
    description: str,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": OPENROUTER_APP_TITLE,
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": DISTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": policy.response_format(
            "counter_policy_distractors", distractor_schema(count)
        ),
        "reasoning": {"effort": policy.REASONING_EFFORT, "exclude": True},
        "provider": {
            "sort": policy.PROVIDER_SORT,
            "require_parameters": policy.REQUIRE_PROVIDER_PARAMETERS,
        },
        "max_tokens": policy.MAX_OUTPUT_TOKENS,
    }
    for attempt in range(1, MAX_REQUEST_RETRIES + 1):
        try:
            async with semaphore:
                response = await client.post(
                    policy.OPENROUTER_API_URL, headers=headers, json=payload
                )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("response content is not an object")
            print(
                f"{description}: response from {body.get('provider', 'unknown provider')}",
                flush=True,
            )
            return parsed
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 402:
                raise RuntimeError(
                    "OpenRouter returned HTTP 402 Payment Required. Fund the account "
                    "or raise its spending limit, then rerun this command to resume "
                    "from the saved checkpoints."
                ) from error
            if attempt == MAX_REQUEST_RETRIES:
                raise RuntimeError(
                    f"{description} failed after {attempt} attempts: {error}"
                ) from error
            wait_seconds = min(policy.RETRY_MAX_WAIT_SECONDS, 2 ** (attempt - 1))
            print(
                f"{description} attempt {attempt}/{MAX_REQUEST_RETRIES} failed: "
                f"{error}; retrying in {wait_seconds}s",
                flush=True,
            )
            await asyncio.sleep(wait_seconds)
    raise AssertionError("unreachable")


def load_checkpoint(
    path: Path,
    row: dict[str, Any],
    train_count: int,
    test_count: int,
) -> dict[str, Any]:
    fingerprint = source_fingerprint(row, train_count, test_count)
    if not path.exists():
        return {
            "source_fingerprint": fingerprint,
            "institution": row[policy.INSTITUTION_COLUMN],
            "train": [],
            "test": [],
        }
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("source_fingerprint") != fingerprint:
        raise ValueError(
            f"Checkpoint source mismatch: {path}. Use a new --output-dir for a "
            "different source dataset or question count."
        )
    return checkpoint


async def generate_split(
    split_name: str,
    source_questions: list[dict[str, Any]],
    institution: str,
    handbook: str,
    questions_per_request: int,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    prompt_dir: Path,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
) -> None:
    generated = checkpoint[split_name]
    while len(generated) < len(source_questions):
        start = len(generated)
        batch = source_questions[start : start + questions_per_request]
        prompt = make_prompt(institution, handbook, batch, start)
        batch_number = start // questions_per_request + 1
        prompt_path = prompt_dir / f"{split_name}_{batch_number:03d}"
        policy.write_prompt(
            prompt_path, OPENROUTER_MODEL, DISTRACTOR_SYSTEM_PROMPT, prompt
        )

        last_rejection = "unknown validation error"
        attempt_prompt = prompt
        for validation_attempt in range(1, MAX_REQUEST_RETRIES + 1):
            parsed = await post_json(
                client,
                semaphore,
                api_key,
                attempt_prompt,
                len(batch),
                f"{institution.splitlines()[0]} {split_name} batch {batch_number}",
            )
            raw_items = parsed.get("items")
            if not isinstance(raw_items, list) or len(raw_items) != len(batch):
                last_rejection = "wrong item count"
                attempt_prompt = make_retry_prompt(prompt, last_rejection)
                continue
            accepted = []
            for offset, (raw, source_question) in enumerate(zip(raw_items, batch)):
                item, rejection = validate_generated_item(
                    raw,
                    start + offset,
                    source_correct_answer(source_question),
                )
                if rejection is not None:
                    last_rejection = f"item {start + offset}: {rejection}"
                    break
                accepted.append(item)
            if len(accepted) == len(batch):
                generated.extend(accepted)
                checkpoint[split_name] = generated
                policy.atomic_write_json(checkpoint_path, checkpoint)
                print(
                    f"{institution.splitlines()[0]} {split_name}: "
                    f"{len(generated)}/{len(source_questions)}",
                    flush=True,
                )
                break
            print(
                f"{institution.splitlines()[0]} {split_name} validation attempt "
                f"{validation_attempt}/{MAX_REQUEST_RETRIES} rejected: {last_rejection}",
                flush=True,
            )
            attempt_prompt = make_retry_prompt(prompt, last_rejection)
        else:
            raise RuntimeError(
                f"Could not generate valid {split_name} batch {batch_number}: "
                f"{last_rejection}"
            )


async def generate_row(
    row_index: int,
    row: dict[str, Any],
    train_count: int,
    test_count: int,
    questions_per_request: int,
    checkpoint_dir: Path,
    prompt_root: Path,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
) -> dict[str, Any]:
    institution = row[policy.INSTITUTION_COLUMN]
    handbook = row[policy.HANDBOOK_COLUMN]
    name = institution.splitlines()[0].removeprefix("Name: ")
    slug = policy.slugify(name)
    checkpoint_path = checkpoint_dir / f"{row_index:03d}_{slug}.json"
    prompt_dir = prompt_root / f"{row_index:03d}_{slug}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path, row, train_count, test_count)
    policy.atomic_write_json(checkpoint_path, checkpoint)

    source_train = list(row[policy.TRAIN_DATA_COLUMN][:train_count])
    source_test = list(row[policy.TEST_DATA_COLUMN][:test_count])
    await generate_split(
        "train", source_train, institution, handbook, questions_per_request,
        checkpoint, checkpoint_path, prompt_dir, client, semaphore, api_key
    )
    await generate_split(
        "test", source_test, institution, handbook, questions_per_request,
        checkpoint, checkpoint_path, prompt_dir, client, semaphore, api_key
    )

    return {
        policy.INSTITUTION_COLUMN: institution,
        policy.HANDBOOK_COLUMN: handbook,
        policy.NUM_TRAIN_COLUMN: train_count,
        policy.NUM_TEST_COLUMN: test_count,
        policy.TRAIN_DATA_COLUMN: [
            build_public_question(question, generated, institution)
            for question, generated in zip(source_train, checkpoint["train"])
        ],
        policy.TEST_DATA_COLUMN: [
            build_public_question(question, generated, institution)
            for question, generated in zip(source_test, checkpoint["test"])
        ],
    }


async def generate_all(
    rows: list[dict[str, Any]],
    train_count: int,
    test_count: int,
    questions_per_request: int,
    checkpoint_dir: Path,
    prompt_dir: Path,
    api_key: str,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    timeout = httpx.Timeout(policy.REQUEST_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await asyncio.gather(
            *[
                generate_row(
                    index, row, train_count, test_count, questions_per_request,
                    checkpoint_dir, prompt_dir, client, semaphore, api_key
                )
                for index, row in enumerate(rows)
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", default=SOURCE_DATASET)
    parser.add_argument("--source-config", default=SOURCE_CONFIG)
    parser.add_argument("--source-split", default=SOURCE_SPLIT)
    parser.add_argument("--institution-limit", type=int, default=policy.NUM_INSTITUTIONS)
    parser.add_argument("--train-count", type=int, default=policy.TRAIN_QUESTIONS)
    parser.add_argument("--test-count", type=int, default=policy.TEST_QUESTIONS)
    parser.add_argument("--questions-per-request", type=int, default=QUESTIONS_PER_REQUEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dataset", default=OUTPUT_DATASET)
    parser.add_argument(
        "--upload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload the completed production dataset (default: true).",
    )
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.institution_limit < 1:
        raise ValueError("--institution-limit must be positive")
    if args.train_count < 0 or args.test_count < 0:
        raise ValueError("Question counts cannot be negative")
    if args.train_count + args.test_count == 0:
        raise ValueError("At least one train or test question is required")
    if not 1 <= args.questions_per_request <= QUESTIONS_PER_REQUEST:
        raise ValueError(
            f"--questions-per-request must be between 1 and {QUESTIONS_PER_REQUEST}"
        )

    source = datasets.load_dataset(
        args.source_dataset,
        args.source_config,
        split=args.source_split,
    )
    if args.institution_limit > len(source):
        raise ValueError(
            f"--institution-limit={args.institution_limit} exceeds the "
            f"{len(source)} source rows"
        )
    rows = [dict(row) for row in source.select(range(args.institution_limit))]
    for row_index, row in enumerate(rows):
        if len(row[policy.TRAIN_DATA_COLUMN]) < args.train_count:
            raise ValueError(f"Source row {row_index} has too few train questions")
        if len(row[policy.TEST_DATA_COLUMN]) < args.test_count:
            raise ValueError(f"Source row {row_index} has too few test questions")

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    api_key = os.environ.get(policy.OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"{policy.OPENROUTER_API_KEY_ENV} is missing from {repo_root / '.env'}"
        )

    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / CHECKPOINT_DIR_NAME
    prompt_dir = output_dir / PROMPT_DIR_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    generated = asyncio.run(
        generate_all(
            rows,
            args.train_count,
            args.test_count,
            args.questions_per_request,
            checkpoint_dir,
            prompt_dir,
            api_key,
        )
    )
    dataset = datasets.Dataset.from_list(generated, features=policy.OUTPUT_FEATURES)
    jsonl_path = output_dir / DATASET_JSONL_NAME
    samples_path = output_dir / SAMPLES_MARKDOWN_NAME
    policy.write_jsonl(dataset, jsonl_path)
    policy.write_readable_samples(dataset, samples_path)
    print(f"Wrote {jsonl_path}", flush=True)
    print(f"Wrote {samples_path}", flush=True)

    if args.upload:
        if len(dataset) != policy.NUM_INSTITUTIONS:
            raise ValueError(
                f"Refusing to upload a smoke-test dataset with {len(dataset)} rows; "
                f"production requires {policy.NUM_INSTITUTIONS}"
            )
        if args.train_count != policy.TRAIN_QUESTIONS or args.test_count != policy.TEST_QUESTIONS:
            raise ValueError("Refusing to upload non-production train/test counts")
        result = dataset.push_to_hub(
            args.output_dataset,
            config_name=policy.OUTPUT_CONFIG,
            split=policy.OUTPUT_SPLIT,
            private=args.private,
        )
        print(result, flush=True)


if __name__ == "__main__":
    main()
