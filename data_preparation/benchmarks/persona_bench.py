"""Generate the PersonaBench few-shot benchmark and upload it to the Hub.

The script is resumable: accepted conversations are checkpointed after every
OpenRouter response.  A production invocation uses the constants below; the
CLI count overrides are intended for inexpensive smoke tests.
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

import datasets
import httpx
from dotenv import load_dotenv
from transformers import AutoTokenizer, PreTrainedTokenizerBase


SOURCE_DATASET = "nvidia/Nemotron-Personas-USA"
SOURCE_SPLIT = "train"
OUTPUT_DATASET = "aklein4/PersonaBench"
OUTPUT_CONFIG = "default"
OUTPUT_SPLIT = "train"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "~deepseek/deepseek-v4-flash-latest"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_APP_TITLE = "PersonaBench dataset generation"

TOKENIZER_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_CHAT_TOKENS = 1024

NUM_PERSONAS = 64
TRAIN_CONVERSATIONS = 256
TEST_CONVERSATIONS = 100
CONVERSATIONS_PER_PERSONA = TRAIN_CONVERSATIONS + TEST_CONVERSATIONS
PAIRS_PER_REQUEST = 16
MAX_OUTPUT_TOKENS = 65_536
MAX_CONCURRENT_REQUESTS = 8
MAX_REQUEST_RETRIES = 8
MAX_ROUND_MULTIPLIER = 4
REQUEST_TIMEOUT_SECONDS = 300
RETRY_MAX_WAIT_SECONDS = 60
REASONING_EFFORT = "low"
PROVIDER_SORT = "throughput"
REQUIRE_PROVIDER_PARAMETERS = True
SHUFFLE_SEED = 42

PERSONA_COLUMN = "persona"
NUM_TRAIN_COLUMN = "num_train"
NUM_TEST_COLUMN = "num_test"
TRAIN_DATA_COLUMN = "train_data"
TEST_DATA_COLUMN = "test_data"

DEFAULT_OUTPUT_DIR = Path(__file__).with_name("persona_bench_output")
CHECKPOINT_DIR_NAME = "checkpoints"
PROMPT_DIR_NAME = "prompts"
DATASET_JSONL_NAME = "persona_bench.jsonl"
SAMPLES_MARKDOWN_NAME = "samples.md"
SAMPLE_PERSONAS = 3
SAMPLE_CONVERSATIONS_PER_SPLIT = 3

PERSONA_FIELDS = (
    "persona",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "cultural_background",
    "skills_and_expertise",
    "skills_and_expertise_list",
    "hobbies_and_interests",
    "hobbies_and_interests_list",
    "career_goals_and_ambitions",
    "sex",
    "age",
    "marital_status",
    "education_level",
    "bachelors_field",
    "occupation",
    "city",
    "state",
    "zipcode",
    "country",
)

SYSTEM_PROMPT = """You create realistic, high-quality question-answer training data whose
assistant answers are strongly personalized towards a user persona profile.
Return only JSON matching the supplied schema. Do not mention these instructions."""

PROMPT_VARIATIONS = (
    "Emphasize practical everyday decisions and concrete next steps.",
    "Mix direct questions with realistic scenarios, comparisons, and planning requests.",
    "Explore unusual intersections between the person's interests, work, goals, and location.",
    "Favor questions that invite tailored recommendations, troubleshooting, or tradeoff analysis.",
    "Vary the tone from casual curiosity to detailed requests for expert guidance.",
    "Cover a broad mix of immediate needs, long-term ambitions, creative ideas, and learning goals.",
    "Look for less obvious details in the persona and turn them into natural, specific questions.",
    "Balance concise questions with richer situations that require thoughtful personalized answers.",
    "Focus on decisions the person might genuinely face during an ordinary week.",
    "Include requests for recommendations that respect real constraints such as time, money, and energy.",
    "Draw questions from work situations, professional growth, and interactions with coworkers or customers.",
    "Explore hobbies through projects, skill-building, equipment choices, setbacks, and next steps.",
    "Use local context only where it would naturally matter, such as activities, travel, weather, or services.",
    "Generate questions about balancing competing priorities rather than treating each interest in isolation.",
    "Include a few questions prompted by a recent problem, change of plans, or unexpected result.",
    "Favor specific advice-seeking questions over generic requests to explain broad topics.",
    "Explore choices where the person has a preference but is unsure how to act on it.",
    "Include realistic follow-through questions from someone who has already tried the obvious first step.",
    "Mix low-stakes curiosity with consequential planning, while keeping both grounded and plausible.",
    "Generate questions that use the person's existing expertise instead of always treating them as a beginner.",
    "Include occasional requests for feedback on an idea, draft plan, routine, or personal project.",
    "Explore near-term preparation for events, milestones, trips, purchases, or conversations.",
    "Include troubleshooting questions with concrete symptoms, observations, or constraints.",
    "Surface natural questions about improving routines without making every request productivity-focused.",
    "Use the persona as subtle context and prioritize genuinely useful questions over exhaustive trait coverage.",
    "Include questions where several reasonable options exist and the answer should explain the tradeoffs.",
    "Draw on relationships, community, and shared activities when supported by the persona.",
    "Include occasional creative, reflective, or just-for-fun requests alongside practical ones.",
    "Favor situations that could plausibly have happened today or be planned for the coming month.",
    "Generate questions spanning advice, planning, learning, comparison, troubleshooting, and brainstorming.",
    "Seek underrepresented but plausible needs instead of repeating the persona's most prominent ambition.",
    "Make the batch feel like messages from one multifaceted person, not a checklist of profile attributes.",
)

LENGTH_VARIATIONS = (
    "Make most questions one short sentence and most answers a compact 2-4 sentences.",
    "Mix very short questions with medium-length answers of roughly 80-160 words.",
    "Use context-rich questions for some pairs, but answer them directly and concisely.",
    "Create a broad length distribution: quick exchanges, medium explanations, and a few detailed answers.",
    "Pair several brief questions with thorough 180-300 word answers when the topic warrants depth.",
    "Include detailed 2-4 sentence questions followed by focused answers that avoid restating the setup.",
    "Favor medium-length conversational questions and vary answers from one paragraph to several paragraphs.",
    "Use a bimodal mix: some very concise exchanges and some substantial, nuanced exchanges.",
    "Let simple questions receive short answers and reserve longer answers for genuinely complex requests.",
    "Vary length organically within the batch; do not make every question or answer resemble its neighbors.",
    "Include a few questions with just enough backstory to explain a constraint, plus several quick questions.",
    "Keep some answers under 60 words while allowing a few useful step-by-step answers of 200-350 words.",
)

REALISM_VARIATIONS = (
    "Write like a real chat user: natural phrasing, contractions where appropriate, and no profile-summary language.",
    "Put only details the person would naturally volunteer in the question; the answer may use additional remembered details.",
    "Make personalization feel like useful memory rather than a recitation of profile fields.",
    "Give each question a believable motivation, constraint, or desired outcome when one is needed.",
    "Vary how much context the user supplies, while keeping every exchange understandable on its own.",
    "Avoid survey-like wording, artificial transitions, and questions designed merely to expose a persona trait.",
    "Use the person's likely vocabulary and expertise level without caricaturing their background.",
    "Make answers candid about tradeoffs and uncertainty instead of presenting every suggestion as certain.",
    "Let the assistant confidently use relevant profile details that the user did not repeat in the question.",
    "Prefer actionable, situation-aware answers over generic encouragement or repeated summaries of the question.",
    "Allow ordinary, imperfect circumstances and modest goals; not every exchange needs to be aspirational.",
    "Keep the interaction warm but matter-of-fact, as in a useful everyday assistant conversation.",
)

PERSONALIZATION_VARIATIONS = (
    "Ground each answer in a combination of the person's practical constraints and longer-term goals.",
    "Calibrate advice to the person's existing expertise, skipping beginner material they would already know.",
    "Choose concrete examples, places, activities, or options that fit the person's location and preferences.",
    "Connect immediate advice to a relevant routine, habit, project, or ambition from the persona.",
    "Use the person's tastes to select among otherwise reasonable recommendations and explain why they fit.",
    "Tailor timing, cost, effort, and risk to the person's actual work schedule and priorities.",
    "Draw on two different persona sections in each answer when they combine naturally.",
    "Personalize through specific next steps the person is unusually well prepared to take.",
    "Account for the person's prior skills and likely resources instead of giving one-size-fits-all advice.",
    "Use remembered context to anticipate a constraint or preference that a generic answer would miss.",
    "Where useful, relate the answer to the person's community, cultural background, or local environment.",
    "Make recommendations consistent with both the person's near-term situation and future plans.",
    "Select examples that match the person's hobbies and aesthetic tastes rather than generic popular choices.",
    "Use personalization to change the substance of the answer, not merely its greeting or final sentence.",
    "Treat the exchange as part of an ongoing assistant relationship that remembers what matters to this user.",
    "Avoid repeatedly relying on the single most obvious persona trait; use less prominent details too.",
)

USER_PROMPT_TEMPLATE = """Use the complete persona below to create {count} distinct questions this
person might naturally ask an assistant, each followed by a useful personalized answer.

Requirements:
- The assistant answer is the primary target. It must depend heavily on the persona and should
  be substantially different from the answer that would be given to an unknown, generic user.
- Treat the persona as persistent memory available to the assistant. The answer may and should
  use relevant facts that the person did not repeat in the current question.
- Normally weave at least one or two concrete, relevant persona details into each answer. Use them to
  change the recommendation, examples, priorities, level of explanation, or proposed next steps;
  superficial name-dropping does not count as personalization.
- Questions are secondary: keep them realistic and natural. At least half should be ordinary,
  broadly phrased requests containing no more than one persona-specific detail. They do not need
  to restate the facts that the personalized answer draws upon.
- Across the batch, spread personalization across different sections of the persona instead of
  repeatedly using only the most prominent hobby, job, or ambition. Unless truly necessary, do
  not reuse one detail or motif in more than one third of the answers.
- Never say that you were given a persona/profile, never enumerate profile fields, and never
  reproduce the profile wholesale. Integrate remembered facts as naturally as a familiar assistant.
- Never invent remembered facts. Do not fabricate schedules, affiliations, past events, exact
  local details, or preferences that are not stated in the persona or current question.
- Write answers in the assistant's voice, addressing the person as "you" where appropriate.
  Never impersonate the person or answer as though the assistant has the person's identity.
- Make each question understandable on its own, but do not stuff it with biographical details
  merely to justify personalization in the answer.
- Do not put the person's name into every question. Avoid near-duplicates and repetitive templates.
- Keep lengths natural for the request. Questions may range from a few words to a context-rich
  paragraph, and answers may range from a few sentences to a detailed explanation when warranted.
- Before returning a pair, ask: "Would this answer still be essentially the same for a generic
  user?" If yes, revise it so relevant persona memory materially shapes the answer.
- Do not add system messages, speaker labels, markdown wrappers around the JSON, or extra fields.

This is generation batch {batch_number}, covering conceptual items {start_index}-{end_index}.
Topic and task direction for this independent batch:
{variation}

Length-profile direction:
{length_variation}

Realism direction:
{realism_variation}

Answer-personalization direction:
{personalization_variation}

Complete persona:

{persona}
"""

MESSAGE_FEATURE = {
    "role": datasets.Value("string"),
    "content": datasets.Value("string"),
}
OUTPUT_FEATURES = datasets.Features(
    {
        PERSONA_COLUMN: datasets.Value("string"),
        NUM_TRAIN_COLUMN: datasets.Value("int64"),
        NUM_TEST_COLUMN: datasets.Value("int64"),
        TRAIN_DATA_COLUMN: [[MESSAGE_FEATURE]],
        TEST_DATA_COLUMN: [[MESSAGE_FEATURE]],
    }
)


def normalize_whitespace(value: Any) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def field_label(field: str) -> str:
    return field.replace("_", " ").title()


def format_persona(row: dict[str, Any]) -> str:
    """Preserve the full source row as an easy-to-read, labeled persona."""
    missing = [field for field in PERSONA_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Source persona is missing fields: {missing}")

    sections = []
    for field in PERSONA_FIELDS:
        value = row[field]
        rendered = "" if value is None else normalize_whitespace(value)
        if field == "persona":
            sections.append(f"## General Persona\n{rendered}")
        else:
            sections.append(f"## {field_label(field)}\n{rendered}")
    return "\n\n".join(sections)


def load_personas(limit: int) -> list[dict[str, Any]]:
    source = datasets.load_dataset(SOURCE_DATASET, split=SOURCE_SPLIT, streaming=True)
    rows = list(source.take(limit))
    if len(rows) != limit:
        raise RuntimeError(f"Loaded {len(rows)} personas; expected {limit}")
    return rows


def format_messages(question: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": normalize_whitespace(question)},
        {"role": "assistant", "content": normalize_whitespace(answer)},
    ]


def chat_token_count(
    messages: list[dict[str, str]], tokenizer: PreTrainedTokenizerBase
) -> int:
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return len(input_ids)


def question_key(question: str) -> str:
    return re.sub(r"\W+", " ", question.casefold()).strip()


def validate_pair(
    pair: Any,
    tokenizer: PreTrainedTokenizerBase,
    seen_questions: set[str],
) -> tuple[list[dict[str, str]] | None, str | None]:
    if not isinstance(pair, dict):
        return None, "not an object"

    question = normalize_whitespace(pair.get("question", ""))
    answer = normalize_whitespace(pair.get("answer", ""))
    if not question or not answer:
        return None, "empty question or answer"

    key = question_key(question)
    if key in seen_questions:
        return None, "duplicate question"

    messages = format_messages(question, answer)
    if chat_token_count(messages, tokenizer) > MAX_CHAT_TOKENS:
        return None, "over token limit"

    return messages, None


def response_schema(count: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "persona_qa_pairs",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "pairs": {
                        "type": "array",
                        "minItems": count,
                        "maxItems": count,
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "answer": {"type": "string"},
                            },
                            "required": ["question", "answer"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["pairs"],
                "additionalProperties": False,
            },
        },
    }


def parse_response(response: httpx.Response) -> list[dict[str, str]]:
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    parsed = json.loads(content)
    pairs = parsed.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("Model response did not contain a pairs array")
    return pairs


async def request_pairs(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    persona: str,
    count: int,
    batch_number: int,
    start_index: int,
    prompt_path: Path,
) -> list[dict[str, str]]:
    variation_rng = random.SystemRandom()
    prompt = USER_PROMPT_TEMPLATE.format(
        count=count,
        batch_number=batch_number,
        start_index=start_index,
        end_index=start_index + count - 1,
        variation=variation_rng.choice(PROMPT_VARIATIONS),
        length_variation=variation_rng.choice(LENGTH_VARIATIONS),
        realism_variation=variation_rng.choice(REALISM_VARIATIONS),
        personalization_variation=variation_rng.choice(PERSONALIZATION_VARIATIONS),
        persona=persona,
    )
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": response_schema(count),
        "reasoning": {
            "effort": REASONING_EFFORT,
            "exclude": True,
        },
        "provider": {
            "sort": PROVIDER_SORT,
            "require_parameters": REQUIRE_PROVIDER_PARAMETERS,
        },
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    atomic_write_json(prompt_path, payload)
    write_readable_prompt(
        prompt_path.with_suffix(".md"),
        model=OPENROUTER_MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": OPENROUTER_APP_TITLE,
    }

    for attempt in range(1, MAX_REQUEST_RETRIES + 1):
        try:
            async with semaphore:
                response = await client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                )
            response.raise_for_status()
            pairs = parse_response(response)
            provider = response.json().get("provider", "unknown provider")
            print(
                f"Batch {batch_number}: received {len(pairs)} pairs from {provider}",
                flush=True,
            )
            return pairs
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            if attempt == MAX_REQUEST_RETRIES:
                raise RuntimeError(
                    f"OpenRouter request failed after {attempt} attempts: {error}"
                ) from error
            retry_after = 0
            if isinstance(error, httpx.HTTPStatusError):
                try:
                    retry_after = int(error.response.headers.get("retry-after", "0"))
                except ValueError:
                    retry_after = 0
            wait_seconds = min(
                RETRY_MAX_WAIT_SECONDS,
                max(retry_after, 2 ** (attempt - 1)),
            )
            print(
                f"OpenRouter attempt {attempt}/{MAX_REQUEST_RETRIES} failed: {error}; "
                f"retrying in {wait_seconds}s",
                flush=True,
            )
            await asyncio.sleep(wait_seconds)

    raise AssertionError("unreachable")


def atomic_write_json(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_readable_prompt(
    path: Path,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# PersonaBench prompt example",
                "",
                f"Model: `{model}`",
                "",
                "## System message",
                "",
                system_prompt,
                "",
                "## User message",
                "",
                user_prompt,
                "",
            ]
        ),
        encoding="utf-8",
    )


def load_checkpoint(path: Path, persona: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "persona": persona,
            "conversations": [],
            "rejections": {},
            "rounds": 0,
        }
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("persona") != persona:
        raise ValueError(f"Checkpoint persona mismatch: {path}")
    return checkpoint


async def generate_for_persona(
    persona_index: int,
    persona_row: dict[str, Any],
    target_count: int,
    pairs_per_request: int,
    tokenizer: PreTrainedTokenizerBase,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    checkpoint_dir: Path,
    prompt_dir: Path,
) -> dict[str, Any]:
    persona = format_persona(persona_row)
    uuid = normalize_whitespace(persona_row["uuid"])
    checkpoint_path = checkpoint_dir / f"{persona_index:03d}_{uuid}.json"
    persona_prompt_dir = prompt_dir / f"{persona_index:03d}_{uuid}"
    persona_prompt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path, persona)
    conversations = checkpoint["conversations"][:target_count]
    seen_questions = {
        question_key(conversation[0]["content"]) for conversation in conversations
    }
    rejection_counts = checkpoint.get("rejections", {})
    rounds = int(checkpoint.get("rounds", 0))
    max_rounds = math.ceil(target_count / pairs_per_request) * MAX_ROUND_MULTIPLIER

    while len(conversations) < target_count:
        if rounds >= max_rounds:
            raise RuntimeError(
                f"Persona {persona_index} has only {len(conversations)}/{target_count} valid "
                f"pairs after {rounds} rounds; rejections={rejection_counts}"
            )

        shortfall = target_count - len(conversations)
        available_rounds = max_rounds - rounds
        wave_size = min(
            MAX_CONCURRENT_REQUESTS,
            math.ceil(shortfall / pairs_per_request),
            available_rounds,
        )
        wave_start = len(conversations) + 1
        requests = []
        request_metadata = []
        for wave_index in range(wave_size):
            requested = min(
                pairs_per_request,
                shortfall - wave_index * pairs_per_request,
            )
            batch_number = rounds + wave_index + 1
            start_index = wave_start + wave_index * pairs_per_request
            request_metadata.append((batch_number, requested))
            requests.append(
                request_pairs(
                    client=client,
                    semaphore=semaphore,
                    api_key=api_key,
                    persona=persona,
                    count=requested,
                    batch_number=batch_number,
                    start_index=start_index,
                    prompt_path=persona_prompt_dir / f"round_{batch_number:03d}.json",
                )
            )

        rounds += wave_size
        print(
            f"Persona {persona_index + 1}: dispatching {wave_size} independent batches "
            f"for a shortfall of {shortfall}",
            flush=True,
        )
        results = await asyncio.gather(*requests, return_exceptions=True)

        for (batch_number, requested), result in zip(request_metadata, results):
            if len(conversations) >= target_count:
                break
            if isinstance(result, BaseException):
                rejection_counts["request failure"] = (
                    rejection_counts.get("request failure", 0) + requested
                )
                print(
                    f"Persona {persona_index + 1}, batch {batch_number}: {result}",
                    flush=True,
                )
                continue

            accepted_this_batch = 0
            for pair in result:
                messages, rejection = validate_pair(pair, tokenizer, seen_questions)
                if rejection is not None:
                    rejection_counts[rejection] = rejection_counts.get(rejection, 0) + 1
                    continue
                question = messages[0]["content"]
                seen_questions.add(question_key(question))
                conversations.append(messages)
                accepted_this_batch += 1
                if len(conversations) == target_count:
                    break

            checkpoint = {
                "persona": persona,
                "conversations": conversations,
                "rejections": rejection_counts,
                "rounds": rounds,
            }
            atomic_write_json(checkpoint_path, checkpoint)
            print(
                f"Persona {persona_index + 1}, batch {batch_number}: accepted "
                f"{accepted_this_batch}/{len(result)}; total "
                f"{len(conversations)}/{target_count}",
                flush=True,
            )

    return {"persona": persona, "conversations": conversations}


async def generate_all(
    persona_rows: list[dict[str, Any]],
    target_count: int,
    pairs_per_request: int,
    tokenizer: PreTrainedTokenizerBase,
    api_key: str,
    checkpoint_dir: Path,
    prompt_dir: Path,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            generate_for_persona(
                persona_index=index,
                persona_row=row,
                target_count=target_count,
                pairs_per_request=pairs_per_request,
                tokenizer=tokenizer,
                client=client,
                semaphore=semaphore,
                api_key=api_key,
                checkpoint_dir=checkpoint_dir,
                prompt_dir=prompt_dir,
            )
            for index, row in enumerate(persona_rows)
        ]
        return await asyncio.gather(*tasks)


def pack_dataset(
    generated: list[dict[str, Any]], train_count: int, test_count: int
) -> datasets.Dataset:
    rows = []
    for row_index, item in enumerate(generated):
        conversations = list(item["conversations"])
        expected = train_count + test_count
        if len(conversations) != expected:
            raise ValueError(f"Expected {expected} conversations, got {len(conversations)}")
        random.Random(SHUFFLE_SEED + row_index).shuffle(conversations)
        rows.append(
            {
                PERSONA_COLUMN: item["persona"],
                NUM_TRAIN_COLUMN: train_count,
                NUM_TEST_COLUMN: test_count,
                TRAIN_DATA_COLUMN: conversations[:train_count],
                TEST_DATA_COLUMN: conversations[train_count:],
            }
        )
    return datasets.Dataset.from_list(rows, features=OUTPUT_FEATURES)


def write_jsonl(dataset: datasets.Dataset, path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in dataset:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_readable_samples(dataset: datasets.Dataset, path: Path) -> None:
    lines = ["# PersonaBench generation samples", ""]
    for row_index, row in enumerate(dataset.select(range(min(SAMPLE_PERSONAS, len(dataset))))):
        lines.extend([f"## Row {row_index}", "", "### Persona", "", row[PERSONA_COLUMN], ""])
        for split_column, split_label in (
            (TRAIN_DATA_COLUMN, "Train"),
            (TEST_DATA_COLUMN, "Test"),
        ):
            conversations = row[split_column][:SAMPLE_CONVERSATIONS_PER_SPLIT]
            for conversation_index, conversation in enumerate(conversations):
                lines.extend(
                    [
                        f"### {split_label} conversation {conversation_index}",
                        "",
                        f"**User:** {conversation[0]['content']}",
                        "",
                        f"**Assistant:** {conversation[1]['content']}",
                        "",
                    ]
                )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona-limit", type=int, default=NUM_PERSONAS)
    parser.add_argument("--train-count", type=int, default=TRAIN_CONVERSATIONS)
    parser.add_argument("--test-count", type=int, default=TEST_CONVERSATIONS)
    parser.add_argument("--pairs-per-request", type=int, default=PAIRS_PER_REQUEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dataset", default=OUTPUT_DATASET)
    parser.add_argument(
        "--upload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload the completed single-config train split (default: true).",
    )
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.persona_limit <= NUM_PERSONAS:
        raise ValueError(f"--persona-limit must be between 1 and {NUM_PERSONAS}")
    if args.train_count < 0 or args.test_count < 0:
        raise ValueError("Conversation counts cannot be negative")
    target_count = args.train_count + args.test_count
    if target_count == 0:
        raise ValueError("At least one train or test conversation is required")
    if not 1 <= args.pairs_per_request <= PAIRS_PER_REQUEST:
        raise ValueError(f"--pairs-per-request must be between 1 and {PAIRS_PER_REQUEST}")

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{OPENROUTER_API_KEY_ENV} is missing from {repo_root / '.env'}")

    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / CHECKPOINT_DIR_NAME
    prompt_dir = output_dir / PROMPT_DIR_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading the first {args.persona_limit} personas from {SOURCE_DATASET}...", flush=True)
    persona_rows = load_personas(args.persona_limit)
    print(f"Loading tokenizer {TOKENIZER_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    generated = asyncio.run(
        generate_all(
            persona_rows=persona_rows,
            target_count=target_count,
            pairs_per_request=args.pairs_per_request,
            tokenizer=tokenizer,
            api_key=api_key,
            checkpoint_dir=checkpoint_dir,
            prompt_dir=prompt_dir,
        )
    )
    dataset = pack_dataset(generated, args.train_count, args.test_count)

    jsonl_path = output_dir / DATASET_JSONL_NAME
    samples_path = output_dir / SAMPLES_MARKDOWN_NAME
    write_jsonl(dataset, jsonl_path)
    write_readable_samples(dataset, samples_path)
    print(f"Wrote {jsonl_path}", flush=True)
    print(f"Wrote {samples_path}", flush=True)

    if args.upload:
        if len(dataset) != NUM_PERSONAS:
            raise ValueError(
                f"Refusing to upload a smoke-test dataset with {len(dataset)} rows; "
                f"production requires {NUM_PERSONAS}"
            )
        if args.train_count != TRAIN_CONVERSATIONS or args.test_count != TEST_CONVERSATIONS:
            raise ValueError("Refusing to upload non-production train/test counts")
        print(f"Uploading {len(dataset)} rows to {args.output_dataset}...", flush=True)
        result = dataset.push_to_hub(
            args.output_dataset,
            config_name=OUTPUT_CONFIG,
            split=OUTPUT_SPLIT,
            private=args.private,
        )
        print(result, flush=True)


if __name__ == "__main__":
    main()
