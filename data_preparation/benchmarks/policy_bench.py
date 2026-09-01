"""Generate PolicyBench, a multiple-choice in-context policy-learning benchmark.

Each row is a fictional institution.  An API model first writes that institution's
private procedure handbook, then writes onboarding questions grounded in numbered
rules from it.  The handbook is retained as generation metadata; benchmark models
are intended to infer the local rules from the train examples, not read the handbook.

Generation is resumable.  A production invocation uses the constants below; the CLI
count overrides are intended for inexpensive smoke tests.
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import datasets
import httpx
from dotenv import load_dotenv


OUTPUT_DATASET = "aklein4/PolicyBench"
OUTPUT_CONFIG = "default"
OUTPUT_SPLIT = "train"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "~deepseek/deepseek-v4-flash-latest"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_APP_TITLE = "PolicyBench dataset generation"

NUM_INSTITUTIONS = 64
TRAIN_QUESTIONS = 256
TEST_QUESTIONS = 100
QUESTIONS_PER_REQUEST = 16
HANDBOOK_RULE_COUNT = 16
HANDBOOK_MIN_WORDS = 2_000
HANDBOOK_MAX_WORDS = 8_000
MAX_OUTPUT_TOKENS = 65_536
MAX_CONCURRENT_REQUESTS = 8
MAX_REQUEST_RETRIES = 8
MAX_QUESTION_ROUND_MULTIPLIER = 4
REQUEST_TIMEOUT_SECONDS = 300
RETRY_MAX_WAIT_SECONDS = 60
REASONING_EFFORT = "low"
PROVIDER_SORT = "throughput"
REQUIRE_PROVIDER_PARAMETERS = True
SHUFFLE_SEED = 42

INSTITUTION_COLUMN = "Institution"
HANDBOOK_COLUMN = "Handbook"
NUM_TRAIN_COLUMN = "num_train"
NUM_TEST_COLUMN = "num_test"
TRAIN_DATA_COLUMN = "train_data"
TEST_DATA_COLUMN = "test_data"

DEFAULT_OUTPUT_DIR = Path(__file__).with_name("policy_bench_output")
CHECKPOINT_DIR_NAME = "checkpoints"
PROMPT_DIR_NAME = "prompts"
DATASET_JSONL_NAME = "policy_bench.jsonl"
SAMPLES_MARKDOWN_NAME = "samples.md"
SAMPLE_INSTITUTIONS = 3


# These are deliberately fictional and span settings in which local procedure matters.
# The short brief constrains the handbook's domain without pre-specifying its policies.
INSTITUTIONS: tuple[dict[str, str], ...] = (
    {"name": "Alderwick Community Hospital", "sector": "regional healthcare", "brief": "A 180-bed hospital coordinating inpatient care, visitors, records, facilities, and clinical support services."},
    {"name": "Beacon Harbor Credit Union", "sector": "retail financial services", "brief": "A member-owned credit union handling branches, remote service, fraud reports, lending, and account maintenance."},
    {"name": "Cedar Vale Public Library Network", "sector": "public libraries", "brief": "A seven-branch library system managing circulation, rooms, events, archives, devices, and patron concerns."},
    {"name": "Driftwood Regional Transit Authority", "sector": "public transportation", "brief": "A bus and light-rail operator managing riders, service disruptions, lost property, accessibility, and field incidents."},
    {"name": "Eastmere Polytechnic Institute", "sector": "higher education", "brief": "A technical college operating labs, workshops, student services, placements, assessment, and campus facilities."},
    {"name": "Foxglove County Benefits Office", "sector": "public benefits administration", "brief": "A county office processing applications, evidence, renewals, appeals, appointments, and vulnerable-client cases."},
    {"name": "Granite Peak Outdoor Cooperative", "sector": "consumer retail cooperative", "brief": "A member-owned outdoor retailer offering rentals, repairs, classes, returns, and guided local trips."},
    {"name": "Hearthline Family Housing Trust", "sector": "affordable housing", "brief": "A nonprofit landlord handling applications, repairs, inspections, transfers, resident support, and contractors."},
    {"name": "Ironwood Municipal Water Service", "sector": "public utilities", "brief": "A water utility handling meters, leaks, billing, field access, boil notices, and commercial accounts."},
    {"name": "Juniper Bay Animal Rescue", "sector": "animal welfare", "brief": "A rescue operating intake, foster care, adoptions, veterinary coordination, volunteers, and donor communications."},
    {"name": "Kingswell Museum of Industry", "sector": "museum and cultural heritage", "brief": "A museum managing collections, loans, researchers, public programs, conservation, and venue operations."},
    {"name": "Larkspur Child Development Center", "sector": "early childhood education", "brief": "A childcare center managing authorized pickup, health events, family communication, staffing, and activities."},
    {"name": "Morrowgate Parcel Cooperative", "sector": "logistics and delivery", "brief": "A local parcel network coordinating depots, couriers, damaged goods, address corrections, and secure deliveries."},
    {"name": "Northstar Civic Theatre", "sector": "performing arts", "brief": "A producing theatre managing rehearsals, ticketing, accessibility, volunteers, performers, and venue safety."},
    {"name": "Orchard Hill Food Bank", "sector": "food assistance", "brief": "A food bank running intake, distributions, partner agencies, dietary requests, volunteers, and recalls."},
    {"name": "Pinehaven Veterinary Referral Center", "sector": "veterinary medicine", "brief": "A specialty veterinary center coordinating referrals, records, urgent arrivals, owners, pharmacy, and follow-up."},
    {"name": "Quarry Lane Construction Group", "sector": "commercial construction", "brief": "A contractor managing job sites, permits, subcontractors, change orders, equipment, and incident reporting."},
    {"name": "Redfern Research Foundation", "sector": "scientific research", "brief": "A nonprofit research institute overseeing laboratories, data, samples, collaborators, purchasing, and publication."},
    {"name": "Silver Birch Senior Living", "sector": "residential elder care", "brief": "An assisted-living community coordinating residents, families, activities, belongings, vendors, and care escalation."},
    {"name": "Tern Island Conservation Society", "sector": "environmental conservation", "brief": "A conservation nonprofit managing fieldwork, wildlife sightings, volunteers, land access, data, and public reports."},
    {"name": "Umberfield Community Pharmacy", "sector": "community pharmacy", "brief": "A pharmacy handling prescriptions, pickups, deliveries, stock exceptions, patient queries, and prescriber contact."},
    {"name": "Verdant Row Property Management", "sector": "commercial property management", "brief": "A property manager coordinating tenants, access, repairs, contractors, rent issues, and building emergencies."},
    {"name": "Westhaven Emergency Shelter", "sector": "homelessness services", "brief": "A shelter managing admission, beds, belongings, visitors, referrals, conflicts, and severe-weather operations."},
    {"name": "Yarrow Creek School District", "sector": "primary and secondary education", "brief": "A school district coordinating attendance, transport, devices, field trips, records, and family requests."},
    {"name": "Zephyr Point Marina Authority", "sector": "marina operations", "brief": "A public marina managing berths, fueling, weather response, contractors, visitors, and environmental incidents."},
    {"name": "Ashcombe Legal Aid Clinic", "sector": "legal services", "brief": "A legal-aid clinic handling intake, conflicts, documents, appointments, interpreters, and case communications."},
    {"name": "Blue Heron Fisheries Exchange", "sector": "food supply and wholesale", "brief": "A wholesale exchange coordinating landings, cold storage, inspections, buyers, rejected lots, and traceability."},
    {"name": "Copperfield Data Hosting", "sector": "cloud infrastructure", "brief": "A hosting provider managing access, incidents, maintenance, customer requests, backups, and security reports."},
    {"name": "Dovetail Home Care Agency", "sector": "home care", "brief": "A home-care provider scheduling visits, handling missed calls, client keys, family updates, and worker safety."},
    {"name": "Evergreen Arts Council", "sector": "arts funding", "brief": "A grant maker overseeing applications, panels, conflicts, awards, changes, reporting, and public events."},
    {"name": "Fairwind Municipal Airport", "sector": "airport operations", "brief": "A small airport coordinating terminal operations, badges, contractors, lost property, weather, and ground incidents."},
    {"name": "Glenhaven Community Bank", "sector": "commercial banking", "brief": "A community bank serving small businesses through branches, lending, cash operations, complaints, and fraud support."},
    {"name": "Highwater Aquatics Center", "sector": "public recreation", "brief": "A municipal pool complex managing lessons, memberships, lockers, incidents, water quality, and group bookings."},
    {"name": "Inkwell University Press", "sector": "academic publishing", "brief": "A publisher coordinating manuscripts, peer review, permissions, production, author changes, and corrections."},
    {"name": "Jasper Ridge Energy Cooperative", "sector": "electric utility", "brief": "A rural electric cooperative handling outages, field work, member accounts, vegetation, contractors, and safety."},
    {"name": "Kestrel County Election Service", "sector": "election administration", "brief": "A county service managing poll workers, ballots, observers, equipment, chain of custody, and voter assistance."},
    {"name": "Longmeadow Rehabilitation Hospital", "sector": "rehabilitation healthcare", "brief": "A rehabilitation hospital coordinating therapy schedules, equipment, visitors, discharge support, and records."},
    {"name": "Mossbank Waste Recovery", "sector": "waste and recycling", "brief": "A materials-recovery operator managing collections, contamination, hazardous finds, weighbridge records, and customers."},
    {"name": "Nightingale Language Services", "sector": "translation and interpreting", "brief": "A language-services agency assigning linguists, protecting files, handling revisions, certification, and urgent requests."},
    {"name": "Oakbridge Public Defender Office", "sector": "public legal defense", "brief": "A defender office managing clients, discovery, court deadlines, investigators, conflicts, and external communications."},
    {"name": "Port Meridian Customs Brokerage", "sector": "trade and customs services", "brief": "A broker coordinating declarations, client documents, holds, inspections, corrections, and restricted shipments."},
    {"name": "Queensmill Community College", "sector": "higher education", "brief": "A community college managing enrollment, advising, accommodations, assessments, labs, and student employment."},
    {"name": "Riverglass Diagnostics Laboratory", "sector": "diagnostic laboratory", "brief": "A laboratory managing specimens, orders, quality exceptions, results, couriers, and client communications."},
    {"name": "Stonecrop Regional Archive", "sector": "archives and records", "brief": "A public archive handling deposits, access, restrictions, digitization, researchers, and preservation incidents."},
    {"name": "Thistlewood Forestry Partnership", "sector": "forestry", "brief": "A forestry partnership coordinating crews, landowners, harvesting, road access, wildlife constraints, and equipment."},
    {"name": "Union Wharf Market Authority", "sector": "public market management", "brief": "A market authority managing traders, inspections, deliveries, events, complaints, and shared facilities."},
    {"name": "Violet Crown Convention Center", "sector": "events and hospitality", "brief": "A convention venue coordinating clients, rooms, vendors, credentials, deliveries, and event-day changes."},
    {"name": "Willowbend Mental Health Network", "sector": "community mental health", "brief": "A community network coordinating referrals, appointments, records, welfare concerns, interpreters, and partners."},
    {"name": "Amberline Bicycle Share", "sector": "shared mobility", "brief": "A bicycle-share operator handling memberships, damaged cycles, rebalancing, refunds, safety reports, and lost items."},
    {"name": "Briarstone Agricultural Extension", "sector": "agricultural support", "brief": "An extension service providing field visits, sample handling, demonstrations, grants guidance, and farmer records."},
    {"name": "Cloudrest Mountain Lodge", "sector": "hospitality", "brief": "A remote lodge coordinating reservations, weather disruptions, activities, guest requests, staff housing, and suppliers."},
    {"name": "Dunlin Coastal Ferry Service", "sector": "passenger transport", "brief": "A ferry operator managing bookings, boarding, vehicles, accessibility, weather changes, and unattended property."},
    {"name": "Elmstead Makers Guild", "sector": "community workshops", "brief": "A membership workshop managing inductions, machinery, project storage, guests, materials, and incidents."},
    {"name": "Frostline District Heating", "sector": "energy utility", "brief": "A district-heating provider handling service faults, planned works, meter disputes, building managers, and contractors."},
    {"name": "Golden Elm Insurance Mutual", "sector": "insurance", "brief": "A mutual insurer managing claims intake, evidence, repairs, complaints, renewals, and suspected misrepresentation."},
    {"name": "Harborlight Youth Sports League", "sector": "youth recreation", "brief": "A sports league coordinating registration, volunteers, fixtures, conduct, weather cancellations, and safeguarding."},
    {"name": "Ivory Gate Procurement Consortium", "sector": "public procurement", "brief": "A purchasing consortium handling requisitions, quotations, conflicts, vendor records, urgent buys, and contract changes."},
    {"name": "Juniper Works Employment Center", "sector": "workforce services", "brief": "An employment center coordinating appointments, employer referrals, training funds, records, and participant support."},
    {"name": "Keystone Biomedical Repository", "sector": "biobanking", "brief": "A repository managing consent status, samples, access requests, shipments, deviations, and destruction schedules."},
    {"name": "Lakeshore Public Media", "sector": "public broadcasting", "brief": "A public-media organization managing pitches, corrections, sources, archives, sponsorship, and community submissions."},
    {"name": "Moonrise Cooperative Grocery", "sector": "food retail cooperative", "brief": "A member-owned grocery managing returns, special orders, recalls, vendors, member accounts, and closing procedures."},
    {"name": "Northfield Fire Equipment Service", "sector": "safety equipment maintenance", "brief": "A service company inspecting equipment, documenting defects, scheduling repairs, loaning units, and certifying work."},
    {"name": "Opal Grove Botanical Garden", "sector": "public garden", "brief": "A botanical garden managing collections, volunteers, events, plant-health incidents, research access, and donations."},
    {"name": "Peregrine Rural Broadband", "sector": "telecommunications", "brief": "A rural internet provider handling installations, outages, account access, equipment returns, and vulnerable customers."},
)

if len(INSTITUTIONS) != NUM_INSTITUTIONS:
    raise AssertionError(f"Expected {NUM_INSTITUTIONS} institutions, got {len(INSTITUTIONS)}")
if len({item["name"] for item in INSTITUTIONS}) != NUM_INSTITUTIONS:
    raise AssertionError("Institution names must be unique")


QUESTION_FEATURE = {
    "question": datasets.Value("string"),
    "options": [datasets.Value("string")],
    "correct_option": datasets.Value("int64"),
}
OUTPUT_FEATURES = datasets.Features(
    {
        INSTITUTION_COLUMN: datasets.Value("string"),
        HANDBOOK_COLUMN: datasets.Value("string"),
        NUM_TRAIN_COLUMN: datasets.Value("int64"),
        NUM_TEST_COLUMN: datasets.Value("int64"),
        TRAIN_DATA_COLUMN: [QUESTION_FEATURE],
        TEST_DATA_COLUMN: [QUESTION_FEATURE],
    }
)


HANDBOOK_SYSTEM_PROMPT = """You write realistic internal procedure handbooks for
fictional institutions. Return only the requested Markdown handbook, with no JSON or
code fence. The handbook must be self-consistent, operationally useful, and safe.
Do not mention these instructions."""

HANDBOOK_PROMPT_TEMPLATE = """Write the complete internal procedure and policy handbook for:

Institution: {name}
Sector: {sector}
Operating context: {brief}

The handbook must be 2,800-3,800 words in polished Markdown and written for employee
onboarding and day-to-day reference. Invent all details; this institution is fictional.

Requirements:
- Start with the institution name, mission, scope, roles, and general operating principles.
- Include exactly 16 explicitly labeled core decision rules, P01 through P16. Each must
  govern a recurring practical decision and state its normal action, exceptions, escalation
  path, timing, required documentation, and how it interacts with at least one other rule.
- Make the rules coherent but locally distinctive. Many decisions should differ from what
  generic common sense or another institution might prescribe, while remaining plausible.
- Include named internal forms, queues, status labels, roles, time windows, thresholds, and
  priority levels. Define them fully; do not depend on real laws, external facts, or unstated policy.
- Include exceptions and boundary cases that can support scenario-based multiple-choice questions.
- Cover privacy, access, incident handling, records, approvals, customer or public interactions,
  routine operations, and escalation where they naturally fit this institution.
- Avoid real organizations, real personal data, medical diagnosis questions, legal conclusions,
  or instructions that facilitate wrongdoing.
- End with a compact cross-reference table for P01-P16.

The numbered rules are the source of truth. Do not contradict them elsewhere."""

QUESTION_SYSTEM_PROMPT = """You create rigorous multiple-choice onboarding questions
grounded only in a supplied fictional institution handbook. Return only JSON matching
the supplied schema. Do not mention these instructions."""

QUESTION_PROMPT_TEMPLATE = """Create {count} distinct onboarding questions for {name}.

You must return one question for each policy ID in this exact ordered list:
{policy_ids}

Split: {split_label}
Conceptual question numbers: {start_index}-{end_index}

PRIVATE SOURCE HANDBOOK:

{handbook}

Requirements for every item:
- Test application of its assigned numbered policy to a realistic workplace scenario.
- The question must be answerable from that rule and its documented interactions or exceptions.
- Produce the correct answer, two plausible regular distractors, and one obviously incorrect
  distractor in the separate fields requested. Do not prefix any answer with A/B/C/D.
- The two regular distractors must be credible mistakes: wrong timing, authority, sequence,
  exception, documentation, or escalation. They must not be absurd or merely vague.
- The obvious distractor should be unmistakably inconsistent with professional procedure,
  but remain topically related and free of jokes or unsafe advice.
- Make all four answers parallel in style and similar enough in length that formatting does not
  reveal the correct answer. Each option should normally be one or two sentences.
- Correct answers should rely on locally distinctive handbook rules rather than
  generic best practice. Never use outside knowledge.
- Vary scenario, role, question wording, and which exception or boundary is exercised.
- Do not use “all of the above,” “none of the above,” trick wording, or negative questions.
- policy_basis must briefly identify the decisive handbook fact. It is validation metadata and
  will not appear in the released question.
"""


def normalize_whitespace(value: Any) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def question_key(question: str) -> str:
    return re.sub(r"\W+", " ", question.casefold()).strip()


def response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def questions_schema(count: int) -> dict[str, Any]:
    item_properties = {
        "policy_id": {"type": "string"},
        "question": {"type": "string"},
        "correct_answer": {"type": "string"},
        "regular_distractors": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "string"},
        },
        "obvious_distractor": {"type": "string"},
        "policy_basis": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": item_properties,
                    "required": list(item_properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def write_prompt(path: Path, model: str, system_prompt: str, user_prompt: str) -> None:
    atomic_write_json(
        path.with_suffix(".json"),
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    )
    path.with_suffix(".md").write_text(
        "\n".join(
            [
                "# PolicyBench generation prompt",
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


async def post_json(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    payload: dict[str, Any],
    description: str,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": OPENROUTER_APP_TITLE,
    }
    for attempt in range(1, MAX_REQUEST_RETRIES + 1):
        try:
            async with semaphore:
                response = await client.post(
                    OPENROUTER_API_URL, headers=headers, json=payload
                )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("response content is not an object")
            print(
                f"{description}: response from {body.get('provider', 'unknown provider')}",
                flush=True,
            )
            return parsed
        except (
            httpx.HTTPError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            if attempt == MAX_REQUEST_RETRIES:
                raise RuntimeError(
                    f"{description} failed after {attempt} attempts: {error}"
                ) from error
            retry_after = 0
            if isinstance(error, httpx.HTTPStatusError):
                try:
                    retry_after = int(error.response.headers.get("retry-after", "0"))
                except ValueError:
                    pass
            wait_seconds = min(
                RETRY_MAX_WAIT_SECONDS, max(retry_after, 2 ** (attempt - 1))
            )
            print(
                f"{description} attempt {attempt}/{MAX_REQUEST_RETRIES} failed: "
                f"{error}; retrying in {wait_seconds}s",
                flush=True,
            )
            await asyncio.sleep(wait_seconds)
    raise AssertionError("unreachable")


async def post_text(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    payload: dict[str, Any],
    description: str,
) -> str:
    """Request long free-form text without forcing it through JSON string escaping."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": OPENROUTER_APP_TITLE,
    }
    for attempt in range(1, MAX_REQUEST_RETRIES + 1):
        try:
            async with semaphore:
                response = await client.post(
                    OPENROUTER_API_URL, headers=headers, json=payload
                )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError("response content is empty")
            print(
                f"{description}: response from {body.get('provider', 'unknown provider')}",
                flush=True,
            )
            return content
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            if attempt == MAX_REQUEST_RETRIES:
                raise RuntimeError(
                    f"{description} failed after {attempt} attempts: {error}"
                ) from error
            wait_seconds = min(RETRY_MAX_WAIT_SECONDS, 2 ** (attempt - 1))
            print(
                f"{description} attempt {attempt}/{MAX_REQUEST_RETRIES} failed: "
                f"{error}; retrying in {wait_seconds}s",
                flush=True,
            )
            await asyncio.sleep(wait_seconds)
    raise AssertionError("unreachable")


def api_payload(
    system_prompt: str, user_prompt: str, schema_name: str, schema: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": response_format(schema_name, schema),
        "reasoning": {"effort": REASONING_EFFORT, "exclude": True},
        "provider": {
            "sort": PROVIDER_SORT,
            "require_parameters": REQUIRE_PROVIDER_PARAMETERS,
        },
        "max_tokens": MAX_OUTPUT_TOKENS,
    }


def validate_handbook(value: Any) -> str:
    handbook = normalize_whitespace(value)
    words = len(handbook.split())
    if not HANDBOOK_MIN_WORDS <= words <= HANDBOOK_MAX_WORDS:
        raise ValueError(
            f"handbook has {words} words; expected {HANDBOOK_MIN_WORDS}-{HANDBOOK_MAX_WORDS}"
        )
    for number in range(1, HANDBOOK_RULE_COUNT + 1):
        policy_id = f"P{number:02d}"
        if policy_id not in handbook:
            raise ValueError(f"handbook is missing {policy_id}")
    return handbook


async def generate_handbook(
    institution: dict[str, str],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    prompt_dir: Path,
) -> str:
    prompt = HANDBOOK_PROMPT_TEMPLATE.format(**institution)
    prompt_path = prompt_dir / "handbook"
    write_prompt(prompt_path, OPENROUTER_MODEL, HANDBOOK_SYSTEM_PROMPT, prompt)
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": HANDBOOK_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "reasoning": {"effort": REASONING_EFFORT, "exclude": True},
        "provider": {"sort": PROVIDER_SORT},
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    last_error: ValueError | None = None
    for generation_attempt in range(1, 4):
        generated = await post_text(
            client,
            semaphore,
            api_key,
            payload,
            f"{institution['name']} handbook",
        )
        try:
            return validate_handbook(generated)
        except ValueError as error:
            last_error = error
            print(
                f"{institution['name']} handbook validation attempt "
                f"{generation_attempt}/3 failed: {error}",
                flush=True,
            )
    raise RuntimeError(f"Could not generate valid handbook: {last_error}")


def policy_schedule(count: int, active_rule_count: int, seed: int) -> list[str]:
    policy_ids = [f"P{number:02d}" for number in range(1, active_rule_count + 1)]
    schedule = [policy_ids[index % len(policy_ids)] for index in range(count)]
    random.Random(seed).shuffle(schedule)
    return schedule


def correct_position_schedule(count: int, seed: int) -> list[int]:
    positions = [index % 4 for index in range(count)]
    random.Random(seed).shuffle(positions)
    return positions


def validate_and_shuffle_question(
    raw: Any,
    expected_policy_id: str,
    correct_position: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict):
        return None, "not an object"
    if normalize_whitespace(raw.get("policy_id", "")).upper() != expected_policy_id:
        return None, "wrong policy ID"
    question = normalize_whitespace(raw.get("question", ""))
    correct = normalize_whitespace(raw.get("correct_answer", ""))
    regular = raw.get("regular_distractors")
    obvious = normalize_whitespace(raw.get("obvious_distractor", ""))
    basis = normalize_whitespace(raw.get("policy_basis", ""))
    if not question or not correct or not obvious or not basis:
        return None, "empty field"
    if not isinstance(regular, list) or len(regular) != 2:
        return None, "regular distractor count"
    regular = [normalize_whitespace(item) for item in regular]
    if not all(regular):
        return None, "empty regular distractor"
    canonical_options = [correct, regular[0], regular[1], obvious]
    if len({option.casefold() for option in canonical_options}) != 4:
        return None, "duplicate option"

    # Move the correct answer to its preassigned position, then shuffle only distractors.
    distractors = canonical_options[1:]
    distractor_seed = int.from_bytes(
        f"{question}|{expected_policy_id}".encode("utf-8"), "little"
    ) % (2**32)
    random.Random(distractor_seed).shuffle(distractors)
    options = list(distractors)
    options.insert(correct_position, correct)
    public = {
        "question": question,
        "options": options,
        "correct_option": correct_position,
    }
    private = {"policy_id": expected_policy_id, "policy_basis": basis}
    return {"public": public, "private": private}, None


async def request_question_batch(
    institution: dict[str, str],
    handbook: str,
    split_label: str,
    policy_ids: list[str],
    start_index: int,
    batch_number: int,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    prompt_dir: Path,
) -> list[dict[str, Any]]:
    count = len(policy_ids)
    prompt = QUESTION_PROMPT_TEMPLATE.format(
        count=count,
        name=institution["name"],
        policy_ids=json.dumps(policy_ids),
        split_label=split_label,
        start_index=start_index,
        end_index=start_index + count - 1,
        handbook=handbook,
    )
    prompt_path = prompt_dir / f"{split_label}_round_{batch_number:03d}"
    write_prompt(prompt_path, OPENROUTER_MODEL, QUESTION_SYSTEM_PROMPT, prompt)
    parsed = await post_json(
        client,
        semaphore,
        api_key,
        api_payload(
            QUESTION_SYSTEM_PROMPT,
            prompt,
            "policy_onboarding_questions",
            questions_schema(count),
        ),
        f"{institution['name']} {split_label} batch {batch_number}",
    )
    questions = parsed.get("questions")
    if not isinstance(questions, list):
        raise ValueError("response did not contain a questions array")
    return questions


def load_checkpoint(path: Path, institution: dict[str, str]) -> dict[str, Any]:
    if not path.exists():
        return {
            "institution": institution,
            "handbook": "",
            "train_questions": [],
            "test_questions": [],
            "train_private_metadata": [],
            "test_private_metadata": [],
            "rejections": {},
            "rounds": {"train": 0, "test": 0},
            "generation_counts": None,
        }
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("institution") != institution:
        raise ValueError(f"Checkpoint institution mismatch: {path}")
    return checkpoint


async def generate_split(
    institution_index: int,
    institution: dict[str, str],
    handbook: str,
    split_label: str,
    target_count: int,
    policy_ids: list[str],
    correct_positions: list[int],
    questions_per_request: int,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    prompt_dir: Path,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
) -> None:
    question_key_name = f"{split_label}_questions"
    metadata_key_name = f"{split_label}_private_metadata"
    questions = checkpoint[question_key_name][:target_count]
    metadata = checkpoint.get(metadata_key_name, [])[:target_count]
    if len(metadata) != len(questions):
        raise ValueError(f"Checkpoint has mismatched {split_label} metadata")
    seen = {
        question_key(item["question"])
        for key in ("train_questions", "test_questions")
        for item in checkpoint.get(key, [])
    }
    rejections = checkpoint.get("rejections", {})
    rounds = checkpoint.get("rounds", {"train": 0, "test": 0})
    max_rounds = max(
        1,
        math.ceil(target_count / questions_per_request)
        * MAX_QUESTION_ROUND_MULTIPLIER,
    )

    while len(questions) < target_count:
        if int(rounds.get(split_label, 0)) >= max_rounds:
            raise RuntimeError(
                f"{institution['name']} has only {len(questions)}/{target_count} "
                f"valid {split_label} questions after {max_rounds} rounds; "
                f"rejections={rejections}"
            )
        start = len(questions)
        requested = min(questions_per_request, target_count - start)
        expected_policy_ids = policy_ids[start : start + requested]
        batch_number = int(rounds.get(split_label, 0)) + 1
        rounds[split_label] = batch_number
        try:
            raw_questions = await request_question_batch(
                institution,
                handbook,
                split_label,
                expected_policy_ids,
                start + 1,
                batch_number,
                client,
                semaphore,
                api_key,
                prompt_dir,
            )
        except Exception as error:
            rejection_key = f"{split_label}: request failure"
            rejections[rejection_key] = rejections.get(rejection_key, 0) + requested
            checkpoint.update({"rejections": rejections, "rounds": rounds})
            atomic_write_json(checkpoint_path, checkpoint)
            print(f"{institution['name']} {split_label}: {error}", flush=True)
            continue

        accepted = 0
        # Strict ordering lets the deterministic schedules survive retries and resumption.
        for offset, expected_policy_id in enumerate(expected_policy_ids):
            if offset >= len(raw_questions):
                rejection = "missing returned question"
                rejections[rejection] = rejections.get(rejection, 0) + 1
                break
            item, rejection = validate_and_shuffle_question(
                raw_questions[offset],
                expected_policy_id,
                correct_positions[len(questions)],
            )
            if rejection is not None:
                rejections[rejection] = rejections.get(rejection, 0) + 1
                break
            key = question_key(item["public"]["question"])
            if key in seen:
                rejection = "duplicate question"
                rejections[rejection] = rejections.get(rejection, 0) + 1
                break
            seen.add(key)
            questions.append(item["public"])
            metadata.append(item["private"])
            accepted += 1

        checkpoint.update(
            {
                question_key_name: questions,
                metadata_key_name: metadata,
                "rejections": rejections,
                "rounds": rounds,
            }
        )
        atomic_write_json(checkpoint_path, checkpoint)
        print(
            f"Institution {institution_index + 1} {split_label}: accepted "
            f"{accepted}/{len(raw_questions)}; total {len(questions)}/{target_count}",
            flush=True,
        )


async def generate_for_institution(
    institution_index: int,
    institution: dict[str, str],
    train_count: int,
    test_count: int,
    questions_per_request: int,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    checkpoint_dir: Path,
    prompt_root: Path,
) -> dict[str, Any]:
    slug = slugify(institution["name"])
    checkpoint_path = checkpoint_dir / f"{institution_index:03d}_{slug}.json"
    prompt_dir = prompt_root / f"{institution_index:03d}_{slug}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path, institution)

    requested_counts = {"train": train_count, "test": test_count}
    saved_counts = checkpoint.get("generation_counts")
    has_questions = bool(
        checkpoint.get("train_questions") or checkpoint.get("test_questions")
    )
    if has_questions and saved_counts is None:
        saved_counts = {
            "train": len(checkpoint.get("train_questions", [])),
            "test": len(checkpoint.get("test_questions", [])),
        }
    if has_questions and saved_counts != requested_counts:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was generated with counts {saved_counts}; "
            f"requested {requested_counts}. Use a new --output-dir so deterministic "
            "label schedules and split coverage remain valid."
        )
    checkpoint["generation_counts"] = requested_counts
    atomic_write_json(checkpoint_path, checkpoint)

    handbook = normalize_whitespace(checkpoint.get("handbook", ""))
    if not handbook:
        handbook = await generate_handbook(
            institution, client, semaphore, api_key, prompt_dir
        )
        checkpoint["handbook"] = handbook
        atomic_write_json(checkpoint_path, checkpoint)
        print(
            f"Institution {institution_index + 1}: saved {len(handbook.split())}-word handbook",
            flush=True,
        )
    else:
        validate_handbook(handbook)

    if train_count and test_count:
        active_rules = min(HANDBOOK_RULE_COUNT, train_count, test_count)
    else:
        active_rules = min(HANDBOOK_RULE_COUNT, max(train_count, test_count))
    train_policies = policy_schedule(
        train_count, active_rules, SHUFFLE_SEED + institution_index * 17
    )
    test_policies = policy_schedule(
        test_count, active_rules, SHUFFLE_SEED + institution_index * 17 + 1
    )
    all_positions = correct_position_schedule(
        train_count + test_count, SHUFFLE_SEED + institution_index * 31
    )
    await generate_split(
        institution_index,
        institution,
        handbook,
        "train",
        train_count,
        train_policies,
        all_positions[:train_count],
        questions_per_request,
        checkpoint,
        checkpoint_path,
        prompt_dir,
        client,
        semaphore,
        api_key,
    )
    await generate_split(
        institution_index,
        institution,
        handbook,
        "test",
        test_count,
        test_policies,
        all_positions[train_count:],
        questions_per_request,
        checkpoint,
        checkpoint_path,
        prompt_dir,
        client,
        semaphore,
        api_key,
    )
    return checkpoint


async def generate_all(
    institutions: tuple[dict[str, str], ...],
    train_count: int,
    test_count: int,
    questions_per_request: int,
    api_key: str,
    checkpoint_dir: Path,
    prompt_dir: Path,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await asyncio.gather(
            *[
                generate_for_institution(
                    index,
                    institution,
                    train_count,
                    test_count,
                    questions_per_request,
                    client,
                    semaphore,
                    api_key,
                    checkpoint_dir,
                    prompt_dir,
                )
                for index, institution in enumerate(institutions)
            ]
        )


def pack_dataset(
    generated: list[dict[str, Any]], train_count: int, test_count: int
) -> datasets.Dataset:
    rows = []
    for item in generated:
        train = item["train_questions"][:train_count]
        test = item["test_questions"][:test_count]
        if len(train) != train_count or len(test) != test_count:
            raise ValueError("Generated question count does not match requested split")
        positions = Counter(
            question["correct_option"] for question in [*train, *test]
        )
        if positions and max(positions.values()) - min(
            positions.get(index, 0) for index in range(4)
        ) > 1:
            raise ValueError(f"Unbalanced correct-option positions: {positions}")
        institution = item["institution"]
        rows.append(
            {
                INSTITUTION_COLUMN: "\n".join(
                    [
                        f"Name: {institution['name']}",
                        f"Sector: {institution['sector']}",
                        f"Operating context: {institution['brief']}",
                    ]
                ),
                HANDBOOK_COLUMN: item["handbook"],
                NUM_TRAIN_COLUMN: train_count,
                NUM_TEST_COLUMN: test_count,
                TRAIN_DATA_COLUMN: train,
                TEST_DATA_COLUMN: test,
            }
        )
    return datasets.Dataset.from_list(rows, features=OUTPUT_FEATURES)


def write_jsonl(dataset: datasets.Dataset, path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in dataset:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_question(lines: list[str], label: str, index: int, item: dict[str, Any]) -> None:
    lines.extend([f"#### {label} question {index + 1}", "", item["question"], ""])
    for option_index, option in enumerate(item["options"]):
        marker = " ✓" if option_index == item["correct_option"] else ""
        lines.append(f"{chr(65 + option_index)}. {option}{marker}")
    lines.append("")


def write_readable_samples(dataset: datasets.Dataset, path: Path) -> None:
    lines = [
        "# PolicyBench smoke-test samples",
        "",
        "The handbook is private generation metadata. Check marks expose labels only in this human-readable audit file.",
        "",
    ]
    for row_index, row in enumerate(
        dataset.select(range(min(SAMPLE_INSTITUTIONS, len(dataset))))
    ):
        lines.extend(
            [
                f"## Row {row_index + 1}: {row[INSTITUTION_COLUMN].splitlines()[0][6:]}",
                "",
                row[INSTITUTION_COLUMN],
                "",
                "### Handbook",
                "",
                row[HANDBOOK_COLUMN],
                "",
                "### Questions",
                "",
            ]
        )
        for split_column, label in (
            (TRAIN_DATA_COLUMN, "Train"),
            (TEST_DATA_COLUMN, "Test"),
        ):
            for index, item in enumerate(row[split_column]):
                append_question(lines, label, index, item)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--institution-limit", type=int, default=NUM_INSTITUTIONS)
    parser.add_argument("--train-count", type=int, default=TRAIN_QUESTIONS)
    parser.add_argument("--test-count", type=int, default=TEST_QUESTIONS)
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
    if not 1 <= args.institution_limit <= NUM_INSTITUTIONS:
        raise ValueError(
            f"--institution-limit must be between 1 and {NUM_INSTITUTIONS}"
        )
    if args.train_count < 0 or args.test_count < 0:
        raise ValueError("Question counts cannot be negative")
    if args.train_count + args.test_count == 0:
        raise ValueError("At least one train or test question is required")
    if not 1 <= args.questions_per_request <= QUESTIONS_PER_REQUEST:
        raise ValueError(
            f"--questions-per-request must be between 1 and {QUESTIONS_PER_REQUEST}"
        )

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

    selected = INSTITUTIONS[: args.institution_limit]
    generated = asyncio.run(
        generate_all(
            selected,
            args.train_count,
            args.test_count,
            args.questions_per_request,
            api_key,
            checkpoint_dir,
            prompt_dir,
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
        if len(dataset) != NUM_INSTITUTIONS:
            raise ValueError(
                f"Refusing to upload a smoke-test dataset with {len(dataset)} rows; "
                f"production requires {NUM_INSTITUTIONS}"
            )
        if args.train_count != TRAIN_QUESTIONS or args.test_count != TEST_QUESTIONS:
            raise ValueError("Refusing to upload non-production train/test counts")
        result = dataset.push_to_hub(
            args.output_dataset,
            config_name=OUTPUT_CONFIG,
            split=OUTPUT_SPLIT,
            private=args.private,
        )
        print(result, flush=True)


if __name__ == "__main__":
    main()
