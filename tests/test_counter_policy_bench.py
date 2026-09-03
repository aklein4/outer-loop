from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "data_preparation" / "benchmarks"))

import counter_policy_bench as counter


def source_question():
    return {
        "question": "Who approves the exception?",
        "options": ["The visitor", "The General Manager", "Nobody", "The courier"],
        "correct_option": 1,
    }


def generated_item():
    return {
        "distractors": [
            "The Group Supervisor",
            "The Duty Manager",
            "The Security Coordinator",
        ],
        "changed_details": [
            "Changes General Manager to Group Supervisor.",
            "Changes General Manager to Duty Manager.",
            "Uses a different but plausible approval chain.",
        ],
    }


def test_prompt_items_exclude_source_distractors():
    items = counter.prompt_items([source_question()], start=7)

    assert items == [
        {
            "item_id": 7,
            "question": "Who approves the exception?",
            "correct_answer": "The General Manager",
        }
    ]
    assert "The visitor" not in str(items)


def test_prompt_requires_realistic_common_sense_counter_policies():
    prompt = counter.make_prompt(
        "Example Institution", "The General Manager approves exceptions.",
        [source_question()], start=0
    )

    assert "consistent with ordinary professional common sense" in prompt
    assert "well-run real institution could reasonably adopt" in prompt
    assert "using only common sense should find every option" in prompt


def test_retry_prompt_includes_actionable_validation_feedback():
    original = "Generate this batch."
    prompt = counter.make_retry_prompt(original, "item 5: option length mismatch")

    assert prompt.startswith(original)
    assert "item 5: option length mismatch" in prompt
    assert "Regenerate the entire batch" in prompt


def test_build_public_question_preserves_question_answer_and_position():
    source = source_question()
    public = counter.build_public_question(source, generated_item(), "Example Institution")

    assert public["question"] == source["question"]
    assert public["correct_option"] == source["correct_option"]
    assert public["options"][public["correct_option"]] == "The General Manager"
    assert set(public["options"]) == {
        "The General Manager",
        "The Group Supervisor",
        "The Duty Manager",
        "The Security Coordinator",
    }


def test_build_public_question_is_deterministic():
    first = counter.build_public_question(
        source_question(), generated_item(), "Example Institution"
    )
    second = counter.build_public_question(
        source_question(), generated_item(), "Example Institution"
    )

    assert first == second


def test_validate_generated_item_retains_private_audit_details():
    raw = {
        "item_id": 3,
        "near_miss_1": "The Group Supervisor",
        "near_miss_1_changed_detail": "Changes the approving role.",
        "near_miss_2": "The Duty Manager",
        "near_miss_2_changed_detail": "Changes the approving role again.",
        "counter_policy": "The Security Coordinator",
        "counter_policy_changed_detail": "Uses another plausible approval chain.",
    }

    item, rejection = counter.validate_generated_item(
        raw, expected_id=3, correct_answer="The General Manager"
    )

    assert rejection is None
    assert item == {
        "distractors": [
            "The Group Supervisor",
            "The Duty Manager",
            "The Security Coordinator",
        ],
        "changed_details": [
            "Changes the approving role.",
            "Changes the approving role again.",
            "Uses another plausible approval chain.",
        ],
    }


def test_validate_generated_item_rejects_duplicate_correct_answer():
    raw = {
        "item_id": 3,
        "near_miss_1": "The General Manager",
        "near_miss_1_changed_detail": "No actual change.",
        "near_miss_2": "The Duty Manager",
        "near_miss_2_changed_detail": "Changes the approver.",
        "counter_policy": "The Security Coordinator",
        "counter_policy_changed_detail": "Uses another approval chain.",
    }

    item, rejection = counter.validate_generated_item(
        raw, expected_id=3, correct_answer="The General Manager"
    )

    assert item is None
    assert rejection == "duplicate option"


def test_length_rejection_reports_actionable_word_counts():
    raw = {
        "item_id": 3,
        "near_miss_1": "20 minutes.",
        "near_miss_1_changed_detail": "Changes 30 to 20 minutes.",
        "near_miss_2": "45 minutes.",
        "near_miss_2_changed_detail": "Changes 30 to 45 minutes.",
        "counter_policy": (
            "Open another accessible station once the voter has waited for "
            "a total of sixty minutes."
        ),
        "counter_policy_changed_detail": "Changes 30 to 60 minutes.",
    }

    item, rejection = counter.validate_generated_item(
        raw, expected_id=3, correct_answer="30 minutes."
    )

    assert item is None
    assert "word counts" in rejection
    assert "[2, 2, 2, 15]" in rejection
