from __future__ import annotations

import random
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import evaluate_continual_policy as continual


def make_rows(num_rows=8, num_examples=12):
    return [
        {
            "subset": "default",
            "institution": f"institution-{row_idx}",
            "train_data": [f"row-{row_idx}-example-{example_idx}" for example_idx in range(num_examples)],
            "test_data": [f"row-{row_idx}-test"],
        }
        for row_idx in range(num_rows)
    ]


def test_distinct_column_permutations_are_complete_and_distinct():
    columns = continual._distinct_column_permutations(8, 4, random.Random(7))

    assert all(sorted(column) == list(range(8)) for column in columns)
    assert all(len({column[row_idx] for column in columns}) == 4 for row_idx in range(8))


def test_make_trajectories_uses_each_chunk_once_and_distinct_rows():
    rows = make_rows()
    trajectories = continual.make_trajectories(rows, n_tasks=4, trajectory_length=12, seed=3)

    assert len(trajectories) == len(rows)
    used = set()
    for trajectory in trajectories:
        assert len(trajectory["tasks"]) == 4
        assert len({task["source_index"] for task in trajectory["tasks"]}) == 4
        assert len(trajectory["train_data"]) == 12
        for task_idx, task in enumerate(trajectory["tasks"]):
            assert task["subset"] == f"task_{task_idx + 1}"
            assert len(task["train_data"]) == 3
            used.add((task["source_index"], task_idx))

    assert used == {(row_idx, task_idx) for row_idx in range(8) for task_idx in range(4)}


def test_make_trajectories_is_seeded_without_mutating_rows():
    rows = make_rows()
    original = [list(row["train_data"]) for row in rows]

    first = continual.make_trajectories(rows, n_tasks=4, trajectory_length=12, seed=11)
    second = continual.make_trajectories(rows, n_tasks=4, trajectory_length=12, seed=11)

    assert first == second
    assert [row["train_data"] for row in rows] == original


def test_make_trajectories_rejects_non_divisible_length():
    try:
        continual.make_trajectories(make_rows(), n_tasks=4, trajectory_length=10, seed=0)
    except ValueError as error:
        assert "must be divisible" in str(error)
    else:
        raise AssertionError("expected a divisibility error")


def test_messages_include_source_institution_system_prompt():
    item = {
        "question": "What is allowed?",
        "options": ["One", "Two", "Three", "Four"],
        "correct_option": 1,
    }

    adaptation = continual.format_adaptation_messages(item, "Acme")
    evaluation = continual.format_evaluation_messages(item, "Acme")

    expected = "Answer the question by following the rules in the Acme handbook."
    assert adaptation[0] == {"role": "system", "content": expected}
    assert evaluation[0] == {"role": "system", "content": expected}
    assert adaptation[-1]["role"] == "assistant"
    assert evaluation[-1]["role"] == "user"
