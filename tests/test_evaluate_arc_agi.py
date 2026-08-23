import json
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import evaluate_arc_agi as arc


def test_prompt_and_response_format():
    examples = [{"input": [[0, 1], [2, 3]], "output": [[4], [5]]}]
    question = {"input": [[6, 7]], "output": [[8, 9]]}

    messages = arc.conversation(examples, question)

    assert messages[0]["content"] == (
        "These grids follow a hidden input-output rule. Infer the output of the last input grid.\n\n"
        "# Example 1\n\n## Input\n\n01\n23\n\n## Output\n\n4\n5\n\n"
        "# Question\n\n## Input\n\n67\n\n## Output"
    )
    assert messages[1] == {"role": "assistant", "content": "89"}


def test_paper_transform_set_is_complete_and_unique():
    names = [transform.name for transform in arc.paper_transforms()]
    assert len(names) == 24
    assert len(set(names)) == len(names)
    assert {"rotate-90", "rotate-270", "translate-xy", "repeat-both-2"} <= set(names)


def test_load_rows_duplicates_test_pairs(tmp_path):
    split = tmp_path / "evaluation"
    split.mkdir()
    (split / "task.json").write_text(
        json.dumps(
            {
                "train": [{"input": [[0]], "output": [[1]]}],
                "test": [
                    {"input": [[2]], "output": [[3]]},
                    {"input": [[4]], "output": [[5]]},
                ],
            }
        )
    )

    rows = arc.load_rows(tmp_path, ["validation", "public-test"])

    assert [(row.task_id, row.test_index) for row in rows] == [
        ("task", 0),
        ("task", 1),
    ]


def test_inactive_state_rows_are_restored():
    class Model:
        def __init__(self):
            self.state = torch.tensor([[[1.0]], [[2.0]]])

        def state_containers(self):
            yield self.state

    model = Model()
    frozen = [model.state.clone()]
    model.state.add_(10)

    arc._restore_inactive_states(
        model, frozen, torch.tensor([True, False], dtype=torch.bool)
    )

    assert model.state.tolist() == [[[11.0]], [[2.0]]]


def test_result_starts_with_compact_safe_overrides(tmp_path):
    path = tmp_path / "result.json"
    payload = {
        "model_config_overrides": {"base_lr": 0.001, "name": "a\nb"},
        "config": {"batch_size": 8},
    }

    arc.write_results(path, payload)

    lines = path.read_text().splitlines()
    assert lines[1] == '  "model_config_overrides": {"base_lr":0.001,"name":"a\\nb"},'
    assert json.loads(path.read_text()) == payload


def test_default_result_path_uses_checkpoint_step_and_kwargs(tmp_path):
    assert arc.default_result_path(
        tmp_path,
        "aklein4/horizon-v2_baseline",
        1600,
        {"base_lr": 0.0003},
    ) == (
        tmp_path
        / "aklein4--horizon-v2_baseline"
        / "000000001600"
        / "base_lr=0.0003.json"
    )
    assert arc.model_kwargs_filename({}) == "default.json"
