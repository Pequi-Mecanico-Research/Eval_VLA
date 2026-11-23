import csv

import pytest

from eval_vla.config import SetupConfig
from eval_vla.metrics.long_horizon import LongHorizon

CONFIG = SetupConfig(
    robot_embodiment={}, camera_configuration={}, action_space={}, control_frequency=10.0,
    object_set={}, scene_layout={}, reset_procedure="manual", success_criteria="x",
    model_name="m", experiment_name="exp", task_name="t",
)


def run(lh, completed):
    out = {}
    for name in ("subtask_completion_rate", "average_completed_sequence_length", "failure_position_distribution"):
        out.update(getattr(lh, name)(completed))
    lh.end_episode()
    return out


@pytest.fixture
def lh(tmp_path):
    return LongHorizon(CONFIG, run_dir=tmp_path)


def test_failure_example(lh):
    assert run(lh, [True, True, False, False]) == {
        "subtask_completion_rate": 0.5, "average_completed_sequence_length": 2.0, "failure_position": 3}


def test_all_completed(lh):
    out = run(lh, [True] * 4)
    assert out["subtask_completion_rate"] == 1.0 and out["average_completed_sequence_length"] == 4.0
    assert out["failure_position"] is None


def test_aggregates_and_first_failure_only(lh):
    run(lh, [True, True, False, False])
    out = run(lh, [False, True, False, True])
    assert out["failure_position"] == 1
    assert out["subtask_completion_rate"] == 4 / 8
    assert out["average_completed_sequence_length"] == 1.0
    assert lh.failure_positions == [3, 1]


def test_empty_raises(lh):
    for name in ("subtask_completion_rate", "average_completed_sequence_length", "failure_position_distribution"):
        with pytest.raises(ValueError):
            getattr(lh, name)([])


def test_csv_columns(lh):
    run(lh, [True, True, False, False])
    run(lh, [True] * 3)
    rows = list(csv.DictReader(lh.csv_path.open()))
    assert [r["failure_position"] for r in rows] == ["3", ""]
    assert rows[0]["subtask_completion_rate"] == "0.5"
    assert float(rows[1]["average_completed_sequence_length"]) == 2.5
    assert (lh.output_dir / "long_horizon.png").exists()
