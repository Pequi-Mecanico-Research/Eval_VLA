import csv
import io
import json

import cv2
import numpy as np
import pytest

from eval_vla.config import SetupConfig
from eval_vla.metrics import long_horizon as lh_module
from eval_vla.metrics.long_horizon import LongHorizon
from eval_vla.success_detection import subtask_judge

CONFIG = SetupConfig(
    robot_embodiment={}, camera_configuration={}, action_space={}, control_frequency=10.0,
    object_set={}, scene_layout={}, reset_procedure="manual", success_criteria="x",
    model_name="m", experiment_name="exp", task_name="t",
)
SUBTASKS = ["reach", "grasp", "lift", "place"]


@pytest.fixture
def lh(tmp_path, monkeypatch):
    replies = []
    monkeypatch.setattr(lh_module, "ollama_subtask_times", lambda *a, **k: (replies.pop(0), "raw"))
    lh = LongHorizon(CONFIG, run_dir=tmp_path, subtasks=SUBTASKS)
    lh.replies = replies
    return lh


def run(lh, times):
    lh.replies.append(times)
    out = {}
    out.update(lh.subtask_completion_rate("video.mp4"))
    out.update(lh.average_completed_sequence_length())
    out.update(lh.failure_position_distribution())
    lh.end_episode()
    return out


def test_failure_time_and_step(lh):
    out = run(lh, [1.0, 2.5, None, None])
    assert out["subtask_completion_rate"] == 0.5 and out["average_completed_sequence_length"] == 2.0
    assert out["t_grasp"] == 2.5 and out["t_lift"] is None
    assert (out["failure_position"], out["failure_subtask"], out["failure_time_s"], out["failure_step"]) == (3, "lift", 2.5, 25)


def test_first_subtask_fails_at_zero(lh):
    out = run(lh, [None, 3.0, None, 4.0])
    assert (out["failure_position"], out["failure_time_s"], out["failure_step"]) == (1, 0.0, 0)
    assert out["episode_subtask_completion_rate"] == 0.5


def test_all_completed(lh):
    out = run(lh, [1.0, 2.0, 3.0, 4.0])
    assert out["subtask_completion_rate"] == 1.0 and out["average_completed_sequence_length"] == 4.0
    assert out["failure_position"] is out["failure_time_s"] is out["failure_step"] is None


def test_aggregates_csv_and_plot(lh):
    run(lh, [1.0, 2.0, None, None])
    run(lh, [1.0, 2.0, 3.0, 4.0])
    assert lh.failure_positions == [3, None] and lh.failure_times == [2.0, None]
    rows = list(csv.DictReader(lh.csv_path.open()))
    assert [r["failure_subtask"] for r in rows] == ["lift", ""]
    assert float(rows[1]["subtask_completion_rate"]) == 0.75
    assert (lh.output_dir / "long_horizon.png").exists()


def test_needs_analysis_first(tmp_path):
    lh = LongHorizon(CONFIG, run_dir=tmp_path, subtasks=SUBTASKS)
    with pytest.raises(ValueError):
        lh.failure_position_distribution()
    with pytest.raises(ValueError):
        lh.average_completed_sequence_length([])
    assert lh.average_completed_sequence_length([True, False]) == {"average_completed_sequence_length": 1.0}


def test_subtasks_from_task_name(tmp_path):
    with pytest.raises(ValueError):
        LongHorizon(CONFIG, run_dir=tmp_path)
    with pytest.raises(ValueError):
        LongHorizon(CONFIG, run_dir=tmp_path, subtasks=SUBTASKS, judge="gpt")
    config = SetupConfig(**{**CONFIG.__dict__, "task_name": "Organize_the_table"})
    assert LongHorizon(config, run_dir=tmp_path).subtasks == lh_module.TASK_SUBTASKS["Organize_the_table"]


def test_parse_frame_indices():
    parse = subtask_judge.parse_frame_indices
    assert parse('ok {"Reach": 2, "grasp": null, "lift": 99, "place": true}', SUBTASKS, 10) == [2, None, None, None]
    assert parse("no json here", SUBTASKS, 10) == [None] * 4


def test_first_crossing():
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert subtask_judge.first_crossing([0.1, 0.2, 0.7, 0.9, 1.0], times, 0.5) == 2.0
    assert subtask_judge.first_crossing([0.1, 0.9, 1.0], times, 0.5) == 2.0  # 3 scores spread over 0-4 s
    assert subtask_judge.first_crossing([0.1, 0.2], times, 0.5) is None


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(50):
        writer.write(np.full((48, 64, 3), i * 5, np.uint8))
    writer.release()
    return path


def test_read_frames(video):
    times, frames = subtask_judge.read_frames(video, n_frames=5)
    assert times == pytest.approx([0.0, 1.2, 2.4, 3.7, 4.9], abs=0.06)
    assert len(frames) == 5 and frames[0].shape == (48, 64, 3)


def test_robometer_request(video, monkeypatch):
    import requests

    sent = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"outputs_progress": {"progress_pred": [[0] * 5, [0] * 5]},
                    "outputs_success": {"success_probs": [[0.1, 0.6, 0.9, 1, 1], [0.1, 0.1, 0.2, 0.3, 0.4]]}}

    def post(url, files, data, timeout):
        sent.update(url=url, files=files, data=data)
        return Response()

    monkeypatch.setattr(requests, "post", post)
    times, _ = subtask_judge.robometer_subtask_times(video, ["reach", "grasp"], url="http://gpu:8000/", n_frames=5)
    assert times[0] == pytest.approx(1.2, abs=0.06) and times[1] is None
    assert sent["url"] == "http://gpu:8000/evaluate_batch_npy"
    sample = json.loads(sent["data"]["sample_1"])
    assert sample["trajectory"]["task"] == "grasp" and sample["trajectory"]["frames_shape"] == [5, 48, 64, 3]
    assert np.load(io.BytesIO(sent["files"]["sample_1_trajectory_frames"][1])).shape == (5, 48, 64, 3)
