import csv
import json
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from eval_vla.config import SetupConfig
from eval_vla.metrics.drift import Drift, build_reference

MEANS = [0.6, 0.4, 0.2]
CAMS = ("cam_high", "cam_wrist", "cam_low")
CONFIG = SetupConfig({}, {}, {}, 0.0, {}, {}, "", "", "", "drift_test", "")


def make_dataset(root, rng):
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    names = [f"s{i}" for i in range(14)]
    (root / "meta/info.json").write_text(json.dumps({"fps": 30, "features": {
        "observation.state": {"names": names}, "action": {"names": names[:7]}}}))
    (root / "meta/stats.json").write_text(json.dumps(
        {f"observation.images.{c}": {"mean": [[[m]] for m in MEANS]} for c in CAMS}))
    (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "Organize the table"}\n')
    pq.write_table(pa.table({
        "observation.state": rng.normal(size=(2000, 14)).tolist(),
        "action": rng.normal(size=(2000, 7)).tolist(),
    }), root / "data/chunk-000/episode_000000.parquet")


def make_episode(rng, state_dim=14, shift=0.0, bgr=False, stretched=False, prompt="Organize the table"):
    img = np.zeros((20, 3, 224, 224), np.uint8)
    rows = slice(None) if stretched else slice(28, 196)
    for c, m in enumerate(MEANS[::-1] if bgr else MEANS):
        img[:, c, rows] = m * 255
    return {
        "state": rng.normal(size=(500, state_dim)) + shift,
        "actions": rng.normal(size=(10, 50, 7)) + shift,
        "prompt": np.array(prompt),
        **{c: img for c in CAMS},
    }


def make_reference(tmp_path, rng):
    make_dataset(tmp_path / "ds", rng)
    build_reference(tmp_path / "ds", tmp_path / "ref.npz")
    return tmp_path / "ref.npz"


def test_drift(tmp_path):
    rng = np.random.default_rng(0)
    drift = Drift(CONFIG, make_reference(tmp_path, rng), tmp_path / "out")

    good = make_episode(rng)
    assert drift.contract(good) == {"contract_ok": True, "contract_issues": ""}
    assert drift.state_drift(good)["state_wasserstein_std_max"] < 0.2

    bad = make_episode(rng, state_dim=7, shift=2.0, bgr=True, stretched=True, prompt="move the arm to the left")
    res = drift.contract(bad)
    assert res["contract_ok"] is False
    issues = res["contract_issues"].split("; ")
    assert any("state dim 7 != 14" in i for i in issues)
    assert any("prompt" in i for i in issues)
    assert sum("letterbox" in i for i in issues) == 3
    assert sum("BGR" in i for i in issues) == 3
    state = drift.state_drift(bad)
    assert len(drift.state_detail) == 7
    assert state["state_wasserstein_std_mean"] > 1.5 and state["state_frac_outside_max"] > 0.3
    assert state["state_worst_dim"] in drift.state_detail
    assert drift.action_drift(bad)["action_wasserstein_std_mean"] > 1.5
    assert json.loads(drift.detail_path.read_text())["state"] == drift.state_detail


def test_csv_columns(tmp_path):
    rng = np.random.default_rng(0)
    drift = Drift(CONFIG, make_reference(tmp_path, rng), run_dir=tmp_path / "run", name="drift")
    episode = make_episode(rng)
    for m in (drift.contract, drift.state_drift, drift.action_drift):
        m(episode)
    drift.end_episode()
    (row,) = csv.DictReader((tmp_path / "run/drift_episodes.csv").open())
    assert {"contract_ok", "contract_issues", "state_wasserstein_std_max", "state_worst_dim",
            "action_frac_outside_max", "action_wasserstein_std_mean"} <= row.keys()
    assert (tmp_path / "run/drift_drift_detail.json").exists()


def test_cli_check(tmp_path):
    rng = np.random.default_rng(0)
    ref = make_reference(tmp_path, rng)
    np.savez(tmp_path / "ep.npz", **make_episode(rng))
    out = subprocess.run([sys.executable, "-m", "eval_vla.metrics.drift", "check", str(ref), str(tmp_path / "ep.npz"),
                          "--out", str(tmp_path / "out")], capture_output=True, text=True, check=True).stdout
    assert json.loads(out)["contract"]["contract_ok"] is True
    assert list((tmp_path / "out").glob("*/episodes.csv"))


import pytest


def drift_for(tmp_path, seed=0, **kw):
    return Drift(CONFIG, make_reference(tmp_path, np.random.default_rng(seed)), tmp_path / "out", **kw)


def test_build_reference_reads_all_chunks_and_meta(tmp_path):
    rng = np.random.default_rng(1)
    make_dataset(tmp_path / "ds", rng)
    (tmp_path / "ds/data/chunk-001").mkdir()
    pq.write_table(pa.table({"observation.state": rng.normal(size=(300, 14)).tolist(),
                             "action": rng.normal(size=(300, 7)).tolist()}),
                   tmp_path / "ds/data/chunk-001/episode_000001.parquet")
    (tmp_path / "ds/meta/tasks.jsonl").write_text('{"task": "Organize the table"}\n\n{"task": "Stack the cups"}\n')
    build_reference(tmp_path / "ds", tmp_path / "ref.npz")
    with np.load(tmp_path / "ref.npz") as ref:
        assert ref["state"].shape == (2300, 14) and ref["action"].shape == (2300, 7)
        assert ref["state"].dtype == np.float32
        assert ref["state_names"].tolist() == [f"s{i}" for i in range(14)]
        assert ref["action_names"].tolist() == [f"s{i}" for i in range(7)]
        assert int(ref["fps"]) == 30
        assert ref["prompts"].tolist() == ["Organize the table", "Stack the cups"]
        for cam in CAMS:
            np.testing.assert_allclose(ref[f"mean_{cam}"], MEANS)


def test_missing_reference_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Drift(CONFIG, tmp_path / "nope.npz", tmp_path / "out")


def _break(kind, ep):
    img = ep["cam_high"]
    if kind == "missing_cam":
        del ep["cam_high"]
    elif kind == "wrong_shape":
        ep["cam_high"] = np.zeros((20, 3, 256, 256), np.uint8)
    elif kind == "wrong_dtype":
        ep["cam_high"] = img.astype(np.float32) / 255
    elif kind == "channels_last":
        ep["cam_high"] = img.transpose(0, 2, 3, 1)
    elif kind == "no_time_axis":
        ep["cam_high"] = img[0]
    elif kind == "stretched":
        ep["cam_high"] = np.broadcast_to(img[:, :, 100:101, :], img.shape).copy()
    elif kind == "bgr":
        ep["cam_high"] = img[:, ::-1].copy()
    elif kind == "short_state":
        ep["state"] = ep["state"][:, :12]
    elif kind == "long_state":
        ep["state"] = np.concatenate([ep["state"], ep["state"][:, :2]], axis=1)
    elif kind == "unknown_prompt":
        ep["prompt"] = np.array("organize the table")
    return ep


@pytest.mark.parametrize("kind, expected", [
    ("missing_cam", "cam_high missing"),
    ("wrong_shape", "cam_high is (3, 256, 256) uint8"),
    ("wrong_dtype", "cam_high is (3, 224, 224) float32"),
    ("channels_last", "cam_high is (224, 224, 3) uint8"),
    ("no_time_axis", "cam_high is (224, 224) uint8"),
    ("stretched", "cam_high has no letterbox"),
    ("bgr", "cam_high channels match training better when reversed"),
    ("short_state", "state dim 12 != 14, missing ['s12', 's13']"),
    ("long_state", "state dim 16 != 14"),
    ("unknown_prompt", "prompt 'organize the table' not in training prompts"),
])
def test_contract_flags_each_fault_alone(tmp_path, kind, expected):
    drift = drift_for(tmp_path)
    res = drift.contract(_break(kind, make_episode(np.random.default_rng(2))))
    assert res["contract_ok"] is False
    issues = res["contract_issues"].split("; ")
    assert len(issues) == 1 and issues[0].startswith(expected), issues
    assert drift.contract_issues == issues


def test_contract_accepts_plain_str_prompt_and_gray_scene(tmp_path):
    drift = drift_for(tmp_path)
    ep = make_episode(np.random.default_rng(3), prompt="Organize the table")
    ep["prompt"] = "Organize the table"
    assert drift.contract(ep)["contract_ok"] is True
    gray = np.zeros((5, 3, 224, 224), np.uint8)
    gray[:, :, 28:196] = 128
    ep.update({c: gray for c in CAMS})
    assert drift.contract(ep)["contract_issues"] == ""


@pytest.mark.parametrize("shift", [0.5, 1.0, 2.0, 4.0])
def test_wasserstein_tracks_mean_shift_in_std_units(tmp_path, shift):
    drift = drift_for(tmp_path)
    res = drift.state_drift({"state": np.random.default_rng(4).normal(size=(5000, 14)) + shift})
    assert res["state_wasserstein_std_mean"] == pytest.approx(shift, abs=0.1)


def test_same_distribution_has_low_drift(tmp_path):
    drift = drift_for(tmp_path)
    res = drift.state_drift({"state": np.random.default_rng(5).normal(size=(5000, 14))})
    assert res["state_wasserstein_std_max"] < 0.1
    assert res["state_frac_outside_max"] < 0.05


def test_scale_change_without_shift_is_detected(tmp_path):
    drift = drift_for(tmp_path)
    res = drift.state_drift({"state": np.random.default_rng(6).normal(size=(5000, 14)) * 3})
    assert res["state_wasserstein_std_mean"] > 1.0 and res["state_frac_outside_max"] > 0.3


def test_single_dimension_drift_is_localized(tmp_path):
    drift = drift_for(tmp_path)
    state = np.random.default_rng(7).normal(size=(5000, 14))
    state[:, 9] += 3
    res = drift.state_drift({"state": state})
    assert res["state_worst_dim"] == "s9"
    others = [v["wasserstein_std"] for k, v in drift.state_detail.items() if k != "s9"]
    assert drift.state_detail["s9"]["wasserstein_std"] > 2.5 and max(others) < 0.15


def test_state_with_extra_or_missing_dims_compares_the_overlap(tmp_path):
    drift = drift_for(tmp_path)
    rng = np.random.default_rng(8)
    drift.state_drift({"state": rng.normal(size=(500, 16))})
    assert list(drift.state_detail) == [f"s{i}" for i in range(14)]
    drift.state_drift({"state": rng.normal(size=(500, 5))})
    assert list(drift.state_detail) == [f"s{i}" for i in range(5)]


def test_constant_dimension_stays_finite(tmp_path):
    rng = np.random.default_rng(9)
    make_dataset(tmp_path / "ds", rng)
    state = rng.normal(size=(2000, 14))
    state[:, 0] = 0.25
    pq.write_table(pa.table({"observation.state": state.tolist(), "action": rng.normal(size=(2000, 7)).tolist()}),
                   tmp_path / "ds/data/chunk-000/episode_000000.parquet")
    build_reference(tmp_path / "ds", tmp_path / "ref.npz")
    drift = Drift(CONFIG, tmp_path / "ref.npz", tmp_path / "out")
    same = drift.state_drift({"state": np.where(np.arange(14) == 0, 0.25, rng.normal(size=(500, 14)))})
    assert np.isfinite(same["state_wasserstein_std_max"]) and drift.state_detail["s0"]["wasserstein_std"] < 1e-3
    moved = drift.state_drift({"state": np.where(np.arange(14) == 0, 0.5, rng.normal(size=(500, 14)))})
    assert np.isfinite(moved["state_wasserstein_std_max"]) and moved["state_worst_dim"] == "s0"


def test_action_chunks_are_flattened(tmp_path):
    drift = drift_for(tmp_path)
    actions = np.random.default_rng(10).normal(size=(10, 50, 7)) + 1
    chunked = drift.action_drift({"actions": actions})
    flat = drift.action_drift({"actions": actions.reshape(-1, 7)})
    assert chunked == flat
    assert chunked["action_wasserstein_std_mean"] == pytest.approx(1.0, abs=0.15)


def test_results_are_deterministic(tmp_path):
    drift = drift_for(tmp_path)
    ep = make_episode(np.random.default_rng(11), shift=0.7)
    assert drift.state_drift(ep) == drift.state_drift(ep) and drift.action_drift(ep) == drift.action_drift(ep)


def test_several_episodes_append_rows_and_keep_last_detail(tmp_path):
    rng = np.random.default_rng(12)
    drift = Drift(CONFIG, make_reference(tmp_path, rng), run_dir=tmp_path / "run")
    for shift in (0.0, 3.0):
        ep = make_episode(rng, shift=shift)
        for m in (drift.contract, drift.state_drift, drift.action_drift):
            m(ep)
        drift.end_episode()
    rows = list(csv.DictReader((tmp_path / "run/episodes.csv").open()))
    assert [r["episode"] for r in rows] == ["1", "2"]
    assert float(rows[0]["state_wasserstein_std_mean"]) < 0.3 < 2.5 < float(rows[1]["state_wasserstein_std_mean"])
    detail = json.loads((tmp_path / "run/drift_detail.json").read_text())
    assert detail["state"]["s0"]["wasserstein_std"] > 2.5
    report = json.loads((tmp_path / "run/results.json").read_text())
    assert report["episodes_completed"] == 2 and report["metrics"]["contract"]["contract_ok"] is True


def _cli(*args, check=True):
    return subprocess.run([sys.executable, "-m", "eval_vla.metrics.drift", *map(str, args)],
                          capture_output=True, text=True, check=check)


def test_cli_build_then_check_bad_episode(tmp_path):
    make_dataset(tmp_path / "ds", np.random.default_rng(13))
    assert _cli("build", tmp_path / "ds", tmp_path / "ref.npz").stdout == ""
    assert (tmp_path / "ref.npz").exists()
    np.savez(tmp_path / "bad.npz", **make_episode(np.random.default_rng(14), shift=3.0, bgr=True))
    run = _cli("check", tmp_path / "ref.npz", tmp_path / "bad.npz", "--out", tmp_path / "out")
    result = json.loads(run.stdout)
    assert result["contract"]["contract_ok"] is False
    assert result["state_drift"]["state_wasserstein_std_mean"] > 2.5
    assert "results in" in run.stderr and "results in" not in run.stdout


def test_cli_rejects_bad_usage(tmp_path):
    assert _cli(check=False).returncode == 2
    assert _cli("check", tmp_path / "missing.npz", tmp_path / "ep.npz", check=False).returncode != 0


def test_wasserstein_is_in_units_of_training_std(tmp_path):
    rng = np.random.default_rng(15)
    make_dataset(tmp_path / "ds", rng)
    pq.write_table(pa.table({"observation.state": (rng.normal(size=(4000, 14)) * 10).tolist(),
                             "action": rng.normal(size=(4000, 7)).tolist()}),
                   tmp_path / "ds/data/chunk-000/episode_000000.parquet")
    build_reference(tmp_path / "ds", tmp_path / "ref.npz")
    drift = Drift(CONFIG, tmp_path / "ref.npz", tmp_path / "out")
    res = drift.state_drift({"state": rng.normal(size=(5000, 14)) * 10 + 10})
    assert res["state_wasserstein_std_mean"] == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("band", ["top", "bottom"])
def test_letterbox_checks_each_band(tmp_path, band):
    drift = drift_for(tmp_path)
    ep = make_episode(np.random.default_rng(16))
    img = ep["cam_high"].copy()
    rows = slice(0, 28) if band == "top" else slice(196, 224)
    img[:, :, rows] = 200
    ep["cam_high"] = img
    issues = drift.contract(ep)["contract_issues"].split("; ")
    assert any(i.startswith("cam_high has no letterbox") for i in issues), issues
