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
