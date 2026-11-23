from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..base import EvalVLA, metric
from ..config import SetupConfig

CAMERAS = ("cam_high", "cam_wrist", "cam_low")
PAD_ROWS = 28


def build_reference(dataset_root: str | Path, out_path: str | Path) -> None:
    root = Path(dataset_root)
    info = json.loads((root / "meta/info.json").read_text())
    stats = json.loads((root / "meta/stats.json").read_text())
    prompts = [json.loads(line)["task"] for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line]
    files = sorted(root.glob("data/**/*.parquet"))

    def column(name):
        return np.concatenate([np.array(pq.read_table(f, columns=[name])[name].to_pylist(), dtype=np.float32) for f in files])

    np.savez(
        out_path,
        state=column("observation.state"),
        action=column("action"),
        state_names=info["features"]["observation.state"]["names"],
        action_names=info["features"]["action"]["names"],
        fps=info["fps"],
        prompts=prompts,
        **{f"mean_{cam}": np.ravel(stats[f"observation.images.{cam}"]["mean"]) for cam in CAMERAS},
    )


class Drift(EvalVLA):
    """Production episode vs. training reference (built by `build_reference`).

    Layer 0: input contract (state dim, prompt, image format, letterbox, RGB/BGR).

    Layer 1: state/action distribution (fraction outside the training [q01, q99] and Wasserstein / training std).
    """

    def __init__(self, config: SetupConfig, reference_path: str | Path, output_dir: str | Path = "eval_results",
                 run_dir: str | Path | None = None, name: str | None = None) -> None:
        super().__init__(config, output_dir, run_dir, name)
        with np.load(reference_path) as data:
            self.ref = dict(data)
        self.contract_issues: list[str] = []
        self.state_detail: dict[str, dict[str, float]] = {}
        self.action_detail: dict[str, dict[str, float]] = {}
        self.detail_path = self.output_dir / (f"{name}_drift_detail.json" if name else "drift_detail.json")

    def _save_detail(self) -> None:
        self.detail_path.write_text(json.dumps(
            {"contract_issues": self.contract_issues, "state": self.state_detail, "action": self.action_detail}, indent=2))

    @metric
    def contract(self, episode: dict) -> dict:
        """Layer 0 checks of the episode inputs.

        Call once per episode.

        Args:
            episode: dict with `state`, `prompt` and one image array per camera (see README).

        Returns:
            dict with `contract_ok` and `contract_issues` (issues joined by "; ", "" when ok).
        """
        issues = []
        dim, ref_dim = episode["state"].shape[-1], self.ref["state"].shape[-1]
        if dim != ref_dim:
            issues.append(f"state dim {dim} != {ref_dim}, missing {self.ref['state_names'][dim:].tolist()}")
        if str(episode["prompt"]) not in self.ref["prompts"]:
            issues.append(f"prompt {str(episode['prompt'])!r} not in training prompts {self.ref['prompts'].tolist()}")
        for cam in CAMERAS:
            if cam not in episode:
                issues.append(f"{cam} missing")
                continue
            img = episode[cam]
            if img.shape[1:] != (3, 224, 224) or img.dtype != np.uint8:
                issues.append(f"{cam} is {img.shape[1:]} {img.dtype}, expected (3, 224, 224) uint8")
                continue
            if img[..., :PAD_ROWS, :].mean() > 5 or img[..., -PAD_ROWS:, :].mean() > 5:
                issues.append(f"{cam} has no letterbox padding (stretched resize?)")
            # ponytail: channel-mean heuristic, unreliable on gray scenes; Layer 2 embeddings replace it
            means = img[..., PAD_ROWS:-PAD_ROWS, :].mean(axis=(0, 2, 3)) / 255
            ref = self.ref[f"mean_{cam}"]
            if np.abs(means[::-1] - ref).sum() < np.abs(means - ref).sum():
                issues.append(f"{cam} channels match training better when reversed (BGR?)")
        self.contract_issues = issues
        self._save_detail()
        return {"contract_ok": not issues, "contract_issues": "; ".join(issues)}

    @metric
    def state_drift(self, episode: dict) -> dict:
        """Layer 1 drift of the state distribution against the training state.

        Call once per episode.

        Args:
            episode: dict with `state`, shape (T, state_dim).

        Returns:
            dict with `state_wasserstein_std_max`, `state_wasserstein_std_mean`, `state_frac_outside_max`, `state_worst_dim`.
        """
        self.state_detail = self._drift(episode["state"], self.ref["state"], self.ref["state_names"])
        self._save_detail()
        return self._summary("state", self.state_detail)

    @metric
    def action_drift(self, episode: dict) -> dict:
        """Layer 1 drift of the predicted actions against the training actions.

        Call once per episode.

        Args:
            episode: dict with `actions`, shape (..., action_dim); leading axes are flattened.

        Returns:
            dict with `action_wasserstein_std_max`, `action_wasserstein_std_mean`, `action_frac_outside_max`, `action_worst_dim`.
        """
        actions = episode["actions"]
        self.action_detail = self._drift(actions.reshape(-1, actions.shape[-1]), self.ref["action"], self.ref["action_names"])
        self._save_detail()
        return self._summary("action", self.action_detail)

    @staticmethod
    def _summary(prefix: str, detail: dict) -> dict:
        w = {k: v["wasserstein_std"] for k, v in detail.items()}
        return {
            f"{prefix}_wasserstein_std_max": max(w.values()),
            f"{prefix}_wasserstein_std_mean": float(np.mean(list(w.values()))),
            f"{prefix}_frac_outside_max": max(v["outside_q01_q99"] for v in detail.values()),
            f"{prefix}_worst_dim": max(w, key=w.get),
        }

    @staticmethod
    def _drift(prod: np.ndarray, ref: np.ndarray, names: np.ndarray) -> dict:
        q = np.linspace(0, 1, 101)
        lo, hi = np.quantile(ref, [0.01, 0.99], axis=0)
        out = {}
        for i in range(min(prod.shape[-1], ref.shape[-1])):
            x = prod[:, i]
            out[str(names[i])] = {
                "outside_q01_q99": float(np.mean((x < lo[i]) | (x > hi[i]))),
                "wasserstein_std": float(np.abs(np.quantile(x, q) - np.quantile(ref[:, i], q)).mean() / (ref[:, i].std() + 1e-6)),
            }
        return out


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval_vla.metrics.drift")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build")
    build.add_argument("dataset")
    build.add_argument("reference")
    check = sub.add_parser("check")
    check.add_argument("reference")
    check.add_argument("episode")
    check.add_argument("--out", default="eval_results")
    args = parser.parse_args()
    if args.cmd == "build":
        build_reference(args.dataset, args.reference)
        return
    config = SetupConfig({}, {}, {}, 0.0, {}, {}, "", "", "", "drift_check", "")
    drift = Drift(config, args.reference, args.out)
    with np.load(args.episode) as data:
        episode = dict(data)
    result = {m: getattr(drift, m)(episode) for m in ("contract", "state_drift", "action_drift")}
    drift.end_episode()
    print(json.dumps(result, indent=2))
    print(f"results in {drift.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
