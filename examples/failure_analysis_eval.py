"""Diagnoses the failure type of recorded episodes.

    python examples/failure_analysis_eval.py eval_results/pick_cube-26-10-2026/ep0.mp4 --window 30 --threshold 0.02
"""

from __future__ import annotations

import argparse

from vla_eval.config import SetupConfig
from vla_eval.metrics.failure_analysis import FailureAnalysis
import time


def main(video: str, window_s: float = 30.0, motion_threshold: float = 0.02) -> None:
    config = SetupConfig(
        robot_embodiment={"manufacturer": "Trossen", "model": "WidowX", "dof": 6, "end_effector": "Precision Gripper"},
        camera_configuration={"count": 3, "positions": ["wrist", "front", "high"], "resolution": [640, 480], "wrist": True},
        action_space={"type": "end_effector_pose", "dim": 7},
        control_frequency=10.0,
        object_set={"count": 1, "materials": ["plastic"], "seen_in_training": True},
        scene_layout={"workspace_m": "Grey Table with 100 meters", "lighting": "indoor"},
        reset_procedure="VLA",
        success_criteria="objects inside the tots",
        model_name="openpi-finetune",
        experiment_name="Failure_Planning2_30frames",
        task_name="Organize_the_table",
        perturbation_type=None, 
    )

    start_time = time.perf_counter()

    analysis = FailureAnalysis(config, video, window_s=window_s, debug_video=True)

    print(f"\n{video}")
    print("  failure type      :", analysis.failure_type)
    print("  results           :", analysis._results)
    print("  inference_time           :", time.perf_counter() - start_time)

    analysis.end_episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="episode video to analyze")
    parser.add_argument("--window", type=float, default=30.0, help="seconds at the end of the video to analyze")
    parser.add_argument("--threshold", type=float, default=0.02, help="motion std (fraction of the frame diagonal) below which the arm counts as still")
    args = parser.parse_args()
    main(args.video, args.window, args.threshold)
