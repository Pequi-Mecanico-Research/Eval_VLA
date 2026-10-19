from __future__ import annotations

import json
import random
import time

from ..src.vla_eval.config import SetupConfig
from ..src.vla_eval.metrics.statistical import Statistical


def fake_policy(obs):
    """Simula a inferência de um VLA (5-15 ms)."""
    time.sleep(random.uniform(0.005, 0.015))
    return [0.0] * 7


def main(n_episodes: int = 10, steps_per_episode: int = 5) -> None:
    task_name = "Organize_the_table"

    config = SetupConfig(
        robot_embodiment={"manufacturer": "Trossen", "model": "WidowX", "dof": 6, "end_effector": "Precision Gripper"},
        camera_configuration={"count": 3, "positions": ["wrist", "front", "high"], "resolution": [640, 480], "wrist": True},
        action_space={"type": "end_effector_pose", "dim": 7},
        control_frequency=10.0,
        object_set={"count": 1, "materials": ["plastic"], "seen_in_training": True},
        scene_layout={"workspace_m": [0.8, 0.6], "lighting": "indoor"},
        reset_procedure="manual",
        success_criteria="objects inside the tots",
        model_name="openpi-pick_cube",
        experiment_name="pick_cube-05/10/2026",
        task_name=task_name
    )

    stats = Statistical(config)

    for ep in range(n_episodes):
        start_time = time.perf_counter()
        for _ in range(steps_per_episode):
            inferece_start = time.perf_counter()

            fake_policy(obs=None)

            stats.inference_latency(inferece_start, time.perf_counter(), steps_per_episode=steps_per_episode)

        stats.task_completion_time(start_time, time.perf_counter())

        stats.success_rate(task_name=task_name)

        stats.end_episode()


if __name__ == "__main__":
    main()
