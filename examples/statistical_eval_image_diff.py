from __future__ import annotations

import json
import random
import time
from pathlib import Path

from vla_eval.config import SetupConfig
from vla_eval.metrics.statistical import Statistical
from vla_eval.vision.camera import Camera, load_image
from vla_eval.vision.roi_selector import load_rois, create_rois


def fake_policy(obs):
    """Simula a inferência de um VLA (5-15 ms)."""
    time.sleep(random.uniform(0.005, 0.015))
    return [0.0] * 7


def main(n_episodes: int = 2, steps_per_episode: int = 5) -> None:
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

    #TODO: Facilitar a escolha de câmera dentro do código, podendo escolher pelo nome além de (0 na lista). 
    camera = Camera(0).open()

    reference_image_path = "eval_results/pick_cube-05-10-2026/reference.png"
    rois_path = "eval_results/pick_cube-05-10-2026/rois.json"
    debug_plots = True  

    if reference_image_path and Path(reference_image_path).exists():
        reference_image = load_image(reference_image_path)
    else:
        input("Criando imagem de referência: Deixe somente os objetos fixos na cena e aperte Enter...")
        reference_image = camera.capture(stats.output_dir / "reference.png", countdown_s=2)

    if rois_path and Path(rois_path).exists():
        rois = load_rois(rois_path)
    else:
        rois = create_rois(reference_image, ["table", "container"], stats.output_dir / "rois.json")


    for ep in range(n_episodes):
        input(f"Episode {ep}: press Enter to run (Adicione os objetos na mesa para iniciar o eval.)...")
        before_frame = camera.capture(stats.output_dir / f"ep{ep}_before.png")
        start_time = time.perf_counter()
        for _ in range(steps_per_episode):
            inference_start = time.perf_counter()

            fake_policy(obs=None)

            stats.inference_latency(inference_start, time.perf_counter(), steps_per_episode=steps_per_episode)

        stats.task_completion_time(start_time, time.perf_counter())

        input(f"Episode {ep}: press Enter to run (Adicione os objetos ao Tot antes de finalizar a task.)...")
        after_frame = camera.capture(stats.output_dir / f"ep{ep}_after.png")

        stats.success_rate(task_name, "image_diff", before_frame, after_frame, reference_frame=reference_image, rois=rois, debug_plots=debug_plots)

        stats.end_episode()

    camera.close()


if __name__ == "__main__":
    main()
