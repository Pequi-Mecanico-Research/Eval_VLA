from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from vla_eval.config import SetupConfig
from vla_eval.metrics.statistical import Statistical
from vla_eval.vision.camera import Camera, load_image
from vla_eval.vision.viewer import make_viewer
from vla_eval.vision.roi_selector import RoiSelector

CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration"


def fake_policy(obs):
    """Simula a inferência de um VLA (5-15 ms)."""
    time.sleep(random.uniform(0.005, 0.015))
    return [0.0] * 7


def load_or_capture_reference(camera, reference_image_path):
    if Path(reference_image_path).exists():
        return load_image(reference_image_path)
    input("Criando imagem de referência: Deixe somente os objetos fixos na cena e aperte Enter...")
    return camera.capture(reference_image_path, countdown_s=2)


def main(n_episodes: int = 2, steps_per_episode: int = 5, viewer: str = "rerun") -> None:
    task_name = "Organize_the_table"

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
        experiment_name="Changing_cube_locations",
        task_name=task_name,
    )

    stats = Statistical(config)

    camera = Camera(0).open()
    view = make_viewer(viewer, camera)

    reference_image_path = CALIBRATION_DIR / task_name / "reference.png"
    rois_path = CALIBRATION_DIR / task_name / "rois.json"
    debug_plots = True  

    reference_image = load_or_capture_reference(camera, reference_image_path)
    rois = RoiSelector.load_or_create(rois_path, ["table", "container"], reference_image)

    for ep in range(n_episodes):
        before_frame = camera.capture(stats.output_dir / f"ep{ep}_before.png")
        camera.start_recording(stats.output_dir / f"ep{ep}_debug.mp4", preview=view.preview)  
        
        start_time = time.perf_counter()
        for _ in range(steps_per_episode):
            inference_start = time.perf_counter()

            view.update()
            fake_policy(obs=None)

            stats.inference_latency(inference_start, time.perf_counter(), steps_per_episode=steps_per_episode)

        stats.task_completion_time(start_time, time.perf_counter())

        camera.wait_for_enter(f"Episode {ep}: press Enter to run (Adicione os objetos ao Tot antes de finalizar a task.)...", on_tick=view.update)
        
        after_frame = camera.capture(stats.output_dir / f"ep{ep}_after.png")
        camera.stop_recording()

        stats.success_rate(task_name, "image_diff", before_frame, after_frame, reference_frame=reference_image, rois=rois, debug_plots=debug_plots)

        stats.end_episode()

    view.close()
    camera.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=2)
    parser.add_argument("--steps_per_episode", type=int, default=5)
    parser.add_argument("--viewer", choices=["opencv", "rerun"], default=argparse.SUPPRESS)
    main(**vars(parser.parse_args()))
