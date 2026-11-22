"""Runs every metric family on the same episodes and plots their CSVs (graphs are refreshed after each episode).

    uv run -m examples.full_eval --n_episodes 3 --steps_per_episode 50 --ip 192.168.1.2
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from examples.operation_eval import JOINT_NAMES
from examples.statistical_eval_image_diff import CALIBRATION_DIR, fake_policy, load_or_capture_reference
from examples.widowx_driver_probe import connect_driver
from eval_vla.config import SetupConfig
from eval_vla.metrics.failure_analysis import FailureAnalysis
from eval_vla.metrics.long_horizon import LongHorizon
from eval_vla.metrics.operation import Operation
from eval_vla.metrics.statistical import Statistical
from eval_vla.vision.camera import Camera
from eval_vla.vision.roi_selector import RoiSelector
from eval_vla.vision.viewer import make_viewer


def main(n_episodes: int = 5, steps_per_episode: int = 50, ip: str = "192.168.1.2", window_s: float = 30.0,
         debug_video: bool = False, viewer: str = "rerun") -> None:
    start_eval_time = time.perf_counter()
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
        experiment_name="Full_eval_Complete",
        task_name=task_name,
        perturbation_type=None,  # list of: object, layout, camera, lighting, background, initial_state, sensor_noise
    )

    # one folder for the whole run: <name>_episodes.csv / <name>.png per family
    run_dir = Path("eval_results") / f"{config.experiment_name}_{datetime.now():%d-%m_%H:%M:%S}"

    stats = Statistical(config, run_dir=run_dir, name="statistical")
    operation = Operation(config, episodes=n_episodes, run_dir=run_dir, name="operation")
    long_horizon = LongHorizon(config, run_dir=run_dir, name="long_horizon")

    camera = Camera(0).open()
    view = make_viewer(viewer, camera)
    driver = connect_driver(ip)

    reference_image = load_or_capture_reference(camera, CALIBRATION_DIR / task_name / "reference.png")
    rois = RoiSelector.load_or_create(CALIBRATION_DIR / task_name / "rois.json", ["table", "container"], reference_image)

    for ep in range(n_episodes):
        video_path = run_dir / f"ep{ep}.mp4"
        before_frame = camera.capture(run_dir / f"ep{ep}_before.png")
        camera.start_recording(video_path, preview=view.preview)

        start_time = time.perf_counter()
        for step in range(steps_per_episode):
            inference_start = time.perf_counter()
            fake_policy(obs=None)
            stats.inference_latency(inference_start, time.perf_counter(), steps_per_episode=steps_per_episode)

            external_efforts = dict(zip(JOINT_NAMES, (round(e, 2) for e in driver.get_all_external_efforts())))
            operation.collision_rate(driver, mode="gmo")
            operation.jerks_rate(driver)

            view.update(
                external_effort=external_efforts,
                gmo_residual=dict(zip(JOINT_NAMES, operation.gmo_residual)),
                jerk_peaks=dict(zip([*JOINT_NAMES, "linear", "angular"], operation.jerks_peaks)),
            )

        stats.task_completion_time(start_time, time.perf_counter())

        camera.wait_for_enter(f"Episode {ep}: add the objects to the tot before finishing the task, then press Enter...", on_tick=view.update)
        after_frame = camera.capture(run_dir / f"ep{ep}_after.png")
        camera.stop_recording()

        stats.success_rate(task_name, "model_segmentation", before_frame, after_frame, prompt=["Cube", "black clock cube"], rois=rois, model="sam3")
        #operation.reset_failure_rate(driver)
        operation.human_intervention_rate(video_path, prompt=["hand", "human arm"], model="sam3")

        subtasks_completed = [bool(stats.success_rate_value)] 
        long_horizon.subtask_completion_rate(subtasks_completed)
        long_horizon.average_completed_sequence_length(subtasks_completed)
        long_horizon.failure_position_distribution(subtasks_completed)

        stats.end_episode()
        operation.end_episode()
        long_horizon.end_episode()

        if not stats.success_rate_value:
            analysis = FailureAnalysis(config, video_path, window_s=window_s, debug_video=debug_video,
                                       run_dir=run_dir, name="failure_analysis")
            print(f"Episode {ep} failure type:", analysis.failure_type)
            analysis.end_episode(episode=ep + 1)

    view.close()
    camera.close()
    print("Final eval time (segundos):", round(time.perf_counter() - start_eval_time, 3))
    print("Results and graphs in", run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=2)
    parser.add_argument("--steps_per_episode", type=int, default=50)
    parser.add_argument("--ip", default="192.168.1.2")
    parser.add_argument("--window_s", type=float, default=30.0, help="seconds at the end of a failed episode sent to the VLM")
    parser.add_argument("--debug_video", action="store_true", help="save the frames sent to the VLM")
    parser.add_argument("--viewer", choices=["opencv", "rerun"], default=argparse.SUPPRESS)
    main(**vars(parser.parse_args()))
