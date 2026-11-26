from __future__ import annotations

import argparse
import time
from pathlib import Path

from eval_vla.config import SetupConfig
from eval_vla.metrics.operation import Operation
from eval_vla.vision.camera import Camera
from eval_vla.vision.viewer import make_viewer
from examples.widowx_driver_probe import connect_driver


JOINT_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]  # order of the driver's per-joint vectors


def fake_policy(obs):
    """Simulates a policy step at the control frequency."""
    time.sleep(0.1)
    return [0.0] * 7


COLLISION_CALIBRATION = Path("calibration") / "collision_gmo.json"


def main(n_episodes: int = 2, steps_per_episode: int = 100, ip: str = "192.168.1.2", viewer: str = "rerun",
         calibrate_collision: bool = False) -> None:
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
        task_name="Organize_the_table",
        perturbation_type=None,  
    )

    operation = Operation(config, episodes=n_episodes)
    camera = Camera(0).open()
    view = make_viewer(viewer, camera)

    driver = connect_driver(ip)

    # without a calibration file the fixed thresholds are used
    calibration = COLLISION_CALIBRATION if COLLISION_CALIBRATION.exists() and not calibrate_collision else None

    for ep in range(n_episodes):
        video_path = operation.output_dir / f"ep{ep}.mp4"       
        camera.start_recording(video_path, preview=view.preview)

        for step in range(steps_per_episode):
            fake_policy(obs=None)

            external_efforts = dict(zip(JOINT_NAMES, (round(e, 2) for e in driver.get_all_external_efforts())))
            print(f"[ep{ep} step{step}] external_effort (Nm; N gripper):", external_efforts)

            collision = operation.collision_rate(driver, mode="gmo", calibration=calibration)
            jerks = operation.jerks_rate(driver)

            view.update(
                external_effort=external_efforts,
                gmo_residual=dict(zip(JOINT_NAMES, operation.gmo_residual)),
                jerk=dict(zip(JOINT_NAMES, operation.jerks_values)),
            )

        camera.stop_recording()

        print(collision)
        print(jerks)

        print(operation.reset_failure_rate(driver))

        print(operation.human_intervention_rate(video_path))
        if calibrate_collision:  # the episode must have no contact
            print("Collision calibration saved to", operation.save_collision_calibration(COLLISION_CALIBRATION))
        operation.end_episode()

    view.close()
    camera.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=2)
    parser.add_argument("--steps_per_episode", type=int, default=100)
    parser.add_argument("--ip", default="192.168.1.2")
    parser.add_argument("--viewer", choices=["opencv", "rerun"], default=argparse.SUPPRESS)
    parser.add_argument("--calibrate_collision", action="store_true",
                        help="record the collision thresholds from episodes without any contact")
    main(**vars(parser.parse_args()))
