from __future__ import annotations

import time

from vla_eval.config import SetupConfig
from vla_eval.metrics.operation import Operation
from vla_eval.vision.camera import Camera
from examples.widowx_driver_probe import connect_driver


# TODO: placeholder thresholds, calibrate them with the TEMP prints in the loop below.
EXTERNAL_EFFORT_THRESHOLDS = [3.0] * 6 + [20.0]       # Nm per arm joint, N for the gripper
JOINT_ACCELERATION_THRESHOLDS = [50.0] * 6 + [5.0]    # rad/s^2 per arm joint, m/s^2 for the gripper
LINEAR_ACCELERATION_THRESHOLD = 5.0                   # m/s^2 (end effector)
ANGULAR_ACCELERATION_THRESHOLD = 50.0                 # rad/s^2 (end effector)
SLEEP_POSITIONS = [0.0] * 7                           # TODO: read it with the arm resting in its sleep pose


def fake_policy(obs):
    """Simulates a policy step at the control frequency."""
    time.sleep(0.1)
    return [0.0] * 7


def main(n_episodes: int = 2, steps_per_episode: int = 100, ip: str = "192.168.1.2") -> None:
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
        experiment_name="pick_cube-26/10/2026",
        task_name="Organize_the_table",
    )

    operation = Operation(config)
    camera = Camera(0).open()

    driver = connect_driver(ip)

    for ep in range(n_episodes):
        input(f"Episode {ep}: press Enter to start recording...")
        video_path = operation.output_dir / f"ep{ep}.mp4"
        
        camera.start_recording(video_path, preview=True)

        for step in range(steps_per_episode):
            fake_policy(obs=None)

            ext = list(driver.get_all_external_efforts())
            cart_ext = list(driver.get_cartesian_external_efforts())
            acc = list(driver.get_all_accelerations())
            cart_acc = list(driver.get_cartesian_accelerations())
            print(f"[ep{ep} step{step}] external_efforts      (Nm; N gripper):", [round(v, 2) for v in ext])
            print(f"[ep{ep} step{step}] cartesian_ext_efforts (F N | T Nm)   :", [round(v, 2) for v in cart_ext])
            print(f"[ep{ep} step{step}] joint_accelerations   (rad/s^2)      :", [round(v, 2) for v in acc])
            print(f"[ep{ep} step{step}] cartesian_accelerations lin|ang      :", [round(v, 2) for v in cart_acc],
                  "| |lin| =", round(sum(v * v for v in cart_acc[:3]) ** 0.5, 2),
                  "|ang| =", round(sum(v * v for v in cart_acc[3:]) ** 0.5, 2))

            collision = operation.collision_rate(driver, EXTERNAL_EFFORT_THRESHOLDS)
            jerks = operation.jerks_rate(driver, JOINT_ACCELERATION_THRESHOLDS, LINEAR_ACCELERATION_THRESHOLD, ANGULAR_ACCELERATION_THRESHOLD)

        camera.stop_recording()
        print(collision)
        print(jerks)

        input(f"Episode {ep}: move the arm to its sleep pose and press Enter to check the reset...")
        print(operation.reset_failure_rate(driver, SLEEP_POSITIONS))

        print(operation.human_intervention_rate(video_path))
        operation.end_episode()

    camera.close()


if __name__ == "__main__":
    main()
