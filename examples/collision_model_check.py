"""Read-only check of the dynamic model used by the collision residual, with the arm still and without contact.

Compares the gravity + friction of the model with the compensation efforts computed by the driver. Large
differences point to a wrong end effector, joint offsets or effort sign. Never commands the arm.

    uv run -m examples.collision_model_check --ip 192.168.1.2 --end_effector wxai_v0_follower
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pinocchio as pin
import trossen_arm

from eval_vla.metrics.operation import URDF_PATH, end_effector_inertia, friction_efforts, read_friction


def main(ip: str = "192.168.1.2", end_effector: str = "wxai_v0_follower", samples: int = 100) -> None:
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(trossen_arm.Model.wxai_v0, getattr(trossen_arm.StandardEndEffector, end_effector), ip, False, timeout=10)

    full = pin.buildModelFromUrdf(str(URDF_PATH))
    carriages = [full.getJointId(n) for n in ("left_carriage_joint", "right_carriage_joint")]
    model = pin.buildReducedModel(full, carriages, pin.neutral(full))
    model.inertias[-1] = end_effector_inertia(driver.get_end_effector(), driver.get_robot_output().joint.all.positions[6])
    data = model.createData()
    friction = read_friction(driver)

    expected, reported, measured = [], [], []
    for _ in range(samples):
        arm = driver.get_robot_output().joint.arm
        q, v = np.array(arm.positions), np.array(arm.velocities)
        inverse_dynamics = pin.rnea(model, data, q, v, np.zeros(model.nv)).copy()
        expected.append(inverse_dynamics + friction_efforts(friction, inverse_dynamics, v))
        reported.append(list(arm.compensation_efforts))
        measured.append(list(arm.efforts))
        time.sleep(0.02)

    expected, reported, measured = (np.mean(x, axis=0) for x in (expected, reported, measured))
    print(f"{'joint':8}{'model':>10}{'driver comp':>14}{'effort':>10}{'model-comp':>13}{'model-effort':>15}  (Nm)")
    for i, (m, c, e) in enumerate(zip(expected, reported, measured)):
        print(f"joint_{i:<2}{m:10.3f}{c:14.3f}{e:10.3f}{m - c:13.3f}{m - e:15.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.1.2")
    parser.add_argument("--end_effector", default="wxai_v0_follower", help="must match the physical end effector")
    parser.add_argument("--samples", type=int, default=100)
    main(**vars(parser.parse_args()))
