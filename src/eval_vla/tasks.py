"""Evaluation tasks: tolerant matching of their names and the collision observer of the operation metrics."""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pinocchio as pin
import trossen_arm

TASKS = ("Organize_the_table",)
_ARTICLES = {"the", "a", "an", "teh"}

CALIBRATION_MIN_SAMPLES = 30
CARRIAGE_X = 0.0865                                   # m, finger carriages position along the last link (URDF)


def _tokens(name: str) -> list[str]:
    """Lowercase words of `name`, without accents; spaces, "_", "-" and CamelCase separate words."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", text.lower())


def _matches(name: str, task: str, cutoff: float) -> bool:
    a, b = _tokens(name), _tokens(task)
    if "".join(a) == "".join(b):
        return True
    a = [t for t in a if t not in _ARTICLES]
    b = [t for t in b if t not in _ARTICLES]
    # Word by word, because on the whole sentence "cable" is as close to "table" as a typo is.
    return len(a) == len(b) and all(
        x[0] == y[0] and SequenceMatcher(None, x, y).ratio() >= cutoff for x, y in zip(a, b))


def resolve_task_name(name: str, cutoff: float = 0.8) -> str | None:
    """Canonical task name for `name`, or None if it is not a known task.

    Args:
        name: Task name as typed, e.g. "organize the table".
        cutoff: Minimum `difflib` similarity (0 to 1) between each pair of words, which must also start
            with the same letter.
    """
    return next((task for task in TASKS if _matches(name, task, cutoff)), None)


def end_effector_inertia(end_effector: trossen_arm.EndEffector, gripper_position: float) -> pin.Inertia:
    """Inertia of the last arm link (palm and both fingers) from the end effector configured in the driver."""
    def link_inertia(link, offset) -> pin.Inertia:
        rotation = pin.rpy.rpyToMatrix(np.array(link.origin_rpy, dtype=float))
        inertia = rotation @ np.array(link.inertia, dtype=float).reshape(3, 3) @ rotation.T
        return pin.Inertia(link.mass, offset + np.array(link.origin_xyz, dtype=float), inertia)

    left = np.array([CARRIAGE_X, end_effector.offset_finger_left + gripper_position, 0.0])
    right = np.array([CARRIAGE_X, end_effector.offset_finger_right - gripper_position, 0.0])
    return (link_inertia(end_effector.palm, np.zeros(3)) + link_inertia(end_effector.finger_left, left)
            + link_inertia(end_effector.finger_right, right))


def read_friction(driver: trossen_arm.TrossenArmDriver, n_joints: int = 6) -> dict[str, np.ndarray]:
    """Friction parameters of the arm joints configured in the driver."""
    return {name: np.array(getattr(driver, f"get_friction_{name}")()[:n_joints], dtype=float)
            for name in ("constant_terms", "coulomb_coefs", "viscous_coefs", "transition_velocities")}


def friction_efforts(friction: dict[str, np.ndarray], model_efforts: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Joint friction as modelled by Trossen (constant + Coulomb + viscous), linear around zero velocity.

    https://docs.trossenrobotics.com/trossen_arm/main/getting_started/configuration.html
    """
    magnitude = (friction["constant_terms"] + friction["coulomb_coefs"] * np.abs(model_efforts)
                 + friction["viscous_coefs"] * np.abs(velocities))
    return magnitude * np.clip(velocities / friction["transition_velocities"], -1.0, 1.0)


class CollisionObserver:
    """Collision state, momentum observer and calibration behind `Operation.collision_rate`.

    Mixed into `Operation`, which creates the model (`gmo_model`, `gmo_data`) and the `collision_*` state.
    """

    def _start_collision_episode(self, driver, output, mode: str, calibration: str | Path | None) -> None:
        data = None
        if calibration is not None:
            data = json.loads(Path(calibration).read_text())
            if data["mode"] != mode:
                raise ValueError(f"calibration {calibration} was recorded in mode {data['mode']!r}, not {mode!r}")

        if mode == "gmo":
            self.gmo_friction = read_friction(driver, self.gmo_model.nv)
            gripper_position = output.joint.all.positions[self.gmo_model.nv]
            self.gmo_model.inertias[-1] = end_effector_inertia(driver.get_end_effector(), gripper_position)

        n = self.gmo_model.nv
        self.collision_episode, self.collision_mode, self.collision_calibration = self._episode, mode, data
        self.collision_detected_value = False
        self.collision_start = self.collision_time = self.gmo_previous = None
        self.gmo_residual, self.collision_values, self.collision_peaks = np.zeros(n), np.zeros(n), np.zeros(n)
        self.collision_consecutive, self.collision_streaks = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        self.collision_samples = []


    def _update_gmo(self, output, time_s: float, gain: float) -> np.ndarray:
        """Advances the momentum observer to the current sample and returns the residual (external torque), in Nm."""
        model, data = self.gmo_model, self.gmo_data
        arm = output.joint.arm
        q, v, acc, tau = (np.array(x, dtype=float) for x in (arm.positions, arm.velocities, arm.accelerations, arm.efforts))

        friction = friction_efforts(self.gmo_friction, pin.rnea(model, data, q, v, acc).copy(), v)
        M = pin.crba(model, data, q)
        momentum = (np.triu(M) + np.triu(M, 1).T) @ v
        coriolis = pin.computeCoriolisMatrix(model, data, q, v).T @ v
        drive = tau - friction + coriolis - pin.computeGeneralizedGravity(model, data, q)  # tau + beta

        if self.gmo_previous is not None:
            dt = time_s - self.collision_time
            previous_momentum, previous_drive = self.gmo_previous
            decay = np.exp(-gain * dt)  # exact discretization: stable for any dt
            external = (momentum - previous_momentum) / dt - (drive + previous_drive) / 2
            self.gmo_residual = decay * self.gmo_residual + (1 - decay) * external

        self.gmo_previous = (momentum, drive)
        return self.gmo_residual


    def save_collision_calibration(self, path: str | Path) -> Path:
        """Saves the mean and standard deviation of the collision signal of the current episode.

        Run an episode with a representative motion and **no contact**, calling `collision_rate` at every
        step, then call this before `end_episode()`. Pass the file as `calibration` in later episodes.

        Raises:
            ValueError: If the episode has fewer than `CALIBRATION_MIN_SAMPLES` samples after the warm-up.
        """
        samples = self.collision_samples if self.collision_episode == self._episode else []
        if len(samples) < CALIBRATION_MIN_SAMPLES:
            raise ValueError(f"calibration needs at least {CALIBRATION_MIN_SAMPLES} samples, got {len(samples)}")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "mode": self.collision_mode,
            "samples": len(samples),
            "mean": np.mean(samples, axis=0).tolist(),
            "std": np.std(samples, axis=0).tolist(),
        }, indent=2))
        return path
