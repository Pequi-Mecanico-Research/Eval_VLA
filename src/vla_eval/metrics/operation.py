from __future__ import annotations

from pathlib import Path

import numpy as np

from ..base import EvalVLA, metric
from ..config import SetupConfig
from ..success_detection.human_detection import detect_human_intervals


class Operation(EvalVLA):
    """Operational metrics.

    Human intervention rate (fraction of trials requiring manual correction during execution).

    Collision rate (fraction of trials in which unintended contact occurs).

    Jerks rate (fraction of trials which show substântial jerk / measure jerk_rate)z.

    Reset failure rate (fraction of trials in which the reset procedure itself fails or introduces bias).
    """

    def __init__(self, config: SetupConfig, output_dir: str | Path = "eval_results"):
        super().__init__(config, output_dir)
        self.human_intervention_count: int = 0
        self.collision_count: int = 0
        self.constraint_violation_count: int = 0
        self.unsafe_contact_count: int = 0
        self.reset_failure_count: int = 0
        self.human_intervention_trials: int = 0
        self.reset_trials: int = 0
        self.collision_trials: int = 0
        self.collision_detected_value: bool = False
        self.collision_episode: int | None = None
        self.collision_peaks: list[float] = []
        self.jerks_count: int = 0
        self.jerks_trials: int = 0
        self.jerks_detected_value: bool = False
        self.jerks_episode: int | None = None
        self.jerks_peaks: list[float] = []


    @metric
    def human_intervention_rate(self, video_path: str | Path, precision: int = 3, **detection_kwargs) -> dict[str, float]:
        """Counts the human hand/arm appearances in the episode video."""

        intervals = detect_human_intervals(video_path, **detection_kwargs)

        self.human_intervention_trials += 1
        self.human_intervention_count += bool(intervals)

        return {
            "human_interventions": len(intervals),
            "human_intervention_sec": round(sum(b - a for a, b in intervals), precision),
            "human_intervention_intervals": [(round(a, precision), round(b, precision)) for a, b in intervals],
            "human_intervention_rate": round(self.human_intervention_count / self.human_intervention_trials, precision),
        }


    @metric
    def collision_rate(self, driver, external_effort_thresholds: list[float], precision: int = 3) -> dict:
        """Flags a collision when the external effort of any joint goes over its threshold. Call it at every control step.

        `external_effort_thresholds` has one value per joint in the order of `driver.get_all_external_efforts()`
        (Nm for the arm joints, N for the gripper). The episode counts as a collision once any reading exceeds a
        threshold, and `collision_joint_x_value` is the highest absolute effort seen in joint x during the episode."""

        joint_efforts = list(driver.get_all_external_efforts())

        if len(joint_efforts) != len(external_effort_thresholds):
            raise ValueError(f"Expected {len(joint_efforts)} external_effort_thresholds, got {len(external_effort_thresholds)}")

        if self.collision_episode != self._episode:
            self.collision_episode = self._episode
            self.collision_trials += 1
            self.collision_detected_value = False
            self.collision_peaks = [0.0] * len(joint_efforts)

        self.collision_peaks = [max(peak, abs(value)) for peak, value in zip(self.collision_peaks, joint_efforts)]

        if not self.collision_detected_value and any(abs(v) > t for v, t in zip(joint_efforts, external_effort_thresholds)):
            self.collision_detected_value = True
            self.collision_count += 1

        result = {
            "collision_detected": self.collision_detected_value,
            "collision_rate": round(self.collision_count / self.collision_trials, precision),
        }
        for i, (threshold, peak) in enumerate(zip(external_effort_thresholds, self.collision_peaks)):
            result[f"collision_joint_{i}_threshold"] = threshold
            result[f"collision_joint_{i}_value"] = round(peak, precision)

        return result


    @metric
    def jerks_rate(self, driver, joint_acceleration_thresholds: list[float], linear_acceleration_threshold: float,
                   angular_acceleration_threshold: float, precision: int = 3) -> dict:
        """Flags jerky motion when an acceleration goes over its threshold. Call it at every control step.

        `joint_acceleration_thresholds` has one value per joint in the order of `driver.get_all_accelerations()`
        (rad/s^2 for the arm joints, m/s^2 for the gripper). The norms of the end effector's linear (m/s^2) and
        angular (rad/s^2) accelerations from `driver.get_cartesian_accelerations()` have their own thresholds.
        `jerks_x_value` is the highest absolute acceleration seen in x during the episode."""

        joint_accelerations = list(driver.get_all_accelerations())

        if len(joint_accelerations) != len(joint_acceleration_thresholds):
            raise ValueError(f"Expected {len(joint_accelerations)} joint_acceleration_thresholds, got {len(joint_acceleration_thresholds)}")

        cartesian = np.array(driver.get_cartesian_accelerations(), dtype=float)
        values = [abs(a) for a in joint_accelerations] + [float(np.linalg.norm(cartesian[:3])), float(np.linalg.norm(cartesian[3:]))]
        thresholds = list(joint_acceleration_thresholds) + [linear_acceleration_threshold, angular_acceleration_threshold]
        names = [f"joint_{i}" for i in range(len(joint_accelerations))] + ["linear", "angular"]

        if self.jerks_episode != self._episode:
            self.jerks_episode = self._episode
            self.jerks_trials += 1
            self.jerks_detected_value = False
            self.jerks_peaks = [0.0] * len(values)

        self.jerks_peaks = [max(peak, value) for peak, value in zip(self.jerks_peaks, values)]

        if not self.jerks_detected_value and any(v > t for v, t in zip(values, thresholds)):
            self.jerks_detected_value = True
            self.jerks_count += 1

        result = {
            "jerks_detected": self.jerks_detected_value,
            "jerks_rate": round(self.jerks_count / self.jerks_trials, precision),
        }
        for name, threshold, peak in zip(names, thresholds, self.jerks_peaks):
            result[f"jerks_{name}_threshold"] = threshold
            result[f"jerks_{name}_value"] = round(peak, precision)

        return result


    @metric
    def reset_failure_rate(self, driver, sleep_positions: list[float], tolerance: float = 0.1, precision: int = 3) -> dict[str, float]:
        """Checks whether the arm returned to the sleep pose at the end of the episode.

        `sleep_positions` are the joint positions (same order and units as `driver.get_all_positions()`) of the
        sleep pose. The reset fails if any joint is further than `tolerance` from it."""

        positions = list(driver.get_all_positions())
        
        if len(positions) != len(sleep_positions):
            raise ValueError(f"Expected {len(positions)} joint positions, got {len(sleep_positions)} in sleep_positions")

        joint_errors = [abs(p - s) for p, s in zip(positions, sleep_positions)]
        failed = max(joint_errors) > tolerance

        self.reset_trials += 1
        self.reset_failure_count += failed

        return {
            "reset_failed": failed,
            "reset_max_joint_error": round(max(joint_errors), precision),
            "reset_joint_errors": [round(e, precision) for e in joint_errors],
            "reset_failure_rate": round(self.reset_failure_count / self.reset_trials, precision),
        }
