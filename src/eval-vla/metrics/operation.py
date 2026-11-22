from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
import trossen_arm

from ..base import EvalVLA, metric
from ..config import SetupConfig
from ..plots import plot_operation
from ..success_detection.sam_inference import ObjectDetection

HUMAN_PROMPTS = ["hand", "human arm"]

URDF_PATH = Path(__file__).parent / "models" / "wxai.urdf"

COLLISION_RESIDUAL_THRESHOLDS = [3.0] * 6             # Nm per arm joint, applied to the GMO residual
GMO_GAIN = 20.0                                       # 1/s, observer gain K_O (residual time constant 1/K_O)
JOINT_JERK_THRESHOLDS = [500.0] * 6 + [50.0]          # rad/s^3 per arm joint, m/s^3 for the gripper
LINEAR_JERK_THRESHOLD = 50.0                          # m/s^3 (end effector)
ANGULAR_JERK_THRESHOLD = 500.0                        # rad/s^3 (end effector)
SLEEP_POSITIONS = [0.0] * 7                         


class Operation(EvalVLA):
    """Operational metrics.

    Human intervention rate (fraction of trials requiring manual correction during execution).

    Collision rate (fraction of trials in which unintended contact occurs).

    Jerks rate (fraction of trials which show substântial jerk / measure jerk_rate)z.

    Reset failure rate (fraction of trials in which the reset procedure itself fails or introduces bias).
    """

    def __init__(self, config: SetupConfig, episodes: int, output_dir: str | Path = "eval_results",
                 run_dir: str | Path | None = None, name: str | None = None):
        super().__init__(config, output_dir, run_dir, name)
        self.episodes = episodes

        self.human_detector: ObjectDetection | None = None
        self.human_intervention_count: int = 0

        model = pin.buildModelFromUrdf(str(URDF_PATH))
        carriages = [model.getJointId(n) for n in ("left_carriage_joint", "right_carriage_joint")]
        self.gmo_model = pin.buildReducedModel(model, carriages, pin.neutral(model))
        self.gmo_data = self.gmo_model.createData()
        self.gmo_residual: np.ndarray | None = None
        self.gmo_integral: np.ndarray | None = None
        self.gmo_p0: np.ndarray | None = None
        self.gmo_time: float = 0.0
        self.collision_count: int = 0
        self.collision_detected_value: bool = False
        self.collision_episode: int | None = None
        self.collision_peaks: list[float] = []

        self.reset_failure_count: int = 0
        self.reset_trials: int = 0

        self.jerks_count: int = 0
        self.jerks_trials: int = 0
        self.jerks_detected_value: bool = False
        self.jerks_episode: int | None = None
        self.jerks_peaks: list[float] = []
        self.jerks_times: list[float] = []
        self.jerks_accelerations: list[list[float]] = []


    def _save_graphs(self) -> None:
        plot_operation(self.csv_path, self.output_dir, self.config.experiment_name)


    @metric
    def human_intervention_rate(self, video_path: str | Path, model: str = "sam3", prompt: list[str] = HUMAN_PROMPTS, precision: int = 3, **detection_kwargs) -> dict[str, float]:
        """Detects whether a human entered the scene during the episode, i.e. needed to intervene manually.

        Call once per episode, on the recorded video.

        Args:
            video_path: Video of the episode.
            model: Segmentation model used to find the human in the video.
            prompt: Descriptions of what counts as a human intervention (e.g. "hand", "human arm").
            precision: Decimal places used to round the returned values.
            **detection_kwargs: Extra arguments passed to `ObjectDetection.run`.

        Returns:
            dict with `human_interventions` (number of time intervals with a human), `human_intervention_sec`
            (total time with a human), `human_intervention_intervals` (list of (start, end) in seconds) and
            `human_intervention_rate` (fraction of the `episodes` with at least one intervention so far).
        """

        self.human_detector = self.human_detector or ObjectDetection("video", model=model)
        intervals = self.human_detector.run(video_path, prompt, **detection_kwargs)

        self.human_intervention_count += bool(intervals)

        return {
            "human_interventions": len(intervals),
            "human_intervention_sec": round(sum(b - a for a, b in intervals), precision),
            "human_intervention_intervals": [(round(a, precision), round(b, precision)) for a, b in intervals],
            "human_intervention_rate": round(self.human_intervention_count / self.episodes, precision),
        }


    @metric
    def collision_rate(self, driver: trossen_arm.TrossenArmDriver, mode: str = "gmo", thresholds: list[float] = COLLISION_RESIDUAL_THRESHOLDS,
                       gain: float = GMO_GAIN, precision: int = 3) -> dict:
        """Detects unintended contact (collision) during the episode from the arm joint torques.

        Call at every control step. It keeps the peak of the signal per joint during the episode and flags
        a collision when any peak exceeds its threshold.

        Modes (`mode`):
            - "gmo": generalized momentum observer. The residual estimates the external torque using the
              arm dynamic model (URDF) and the measured positions, velocities and efforts.
            - "driver": uses the external efforts already reported by the arm driver.

        Args:
            driver: Connected Trossen arm driver, read at each call.
            mode: "gmo" or "driver".
            thresholds: Maximum accepted absolute value per arm joint, in Nm.
            gain: Observer gain K_O of the "gmo" mode, in 1/s. Higher values react faster but add noise.
            precision: Decimal places used to round the returned values.

        Returns:
            dict with `collision_mode`, `collision_detected` (this episode), `collision_rate` (fraction of the
            `episodes` with a collision so far), `collision_thresholds` and `collision_values` (peak per joint).

        Raises:
            ValueError: If `mode` is not "gmo" or "driver".
        """

        output = driver.get_robot_output()

        if self.collision_episode != self._episode:
            self.collision_episode = self._episode
            self.collision_detected_value = False
            self.collision_peaks = [0.0] * len(thresholds)
            self.gmo_p0 = None

        if mode == "gmo":
            model, data = self.gmo_model, self.gmo_data
            q, v, tau = (np.array(x) for x in (output.joint.arm.positions, output.joint.arm.velocities, output.joint.arm.efforts))
            time_s = output.header.timestamp * 1e-6

            M = pin.crba(model, data, q)
            M = np.triu(M) + np.triu(M, 1).T
            p = M @ v

            if self.gmo_p0 is None:
                self.gmo_residual, self.gmo_integral, self.gmo_p0, self.gmo_time = np.zeros_like(p), np.zeros_like(p), p, time_s

            dt = time_s - self.gmo_time
            if dt > 0:
                beta = pin.computeCoriolisMatrix(model, data, q, v).T @ v - pin.computeGeneralizedGravity(model, data, q)
                self.gmo_integral += (tau + beta + self.gmo_residual) * dt
                self.gmo_residual = gain * (p - self.gmo_p0 - self.gmo_integral)
                self.gmo_time = time_s

            signal = self.gmo_residual

        elif mode == "driver":
            signal = np.array(output.joint.arm.external_efforts)
        else:
            raise ValueError(f"mode must be 'gmo' or 'driver', got {mode!r}")

        self.collision_peaks = [max(peak, abs(float(x))) for peak, x in zip(self.collision_peaks, signal)]

        if not self.collision_detected_value and any(peak > t for peak, t in zip(self.collision_peaks, thresholds)):
            self.collision_detected_value = True
            self.collision_count += 1

        return {
            "collision_mode": mode,
            "collision_detected": self.collision_detected_value,
            "collision_rate": round(self.collision_count / self.episodes, precision),
            "collision_thresholds": list(thresholds),
            "collision_values": [round(peak, precision) for peak in self.collision_peaks],
        }


    @metric
    def jerks_rate(self, driver: trossen_arm.TrossenArmDriver, joint_jerk_thresholds: list[float] = JOINT_JERK_THRESHOLDS,
                   linear_jerk_threshold: float = LINEAR_JERK_THRESHOLD,
                   angular_jerk_threshold: float = ANGULAR_JERK_THRESHOLD, precision: int = 3) -> dict:
        """Detects jerky motion (abrupt change of acceleration) during the episode.

        Call at every control step. The jerk is the derivative of the accelerations along the episode;
        it keeps the peak per joint and for the end effector, and flags the episode when any peak
        exceeds its threshold.

        Args:
            driver: Connected Trossen arm driver, read at each call.
            joint_jerk_thresholds: Maximum accepted jerk per joint, in rad/s^3 (m/s^3 for the gripper).
            linear_jerk_threshold: Maximum accepted end-effector linear jerk, in m/s^3.
            angular_jerk_threshold: Maximum accepted end-effector angular jerk, in rad/s^3.
            precision: Decimal places used to round the returned values.

        Returns:
            dict with `jerks_detected` (this episode), `jerks_rate` (fraction of the episodes with jerks so far),
            `jerks_thresholds` and `jerks_values` (peaks per joint, then linear and angular).
        """

        output = driver.get_robot_output()
        joint_accelerations = list(output.joint.all.accelerations)
        n_joints = len(joint_accelerations)

        thresholds = list(joint_jerk_thresholds) + [linear_jerk_threshold, angular_jerk_threshold]

        if self.jerks_episode != self._episode:
            self.jerks_episode = self._episode
            self.jerks_trials += 1
            self.jerks_detected_value = False
            self.jerks_peaks = [0.0] * len(thresholds)
            self.jerks_times = []
            self.jerks_accelerations = []

        time_s = output.header.timestamp * 1e-6  # µs since configuration

        if not self.jerks_times or time_s > self.jerks_times[-1]:  # the gradient divides by dt
            self.jerks_times.append(time_s)
            self.jerks_accelerations.append(joint_accelerations + list(output.cartesian.accelerations))

        if len(self.jerks_times) >= 2:
            jerk = np.gradient(np.array(self.jerks_accelerations), np.array(self.jerks_times), axis=0)
            values = list(np.abs(jerk[:, :n_joints]).max(axis=0))
            values.append(np.linalg.norm(jerk[:, n_joints:n_joints + 3], axis=1).max())
            values.append(np.linalg.norm(jerk[:, n_joints + 3:], axis=1).max())
            self.jerks_peaks = [max(peak, float(value)) for peak, value in zip(self.jerks_peaks, values)]

        if not self.jerks_detected_value and any(p > t for p, t in zip(self.jerks_peaks, thresholds)):
            self.jerks_detected_value = True
            self.jerks_count += 1

        return {
            "jerks_detected": self.jerks_detected_value,
            "jerks_rate": round(self.jerks_count / self.jerks_trials, precision),
            "jerks_thresholds": thresholds,
            "jerks_values": [round(peak, precision) for peak in self.jerks_peaks],
        }


    @metric
    def reset_failure_rate(self, driver: trossen_arm.TrossenArmDriver, sleep_positions: list[float] = SLEEP_POSITIONS, tolerance: float = 0.1, precision: int = 3) -> dict[str, float]:
        """Checks whether the arm is back at the reset (sleep) pose, i.e. whether the reset procedure worked.

        Call once per episode, when the arm should be at the reset pose.

        Args:
            driver: Connected Trossen arm driver, read at the call.
            sleep_positions: Expected joint positions of the reset pose, in the driver units.
            tolerance: Maximum accepted absolute error per joint; above it the reset counts as failed.
            precision: Decimal places used to round the returned values.

        Returns:
            dict with `reset_failed` (this episode), `reset_max_joint_error`, `reset_joint_errors` (per joint)
            and `reset_failure_rate` (fraction of episodes with a failed reset so far).
        """
        positions = list(driver.get_all_positions())

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
