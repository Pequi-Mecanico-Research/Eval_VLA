from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pinocchio as pin
import trossen_arm

from ..base import EvalVLA, metric
from ..config import SetupConfig
from ..plots import plot_operation
from ..success_detection.open_vocab_detection import ObjectDetection
from ..tasks import CollisionObserver, end_effector_inertia, friction_efforts, read_friction

HUMAN_PROMPTS = ["hand", "human arm"]

URDF_PATH = Path(__file__).parent / "models" / "wxai.urdf"

COLLISION_RESIDUAL_THRESHOLDS = [3.0] * 6             # Nm per arm joint, used without a calibration file
COLLISION_K = 5.0                                     # standard deviations of the calibrated threshold
COLLISION_MIN_THRESHOLD = 0.5                         # Nm, floor of the calibrated threshold
COLLISION_MIN_SAMPLES = 5                             # consecutive samples of a same joint that validate a collision
COLLISION_WARMUP_S = 0.5                              # s ignored at the start of the episode, while the residual settles
GMO_GAIN = 20.0                                       # 1/s, observer gain K_O (residual time constant 1/K_O)
JOINT_JERK_THRESHOLDS = [500.0] * 6 + [50.0]          # rad/s^3 per arm joint, m/s^3 for the gripper
JERK_MIN_OCCURRENCES = 3                              # threshold crossings of a same joint that flag the episode
JERK_MEDIAN_WINDOW = 3                                # samples, removes single-sample spikes
JERK_SMOOTHING_WINDOW = 5                             # samples, moving average applied after the median
SLEEP_POSITIONS = [0.0] * 7                         


class Operation(CollisionObserver, EvalVLA):
    """Operational metrics.

    Human intervention rate (fraction of trials requiring manual correction during execution).

    Collision detection (whether unintended contact occurs during the episode).

    Jerk detection (whether a joint repeatedly shows substantial jerk during the episode).

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
        self.gmo_residual: np.ndarray = np.zeros(self.gmo_model.nv)
        self.gmo_previous: tuple[np.ndarray, np.ndarray] | None = None    # momentum and tau + beta of the last sample
        self.gmo_friction: dict[str, np.ndarray] | None = None
        self.collision_detected_value: bool = False
        self.collision_episode: int | None = None
        self.collision_mode: str | None = None
        self.collision_calibration: dict | None = None
        self.collision_start: float | None = None                         # timestamps of the first and last samples
        self.collision_time: float | None = None
        self.collision_values: np.ndarray = np.zeros(self.gmo_model.nv)
        self.collision_consecutive: np.ndarray = np.zeros(self.gmo_model.nv, dtype=int)
        self.collision_streaks: np.ndarray = np.zeros(self.gmo_model.nv, dtype=int)
        self.collision_peaks: np.ndarray = np.zeros(self.gmo_model.nv)
        self.collision_samples: list[np.ndarray] = []                     # signal of the episode, for calibration

        self.reset_failure_count: int = 0
        self.reset_trials: int = 0

        self.jerks_detected_value: bool = False
        self.jerks_episode: int | None = None
        self.jerks_values: list[float] = [0.0] * len(JOINT_JERK_THRESHOLDS)
        self.jerks_exceedances: np.ndarray | None = None
        self.jerks_above: np.ndarray | None = None
        self.jerks_raw: deque | None = None                    # last accelerations, input of the median
        self.jerks_medians: deque | None = None                # last medians, input of the moving average
        self.jerks_smoothed: np.ndarray | None = None          # last filtered acceleration
        self.jerks_time: float | None = None                   # timestamp of the last sample


    def _save_graphs(self) -> None:
        plot_operation(self.csv_path, self.output_dir, self.config.experiment_name)


    @metric
    def human_intervention_rate(self, video_path: str | Path, model: str = "sam3", prompt: list[str] = HUMAN_PROMPTS, precision: int = 3, weights: str | Path | None = None, imgsz: int | None = None, conf: float | None = None, **detection_kwargs) -> dict[str, float]:
        """Detects whether a human entered the scene during the episode, i.e. needed to intervene manually.

        Call once per episode, on the recorded video.

        Args:
            video_path: Video of the episode.
            model: Detection model used to find the human in the video: "sam2", "sam3" or "yoloworld".
            prompt: Descriptions of what counts as a human intervention (e.g. "hand", "human arm").
            precision: Decimal places used to round the returned values.
            weights: Path to the weights file (e.g. `sam3.pt`) of `model`. If None, looks for
                `model_weights/<model>.pt` inside the package, which is not shipped with the pip install;
                download SAM weights first (see the root README); "yoloworld" weights are downloaded to this path
                if missing. Only read the first time the detector is created.
            imgsz: Inference size of `model`. If None, uses 640 for "sam2" and "yoloworld" and 644 for "sam3"
                (multiples of their strides). Only read the first time the detector is created.
            conf: Minimum confidence of a detection. If None, uses 0.70 for the SAM models and 0.5 for
                "yoloworld". Only read the first time the detector is created.
            **detection_kwargs: Extra arguments passed to `ObjectDetection.run`.

        Returns:
            dict with `human_interventions` (number of time intervals with a human), `human_intervention_sec`
            (total time with a human), `human_intervention_intervals` (list of (start, end) in seconds) and
            `human_intervention_rate` (fraction of the `episodes` with at least one intervention so far).
        """

        self.human_detector = self.human_detector or ObjectDetection("video", model=model, weights=weights, imgsz=imgsz, conf=conf)
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
                       gain: float = GMO_GAIN, precision: int = 3, calibration: str | Path | None = None, k: float = COLLISION_K,
                       min_samples: int = COLLISION_MIN_SAMPLES) -> dict:
        """Detects unintended contact (collision) during the episode from the arm joint torques.

        Call at every control step. A collision is validated when the signal of a same joint stays above
        its threshold for `min_samples` consecutive calls. The first `COLLISION_WARMUP_S` seconds of each
        episode and calls with a repeated timestamp are ignored.

        Modes (`mode`):
            - "gmo": generalized momentum observer. The residual estimates the external torque using the
              arm dynamic model (URDF, with the end effector and friction configured in the driver) and the
              measured positions, velocities, accelerations and efforts.
            - "driver": uses the external efforts already reported by the arm driver.

        Args:
            driver: Connected Trossen arm driver, read at each call.
            mode: "gmo" or "driver".
            thresholds: Maximum accepted absolute value per arm joint, in Nm, without `calibration`.
            gain: Observer gain K_O of the "gmo" mode, in 1/s. Higher values react faster but add noise.
            precision: Decimal places used to round the returned values.
            calibration: JSON written by `save_collision_calibration`. The mean of each joint is subtracted
                from the signal and its threshold becomes `max(k * std, COLLISION_MIN_THRESHOLD)`.
                Read at the first call of each episode.
            k: Number of standard deviations of the calibrated threshold.
            min_samples: Consecutive samples above the threshold that validate a collision.

        Returns:
            dict with `collision_mode`, `collision_detected` (this episode), `collision_values` (current signal
            per joint, without the calibrated mean), `collision_thresholds`, `collision_streaks` (longest
            sequence above the threshold per joint in this episode), `collision_min_samples` and
            `collision_peaks` (largest absolute value per joint, for logging only).

        Raises:
            ValueError: If `mode` is not "gmo" or "driver", or the calibration belongs to another mode.
            FileNotFoundError: If `calibration` does not exist.
        """
        if mode not in ("gmo", "driver"):
            raise ValueError(f"mode must be 'gmo' or 'driver', got {mode!r}")

        output = driver.get_robot_output()
        time_s = output.header.timestamp * 1e-6  # µs since configuration

        if self.collision_episode != self._episode or self.collision_mode != mode:
            self._start_collision_episode(driver, output, mode, calibration)

        if self.collision_calibration is None:
            mean, limits = np.zeros(len(thresholds)), np.array(thresholds, dtype=float)
        else:
            mean = np.array(self.collision_calibration["mean"])
            limits = np.maximum(k * np.array(self.collision_calibration["std"]), COLLISION_MIN_THRESHOLD)

        if self.collision_time is None or time_s > self.collision_time:
            if mode == "gmo":
                signal = self._update_gmo(output, time_s, gain)
            else:
                signal = np.array(output.joint.arm.external_efforts, dtype=float)

            if self.collision_start is None:
                self.collision_start = time_s
            self.collision_time = time_s
            self.collision_values = signal - mean

            if time_s - self.collision_start >= COLLISION_WARMUP_S:
                above = np.abs(self.collision_values) > limits
                self.collision_consecutive = np.where(above, self.collision_consecutive + 1, 0)
                self.collision_streaks = np.maximum(self.collision_streaks, self.collision_consecutive)
                self.collision_peaks = np.maximum(self.collision_peaks, np.abs(self.collision_values))
                self.collision_samples.append(signal)

        self.collision_detected_value = self.collision_detected_value or bool((self.collision_streaks >= min_samples).any())

        return {
            "collision_mode": mode,
            "collision_detected": self.collision_detected_value,
            "collision_values": [round(float(x), precision) for x in self.collision_values],
            "collision_thresholds": [round(float(x), precision) for x in limits],
            "collision_streaks": self.collision_streaks.tolist(),
            "collision_min_samples": min_samples,
            "collision_peaks": [round(float(x), precision) for x in self.collision_peaks],
        }


    @metric
    def jerks_rate(self, driver: trossen_arm.TrossenArmDriver, joint_jerk_thresholds: list[float] = JOINT_JERK_THRESHOLDS,
                   min_occurrences: int = JERK_MIN_OCCURRENCES, median_window: int = JERK_MEDIAN_WINDOW,
                   smoothing_window: int = JERK_SMOOTHING_WINDOW, precision: int = 3) -> dict:
        """Detects jerky motion (abrupt change of acceleration) of the joints during the episode.

        Call at every control step. The joint accelerations go through a causal median filter (removes
        single-sample spikes) and a moving average (reduces noise), and the jerk is the gradient of the
        filtered signal. Each time the absolute jerk of a joint goes above its threshold counts as one
        occurrence (it has to come back down to count again). The episode is flagged when a same joint
        reaches `min_occurrences`.

        The windows are in samples: the first `median_window + smoothing_window` calls of an episode
        (minus one) return zero jerk, and the filters delay the signal by about
        `(median_window + smoothing_window) / 2 - 1` samples. Calls with a repeated timestamp are ignored.

        Args:
            driver: Connected Trossen arm driver, read at each call.
            joint_jerk_thresholds: Maximum accepted jerk per joint, in rad/s^3 (m/s^3 for the gripper).
            min_occurrences: Occurrences of a same joint needed to flag the episode.
            median_window: Samples of the median filter. Read at the first call of each episode.
            smoothing_window: Samples of the moving average. Read at the first call of each episode.
            precision: Decimal places used to round the returned values.

        Returns:
            dict with `jerks_detected` (this episode), `jerks_values` (current jerk per joint, signed),
            `jerks_thresholds`, `jerks_exceedances` (occurrences per joint in this episode) and
            `jerks_min_occurrences`.
        """

        output = driver.get_robot_output()
        accelerations = np.array(output.joint.all.accelerations, dtype=float)
        time_s = output.header.timestamp * 1e-6  # µs since configuration

        if self.jerks_episode != self._episode:
            self.jerks_episode = self._episode
            self.jerks_detected_value = False
            self.jerks_values = [0.0] * len(accelerations)
            self.jerks_exceedances = np.zeros(len(accelerations), dtype=int)
            self.jerks_above = np.zeros(len(accelerations), dtype=bool)
            self.jerks_raw = deque(maxlen=median_window)
            self.jerks_medians = deque(maxlen=smoothing_window)
            self.jerks_smoothed = None
            self.jerks_time = None

        if self.jerks_time is None or time_s > self.jerks_time:  # the gradient divides by dt
            previous_time, self.jerks_time = self.jerks_time, time_s
            self.jerks_raw.append(accelerations)

            if len(self.jerks_raw) == self.jerks_raw.maxlen:
                self.jerks_medians.append(np.median(self.jerks_raw, axis=0))

            if len(self.jerks_medians) == self.jerks_medians.maxlen:
                previous, self.jerks_smoothed = self.jerks_smoothed, np.mean(self.jerks_medians, axis=0)
                
                if previous is not None:
                    jerk = np.gradient(np.stack([previous, self.jerks_smoothed]), [previous_time, time_s], axis=0)[-1]
                    above = np.abs(jerk) > np.array(joint_jerk_thresholds)
                    self.jerks_exceedances += above & ~self.jerks_above
                    self.jerks_above = above
                    self.jerks_values = jerk.tolist()

        self.jerks_detected_value = self.jerks_detected_value or bool((self.jerks_exceedances >= min_occurrences).any())

        return {
            "jerks_detected": self.jerks_detected_value,
            "jerks_values": [round(value, precision) for value in self.jerks_values],
            "jerks_thresholds": list(joint_jerk_thresholds),
            "jerks_exceedances": self.jerks_exceedances.tolist(),
            "jerks_min_occurrences": min_occurrences,
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
