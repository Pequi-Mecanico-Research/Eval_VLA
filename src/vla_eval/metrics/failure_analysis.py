from __future__ import annotations

from enum import Enum
from pathlib import Path

from ..base import EvalVLA, metric
from ..config import SetupConfig


class FailureType(str, Enum):
    PERCEPTION = "perception"
    LANGUAGE_UNDERSTANDING = "language_understanding"
    PLANNING = "planning"
    GRASPING = "grasping"
    CONTROL = "control"
    RECOVERY = "recovery"
    HARDWARE = "hardware"


class FailureAnalysis(EvalVLA):
    """Failure analysis (R).

    Perception errors: the policy misidentifies or fails to locate the target object.

    Language-understanding errors: the executed behavior does not match the instruction semantics.

    Planning errors: the policy selects an incorrect or suboptimal action sequence.

    Grasping errors: the end-effector fails to acquire or maintain a stable grasp.

    Control errors: the low-level controller produces jerky, oscillatory, or divergent motions.

    Recovery failures: the policy fails to recover after an intermediate error.

    Hardware or calibration failures: the trial fails due to mechanical, sensor, or calibration issues unrelated to the policy.
    """

    def __init__(self, config: SetupConfig, output_dir: str | Path = "eval_results"):
        super().__init__(config, output_dir)
        self.failure_counts: dict[FailureType, int] = {t: 0 for t in FailureType}
        self.episodes_evaluated: int = 0


    @metric
    def perception_errors(self) -> dict[str, float]:
        pass


    @metric
    def language_understanding_errors(self) -> dict[str, float]:
        pass


    @metric
    def planning_errors(self) -> dict[str, float]:
        pass


    @metric
    def grasping_errors(self) -> dict[str, float]:
        pass


    @metric
    def control_errors(self) -> dict[str, float]:
        pass


    @metric
    def recovery_failures(self) -> dict[str, float]:
        pass


    @metric
    def hardware_failures(self) -> dict[str, float]:
        pass


    @metric
    def failure_distribution(self) -> dict[str, float]:
        """Fraction of failed trials attributed to each FailureType."""
        pass
