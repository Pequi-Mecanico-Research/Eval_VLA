from __future__ import annotations

from pathlib import Path

from ..base import EvalVLA, metric
from ..config import SetupConfig


class LongHorizon(EvalVLA):
    """Long-horizon tasks (C).

    Subtask completion rate (fraction of individual subtasks completed, independent of full-sequence success).

    Average completed sequence length (e.g., CALVIN-style metric).

    Distribution of failure positions along the subtask sequence (e.g., histogram showing at which step failures concentrate).
    """

    def __init__(self, config: SetupConfig, output_dir: str | Path = "eval_results"):
        super().__init__(config, output_dir)
        self.completed_lengths: list[int] = []
        self.failure_positions: list[int] = []


    @metric
    def subtask_completion_rate(self, subtasks_completed: list[bool]) -> dict[str, float]:
        pass


    @metric
    def average_completed_sequence_length(self, subtasks_completed: list[bool]) -> dict[str, float]:
        pass


    @metric
    def failure_position_distribution(self, subtasks_completed: list[bool]) -> dict[str, float]:
        """Histogram of the subtask index (1-based) at which each trial first failed."""
        pass
