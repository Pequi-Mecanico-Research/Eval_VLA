from __future__ import annotations

from pathlib import Path

from ..base import EvalVLA, metric
from ..config import SetupConfig
from ..plots import plot_long_horizon


class LongHorizon(EvalVLA):
    """Long-horizon tasks (C).

    Subtask completion rate (fraction of individual subtasks completed, independent of full-sequence success).

    Average completed sequence length (e.g., CALVIN-style metric).

    Distribution of failure positions along the subtask sequence (e.g., histogram showing at which step failures concentrate).
    """

    def __init__(self, config: SetupConfig, output_dir: str | Path = "eval_results",
                 run_dir: str | Path | None = None, name: str | None = None):
        super().__init__(config, output_dir, run_dir, name)
        self.completed_lengths: list[int] = []
        self.failure_positions: list[int] = []


    def _save_graphs(self) -> None:
        plot_long_horizon(self.csv_path, self.output_dir, self.config.experiment_name)


    @metric
    def subtask_completion_rate(self, subtasks_completed: list[bool]) -> dict[str, float]:
        """Fraction of the individual subtasks completed, independent of the full sequence succeeding.

        Call once per episode.

        Args:
            subtasks_completed: One bool per subtask, in execution order; True if that subtask was completed.

        Returns:
            dict with the subtask completion rate. Not implemented yet.
        """
        pass


    @metric
    def average_completed_sequence_length(self, subtasks_completed: list[bool]) -> dict[str, float]:
        """Average number of consecutive subtasks completed from the start of the sequence (CALVIN-style).

        Call once per episode.

        Args:
            subtasks_completed: One bool per subtask, in execution order; True if that subtask was completed.

        Returns:
            dict with the average completed sequence length. Not implemented yet.
        """
        pass


    @metric
    def failure_position_distribution(self, subtasks_completed: list[bool]) -> dict[str, float]:
        """Histogram of the subtask index (1-based) at which each trial first failed.

        Call once per episode.

        Args:
            subtasks_completed: One bool per subtask, in execution order; True if that subtask was completed.

        Returns:
            dict with the distribution of failure positions. Not implemented yet.
        """
        pass
