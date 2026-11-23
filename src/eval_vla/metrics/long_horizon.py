from __future__ import annotations

from pathlib import Path

import numpy as np

from ..base import EvalVLA, metric
from ..config import SetupConfig
from ..plots import plot_long_horizon


def _validated(subtasks_completed: list[bool]) -> list[bool]:
    if not len(subtasks_completed):
        raise ValueError("subtasks_completed must not be empty")
    return [bool(c) for c in subtasks_completed]


def _first_failure(completed: list[bool]) -> int | None:
    return next((i for i, ok in enumerate(completed) if not ok), None)


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
        self.failure_positions: list[int | None] = []
        self.subtasks_done = 0
        self.subtasks_total = 0


    def _save_graphs(self) -> None:
        plot_long_horizon(self.csv_path, self.output_dir, self.config.experiment_name)


    @metric
    def subtask_completion_rate(self, subtasks_completed: list[bool]) -> dict[str, float]:
        """Fraction of the individual subtasks completed, independent of the full sequence succeeding.

        Call once per episode.

        Args:
            subtasks_completed: One bool per subtask, in execution order; True if that subtask was completed.

        Returns:
            dict with `subtask_completion_rate`: completed subtasks over all subtasks, accumulated
            across the episodes so far.

        Raises:
            ValueError: if `subtasks_completed` is empty.
        """
        completed = _validated(subtasks_completed)
        self.subtasks_done += sum(completed)
        self.subtasks_total += len(completed)
        return {"subtask_completion_rate": self.subtasks_done / self.subtasks_total}


    @metric
    def average_completed_sequence_length(self, subtasks_completed: list[bool]) -> dict[str, float]:
        """Average number of consecutive subtasks completed from the start of the sequence (CALVIN-style).

        Call once per episode.

        Args:
            subtasks_completed: One bool per subtask, in execution order; True if that subtask was completed.

        Returns:
            dict with `average_completed_sequence_length`: mean over the episodes so far of the leading
            run of True values.

        Raises:
            ValueError: if `subtasks_completed` is empty.
        """
        completed = _validated(subtasks_completed)
        first = _first_failure(completed)
        self.completed_lengths.append(len(completed) if first is None else first)
        return {"average_completed_sequence_length": float(np.mean(self.completed_lengths))}


    @metric
    def failure_position_distribution(self, subtasks_completed: list[bool]) -> dict[str, float]:
        """Histogram of the subtask index (1-based) at which each trial first failed.

        Call once per episode.

        Args:
            subtasks_completed: One bool per subtask, in execution order; True if that subtask was completed.

        Returns:
            dict with `failure_position`: 1-based index of the first failed subtask of this episode,
            or None if all were completed. Only the first failure counts. All positions so far are in
            `self.failure_positions`.

        Raises:
            ValueError: if `subtasks_completed` is empty.
        """
        first = _first_failure(_validated(subtasks_completed))
        position = None if first is None else first + 1
        self.failure_positions.append(position)
        return {"failure_position": position}
