from __future__ import annotations

from pathlib import Path

import numpy as np

from ..base import EvalVLA, metric
from ..config import SetupConfig
from ..plots import plot_long_horizon
from ..success_detection.subtask_judge import ollama_subtask_times, robometer_subtask_times

TASK_SUBTASKS: dict[str, list[str]] = {
    "Organize_the_table": [
        "reach the black cube",
        "grasp the black cube",
        "move the cube above the tote",
        "release the cube inside the tote",
    ],
}


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
                 run_dir: str | Path | None = None, name: str | None = None,
                 subtasks: list[str] | None = None, judge: str = "ollama", n_frames: int = 30,
                 vlm_model: str = "qwen3.5:latest", vlm_host: str = "http://localhost:11434",
                 vlm_num_ctx: int = 16384, robometer_url: str = "http://localhost:8000",
                 robometer_threshold: float = 0.5, robometer_frame_steps: bool = False, timeout_s: float = 300.0):
        """`subtasks` defaults to `TASK_SUBTASKS[config.task_name]`. `judge` is "ollama" or "robometer"."""
        super().__init__(config, output_dir, run_dir, name)
        if judge not in ("ollama", "robometer"):
            raise ValueError(f"judge must be 'ollama' or 'robometer', got {judge!r}")
        self.subtasks: list[str] = subtasks if subtasks is not None else TASK_SUBTASKS.get(config.task_name, [])
        if not self.subtasks:
            raise ValueError(f"No subtasks for task {config.task_name!r}: add it to TASK_SUBTASKS or pass `subtasks`")

        self.judge = judge
        self.n_frames = n_frames
        self.vlm_model, self.vlm_host, self.vlm_num_ctx = vlm_model, vlm_host, vlm_num_ctx
        self.robometer_url, self.robometer_threshold = robometer_url, robometer_threshold
        self.robometer_frame_steps = robometer_frame_steps
        self.timeout_s = timeout_s

        self.subtasks_completed: list[bool] | None = None
        self.completion_times: list[float | None] = []
        self.completed_lengths: list[int] = []
        self.failure_positions: list[int | None] = []
        self.failure_times: list[float | None] = []
        self.subtasks_done = 0
        self.subtasks_total = 0


    def _save_graphs(self) -> None:
        plot_long_horizon(self.csv_path, self.output_dir, self.config.experiment_name)


    def _judge_video(self, video_path: str | Path) -> tuple[list[float | None], str]:
        if self.judge == "robometer":
            return robometer_subtask_times(video_path, self.subtasks, url=self.robometer_url, n_frames=self.n_frames,
                                           threshold=self.robometer_threshold, timeout_s=self.timeout_s,
                                           use_frame_steps=self.robometer_frame_steps)
        return ollama_subtask_times(video_path, self.config.task_name, self.subtasks, n_frames=self.n_frames,
                                    model=self.vlm_model, host=self.vlm_host, timeout_s=self.timeout_s,
                                    num_ctx=self.vlm_num_ctx)


    @metric
    def subtask_completion_rate(self, video_path: str | Path) -> dict:
        """Fraction of the task's subtasks completed, judged from the episode video.

        Call once per episode, before `average_completed_sequence_length` and `failure_position_distribution`,
        which reuse this analysis. The subtasks come from `TASK_SUBTASKS[config.task_name]` (or the `subtasks`
        constructor argument); the judge ("ollama" VLM or a Robometer server) gives the time each one was completed.

        Args:
            video_path: The episode video.

        Returns:
            dict with `subtask_completion_rate` (completed subtasks over all subtasks, accumulated across the
            episodes so far), `episode_subtask_completion_rate` (this episode only), `t_<subtask>` (seconds from
            the start of the video at which each subtask was completed, None if never), `judge` and
            `judge_raw_reply`.
        """
        times, reply = self._judge_video(video_path)
        self.completion_times = times
        self.subtasks_completed = [t is not None for t in times]
        self.subtasks_done += sum(self.subtasks_completed)
        self.subtasks_total += len(times)
        return {
            "subtask_completion_rate": self.subtasks_done / self.subtasks_total,
            "episode_subtask_completion_rate": sum(self.subtasks_completed) / len(times),
            **{f"t_{s}": None if t is None else round(t, 2) for s, t in zip(self.subtasks, times)},
            "judge": self.judge,
            "judge_raw_reply": reply.strip(),
        }


    @metric
    def average_completed_sequence_length(self, subtasks_completed: list[bool] | None = None) -> dict[str, float]:
        """Average number of consecutive subtasks completed from the start of the sequence (CALVIN-style).

        Call once per episode.

        Args:
            subtasks_completed: One bool per subtask, in execution order. Defaults to the result of the last
                `subtask_completion_rate` call.

        Returns:
            dict with `average_completed_sequence_length`: mean over the episodes so far of the leading
            run of True values.

        Raises:
            ValueError: if `subtasks_completed` is empty or no video was judged yet.
        """
        completed = _validated(self._completed(subtasks_completed))
        first = _first_failure(completed)
        self.completed_lengths.append(len(completed) if first is None else first)
        return {"average_completed_sequence_length": float(np.mean(self.completed_lengths))}


    @metric
    def failure_position_distribution(self) -> dict:
        """Where and when the episode first failed, from the last `subtask_completion_rate` analysis.

        The failure is the first subtask not completed. Its time is when the policy stopped making progress:
        the completion time of the previous subtask (0 if the first subtask failed). The step is that time
        times `config.control_frequency`. The histograms over episodes are drawn in `_save_graphs`.

        Returns:
            dict with `failure_position` (1-based index of the first failed subtask), `failure_subtask` (its
            name), `failure_time_s` and `failure_step`; all None if every subtask was completed.

        Raises:
            ValueError: if no video was judged yet.
        """
        first = _first_failure(self._completed(None))
        if first is None:
            position = subtask = time_s = step = None
        else:
            position, subtask = first + 1, self.subtasks[first]
            time_s = 0.0 if first == 0 else round(self.completion_times[first - 1] or 0.0, 2)
            step = round(time_s * self.config.control_frequency)
        self.failure_positions.append(position)
        self.failure_times.append(time_s)
        return {"failure_position": position, "failure_subtask": subtask, "failure_time_s": time_s, "failure_step": step}


    def _completed(self, subtasks_completed: list[bool] | None) -> list[bool]:
        if subtasks_completed is not None:
            return subtasks_completed
        if self.subtasks_completed is None:
            raise ValueError("Call subtask_completion_rate(video_path) first")
        return self.subtasks_completed
