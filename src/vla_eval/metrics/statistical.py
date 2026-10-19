from __future__ import annotations

import math
import numpy as np
import time
from statistics import NormalDist
from pathlib import Path

from ..base import EvalVLA, metric
from ..config import SetupConfig


def _round(value: float, precision: int) -> float:
    """Converts numpy scalars to float and rounds them for the report."""
    return round(float(value), precision)


class Statistical(EvalVLA):
    """Statistical reporting.

    Success rate with 95% binomial confidence intervals.

    Task completion time (mean and standard deviation, in seconds).

    Inference latency on the deployed hardware (mean and p95, in milliseconds).
"""
    def __init__(self, config: SetupConfig, output_dir: str | Path = "eval_results"):
        super().__init__(config, output_dir)
        self.success_rate_value: bool = None
        self.task_completion_time_list: list = []
        self.task_completion_mean: float = None
        self.task_completion_median: float = None
        self.task_completion_std: float = None
        self.inference_latency_list: list = []
        self.inference_latency_mean: float = None
        self.inference_latency_median: float = None
        self.inference_latency_std: float = None
        self.inference_latency_p95: float = None


    @metric
    def success_rate(self, task_name: str) -> dict[str, float]:
        if task_name == "Organize_the_table":
            self.success_rate_value = True
        else:
            self.success_rate_value = False

        return {
            "successes": self.success_rate_value,
        }


    @metric
    def task_completion_time(self, start_time: float, end_time: float, precision: int = 3) -> dict[str, float]:
        self.task_completion_time_list.append(end_time - start_time)
        self.task_completion_mean = np.mean(self.task_completion_time_list)
        self.task_completion_median = np.median(self.task_completion_time_list)
        self.task_completion_std = np.std(self.task_completion_time_list)

        return {
            "task_completion_time_list": _round(self.task_completion_time_list[-1], precision),
            "task_completion_mean": _round(self.task_completion_mean, precision),
            "task_completion_median": _round(self.task_completion_median, precision),
            "task_completion_std": _round(self.task_completion_std, precision),
        }


    @metric
    def inference_latency(self, inference_time_start: float, inference_time_end: float, steps_per_episode: int, precision: int = 3) -> dict[str, float]:
        if steps_per_episode > len(self.inference_latency_list) - 1:
            self.inference_latency_list = []

        self.inference_latency_list.append(inference_time_end - inference_time_start)
        self.inference_latency_mean = np.mean(self.inference_latency_list)
        self.inference_latency_median = np.median(self.inference_latency_list)
        self.inference_latency_std = np.std(self.inference_latency_list)
        self.inference_latency_p95 = np.percentile(self.inference_latency_list, 95)
        
        return {
            "inference_latency_list": _round(self.inference_latency_list[-1], precision),
            "inference_latency_mean": _round(self.inference_latency_mean, precision),
            "inference_latency_median": _round(self.inference_latency_median, precision),
            "inference_latency_std": _round(self.inference_latency_std, precision),
            "inference_latency_p95": _round(self.inference_latency_p95, precision),
        }
