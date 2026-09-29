from __future__ import annotations

import math
import numpy as np

from ..base import EvalVLA, metric



class Statistical(EvalVLA):
    """Statistical reporting.

    Number of trials per task (minimum recommended: 50 for primary claims; see discussion below).

    Success rate with 95% binomial confidence intervals.

    Task completion time (mean and standard deviation, in seconds).

    Inference latency on the deployed hardware (mean and p95, in milliseconds).
"""
    @metric
    def trials_per_task(self, recomended: int = 50) -> dict[str, int]:
        pass

    @metric
    def success_rate(self, confidence: float = 0.95) -> dict[str, float]:
        pass

    @metric
    def task_completion_time(self, start_time: float, end_time: float, completed_succesfully: bool = True) -> dict[str, float]:
        pass

    @metric
    def inference_latency(self, inference_time_start: float, inference_time_end: float) -> dict[str, float]:
        pass