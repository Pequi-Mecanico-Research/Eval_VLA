from __future__ import annotations

import math
import numpy as np
import time
from statistics import NormalDist
from pathlib import Path

from ..base import EvalVLA, metric
from ..config import SetupConfig
from ..success_detection.background_subtraction import count_objects_in_rois, debug_pipeline
from ..success_detection.model_segmentation import load_sam_predictor, sam_inference_image


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
        self.task_completion_sec_list: list = []
        self.task_completion_sec_mean: float = None
        self.task_completion_sec_median: float = None
        self.task_completion_sec_std: float = None
        self.inference_latency_sec_list: list = []
        self.inference_latency_sec_mean: float = None
        self.inference_latency_sec_median: float = None
        self.inference_latency_sec_std: float = None
        self.inference_latency_sec_p95: float = None


    @metric
    def success_rate(self, task_name: str, task_mode: str, before_frame: np.ndarray, after_frame: np.ndarray, prompt: list[str] = None, reference_frame: np.ndarray = None, rois = None, debug_plots: bool = False) -> dict[str, float]:
        if task_name == "Organize_the_table":    
            if task_mode == "model_segmentation":
                predictor = load_sam_predictor()
                before = sam_inference_image(before_frame, prompt, rois, predictor)
                after = sam_inference_image(after_frame, prompt, rois, predictor)
                self.success_rate_value = 0 < before["table"] == after["container"]  

            elif task_mode == "image_diff":
                before = count_objects_in_rois(before_frame, reference_frame, rois)
                after = count_objects_in_rois(after_frame, reference_frame, rois)
                self.success_rate_value = 0 < before["table"] == after["container"] and after["table"] == 0
                if debug_plots:
                    for name, frame in (("before", before_frame), ("after", after_frame)):
                        debug_pipeline(frame, reference_frame, rois, save_path=self.output_dir / f"ep{self._episode}_{name}_debug.png")
        else:
            self.success_rate_value = False
            before = None
            after = None

        return {
            "successes": self.success_rate_value,
            "objects_on_table_before": before["table"] if before else None,
            "objects_in_container_after": after["container"] if after else None,
        }


    @metric
    def task_completion_time(self, start_time: float, end_time: float, precision: int = 3) -> dict[str, float]:
        self.task_completion_sec_list.append(end_time - start_time)
        self.task_completion_sec_mean = np.mean(self.task_completion_sec_list)
        self.task_completion_sec_median = np.median(self.task_completion_sec_list)
        self.task_completion_sec_std = np.std(self.task_completion_sec_list)

        return {
            "task_completion_sec_list": _round(self.task_completion_sec_list[-1], precision),
            "task_completion_sec_mean": _round(self.task_completion_sec_mean, precision),
            "task_completion_sec_median": _round(self.task_completion_sec_median, precision),
            "task_completion_sec_std": _round(self.task_completion_sec_std, precision),
        }


    @metric
    def inference_latency(self, inference_time_start: float, inference_time_end: float, steps_per_episode: int, precision: int = 3) -> dict[str, float]:
        if steps_per_episode > len(self.inference_latency_sec_list) - 1:
            self.inference_latency_sec_list = []

        self.inference_latency_sec_list.append(inference_time_end - inference_time_start)
        self.inference_latency_sec_mean = np.mean(self.inference_latency_sec_list)
        self.inference_latency_sec_median = np.median(self.inference_latency_sec_list)
        self.inference_latency_sec_std = np.std(self.inference_latency_sec_list)
        self.inference_latency_sec_p95 = np.percentile(self.inference_latency_sec_list, 95)
        
        return {
            "inference_latency_sec_list": _round(self.inference_latency_sec_list[-1], precision),
            "inference_latency_sec_mean": _round(self.inference_latency_sec_mean, precision),
            "inference_latency_sec_median": _round(self.inference_latency_sec_median, precision),
            "inference_latency_sec_std": _round(self.inference_latency_sec_std, precision),
            "inference_latency_sec_p95": _round(self.inference_latency_sec_p95, precision),
        }
