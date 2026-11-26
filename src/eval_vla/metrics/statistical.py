from __future__ import annotations

import math
import numpy as np
import time
from statistics import NormalDist
from pathlib import Path

from ..base import EvalVLA, metric
from ..config import SetupConfig
from ..plots import plot_statistical
from ..success_detection.background_subtraction import count_objects_in_rois, debug_pipeline
from ..success_detection.open_vocab_detection import ObjectDetection


def _round(value: float, precision: int) -> float:
    """Converts numpy scalars to float and rounds them for the report."""
    return round(float(value), precision)


class Statistical(EvalVLA):
    """Statistical reporting.

    Success rate with 95% binomial confidence intervals.

    Task completion time (mean and standard deviation, in seconds).

    Inference latency on the deployed hardware (mean and p95, in milliseconds).
"""
    def __init__(self, config: SetupConfig, output_dir: str | Path = "eval_results",
                 run_dir: str | Path | None = None, name: str | None = None):
        super().__init__(config, output_dir, run_dir, name)
        self.detector: ObjectDetection | None = None
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


    def _save_graphs(self) -> None:
        plot_statistical(self.csv_path, self.output_dir, self.config.experiment_name)


    @metric
    def success_rate(self, task_name: str, task_mode: str, before_frame: np.ndarray, after_frame: np.ndarray, prompt: list[str] = None, reference_frame: np.ndarray = None, rois = None, model: str = "sam3", debug_plots: bool = True, weights: str | Path | None = None, imgsz: int | None = None, conf: float | None = None) -> dict[str, float]:
        """Checks whether the policy moved the objects from the table into the container.

        Modes (`task_mode`):
            - "model_segmentation": counts the objects in `prompt` inside the ROIs with `model`,
              in `before_frame` and `after_frame`. Needs `prompt` and `rois`.
            - "image_diff": counts the objects by comparing each frame with `reference_frame`
              (empty table). Needs `reference_frame` and `rois`.

        Args:
            task_name: Task being evaluated (currently only "Organize_the_table").
            task_mode: "model_segmentation" or "image_diff".
            before_frame: Table image right before the policy acts.
            after_frame: Table image right after the policy acts.
            prompt: Names/descriptions of the objects to segment.
            reference_frame: Empty-table image used as baseline in "image_diff".
            rois: Regions of interest where the objects start (table) and end (container).
            model: Detection model used in "model_segmentation": "sam3" or "yoloworld".
            debug_plots: Saves the debug images of "image_diff".
            weights: Path to the weights file (e.g. `sam3.pt`) of `model`, used in "model_segmentation".
                If None, looks for `model_weights/<model>.pt` inside the package, which is not shipped
                with the pip install; download it first (see the root README). Only read the first time
                the detector is created. "yoloworld" weights are downloaded to this path if missing.
            imgsz: Inference size of `model`, used in "model_segmentation". If None, uses 644 for "sam3" and
                640 for "yoloworld" (multiples of their strides). Only read the first time the detector is created.
            conf: Minimum confidence of a detection, used in "model_segmentation". If None, uses 0.25 for "sam3"
                and 0.5 for "yoloworld". Only read the first time the detector is created.

        Returns:
            dict with `successes` (bool), `objects_on_table_before` and
            `objects_in_container_after` (int, or None for unknown tasks).
        """
        if task_name == "Organize_the_table":    
            if task_mode == "model_segmentation":
                self.detector = self.detector or ObjectDetection("image", model=model, weights=weights, imgsz=imgsz, conf=conf)
                before = self.detector.run(before_frame, prompt, rois)
                after = self.detector.run(after_frame, prompt, rois)
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
        """Measures how long the policy took to finish the episode and accumulates statistics across episodes.

        Call once per episode, after the policy stops acting.

        Args:
            start_time: Episode start, from `time.perf_counter()`, in seconds.
            end_time: Episode end, from `time.perf_counter()`, in seconds.
            precision: Decimal places used to round the returned values.

        Returns:
            dict with `task_completion_sec_list` (duration of this episode), and
            `task_completion_sec_mean`, `_median` and `_std` over all episodes so far, in seconds.
        """
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
        """Measures the policy inference time of one step and accumulates statistics over the episode.

        Call once per policy step, wrapping only the policy call.

        Args:
            inference_time_start: Time right before the policy call, from `time.perf_counter()`, in seconds.
            inference_time_end: Time right after the policy call, from `time.perf_counter()`, in seconds.
            steps_per_episode: Number of policy steps in one episode, used to restart the accumulated
                latencies for the next episode.
            precision: Decimal places used to round the returned values.

        Returns:
            dict with `inference_latency_sec_list` (latency of this step), and
            `inference_latency_sec_mean`, `_median`, `_std` and `_p95` of the accumulated latencies, in seconds.
        """
        if len(self.inference_latency_sec_list) >= steps_per_episode:
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
