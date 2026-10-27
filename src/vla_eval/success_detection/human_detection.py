from __future__ import annotations

from pathlib import Path

import cv2

from .model_segmentation import SAM_WEIGHTS

HUMAN_PROMPTS = ["hand", "human arm"]


def frames_to_intervals(signal: list[float], fps: float, limiar_on: float = 0.5, limiar_off: float = 0.3,
                        min_dur_s: float = 1, juntar_gap_s: float = 1.5) -> list[tuple[float, float]]:
    """Turns a per-frame confidence signal into (start_s, end_s) intervals.

    Hysteresis: an interval opens when the score reaches `limiar_on` and closes when it drops below `limiar_off`.
    Intervals separated by less than `juntar_gap_s` are merged, then those shorter than `min_dur_s` are discarded."""
    intervals: list[list[float]] = []
    start = None
    for i, score in enumerate(signal):
        if start is None and score >= limiar_on:
            start = i
        elif start is not None and score < limiar_off:
            intervals.append([start / fps, i / fps])
            start = None
    if start is not None:
        intervals.append([start / fps, len(signal) / fps])

    print("Intervalos detectados: ", intervals)

    merged: list[list[float]] = []
    for a, b in intervals:
        if b - a <= min_dur_s:
            pass
        elif merged and a - merged[-1][1] <= juntar_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    return [(a, b) for a, b in merged]


def human_presence_signal(video_path: str | Path, prompt: list[str] = HUMAN_PROMPTS, weights: str = SAM_WEIGHTS,
                          conf: float = 0.70, min_area: float = 400) -> tuple[list[float], float]:
    """Runs SAM3 over the video and returns (per-frame score, video fps).

    The score of a frame is the highest confidence among the boxes larger than `min_area` px²."""
    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    video_path = str(video_path)
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()

    predictor = SAM3VideoSemanticPredictor(overrides={
        "conf": conf, "task": "segment", "mode": "predict",
        "imgsz": 640, "model": weights, "quantize": 16, "save": True, "save_dir": "eval_results/sam3_video_detections"
    })

    signal = []
    for r in predictor(source=video_path, text=prompt, stream=True):  # one Results per frame
        score = 0.0
        if r.boxes is not None and len(r.boxes):
            wh = r.boxes.xywh
            areas = wh[:, 2] * wh[:, 3]
            valid = r.boxes.conf[areas > min_area]
            if len(valid):
                score = float(valid.max())
        signal.append(score)
    return signal, fps


def detect_human_intervals(video_path: str | Path, prompt: list[str] = HUMAN_PROMPTS, min_area: float = 400) -> list[tuple[float, float]]:
    """(start_s, end_s) intervals in which a human hand/arm is visible in the episode video."""

    signal, fps = human_presence_signal(video_path, prompt, min_area=min_area)

    return frames_to_intervals(signal, fps)
