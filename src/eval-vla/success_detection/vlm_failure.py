"""Failure-type classification of an episode video with a vision-language model served by Ollama."""

from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

FAILURE_DESCRIPTIONS = {
    "perception": "the policy misidentifies or fails to locate the target object",
    "language_understanding": "the executed behavior does not match the instruction semantics",
    "planning": "the policy selects an incorrect or suboptimal action sequence",
    "grasping": "the end-effector fails to acquire or maintain a stable grasp",
    "control": "the low-level controller produces jerky, oscillatory, or divergent motions",
    "recovery": "the policy fails to recover after an intermediate error",
    "hardware": "the trial fails due to mechanical, sensor, or calibration issues unrelated to the policy",
}


def sample_frames(video_path: str | Path, window_s: float = 30.0, n_frames: int = 16, max_side: int = 640) -> list[bytes]:
    """`n_frames` evenly spaced JPEG frames from the last `window_s` seconds of the video."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = max(0, total - int(window_s * fps))
    count = min(n_frames, total - first)
    indices = [first + round(i * (total - first - 1) / max(count - 1, 1)) for i in range(count)]

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        scale = max_side / max(frame.shape[:2])
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            frames.append(jpg.tobytes())

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from {video_path}")
    
    return frames


def save_debug_video(frames: list[bytes], path: str | Path, fps: float = 2.0) -> Path:
    """Packs the exact JPEG frames sent to the VLM into an mp4 (low fps so each frame can be inspected)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR) for f in frames]
    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for image in images:
        writer.write(image)
    writer.release()
    return path


def build_prompt(task: str, success_criteria: str, n_frames: int, window_s: float) -> str:
    types = "\n".join(f"- {name}: {desc}" for name, desc in FAILURE_DESCRIPTIONS.items())
    return (
        f"These are {n_frames} frames, in chronological order, from the last {window_s:g} seconds of a robot manipulation "
        f"episode that FAILED.\nTask: {task}\nSuccess criteria: {success_criteria},\n\n"
        f"Classify the main cause of the failure as exactly one of:\n{types}\n\n"
        "Answer with only the failure type name, exactly as written above, and nothing else."
    )


def parse_failure_type(text: str) -> str | None:
    text = text.strip().lower()
    hits = [(text.find(name), name) for name in FAILURE_DESCRIPTIONS if name in text]
    return min(hits)[1] if hits else None


def classify_failure(video_path: str | Path, task: str, success_criteria: str, window_s: float = 30.0,
                     n_frames: int = 16, model: str = "qwen3.5:latest", host: str = "http://localhost:11434",
                     timeout_s: float = 300.0, debug_video_path: str | Path | None = None,
                     num_ctx: int = 16384) -> tuple[str | None, str]:

    frames = sample_frames(video_path, window_s, n_frames)

    if debug_video_path is not None:
        save_debug_video(frames, debug_video_path)

    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [{
            "role": "user",
            "content": build_prompt(task, success_criteria, len(frames), window_s),
            "images": [base64.b64encode(f).decode() for f in frames],
        }],
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }
    
    request = urllib.request.Request(f"{host}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        reply = json.load(response)["message"]["content"]
    
    return parse_failure_type(reply), reply
