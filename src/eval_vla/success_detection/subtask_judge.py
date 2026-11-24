"""When each subtask of an episode was completed, judged from the episode video.

Two judges, same output (one completion time in seconds per subtask, None if never completed):
- "ollama": a general VLM served by Ollama sees timestamped frames and names the first frame where each subtask is done.
- "robometer": a Robometer eval server (https://github.com/robometer/robometer) scores per-frame success for each
  subtask instruction; completion is the first frame whose success probability reaches `threshold`.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def read_frames(video_path: str | Path, n_frames: int, max_side: int = 640) -> tuple[list[float], list[np.ndarray]]:
    """`n_frames` evenly spaced BGR frames over the whole video, with their timestamps in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    times, frames = [], []

    for idx in np.unique(np.linspace(0, max(total - 1, 0), n_frames).round().astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        scale = max_side / max(frame.shape[:2])
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        times.append(idx / fps)
        frames.append(frame)

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from {video_path}")
    return times, frames


def build_prompt(task: str, subtasks: list[str], times: list[float]) -> str:
    listing = "\n".join(f"- {s}" for s in subtasks)
    stamps = ", ".join(f"{i}: {t:.1f}s" for i, t in enumerate(times))
    return (
        f"These are {len(times)} frames, in chronological order, of a robot manipulation episode.\n"
        f"Task: {task}\nFrame index: timestamp -> {stamps}\n\n"
        f"The task is split into these subtasks:\n{listing}\n\n"
        "For each subtask, give the index of the FIRST frame where it is visibly completed correctly, "
        "or null if it is never completed. Answer only with a JSON object mapping each subtask, written "
        "exactly as above, to a frame index or null."
    )


def parse_frame_indices(reply: str, subtasks: list[str], n_frames: int) -> list[int | None]:
    try:
        answer = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
    except ValueError:
        return [None] * len(subtasks)
    answer = {str(k).strip().lower(): v for k, v in answer.items()} if isinstance(answer, dict) else {}
    indices = []
    for s in subtasks:
        v = answer.get(s.lower())
        indices.append(int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v < n_frames else None)
    return indices


def ollama_subtask_times(video_path: str | Path, task: str, subtasks: list[str], n_frames: int = 30,
                         model: str = "qwen3.5:latest", host: str = "http://localhost:11434",
                         timeout_s: float = 300.0, num_ctx: int = 16384) -> tuple[list[float | None], str]:
    times, frames = read_frames(video_path, n_frames)
    images = [base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode() for f in frames]
    payload = {
        "model": model, "stream": False, "think": False, "format": "json",
        "messages": [{"role": "user", "content": build_prompt(task, subtasks, times), "images": images}],
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }
    request = urllib.request.Request(f"{host}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        reply = json.load(response)["message"]["content"]

    return [None if i is None else times[i] for i in parse_frame_indices(reply, subtasks, len(times))], reply


def first_crossing(scores: list[float], times: list[float], threshold: float) -> float | None:
    """Time of the first score >= threshold. Scores are mapped linearly onto `times` when the server returns a
    different number of predictions than frames sent (it subsamples unless `use_frame_steps`)."""
    hit = next((i for i, s in enumerate(scores) if s >= threshold), None)
    if hit is None:
        return None
    if len(scores) == len(times):
        return times[hit]
    return times[0] + hit * (times[-1] - times[0]) / max(len(scores) - 1, 1)


def robometer_subtask_times(video_path: str | Path, subtasks: list[str], url: str = "http://localhost:8000",
                            n_frames: int = 30, threshold: float = 0.5, timeout_s: float = 300.0,
                            use_frame_steps: bool = False) -> tuple[list[float | None], str]:
    """One sample per subtask in a single /evaluate_batch_npy request (the protocol of Robometer's
    scripts/example_inference.py). Falls back to the progress head when the model has no success head."""
    import requests

    times, frames = read_frames(video_path, n_frames)
    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]).astype(np.uint8)
    buf = io.BytesIO()
    np.save(buf, rgb)

    files, data = {}, {"use_frame_steps": "true" if use_frame_steps else "false"}
    for i, subtask in enumerate(subtasks):
        key = f"sample_{i}_trajectory_frames"
        files[key] = (f"{key}.npy", buf.getvalue(), "application/octet-stream")
        data[f"sample_{i}"] = json.dumps({"sample_type": "progress", "trajectory": {
            "frames": {"__numpy_file__": key}, "frames_shape": list(rgb.shape), "task": subtask, "id": str(i),
            "metadata": {"subsequence_length": len(frames)}, "video_embeddings": None}})

    response = requests.post(url.rstrip("/") + "/evaluate_batch_npy", files=files, data=data, timeout=timeout_s)
    response.raise_for_status()
    outputs = response.json()

    scores = (outputs.get("outputs_success") or {}).get("success_probs") or outputs["outputs_progress"]["progress_pred"]
    return [first_crossing(np.ravel(s).tolist(), times, threshold) for s in scores], json.dumps(outputs)
