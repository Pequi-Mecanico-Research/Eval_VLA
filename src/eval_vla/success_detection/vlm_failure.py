"""Failure-type classification of an episode video with a vision-language model.

Two inference modes: "ollama" (model served by an Ollama server) and "hf" (Hugging Face `transformers` in this
process, needs the `hf` extra: `pip install "eval_vla[hf]"`).
"""

from __future__ import annotations

import base64
import functools
import io
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODELS = {"ollama": "qwen3.5:latest", "hf": "Qwen/Qwen3.5-9B"}
HF_WEIGHTS_DIR = Path(__file__).parent / "model_weights"

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


def download_hf_weights(model: str, model_weights: str | Path) -> Path:
    """Downloads the Hugging Face repository `model` into the folder `model_weights`, unless it is already there.

    The folder holds one model: if it has the config and every weight file, nothing is downloaded (no network
    access). An interrupted download is resumed.
    """
    folder = Path(model_weights)
    index = folder / "model.safetensors.index.json"
    if index.exists():
        files = set(json.loads(index.read_text())["weight_map"].values())
    else:
        files = {"model.safetensors"}

    if (folder / "config.json").exists() and all((folder / name).exists() for name in files):
        return folder

    from huggingface_hub import snapshot_download

    print(f"Downloading {model} to {folder} (this happens only once)...")
    snapshot_download(repo_id=model, local_dir=folder)
    return folder


@functools.lru_cache(maxsize=1)
def _load_hf_pipeline(folder: str):
    """The pipeline is kept in memory so that the weights are loaded once, not at every episode."""
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError('mode="hf" needs transformers and accelerate: pip install "eval_vla[hf]"') from exc

    return pipeline("image-text-to-text", model=folder, device_map="auto", dtype="auto")


def _infer_ollama(frames: list[bytes], prompt: str, model: str, host: str, timeout_s: float, num_ctx: int) -> str:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(f).decode() for f in frames],
        }],
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }

    request = urllib.request.Request(f"{host}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.load(response)["message"]["content"]


def _infer_hf(frames: list[bytes], prompt: str, model: str, model_weights: str | Path | None, max_new_tokens: int = 32) -> str:
    from PIL import Image

    folder = download_hf_weights(model, model_weights or HF_WEIGHTS_DIR / model.split("/")[-1])
    pipe = _load_hf_pipeline(str(folder))

    content = [{"type": "image", "image": Image.open(io.BytesIO(f)).convert("RGB")} for f in frames]
    messages = [{"role": "user", "content": [*content, {"type": "text", "text": prompt}]}]

    output = pipe(text=messages, return_full_text=False, enable_thinking=False,
                  generate_kwargs={"max_new_tokens": max_new_tokens, "do_sample": False})
    
    return output[0]["generated_text"]


def classify_failure(video_path: str | Path, task: str, success_criteria: str, window_s: float = 30.0,
                     n_frames: int = 16, model: str | None = None, host: str = "http://localhost:11434",
                     timeout_s: float = 300.0, debug_video_path: str | Path | None = None,
                     num_ctx: int = 16384, mode: str = "ollama",
                     model_weights: str | Path | None = None) -> tuple[str | None, str]:
    """Asks a vision-language model which failure type the end of the video shows.

    Args:
        mode: "ollama" or "hf" (Hugging Face `transformers`).
        model: Model name of the chosen mode. If None, uses `DEFAULT_MODELS[mode]`.
        host, timeout_s, num_ctx: Only used by "ollama".
        model_weights: Only used by "hf". Folder of the weights, downloaded there if missing. If None, uses
            `model_weights/<model name>` inside the package.

    Returns:
        The failure type (or None if the reply was not recognized) and the model reply.

    Raises:
        ValueError: If `mode` is not "ollama" or "hf".
    """
    if mode not in DEFAULT_MODELS:
        raise ValueError(f"mode must be one of {list(DEFAULT_MODELS)}, got {mode!r}")
    model = model or DEFAULT_MODELS[mode]

    frames = sample_frames(video_path, window_s, n_frames)

    if debug_video_path is not None:
        save_debug_video(frames, debug_video_path)

    prompt = build_prompt(task, success_criteria, len(frames), window_s)

    if mode == "hf":
        reply = _infer_hf(frames, prompt, model, model_weights)
    else:
        reply = _infer_ollama(frames, prompt, model, host, timeout_s, num_ctx)

    return parse_failure_type(reply), reply
