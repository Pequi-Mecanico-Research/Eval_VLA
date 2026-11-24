from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from ultralytics.models.fastsam import FastSAMPredictor

import cv2

os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"
import numpy as np
from torchvision.ops import nms

def _patch_clip_tokenizer() -> None:
    """SAM3 (ultralytics) calls `SimpleTokenizer()(texts, context_length=...)`, which only exists in Ultralytics' CLIP fork.

    The PyPI `openai-clip` has the same tokenizer without `__call__`, so add it (same behavior as `clip.tokenize`).
    """
    import clip

    tokenizer = clip.simple_tokenizer.SimpleTokenizer
    if "__call__" not in vars(tokenizer):
        tokenizer.__call__ = lambda self, texts, context_length=77: clip.tokenize(texts, context_length=context_length, truncate=True)


_patch_clip_tokenizer()

WEIGHTS_DIR = Path(__file__).parent / "model_weights"
SAM_WEIGHTS = {
    "sam2": WEIGHTS_DIR / "sam2.1_b.pt", 
    "sam3": WEIGHTS_DIR / "sam3.pt",
}


def frames_to_intervals(signal, fps, limiar_on=0.5, limiar_off=0.3, min_dur_s=1, juntar_gap_s=1.5):
    intervals, start = [], None
    for i, score in enumerate(signal):
        if start is None and score >= limiar_on:
            start = i
        elif start is not None and score < limiar_off:
            intervals.append([start / fps, i / fps])
            start = None
    if start is not None:
        intervals.append([start / fps, len(signal) / fps])

    merged = []
    for a, b in intervals:
        if b - a <= min_dur_s:
            continue
        if merged and a - merged[-1][1] <= juntar_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


class ObjectDetection:
    def __init__(self, mode: str, model: str = "sam2", weights: str | Path | None = None):
        """Segmentation predictor used by the success and human-intervention metrics.

        Args:
            mode: "image" (only with model="sam3") or "video".
            model: "sam2" or "sam3"; picks the predictor and its default weights.
            weights: Path to the weights file. If None, uses `model_weights/<model>.pt` inside the package,
                which is not shipped with the pip install.
        """
        if model not in SAM_WEIGHTS:
            raise ValueError(f"model must be one of {list(SAM_WEIGHTS)}, got {model!r}")

        overrides = {"task": "segment",
                     "mode": "predict",
                     "model": str(weights or SAM_WEIGHTS[model]),
                     "quantize": 16,
                     "save": True,
                     "save_dir": f"eval_results/{model}_results/{model}_{mode}_detections_{datetime.now():%d-%m_%H:%M:%S}"}

        if mode == "image":
            if model != "sam3":
                raise ValueError("image mode only supports model='sam3' (text-prompted SAM3SemanticPredictor)")
            from ultralytics.models.sam import SAM3SemanticPredictor

            self.predictor = SAM3SemanticPredictor(overrides={**overrides, "conf": 0.25})
        elif mode == "video":
            if model == "sam2":
                from ultralytics.models.sam import SAM2VideoPredictor
                self.predictor = SAM2VideoPredictor(overrides={**overrides, "conf": 0.70, "imgsz": 640})
            else:
                from ultralytics.models.sam import SAM3VideoSemanticPredictor
                self.predictor = SAM3VideoSemanticPredictor(overrides={**overrides, "conf": 0.70, "imgsz": 640})

        else:
            raise ValueError(f"mode must be 'image' or 'video', got {mode!r}")
        
        self.mode = mode


    def run(self, source, prompt: list[str], rois: dict | None = None, **kwargs):
        if self.mode == "image":
            return self._image(source, prompt, rois, **kwargs)
        else: 
            return self._video(source, prompt, **kwargs)


    def _image(self, image: np.ndarray, prompt, rois, min_area=200, max_area=50_000, iou_thr=0.5):
        polys = {k: np.array(v, np.int32) for k, v in rois.items()}
        counts = {k: 0 for k in polys}
        self.predictor.set_image(image)
        boxes = self.predictor(text=prompt)[0].boxes
        if boxes is None or not len(boxes):
            return counts

        xyxy = boxes.xyxy.float().cpu()
        for i in nms(xyxy, boxes.conf.float().cpu(), iou_thr).tolist():
            x1, y1, x2, y2 = xyxy[i].tolist()
            if not min_area < (x2 - x1) * (y2 - y1) < max_area:
                continue
            for name, poly in polys.items():
                if cv2.pointPolygonTest(poly, ((x1 + x2) / 2, y2), False) >= 0:
                    counts[name] += 1
                    break
        return counts


    def _video(self, video_path, prompt, min_area=400):
        video_path = str(video_path)
        capture = cv2.VideoCapture(video_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        capture.release()

        signal = []
        for r in self.predictor(source=video_path, text=prompt, stream=True):
            score = 0.0
            if r.boxes is not None and len(r.boxes):
                wh = r.boxes.xywh
                valid = r.boxes.conf[wh[:, 2] * wh[:, 3] > min_area]
                if len(valid):
                    score = float(valid.max())
            signal.append(score)
        return frames_to_intervals(signal, fps)
