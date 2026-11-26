from __future__ import annotations

import os
import warnings
from datetime import datetime
from pathlib import Path
from ultralytics.models.fastsam import FastSAMPredictor

import cv2

os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"
import numpy as np
from torchvision.ops import nms

warnings.filterwarnings("ignore", category=UserWarning, message=r"The '(repr|frozen)' attribute with value \w+ was provided to the `Field\(\)` function")


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
MODEL_WEIGHTS = {
    "sam2": WEIGHTS_DIR / "sam2.1_b.pt", 
    "sam3": WEIGHTS_DIR / "sam3.pt",
    "yoloworld": WEIGHTS_DIR / "yolov8s-world.pt",
}
MODEL_IMGSZ = {"sam2": 640, "sam3": 644, "yoloworld": 640}


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
    def __init__(self, mode: str, model: str = "sam2", weights: str | Path | None = None, imgsz: int | None = None, conf: float | None = None):
        """Open-vocabulary detector (SAM2, SAM3 or YOLO-World) used by the success and human-intervention metrics.

        Args:
            mode: "image" (only with model="sam3" or "yoloworld") or "video".
            model: "sam2", "sam3" or "yoloworld"; picks the predictor and its default weights.
            weights: Path to the weights file. If None, uses the file of `MODEL_WEIGHTS[model]` in `model_weights/`
                inside the package, which is not shipped with the pip install. YOLO-World weights are downloaded
                to this path if missing, as long as the file name is an official one (`yolov8{s,m,l,x}-world.pt`
                or `yolov8{s,m,l,x}-worldv2.pt`). SAM weights must be downloaded manually.
            imgsz: Inference size. If None, uses `MODEL_IMGSZ[model]`.
            conf: Minimum confidence of a detection. If None, uses 0.5 for "yoloworld" and, for the SAM models,
                0.25 in "image" mode and 0.70 in "video" mode.
        """
        if model not in MODEL_WEIGHTS:
            raise ValueError(f"model must be one of {list(MODEL_WEIGHTS)}, got {model!r}")
        if mode not in ("image", "video"):
            raise ValueError(f"mode must be 'image' or 'video', got {mode!r}")
        if mode == "image" and model == "sam2":
            raise ValueError("image mode only supports the text-prompted models 'sam3' and 'yoloworld'")

        if conf is None:
            conf = 0.5 if model == "yoloworld" else 0.25 if mode == "image" else 0.70

        self.mode = mode
        self.model = model
        self._classes: list[str] | None = None

        overrides = {"conf": conf,
                     "quantize": 16,
                     "save": True,
                     "imgsz": imgsz or MODEL_IMGSZ[model],
                     "save_dir": f"eval_results/{model}_results/{model}_{mode}_detections_{datetime.now():%d-%m_%H:%M:%S}"}
        
        weights = str(weights or MODEL_WEIGHTS[model])

        if model == "yoloworld":
            from ultralytics import YOLOWorld

            self.predictor = YOLOWorld(weights)
            self.predict_args = overrides
            return

        overrides = {**overrides, "task": "segment", "mode": "predict", "model": weights}
        if mode == "image":
            from ultralytics.models.sam import SAM3SemanticPredictor

            self.predictor = SAM3SemanticPredictor(overrides=overrides)
        elif model == "sam2":
            from ultralytics.models.sam import SAM2VideoPredictor

            self.predictor = SAM2VideoPredictor(overrides=overrides)
        else:
            from ultralytics.models.sam import SAM3VideoSemanticPredictor

            self.predictor = SAM3VideoSemanticPredictor(overrides=overrides)


    def run(self, source, prompt: list[str], rois: dict | None = None, **kwargs):
        if self.mode == "image":
            return self._image(source, prompt, rois, **kwargs)
        else: 
            return self._video(source, prompt, **kwargs)


    def _set_classes(self, prompt: list[str]) -> None:
        """Sets the YOLO-World vocabulary, only when it changes (it runs the CLIP text encoder)."""
        if prompt != self._classes:
            self.predictor.set_classes(prompt)
            self._classes = list(prompt)


    def _detect_image(self, image: np.ndarray, prompt: list[str]):
        if self.model == "yoloworld":
            self._set_classes(prompt)
            return self.predictor.predict(image, **self.predict_args)[0]
        self.predictor.set_image(image)
        return self.predictor(text=prompt)[0]


    def _detect_video(self, video_path: str, prompt: list[str]):
        if self.model == "yoloworld":
            self._set_classes(prompt)
            return self.predictor.track(source=video_path, stream=True, **self.predict_args)
        return self.predictor(source=video_path, text=prompt, stream=True)


    def _image(self, image: np.ndarray, prompt, rois, min_area=200, max_area=50_000, iou_thr=0.5):
        polys = {k: np.array(v, np.int32) for k, v in rois.items()}
        counts = {k: 0 for k in polys}
        boxes = self._detect_image(image, prompt).boxes
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
        for r in self._detect_video(video_path, prompt):
            score = 0.0
            if r.boxes is not None and len(r.boxes):
                wh = r.boxes.xywh
                valid = r.boxes.conf[wh[:, 2] * wh[:, 3] > min_area]
                if len(valid):
                    score = float(valid.max())
            signal.append(score)
        return frames_to_intervals(signal, fps)
