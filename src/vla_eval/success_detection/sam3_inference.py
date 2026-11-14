from __future__ import annotations

import os
from datetime import datetime

import cv2

os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"
import numpy as np
from torchvision.ops import nms

SAM_WEIGHTS = "src/vla_eval/success_detection/model_weights/sam3.pt"


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
    def __init__(self, mode: str, weights: str = SAM_WEIGHTS):
        overrides = {"task": "segment", 
                     "mode": "predict", 
                     "model": weights, 
                     "quantize": 16,
                     "save": True,
                     "save_dir": f"eval_results/sam3_results/sam3_{mode}_detections_{datetime.now():%d-%m_%H:%M:%S}"}
        
        if mode == "image":
            from ultralytics.models.sam import SAM3SemanticPredictor

            self.predictor = SAM3SemanticPredictor(overrides={**overrides, "conf": 0.25})
        elif mode == "video":
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
