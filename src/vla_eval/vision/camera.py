"""Frame capture from a camera connected to the computer (USB / V4L2) with OpenCV.

    python -m vla_eval.vision.camera --device 0 --out empty_table.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import os
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np


class Camera:
    """Class that helps with camera implementation to save image."""

    def __init__(self, device: int | str = 0, warmup_frames: int = 10) -> None:
        self.device = device
        self.warmup_frames = warmup_frames
        self._capture: cv2.VideoCapture | None = None


    def open(self) -> Camera:
        self._capture = cv2.VideoCapture(self.device)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera {self.device!r}")
        for _ in range(self.warmup_frames):
            self._capture.read()
        return self


    def capture(self, save_path: str | Path | None = None, countdown_s: int = 0) -> np.ndarray:
        """Most recent frame (BGR). `countdown_s` waits that many seconds first, printing a countdown."""
        if self._capture is None:
            raise RuntimeError("Camera is not open; call open() or use `with Camera(...)`")
        
        for remaining in range(countdown_s, 0, -1):
            print(f"Capturing in {remaining}...", flush=True)
            time.sleep(1)

        frame_period = 1 / (self._capture.get(cv2.CAP_PROP_FPS) or 30)

        for _ in range(10):
            start = time.perf_counter()
            self._capture.grab()
            if time.perf_counter() - start > frame_period / 2:
                break

        ok, frame = self._capture.retrieve()

        if not ok:
            raise RuntimeError(f"Could not read a frame from camera {self.device!r}")
        
        if save_path is not None:
            save_image(frame, save_path)

        return frame


    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


    def __enter__(self) -> Camera:
        return self.open()


    def __exit__(self, *exc) -> None:
        self.close()


def save_image(image: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image to {path}")
    return path


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:  
        raise FileNotFoundError(f"Could not read image {path}")
    return image


def capture_frame(device: int | str = 0, save_path: str | Path | None = None,
                  **camera_kwargs) -> np.ndarray:
    """Open the camera, take one frame and close it."""
    with Camera(device, **camera_kwargs) as cam:
        return cam.capture(save_path)

