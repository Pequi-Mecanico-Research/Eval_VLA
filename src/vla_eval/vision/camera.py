"""Frame capture from a camera connected to the computer (USB / V4L2) with OpenCV.

    python -m vla_eval.vision.camera --device 0 --out empty_table.png
"""

from __future__ import annotations

import argparse
import threading
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
        self._recorder: threading.Thread | None = None
        self._stop_recording = threading.Event()
        self._latest_frame: np.ndarray | None = None
        self._record_error: Exception | None = None


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

        if self._recorder is not None:
            frame = self._latest_frame
            if frame is None:
                raise RuntimeError("No frame recorded yet")
            if save_path is not None:
                save_image(frame, save_path)
            return frame.copy()

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


    def start_recording(self, save_path: str | Path, preview: bool = True, window_name: str = "Recording") -> Path:
        """Records the camera to a video file in a background thread, optionally showing a live preview window.

        Frames are written on a wall-clock timeline so the video duration matches real time even if the
        camera delivers fewer frames per second than it advertises. Call `stop_recording()` to finish."""
        if self._capture is None:
            raise RuntimeError("Camera is not open; call open() or use `with Camera(...)`")
        if self._recorder is not None:
            raise RuntimeError("Already recording")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {save_path}")

        self._stop_recording.clear()
        self._record_error = None
        self._latest_frame = None
        self._recorder = threading.Thread(target=self._record_loop, args=(writer, fps, preview, window_name), daemon=True)
        self._recorder.start()
        return save_path


    def stop_recording(self) -> None:
        if self._recorder is None:
            return
        self._stop_recording.set()
        self._recorder.join()
        self._recorder = None
        if self._record_error is not None:
            raise RuntimeError("Recording failed") from self._record_error


    def _record_loop(self, writer: cv2.VideoWriter, fps: float, preview: bool, window_name: str) -> None:
        written = 0
        start = time.perf_counter()
        try:
            while not self._stop_recording.is_set():
                ok, frame = self._capture.read()
                if not ok:
                    raise RuntimeError(f"Could not read a frame from camera {self.device!r}")
                self._latest_frame = frame

                due = int((time.perf_counter() - start) * fps) + 1
                for _ in range(max(due - written, 1)):
                    writer.write(frame)
                    written += 1

                if preview:
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(1)
        except Exception as exc:
            self._record_error = exc
        finally:
            writer.release()
            if preview:
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)


    def close(self) -> None:
        self.stop_recording()
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

