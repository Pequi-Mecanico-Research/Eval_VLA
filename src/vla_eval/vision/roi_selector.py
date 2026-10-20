"""Interactive drawing of polygonal Regions Of Interest (ROIs) on an image.

    python -m vla_eval.vision.roi_selector --image empty_table.png --names table container --out rois.json
    python -m vla_eval.vision.roi_selector --camera 0 --names table container --out rois.json

Controls (OpenCV window):
    left click        add vertex
    right click / z   undo last vertex
    Enter / space     close the current polygon (>= 3 vertices) and go to the next name
    Esc               cancel without saving
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import os
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")


import cv2
import numpy as np

Rois = dict[str, list[list[int]]]

WINDOW = "ROI selector"
KEY_ENTER, KEY_SPACE, KEY_ESC = (13, 10), 32, 27


class RoiSelector:
    """Collects one polygon per name and creates a GUI to draw it."""

    def __init__(self, image: np.ndarray, names: list[str]) -> None:
        if not names:
            raise ValueError("At least one ROI name is required")
        if len(set(names)) != len(names):
            raise ValueError(f"ROI names must be unique: {names}")
        self.image = image
        self.names = list(names)
        self.rois: Rois = {}
        self.points: list[list[int]] = []

    @property
    def current_name(self) -> str | None:
        return self.names[len(self.rois)] if not self.done else None

    @property
    def done(self) -> bool:
        return len(self.rois) == len(self.names)

    def add_point(self, x: int, y: int) -> None:
        if not self.done:
            self.points.append([int(x), int(y)])

    def undo(self) -> None:
        if self.points:
            self.points.pop()

    def close_polygon(self) -> bool:
        """Store the current polygon; returns False if it has fewer than 3 vertices."""
        if self.done or len(self.points) < 3:
            return False
        self.rois[self.current_name] = self.points
        self.points = []
        return True

    def render(self) -> np.ndarray:
        out = self.image.copy()
        for name, points in self.rois.items():
            polygon = np.array(points, np.int32)
            cv2.polylines(out, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(out, name, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
        if self.points:
            cv2.polylines(out, [np.array(self.points, np.int32)], isClosed=False,
                          color=(0, 0, 255), thickness=2)
            for p in self.points:
                cv2.circle(out, tuple(p), 4, (0, 0, 255), -1)
        status = ("done - press Enter" if self.done
                  else f"drawing '{self.current_name}' ({len(self.points)} pts)")
        cv2.putText(out, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return out

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo()

    def run(self) -> Rois | None:
        """Open the window and block until all ROIs are drawn (or None on Esc)."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        try:
            while True:
                cv2.imshow(WINDOW, self.render())
                key = cv2.waitKey(30) & 0xFF
                if key == KEY_ESC:
                    return None
                if key == ord("z"):
                    self.undo()
                elif key in (*KEY_ENTER, KEY_SPACE):
                    if self.done:
                        return self.rois
                    self.close_polygon()
        finally:
            cv2.destroyWindow(WINDOW)


def select_rois(image: np.ndarray, names: list[str]) -> Rois | None:
    """Draw one polygon per name on `image`. Returns None if cancelled."""
    return RoiSelector(image, names).run()


def save_rois(rois: Rois, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rois, indent=2))
    return path


def load_rois(path: str | Path) -> Rois:
    rois = json.loads(Path(path).read_text())
    for name, points in rois.items():
        if len(points) < 3:
            raise ValueError(f"ROI {name!r} in {path} has fewer than 3 vertices")
    return rois


def create_rois(image: np.ndarray, names: list[str], save_path: str | Path) -> Rois:
    """Draw one polygon per name on `image` and save them to JSON."""
    rois = select_rois(image, names)
    if rois is None:
        raise RuntimeError("ROI drawing cancelled, nothing saved.")
    print(f"Saved {list(rois)} to {save_rois(rois, save_path)}")
    return rois


def main() -> None:
    from .camera import capture_frame, save_image

    parser = argparse.ArgumentParser(description="Draw polygonal ROIs and save them to JSON.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="image to draw on")
    source.add_argument("--camera", help="camera index/path to capture the image from", default=0)
    parser.add_argument("--names", nargs="+", required=True, help="ROI names, in drawing order")
    parser.add_argument("--out", default="rois.json")
    parser.add_argument("--save-frame", help="with --camera: also save the captured image")
    args = parser.parse_args()

    
    device = int(args.camera) if args.camera.isdigit() else args.camera
    
    image = capture_frame(device)
    
    if args.save_frame:
        save_image(image, args.save_frame)

    rois = select_rois(image, args.names)

    if rois is None:
        print("Cancelled, nothing saved.")
        return
    
    print(f"Saved {list(rois)} to {save_rois(rois, args.out)}")


if __name__ == "__main__":
    main()
