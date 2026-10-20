"""Object counting by background subtraction against an empty-scene reference image.

Pipeline: blur -> LAB difference to the reference -> threshold -> morphological
cleanup -> connected components -> filter by area -> assign each blob to the ROI
that contains its centroid.

Limitations: objects touching each other merge into one blob, and the camera,
lighting and table must not move between the reference and the evaluated image.
`min_area`/`max_area` are in pixels, so they depend on resolution and camera height.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import os
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")


import cv2
import numpy as np

from ..vision.roi_selector import Rois


@dataclass
class DetectedObject:
    centroid: tuple[float, float]  
    area: int 
    bbox: tuple[int, int, int, int]  


def pipeline_steps(image: np.ndarray, reference: np.ndarray, diff_threshold: float = 30,
                   lightness_weight: float = 0.5, blur_kernel: int = 5,
                   open_kernel: int = 5, close_kernel: int = 7) -> dict[str, np.ndarray]:

    def to_lab(img: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

    steps: dict[str, np.ndarray] = {}
    diff = np.abs(to_lab(image) - to_lab(reference))

    steps["lightness_diff"] = lightness_weight * diff[..., 0]
    steps["color_diff"] = diff[..., 1:].max(axis=-1)
    steps["distance"] = np.maximum(steps["lightness_diff"], steps["color_diff"])
    steps["threshold"] = (steps["distance"] > diff_threshold).astype(np.uint8) * 255

    steps["opened"] = cv2.morphologyEx(steps["threshold"], cv2.MORPH_OPEN,
                                       np.ones((open_kernel, open_kernel), np.uint8))
    steps["closed"] = cv2.morphologyEx(steps["opened"], cv2.MORPH_CLOSE,
                                       np.ones((close_kernel, close_kernel), np.uint8))
    return steps


def foreground_mask(image: np.ndarray, reference: np.ndarray, **mask_kwargs) -> np.ndarray:
    """Binary mask (0/255) of the pixels that differ from the reference."""
    return pipeline_steps(image, reference, **mask_kwargs)["closed"]


def detect_objects(image: np.ndarray, reference: np.ndarray, min_area: int = 150, max_area: int = 8000, **mask_kwargs) -> list[DetectedObject]:
    mask = foreground_mask(image, reference, **mask_kwargs)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    objects = []
    for i in range(1, n): 
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area < area < max_area:
            bbx_left_up, bby_left_up, bbwidth, bbheight = (int(v) for v in stats[i, :4])
            objects.append(DetectedObject(centroid=(float(centroids[i][0]), float(centroids[i][1])),
                                          area=area, bbox=(bbx_left_up, bby_left_up, bbwidth, bbheight)))
    return objects


def count_objects_in_rois(image: np.ndarray, reference: np.ndarray, rois: Rois, **detect_kwargs) -> dict[str, int]:
    polygons = {name: np.array(points, np.int32) for name, points in rois.items()}
    counts = {name: 0 for name in rois}
    
    for obj in detect_objects(image, reference, **detect_kwargs):
        for name, polygon in polygons.items():
            if cv2.pointPolygonTest(polygon, obj.centroid, False) >= 0:
                counts[name] += 1
    
    return counts


def draw_detections(image: np.ndarray, rois: Rois,  objects: list[DetectedObject]) -> np.ndarray:
    out = image.copy()
    for name, points in rois.items():
        polygon = np.array(points, np.int32)
        cv2.polylines(out, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(out, name, tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    for obj in objects:
        x, y, w, h = obj.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, tuple(int(c) for c in obj.centroid), 3, (0, 0, 255), -1)
    return out


def debug_pipeline(image: np.ndarray, reference: np.ndarray, rois: Rois, min_area: int = 150,
                   max_area: int = 8000, save_path: str | Path | None = None, show: bool = True,
                   tile_width: int = 480, **mask_kwargs) -> np.ndarray:
    
    steps = pipeline_steps(image, reference, **mask_kwargs)
    threshold = mask_kwargs.get("diff_threshold", 30)

    n, _, stats, _ = cv2.connectedComponentsWithStats(steps["closed"])
    detections = draw_detections(image, rois, [])
    areas = stats[1:, cv2.CC_STAT_AREA]
    for (x, y, w, h, area) in stats[1:]:
        color = (0, 255, 0) if min_area < area < max_area else (0, 0, 255)
        cv2.rectangle(detections, (x, y), (x + w, y + h), color, 2)
        cv2.putText(detections, str(area), (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    counts = count_objects_in_rois(image, reference, rois, min_area=min_area, max_area=max_area, **mask_kwargs)

    def heatmap(values: np.ndarray) -> np.ndarray:
        scaled = np.clip(values / (2 * threshold) * 255, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

    tiles = {
        "reference": reference,
        "image": image,
        "detections (green=kept, red=area filter)": detections,
        "lightness diff (weighted)": heatmap(steps["lightness_diff"]),
        "color diff": heatmap(steps["color_diff"]),
        f"distance (green = threshold {threshold})": heatmap(steps["distance"]),
        "threshold": steps["threshold"],
        "opened (noise removed)": steps["opened"],
        "closed (holes filled)": steps["closed"],
    }
    rows = []
    for row_names in np.array_split(list(tiles), 3):
        row = []
        for name in row_names:
            tile = tiles[name]
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR) if tile.ndim == 2 else tile
            tile = cv2.resize(tile, (tile_width, int(tile.shape[0] * tile_width / tile.shape[1])))
            tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            cv2.putText(tile, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(tile)
        rows.append(np.hstack(row))
    mosaic = np.vstack(rows)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), mosaic)
    if show:
        cv2.imshow("background subtraction", mosaic)
        cv2.waitKey(0)
        cv2.destroyWindow("background subtraction")
    return mosaic
