from __future__ import annotations

import time
from typing import Mapping

import cv2

from .camera import Camera


class OpenCVViewer:
    preview = True

    def __init__(self, camera: Camera) -> None:
        self.camera = camera

    def update(self, **series: Mapping[str, float]) -> None:
        self.camera.show_preview()

    def close(self) -> None:
        pass


class RerunViewer:
    preview = False

    def __init__(self, camera: Camera, session_name: str = "vla_eval") -> None:
        import rerun as rr

        self.rr = rr
        self.camera = camera
        self.start = time.perf_counter()
        self.series: set[str] = set()
        rr.init(session_name)
        rr.spawn(memory_limit="10%")

    def update(self, **series: Mapping[str, float]) -> None:
        rr = self.rr
        rr.set_time("time", duration=time.perf_counter() - self.start)

        frame = self.camera.latest_frame
        if frame is not None:
            rr.log("camera", rr.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        new = [name for name in series if name not in self.series]
        for name, values in series.items():
            if name in new:
                self.series.add(name)
                rr.log(name, rr.SeriesLines(names=list(values)), static=True)
            rr.log(name, rr.Scalars(list(values.values())))

        if new:
            import rerun.blueprint as rrb

            views = [rrb.Spatial2DView(origin="camera", name="camera")]
            views += [rrb.TimeSeriesView(origin=name, name=name) for name in sorted(self.series)]
            rr.send_blueprint(rrb.Blueprint(rrb.Grid(*views)))

    def close(self) -> None:
        self.rr.rerun_shutdown()


def make_viewer(kind: str, camera: Camera) -> OpenCVViewer | RerunViewer:
    if kind == "opencv":
        return OpenCVViewer(camera)
    if kind == "rerun":
        return RerunViewer(camera)
    raise ValueError(f"viewer must be 'opencv' or 'rerun', got {kind!r}")
