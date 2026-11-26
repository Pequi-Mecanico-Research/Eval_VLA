from __future__ import annotations

from enum import Enum
from pathlib import Path

from ..base import EvalVLA, metric
from ..config import SetupConfig
from ..plots import plot_failure_analysis
from ..success_detection.vlm_failure import DEFAULT_MODELS, classify_failure


class FailureType(str, Enum):
    PERCEPTION = "perception"
    LANGUAGE_UNDERSTANDING = "language_understanding"
    PLANNING = "planning"
    GRASPING = "grasping"
    CONTROL = "control"
    RECOVERY = "recovery"
    HARDWARE = "hardware"


class FailureAnalysis(EvalVLA):
    """Failure analysis.

    Perception errors: the policy misidentifies or fails to locate the target object.

    Language-understanding errors: the executed behavior does not match the instruction semantics.

    Planning errors: the policy selects an incorrect or suboptimal action sequence.

    Grasping errors: the end-effector fails to acquire or maintain a stable grasp.

    Control errors: the low-level controller produces jerky, oscillatory, or divergent motions.

    Recovery failures: the policy fails to recover after an intermediate error.

    Hardware or calibration failures: the trial fails due to mechanical, sensor, or calibration issues unrelated to the policy.
    """

    def __init__(self, config: SetupConfig, video_path: str | Path, output_dir: str | Path = "eval_results",
                window_s: float = 30.0, vlm_model: str | None = None,
                vlm_host: str = "http://localhost:11434", vlm_frames: int = 30,
                debug_video: bool = False, vlm_timeout_s: float = 300.0,
                vlm_num_ctx: int = 16384, run_dir: str | Path | None = None, name: str | None = None,
                mode: str = "ollama", model_weights: str | Path | None = None):
        """Classifies the failure of `video_path` as soon as it is created.

        `mode` is "ollama" (default, uses `vlm_host`, `vlm_timeout_s` and `vlm_num_ctx`) or "hf" (Hugging Face
        `transformers`, weights downloaded into `model_weights` if missing). `vlm_model` None uses the default
        model of the mode.
        """
        super().__init__(config, output_dir, run_dir, name)

        self.vlm_num_ctx = vlm_num_ctx
        self.vlm_timeout_s = vlm_timeout_s
        self.debug_video = debug_video
        self.video_path = video_path
        self.window_s = window_s
        self.failure_counts: dict[FailureType, int] = {t: 0 for t in FailureType}
        self.episodes_evaluated: int = 0
        self.failure_type: FailureType | None = None
        self.mode = mode
        self.model_weights = model_weights
        self.vlm_model = vlm_model or DEFAULT_MODELS.get(mode)
        self.vlm_host = vlm_host
        self.vlm_frames = vlm_frames

        self.main()


    def _save_graphs(self) -> None:
        plot_failure_analysis(self.csv_path, self.output_dir, self.config.experiment_name)


    def main(self) -> FailureType | None:
        """Asks the VLM which failure type the episode video shows and sets `failure_type`."""
        self.episodes_evaluated += 1
        self.failure_type = None

        name = self.vlm_failure_type()["vlm_failure_type"]

        if name is not None:
            self.failure_type = FailureType(name)
            self.failure_counts[self.failure_type] += 1

        return self.failure_type


    @metric
    def vlm_failure_type(self) -> dict[str, str | None]:
        """Asks a vision-language model (VLM, via Ollama or Hugging Face) which failure type the episode video shows.

        Samples `vlm_frames` frames from the last `window_s` seconds of the video and sends them to the VLM
        together with the task name and success criteria from the config. Runs automatically when
        `FailureAnalysis` is created; its inputs are the constructor arguments, not method arguments.

        Returns:
            dict with `vlm_failure_type` (a `FailureType` name, or None if the reply was not recognized),
            `vlm_model` and `vlm_raw_reply` (the model answer as text).
        """
        name, reply = classify_failure(
            self.video_path,
            task=self.config.task_name,
            success_criteria=self.config.success_criteria,
            window_s=self.window_s,
            n_frames=self.vlm_frames,
            model=self.vlm_model,
            mode=self.mode,
            model_weights=self.model_weights,
            host=self.vlm_host,
            timeout_s=self.vlm_timeout_s,
            num_ctx=self.vlm_num_ctx,
            debug_video_path=Path(self.video_path).parent / f"{Path(self.video_path).stem}_vlm_frames.mp4" if self.debug_video else None,
        )
        return {"vlm_failure_type": name, "vlm_model": self.vlm_model, "vlm_raw_reply": reply.strip()}

