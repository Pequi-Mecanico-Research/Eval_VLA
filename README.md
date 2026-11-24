# Pequi_Mêcanico_UFG-eval

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/Pequi-Mecanico-Research/Eval_VLA)](https://github.com/Pequi-Mecanico-Research/Eval_VLA/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Pequi-Mecanico-Research/Eval_VLA)](https://github.com/Pequi-Mecanico-Research/Eval_VLA/issues)

Reproducible metrics for evaluating Vision-Language-Action (VLA) models on real robots. You call the metrics inside your policy's evaluation loop; each episode is logged to JSON/CSV and the plots are refreshed automatically.

## Contents

- [Metric families](#metric-families)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Results and plots](#results-and-plots)
- [Examples](#examples)
- [Full evaluation on a real robot](#full-evaluation-on-a-real-robot)
- [Adding a new metric class](#adding-a-new-metric-class)

## Metric families

Each family is a class in `src/eval_vla/metrics/`. Every method, its inputs and its outputs are described in [`src/eval_vla/metrics/README.md`](src/eval_vla/metrics/README.md).

| Family | What it measures | Needs |
|---|---|---|
| `Statistical` | Success rate, task completion time (mean, std), inference latency (mean, p95) | Before/after camera frames for success (SAM3 or image difference) |
| `Operation` | Human intervention, collision, jerk and reset failure rates | Arm driver (`trossen_arm`); episode video + SAM3 for human intervention |
| `LongHorizon` | Subtask completion rate, average completed sequence length, failure position | Not implemented yet |
| `FailureAnalysis` | Failure type of a failed episode (perception, grasping, planning, ...) | Episode video + a VLM served by [Ollama](https://ollama.com) |

## Installation

**As a package**, inside the environment of the repository that runs your VLA policy:

```bash
uv pip install eval_vla
```

**From source**, to develop or run the examples ([uv](https://docs.astral.sh/uv/getting-started/installation/) recommended):

```bash
git clone https://github.com/Pequi-Mecanico-Research/Eval_VLA.git
cd Eval_VLA
uv venv
source .venv/bin/activate
uv sync
```

### SAM3 weights

The package **does not include the SAM3 weights**. They are only needed by `success_rate` (`"model_segmentation"` mode) and `human_intervention_rate`.

1. Download `sam3.pt` from Meta's Hugging Face repository: https://huggingface.co/facebook/sam3 (access is gated: you must request access on the model page and be logged in).
2. Pass the file path to the metrics' `weights` parameter:

```python
stats.success_rate(..., model="sam3", weights="weights/sam3.pt")
operation.human_intervention_rate(video_path, model="sam3", weights="weights/sam3.pt")
```

If `weights` is not given, the package looks for `model_weights/<model>.pt` inside the package itself, which only exists when running from the repository with the weights placed there.

## Quickstart

The smallest evaluation: describe the setup once, create a metric family, call its methods during each episode and close the episode with `end_episode()`.

```python
import time

from eval_vla.config import SetupConfig
from eval_vla.metrics.statistical import Statistical

config = SetupConfig(  # saved next to every result, so runs stay comparable
    robot_embodiment={"manufacturer": "Trossen", "model": "WidowX", "dof": 6},
    camera_configuration={"count": 1, "positions": ["front"], "resolution": [640, 480]},
    action_space={"type": "end_effector_pose", "dim": 7},
    control_frequency=10.0,
    object_set={"count": 1, "materials": ["plastic"], "seen_in_training": True},
    scene_layout={"lighting": "indoor"},
    reset_procedure="manual",
    success_criteria="cube inside the tote",
    model_name="openpi-finetune",
    experiment_name="quickstart",
    task_name="Organize_the_table",
)

stats = Statistical(config)  # creates eval_results/quickstart_<date>/

for episode in range(3):
    start = time.perf_counter()
    for step in range(50):
        t0 = time.perf_counter()
        action = my_policy(observation)  # your policy call
        stats.inference_latency(t0, time.perf_counter(), steps_per_episode=50)
    stats.task_completion_time(start, time.perf_counter())
    stats.end_episode()  # writes results.json, appends to episodes.csv, refreshes the plots
```

## Results and plots

Every metric family writes to its own folder (or to a shared `run_dir`, see below):

| File | Content |
|---|---|
| `results.json` | Setup config, number of episodes and the latest value of every metric. Overwritten on each `end_episode()`. |
| `episodes.csv` | One row per episode with every metric value. |
| `*.png` | Plots generated from the CSV, refreshed on each `end_episode()`. |

To keep several families in one folder, pass the same `run_dir` and a different `name` to each one. Files are then prefixed: `<name>_results.json`, `<name>_episodes.csv`.

```python
stats = Statistical(config, run_dir=run_dir, name="statistical")
operation = Operation(config, episodes=n_episodes, run_dir=run_dir, name="operation")
```

## Examples

The `examples/` folder has runnable scripts (run them from the repository root):

| Script | What it does |
|---|---|
| `full_eval.py` | Every metric family on the same episodes, with camera, arm and live viewer: `uv run -m examples.full_eval --n_episodes 3 --steps_per_episode 50 --ip 192.168.1.2` |
| `statistical_eval_image_diff.py` | `Statistical` with success detected by image difference |
| `statistical_eval_detect_model.py` | `Statistical` with success detected by a segmentation model |
| `operation_eval.py` | `Operation` metrics on the WidowX arm |
| `failure_analysis_eval.py` | Failure type of a recorded episode: `python examples/failure_analysis_eval.py <episode.mp4> --window 30` |
| `widowx_driver_probe.py` | Read-only probe of what the WidowX driver exposes |

## Full evaluation on a real robot

This is the loop from `examples/full_eval.py`, with every family running on the same episodes. It has three parts: configure the setup, call the metrics during each episode, and close each episode.

<details>
<summary><b>1. Configure the setup</b></summary>

```python
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from examples.operation_eval import JOINT_NAMES
from examples.statistical_eval_image_diff import CALIBRATION_DIR, fake_policy, load_or_capture_reference
from examples.widowx_driver_probe import connect_driver
from eval_vla.config import SetupConfig
from eval_vla.metrics.failure_analysis import FailureAnalysis
from eval_vla.metrics.long_horizon import LongHorizon
from eval_vla.metrics.operation import Operation
from eval_vla.metrics.statistical import Statistical
from eval_vla.vision.camera import Camera
from eval_vla.vision.roi_selector import RoiSelector
from eval_vla.vision.viewer import make_viewer

start_eval_time = time.perf_counter()  # marks the start of the full evaluation

config = SetupConfig(  # environment/robot parameters, saved alongside every result
    robot_embodiment={"manufacturer": "Trossen", "model": "WidowX", "dof": 6, "end_effector": "Precision Gripper"},
    camera_configuration={"count": 3, "positions": ["wrist", "front", "high"], "resolution": [640, 480], "wrist": True},
    action_space={"type": "end_effector_pose", "dim": 7},
    control_frequency=10.0,  # Hz of the hardware in use
    object_set={"count": 1, "materials": ["plastic"], "seen_in_training": True},
    scene_layout={"workspace_m": "Grey Table with 100 meters", "lighting": "indoor"},
    reset_procedure="VLA",  # how the scene is reset between episodes
    success_criteria="objects inside the tots",
    model_name="openpi-finetune",  # format: base_model-finetune
    experiment_name="Full_eval_Complete",  # also names the results folder
    task_name="Organize_the_table",  # one of the Statistical class tasks or a new one
    perturbation_type=None,  # list of: object, layout, camera, lighting, background, initial_state, sensor_noise
)

# one folder for the whole run: <name>_episodes.csv / <name>.png per family
run_dir = Path("eval_results") / f"{config.experiment_name}_{datetime.now():%d-%m_%H:%M:%S}"
```

</details>

<details>
<summary><b>2. Instantiate the metrics and evaluate during each episode</b></summary>

```python
n_episodes, steps_per_episode = 5, 50  # number of episodes and policy steps per episode
SAM3_WEIGHTS = "weights/sam3.pt"  # SAM3 weights, downloaded from https://huggingface.co/facebook/sam3

stats = Statistical(config, run_dir=run_dir, name="statistical")  # name prefixes the family's files
operation = Operation(config, episodes=n_episodes, run_dir=run_dir, name="operation")
long_horizon = LongHorizon(config, run_dir=run_dir, name="long_horizon")

camera = Camera(0).open()  # 0 = camera index
view = make_viewer("rerun", camera)  # live view of the robot signals
driver = connect_driver("192.168.1.2")  # arm driver IP

# reference image and regions of interest (calibrated once, then reused)
reference_image = load_or_capture_reference(camera, CALIBRATION_DIR / config.task_name / "reference.png")
rois = RoiSelector.load_or_create(CALIBRATION_DIR / config.task_name / "rois.json", ["table", "container"], reference_image)

for ep in range(n_episodes):
    video_path = run_dir / f"ep{ep}.mp4"
    before_frame = camera.capture(run_dir / f"ep{ep}_before.png")  # scene before the task
    camera.start_recording(video_path, preview=view.preview)

    start_time = time.perf_counter()
    for step in range(steps_per_episode):
        inference_start = time.perf_counter()
        fake_policy(obs=None)  # replace with your policy call
        stats.inference_latency(inference_start, time.perf_counter(), steps_per_episode=steps_per_episode)

        external_efforts = dict(zip(JOINT_NAMES, (round(e, 2) for e in driver.get_all_external_efforts())))
        operation.collision_rate(driver, mode="gmo")  # collision detection via the generalized momentum observer (GMO)
        operation.jerks_rate(driver)

        view.update(
            external_effort=external_efforts,
            gmo_residual=dict(zip(JOINT_NAMES, operation.gmo_residual)),
            jerk_peaks=dict(zip([*JOINT_NAMES, "linear", "angular"], operation.jerks_peaks)),
        )

    stats.task_completion_time(start_time, time.perf_counter())

    camera.wait_for_enter(f"Episode {ep}: add the objects to the tot before finishing the task, then press Enter...", on_tick=view.update)
    after_frame = camera.capture(run_dir / f"ep{ep}_after.png")  # scene after the task
    camera.stop_recording()

    # success measured by comparing before/after with segmentation (SAM3) inside the ROIs
    stats.success_rate(config.task_name, "model_segmentation", before_frame, after_frame, prompt=["Cube", "black clock cube"], rois=rois, model="sam3", weights=SAM3_WEIGHTS)
    operation.reset_failure_rate(driver)
    operation.human_intervention_rate(video_path, prompt=["hand", "human arm"], model="sam3", weights=SAM3_WEIGHTS)  # detects hand/arm in the video

    subtasks_completed = [bool(stats.success_rate_value)]  # here the task has a single subtask
    long_horizon.subtask_completion_rate(subtasks_completed)
    long_horizon.average_completed_sequence_length(subtasks_completed)
    long_horizon.failure_position_distribution(subtasks_completed)
```

</details>

<details>
<summary><b>3. Close each episode and save logs and plots</b></summary>

```python
    # still inside the episode loop
    stats.end_episode()  # overwrites the JSON, appends a CSV row and refreshes the plots
    operation.end_episode()
    long_horizon.end_episode()

    if not stats.success_rate_value:  # only analyze the failure cause when the episode failed
        analysis = FailureAnalysis(config, video_path, window_s=30.0, debug_video=False,  # window_s: analyzed video window (s)
                                   run_dir=run_dir, name="failure_analysis")
        print(f"Episode {ep} failure type:", analysis.failure_type)
        analysis.end_episode(episode=ep + 1)  # episode is 1-based in the log

# outside the loop: release resources
view.close()
camera.close()
```

</details>

At the end, `run_dir` contains, for each family, `<name>_results.json`, `<name>_episodes.csv` and its plots.

## Adding a new metric class

1. Create `src/eval_vla/metrics/<name>.py` inheriting from `EvalVLA`:

   ```python
   from pathlib import Path

   from ..base import EvalVLA, metric
   from ..config import SetupConfig
   from ..plots import plot_my_family  # optional: function that reads the CSV and saves the plot


   class MyFamily(EvalVLA):
       def __init__(self, config: SetupConfig, output_dir: str | Path = "eval_results",
                    run_dir: str | Path | None = None, name: str | None = None):
           super().__init__(config, output_dir, run_dir, name)  # creates the folder and sets json_path/csv_path

       def _save_graphs(self) -> None:  # called automatically by end_episode()
           plot_my_family(self.csv_path, self.output_dir, self.config.experiment_name)

       @metric  # stores the return value in self._results and writes it to the JSON/CSV
       def my_metric(self, value: float) -> dict[str, float]:
           return {"my_metric": value}  # dict keys become CSV columns
   ```

2. Use the class like the others, calling `end_episode()` at the end of each episode:

   ```python
   my_family = MyFamily(config, run_dir=run_dir, name="my_family")
   my_family.my_metric(0.9)
   my_family.end_episode()
   ```

Methods decorated with `@metric` must return a `dict`; it is saved to `<name>_results.json` and becomes a row of `<name>_episodes.csv` on each `end_episode()`.
