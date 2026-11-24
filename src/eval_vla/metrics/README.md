# Metrics

Each metric class inherits from `EvalVLA`, and each method decorated with `@metric` is a metric.

Structure used for each method:

- **Purpose:**
- **What it does:**
- **Inputs:**
- **Returns:**

---

## Statistical

`statistical.py`: statistical report covering success rate, task completion time (mean and standard deviation, in seconds) and inference latency on the hardware in use (mean and p95).

### `success_rate`

- **Purpose:** Check whether the policy moved the objects from the table into the container.
- **What it does:** Counts the objects in the ROIs before and after the policy acts and compares them. It has two modes (`task_mode`):
  - `"model_segmentation"`: counts the `prompt` objects inside the ROIs using `model`, in `before_frame` and `after_frame`. Requires `prompt` and `rois`.
  - `"image_diff"`: counts the objects by comparing each frame with `reference_frame` (empty table). Requires `reference_frame` and `rois`.
- **Inputs:**

  | Variable | Description |
  |---|---|
  | `task_name` | Task being evaluated (currently only `"Organize_the_table"`). |
  | `task_mode` | `"model_segmentation"` or `"image_diff"`. |
  | `before_frame` | Image of the table right before the policy acts. |
  | `after_frame` | Image of the table right after the policy acts. |
  | `prompt` | Names/descriptions of the objects to segment. |
  | `reference_frame` | Image of the empty table, used as the baseline in `"image_diff"` mode. |
  | `rois` | Regions of interest where the objects start (table) and end up (container). |
  | `model` | Segmentation model used in `"model_segmentation"` mode. |
  | `debug_plots` | Saves the debug images of `"image_diff"` mode. |
  | `weights` | Path to the weights file (e.g. `sam3.pt`) of `model`, used in `"model_segmentation"`. If `None`, looks for `model_weights/<model>.pt` inside the package, which is not shipped with `pip install`; download it from https://huggingface.co/facebook/sam3 (see the root README). Only read the first time the detector is created. |

- **Returns:** dict with `successes` (bool), `objects_on_table_before` and `objects_in_container_after` (int, or `None` for unknown tasks).

### `task_completion_time`

- **Purpose:** Measure how long the policy took to complete the episode.
- **What it does:** Accumulates episode durations and computes mean, median and standard deviation. Call once per episode, after the policy stops acting.
- **Inputs:**

  | Variable | Description |
  |---|---|
  | `start_time` | Episode start, from `time.perf_counter()`, in seconds. |
  | `end_time` | Episode end, from `time.perf_counter()`, in seconds. |
  | `precision` | Decimal places used to round the returned values. |

- **Returns:** dict with `task_completion_sec_list` (this episode's duration) and `task_completion_sec_mean`, `_median` and `_std` over all episodes so far, in seconds.

### `inference_latency`

- **Purpose:** Measure the policy's inference time at each step.
- **What it does:** Accumulates latencies over the episode and computes mean, median, standard deviation and p95. Call once per step, wrapping only the policy call.
- **Inputs:**

  | Variable | Description |
  |---|---|
  | `inference_time_start` | Time right before the policy call, from `time.perf_counter()`, in seconds. |
  | `inference_time_end` | Time right after the policy call, from `time.perf_counter()`, in seconds. |
  | `steps_per_episode` | Number of policy steps in an episode, used to reset the accumulated latencies on the next episode. |
  | `precision` | Decimal places used to round the returned values. |

- **Returns:** dict with `inference_latency_sec_list` (this step's latency) and `inference_latency_sec_mean`, `_median`, `_std` and `_p95` of the accumulated latencies, in seconds.

---

## Operation

`operation.py`: operational metrics covering human intervention rate, collision rate, jerk rate (abrupt movements) and reset failure rate.

### `human_intervention_rate`

- **Purpose:** Detect whether a human entered the scene during the episode, i.e. had to intervene manually.
- **What it does:** Segments the episode video looking for what is in `prompt` and returns the time intervals with a human present. Call once per episode, on the recorded video.
- **Inputs:**

  | Variable | Description |
  |---|---|
  | `video_path` | Episode video. |
  | `model` | Segmentation model used to find the human in the video. |
  | `prompt` | Descriptions of what counts as human intervention (e.g. `"hand"`, `"human arm"`). |
  | `precision` | Decimal places used to round the returned values. |
  | `weights` | Path to the weights file (e.g. `sam3.pt`) of `model`. If `None`, looks for `model_weights/<model>.pt` inside the package, which is not shipped with `pip install`; download it from https://huggingface.co/facebook/sam3 (see the root README). Only read the first time the detector is created. |
  | `**detection_kwargs` | Extra arguments forwarded to `ObjectDetection.run`. |

- **Returns:** dict with `human_interventions` (number of intervals with a human), `human_intervention_sec` (total time with a human), `human_intervention_intervals` (list of (start, end) in seconds) and `human_intervention_rate` (fraction of `episodes` with at least one intervention so far).

### `collision_rate`

- **Purpose:** Detect unintended contact (collision) during the episode from the arm's joint torques.
- **What it does:** Call at every control step. Keeps the per-joint signal peak over the episode and flags a collision when any peak exceeds its threshold. Modes (`mode`):
  - `"gmo"`: generalized momentum observer. The residual estimates the external torque using the arm's dynamic model (URDF) and the measured positions, velocities and efforts.
  - `"driver"`: uses the external efforts already reported by the arm driver.
- **Inputs:**

  | Variable | Description |
  |---|---|
  | `driver` | Connected Trossen arm driver, read on every call. |
  | `mode` | `"gmo"` or `"driver"`. |
  | `thresholds` | Maximum accepted absolute value per arm joint, in Nm. |
  | `gain` | Observer gain K_O in `"gmo"` mode, in 1/s. Higher values react faster but add noise. |
  | `precision` | Decimal places used to round the returned values. |

- **Returns:** dict with `collision_mode`, `collision_detected` (in this episode), `collision_rate` (fraction of `episodes` with a collision so far), `collision_thresholds` and `collision_values` (peak per joint). Raises `ValueError` if `mode` is neither `"gmo"` nor `"driver"`.

### `jerks_rate`

- **Purpose:** Detect abrupt motion (sudden change in acceleration) during the episode.
- **What it does:** Call at every control step. Jerk is the derivative of the accelerations over the episode; it keeps the peak per joint and for the end effector, and flags the episode when any peak exceeds its threshold.
- **Inputs:**

  | Variable | Description |
  |---|---|
  | `driver` | Connected Trossen arm driver, read on every call. |
  | `joint_jerk_thresholds` | Maximum accepted jerk per joint, in rad/s³ (m/s³ for the gripper). |
  | `linear_jerk_threshold` | Maximum accepted linear jerk of the end effector, in m/s³. |
  | `angular_jerk_threshold` | Maximum accepted angular jerk of the end effector, in rad/s³. |
  | `precision` | Decimal places used to round the returned values. |

- **Returns:** dict with `jerks_detected` (in this episode), `jerks_rate` (fraction of episodes with jerks so far), `jerks_thresholds` and `jerks_values` (peaks per joint, then linear and angular).

### `reset_failure_rate`

- **Purpose:** Check whether the arm returned to the reset (sleep) pose, i.e. whether the reset procedure worked.
- **What it does:** Compares the current joint positions with the reset pose and flags a failure if any error exceeds the tolerance. Call once per episode, when the arm should be in the reset pose.
- **Inputs:**

  | Variable | Description |
  |---|---|
  | `driver` | Connected Trossen arm driver, read on the call. |
  | `sleep_positions` | Expected joint positions in the reset pose, in driver units. |
  | `tolerance` | Maximum accepted absolute error per joint; above it the reset counts as failed. |
  | `precision` | Decimal places used to round the returned values. |

- **Returns:** dict with `reset_failed` (in this episode), `reset_max_joint_error`, `reset_joint_errors` (per joint) and `reset_failure_rate` (fraction of episodes with a failed reset so far).

---

## LongHorizon

`long_horizon.py`: long-horizon tasks, covering subtask completion rate, average completed sequence length and the distribution of failure positions along the sequence. **None of the three methods is implemented yet.**

### `subtask_completion_rate`

- **Purpose:** Measure the fraction of individual subtasks completed, regardless of whether the full sequence succeeded.
- **What it does:** Call once per episode. *(Not implemented yet.)*
- **Inputs:**

  | Variable | Description |
  |---|---|
  | `subtasks_completed` | One bool per subtask, in execution order; `True` if that subtask was completed. |

- **Returns:** dict with the subtask completion rate. *(Not implemented yet.)*

### `average_completed_sequence_length`

- **Purpose:** Measure the average number of consecutive subtasks completed from the start of the sequence (CALVIN-style).
- **What it does:** Call once per episode. *(Not implemented yet.)*
- **Inputs:**

  | Variable | Description |
  |---|---|
  | `subtasks_completed` | One bool per subtask, in execution order; `True` if that subtask was completed. |

- **Returns:** dict with the average completed sequence length. *(Not implemented yet.)*

### `failure_position_distribution`

- **Purpose:** Show where in the sequence failures concentrate.
- **What it does:** Histogram of the subtask index (1-based) at which each trial first failed. Call once per episode. *(Not implemented yet.)*
- **Inputs:**

  | Variable | Description |
  |---|---|
  | `subtasks_completed` | One bool per subtask, in execution order; `True` if that subtask was completed. |

- **Returns:** dict with the distribution of failure positions. *(Not implemented yet.)*

---

## FailureAnalysis

`failure_analysis.py`: failure analysis. Classifies the episode into failure types: perception, language understanding, planning, grasping, control, recovery and hardware/calibration.

### `vlm_failure_type`

- **Purpose:** Find out, using a vision-language model (VLM, via Ollama), which failure type the episode video shows.
- **What it does:** Samples `vlm_frames` frames from the last `window_s` seconds of the video and sends them to the VLM together with the task name and the success criterion from the config. Runs automatically when `FailureAnalysis` is created.
- **Inputs:** The method takes no arguments; the inputs are set in `__init__`.

  | Variable (`__init__`) | Description |
  |---|---|
  | `video_path` | Episode video being analyzed. |
  | `window_s` | Final window of the video (in seconds) used in the analysis. |
  | `vlm_model` | Ollama model used (default `qwen3.5:latest`). |
  | `vlm_host` | Ollama server address. |
  | `vlm_frames` | Number of frames sampled for the VLM. |
  | `debug_video` | Saves a video with the frames sent to the VLM. |
  | `vlm_timeout_s` | VLM request timeout, in seconds. |
  | `vlm_num_ctx` | VLM context size (`num_ctx`). |

- **Returns:** dict with `vlm_failure_type` (name of a `FailureType`, or `None` if the reply was not recognized), `vlm_model` and `vlm_raw_reply` (the model's text reply).
