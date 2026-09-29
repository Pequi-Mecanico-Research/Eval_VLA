from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SetupConfig:
    """
    Environment and robot parameters that must be reported with every result.
    
    Robot embodiment (manufacturer, model, DoF, end-effector type).

    Camera configuration (number, mounting positions, resolution, whether wrist cameras are used).

    Action space definition (joint position, joint velocity, end-effector pose, or hybrid).

    Control frequency (Hz) on the deployed hardware.

    Object set (number of objects, material categories, whether objects are seen or unseen during training).

    Scene layout (workspace dimensions, fixture positions, lighting conditions).

    Reset procedure (manual, semi-automatic, or automatic; degree of randomization per trial).

    Success criteria (explicit definition, including tolerance thresholds).

    Model Name (base_model_name-finetune)

    Experiment Name (task-experiment-date(d/m/year))

    """

    robot_embodiment: Mapping[str, Any]
    camera_configuration: Mapping[str, Any]
    action_space: Mapping[str, Any]
    control_frequency: float  #
    object_set: Mapping[str, Any]
    scene_layout: Mapping[str, Any]
    reset_procedure: str
    success_criteria: str
    model_name: str
    experiment_name: str