"""Optional Weights & Biases logging, used when `SetupConfig.wandb` is True."""

from __future__ import annotations

import dataclasses
import numbers
from pathlib import Path

import numpy as np

from .config import SetupConfig


def start_run(config: SetupConfig, run_name: str):
    """The wandb run of this evaluation: created on the first call, shared by every metric family of the process.

    Raises:
        ValueError: If `config.wandb_project` is not set.
    """
    if not config.wandb_project:
        raise ValueError("SetupConfig(wandb=True) needs wandb_project")
    import wandb  # slow import, only paid with config.wandb

    if wandb.run is None:
        run = wandb.init(project=config.wandb_project, config=dataclasses.asdict(config), name=run_name)
        run.define_metric("episode")
        run.define_metric("*", step_metric="episode")  # families log the same episode in separate calls
    return wandb.run


def _scalars(row: dict) -> dict:
    """Columns as wandb values: lists of numbers (e.g. one per joint) become one value per position."""
    values = {}
    for key, value in row.items():
        if isinstance(value, (numbers.Number, str)):
            values[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, numbers.Number) for v in value):
            values.update({f"{key}/{i}": v for i, v in enumerate(value)})
    return values


def log_episode(run, family: str, row: dict, history: dict[str, list[float]], graph: Path | None = None) -> None:
    """Logs the episode `row` of a metric family, its graph, and the mean of each number over the episodes.

    The means go to the run summary, which is what the runs table of the project compares.
    """
    import wandb

    values = _scalars({k: v for k, v in row.items() if k != "episode"})
    data = {f"{family}/{key}": value for key, value in values.items()}
    if graph is not None:
        data[f"{family}/graphs"] = wandb.Image(str(graph))
    run.log({"episode": row["episode"], **data})

    for key, value in values.items():
        if isinstance(value, numbers.Number):
            history.setdefault(key, []).append(float(value))
            run.summary[f"{family}/{key}_mean"] = float(np.mean(history[key]))
