"""Graphs of the episode CSVs written by `EvalVLA.end_episode` (one PNG per metric family).

    plot_statistical(csv_path, out_dir)       -> statistical.png
    plot_operation(csv_path, out_dir)         -> operation.png
    plot_failure_analysis(csv_path, out_dir)  -> failure_analysis.png
    plot_long_horizon(csv_path, out_dir)      -> long_horizon.png

Panels whose columns are missing from the CSV (e.g. metrics not implemented yet) show "no data yet".
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path

import matplotlib
import numpy as np
from cycler import cycler
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .success_detection.vlm_failure import FAILURE_DESCRIPTIONS

COLORS = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
          "sky": "#56B4E9", "purple": "#CC79A7", "grey": "#7F7F7F"}  # Okabe-Ito (color-blind safe)

STYLE = {
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "axes.prop_cycle": cycler(color=list(COLORS.values())),
}


def _parse(value: str):
    """CSV cell -> number, bool, list, None or the original string."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value or None


def _read_csv(path: str | Path) -> dict[str, list]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {key: [_parse(row[key]) for row in rows] for key in (rows[0] if rows else {})}


def _floats(values: list) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values])


def _has(data: dict, *columns: str) -> bool:
    return all(column in data for column in columns)


def _empty(ax, title: str, message: str = "no data yet") -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=COLORS["grey"], transform=ax.transAxes)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _episodes(ax) -> None:
    ax.set_xlabel("episode")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _save(fig: Figure, out_dir: str | Path, name: str) -> Path:
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path)
    return path


def _wilson(successes: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% binomial confidence interval (Wilson score)."""
    p = successes / n
    center = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return center - half, center + half


def plot_statistical(csv_path: str | Path, out_dir: str | Path, title: str = "") -> Path:
    data = _read_csv(csv_path)

    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(15, 4.2), layout="constrained")
        fig.suptitle(title)
        ax_success, ax_time, ax_latency = fig.subplots(1, 3)
        episodes = np.array(data.get("episode", []))

        if _has(data, "successes"):
            outcome = np.array([bool(v) for v in data["successes"]], dtype=float)
            n = np.arange(1, len(outcome) + 1)
            successes = np.cumsum(outcome)
            low, high = _wilson(successes, n)
            ax_success.fill_between(episodes, low, high, color=COLORS["blue"], alpha=0.2, label="95% CI (Wilson)")
            ax_success.plot(episodes, successes / n, marker="o", color=COLORS["blue"], label="cumulative rate")
            ax_success.scatter(episodes, outcome, s=28, zorder=3,
                               color=[COLORS["green"] if o else COLORS["red"] for o in outcome])
            ax_success.set(ylim=(-0.05, 1.05), ylabel="success rate", title="Success rate")
            ax_success.legend(loc="lower right")
            _episodes(ax_success)
        else:
            _empty(ax_success, "Success rate")

        if _has(data, "task_completion_sec_list", "task_completion_sec_mean", "task_completion_sec_std"):
            mean = _floats(data["task_completion_sec_mean"])[-1]
            std = _floats(data["task_completion_sec_std"])[-1]
            ax_time.bar(episodes, _floats(data["task_completion_sec_list"]), width=0.6, color=COLORS["orange"])
            ax_time.axhline(mean, color=COLORS["grey"], ls="--", label=f"mean {mean:.2f} s")
            ax_time.axhspan(mean - std, mean + std, color=COLORS["grey"], alpha=0.15, label=f"std {std:.2f} s")
            ax_time.set(ylabel="seconds", title="Task completion time")
            ax_time.legend()
            _episodes(ax_time)
        else:
            _empty(ax_time, "Task completion time")

        if _has(data, "inference_latency_sec_list", "inference_latency_sec_mean", "inference_latency_sec_p95"):
            ax_latency.plot(episodes, _floats(data["inference_latency_sec_list"]) * 1000, marker="o", label="last step")
            ax_latency.plot(episodes, _floats(data["inference_latency_sec_mean"]) * 1000, ls="--", label="mean")
            ax_latency.plot(episodes, _floats(data["inference_latency_sec_p95"]) * 1000, ls=":", label="p95")
            ax_latency.set(ylabel="milliseconds", title="Inference latency")
            ax_latency.legend()
            _episodes(ax_latency)
        else:
            _empty(ax_latency, "Inference latency")

        return _save(fig, out_dir, "statistical")


def _joint_counts(ax, data: dict, column: str, limit_column: str, title: str, ylabel: str, gripper: bool = False) -> None:
    """Largest count of each joint over the episodes, against the count that flags an episode."""
    rows = [v for v in data.get(column, []) if v]
    if not rows or not _has(data, limit_column):
        return _empty(ax, title)

    counts = np.max(np.array(rows, dtype=int), axis=0)
    limit = int(data[limit_column][-1])
    x = np.arange(len(counts))
    names = [f"joint_{i}" for i in range(len(counts) - gripper)] + ["gripper"] * gripper
    ax.bar(x, counts, width=0.6, color=[COLORS["red"] if c >= limit else COLORS["blue"] for c in counts])
    ax.axhline(limit, color="black", lw=2, label="detection")
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set(title=title, ylabel=ylabel, ylim=(0, max(counts.max(), limit) * 1.3))
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(loc="upper right")


def plot_operation(csv_path: str | Path, out_dir: str | Path, title: str = "") -> Path:
    data = _read_csv(csv_path)

    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(15, 8), layout="constrained")
        fig.suptitle(title)
        axes = fig.subplots(2, 3)
        episodes = np.array(data.get("episode", []))

        ax = axes[0, 0]
        rates = [("reset_failure_rate", "reset failure"),
                 ("human_intervention_rate", "human intervention")]
        shown = [(column, label) for column, label in rates if column in data]
        if shown:
            for column, label in shown:
                ax.plot(episodes, _floats(data[column]), marker="o", label=label)
            ax.set(ylim=(-0.05, 1.05), ylabel="fraction of trials", title="Operation rates")
            ax.legend()
            _episodes(ax)
        else:
            _empty(ax, "Operation rates")

        _joint_counts(axes[0, 1], data, "collision_streaks", "collision_min_samples",
                      "Collision: longest streak (worst episode)", "consecutive samples")
        _joint_counts(axes[0, 2], data, "jerks_exceedances", "jerks_min_occurrences",
                      "Jerk occurrences (worst episode)", "occurrences", gripper=True)

        ax = axes[1, 0]
        if _has(data, "human_intervention_sec"):
            ax.bar(episodes, _floats(data["human_intervention_sec"]), width=0.6, color=COLORS["purple"])
            ax.set(ylabel="seconds", title="Human intervention time")
            _episodes(ax)
        else:
            _empty(ax, "Human intervention time")

        ax = axes[1, 1]
        if _has(data, "reset_max_joint_error"):
            failed = [bool(v) for v in data.get("reset_failed", [False] * len(episodes))]
            ax.bar(episodes, _floats(data["reset_max_joint_error"]), width=0.6,
                   color=[COLORS["red"] if f else COLORS["green"] for f in failed])
            ax.set(ylabel="max joint error (rad)", title="Reset error (red = failed)")
            _episodes(ax)
        else:
            _empty(ax, "Reset error")

        ax = axes[1, 2]
        events = {"collision": data.get("collision_detected"), "jerk": data.get("jerks_detected"),
                  "reset failed": data.get("reset_failed"),
                  "human": [bool(v) for v in data["human_interventions"]] if "human_interventions" in data else None}
        events = {label: [bool(v) for v in values] for label, values in events.items() if values}
        if events:
            ax.imshow(np.array(list(events.values()), dtype=float), aspect="auto", vmin=0, vmax=1,
                      cmap=ListedColormap(["#EEEEEE", COLORS["red"]]))
            ax.set_yticks(range(len(events)), list(events))
            ax.set_xticks(range(len(episodes)), [str(e) for e in episodes])
            ax.set(xlabel="episode", title="Events (red = occurred)")
            ax.grid(False)
        else:
            _empty(ax, "Events")

        return _save(fig, out_dir, "operation")


def plot_failure_analysis(csv_path: str | Path, out_dir: str | Path, title: str = "") -> Path:
    data = _read_csv(csv_path)

    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(7, 4.2), layout="constrained")
        fig.suptitle(title)
        ax = fig.subplots()

        if _has(data, "vlm_failure_type"):
            types = data["vlm_failure_type"]
            categories = list(FAILURE_DESCRIPTIONS)
            counts = [types.count(c) for c in categories]
            bars = ax.barh(categories, counts, color=COLORS["blue"])
            ax.bar_label(bars, padding=3)
            ax.invert_yaxis()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set(xlabel="episodes", title=f"Failure type (n={len(types)}, unclassified={sum(t not in categories for t in types)})")
        else:
            _empty(ax, "Failure type")

        return _save(fig, out_dir, "failure_analysis")


def plot_long_horizon(csv_path: str | Path, out_dir: str | Path, title: str = "") -> Path:
    data = _read_csv(csv_path)

    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(15, 4.2), layout="constrained")
        fig.suptitle(title)
        ax_rate, ax_length, ax_position = fig.subplots(1, 3)
        episodes = np.array(data.get("episode", []))

        for ax, column, label, name in ((ax_rate, "subtask_completion_rate", "fraction of subtasks", "Subtask completion rate"),
                                        (ax_length, "average_completed_sequence_length", "subtasks", "Average completed sequence length")):
            if _has(data, column):
                ax.plot(episodes, _floats(data[column]), marker="o", color=COLORS["blue"])
                ax.set(ylabel=label, title=name)
                _episodes(ax)
            else:
                _empty(ax, name)

        positions = [p for p in data.get("failure_position", []) if p]
        if positions:
            ax_position.hist(positions, bins=np.arange(1, max(positions) + 2) - 0.5, rwidth=0.8, color=COLORS["red"])
            ax_position.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_position.set(xlabel="subtask index", ylabel="episodes", title="Failure positions")
        else:
            _empty(ax_position, "Failure positions")

        return _save(fig, out_dir, "long_horizon")
