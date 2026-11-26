from .config import SetupConfig
from .wandb_logger import log_episode, start_run
from pathlib import Path
import json
import functools
import dataclasses
import csv
from datetime import datetime

def metric(fn):
    """Marks metódo from base class as a métric, and stores its result to use in the report"""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)
        self._results[fn.__name__] = result
        return result
    
    wrapper.__metric__ = True
    return wrapper

class EvalVLA:
    """
    Base classe for VLA Evaluation, which holds the default variables to be used.
    """

    def __init__(self, config: SetupConfig, output_dir: str | Path = "eval_results",
                 run_dir: str | Path | None = None, name: str | None = None) -> None:
        """`run_dir` makes several families share one folder; `name` then prefixes each one's results/CSV files."""
        self.config = config
        self.output_dir = Path(run_dir) if run_dir else Path(output_dir) / f"{config.experiment_name.replace('/', '-')}_{datetime.now():%d-%m_%H:%M:%S}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.output_dir / (f"{name}_results.json" if name else "results.json")
        self.csv_path = self.output_dir / (f"{name}_episodes.csv" if name else "episodes.csv")
        self.name = name or type(self).__name__.lower()
        self._results: dict[str, dict] = {}
        self._episode = 0
        self._history: dict[str, list[float]] = {}             
        self._wandb_run = start_run(config, self.output_dir.name) if config.wandb else None


    def episode_dir(self, episode: int | None = None) -> Path:
        """Folder `ep_<episode>` (0-based) with the files of one episode, created if missing.

        If `episode` is None, uses the episode being evaluated.
        """
        path = self.output_dir / f"ep_{self._episode if episode is None else episode}"
        path.mkdir(parents=True, exist_ok=True)
        return path


    def _save_json(self, folder: Path, episode: int) -> None:
        report = {
            "config": dataclasses.asdict(self.config),
            "episode": episode,
            "updated_at": datetime.now().isoformat(),
            "metrics": self._results,
        }
        path = folder / self.json_path.name
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(path)


    def _row(self, extra: dict) -> dict:
        row = {"episode": self._episode, **extra}
        for result in self._results.values():
            if result:
                row.update(result)
        return row


    def _append_csv(self, path: Path, row: dict) -> None:
        new_file = not path.exists()

        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)



    def _save_graphs(self) -> None:
        """Base implementation for graph analisys for each class."""
        pass


    def end_episode(self, **extra) -> None:
        """Call once per episode.

        Saves the episode metadata (JSON and one-row CSV) in its `ep_<n>` folder, appends it to the CSV of the
        whole run, refreshes the graphs of the run and, with `config.wandb`, sends them to Weights & Biases.
        The metric results are then cleared for the next episode.

        Args:
            **extra: Additional columns. `episode` (1-based) overrides the episode number, e.g. for an instance
                created for a single episode; its folder is `ep_<episode - 1>`.
        """
        self._episode += 1
        row = self._row(extra)
        folder = self.episode_dir(row["episode"] - 1)

        self._save_json(folder, row["episode"] - 1)
        episode_csv = folder / self.csv_path.name
        episode_csv.unlink(missing_ok=True)
        self._append_csv(episode_csv, row)
        self._append_csv(self.csv_path, row)

        graphs = True
        try:
            self._save_graphs()
        except Exception as exc:  
            graphs = False
            print(f"Could not save graphs: {exc!r}")

        if self._wandb_run is not None:
            graph = self.output_dir / f"{self.name}.png"
            try:
                log_episode(self._wandb_run, self.name, row, self._history, graph if graphs and graph.exists() else None)
            except Exception as exc: 
                print(f"Could not log to wandb: {exc!r}")

        self._results = {}
