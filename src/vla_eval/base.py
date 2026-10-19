from .config import SetupConfig
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

    def __init__(self, config: SetupConfig, output_dir: str | Path = "eval_results") -> None:
        self.config = config
        self.output_dir = Path(output_dir) / config.experiment_name.replace("/", "-")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.output_dir / "results.json"
        self.csv_path = self.output_dir / "episodes.csv"
        self._results: dict[str, dict] = {}
        self._episode = 0


    def _save_json(self) -> None:
        report = {
            "config": dataclasses.asdict(self.config),
            "episodes_completed": self._episode,
            "updated_at": datetime.now().isoformat(),
            "metrics": self._results,
        }
        tmp = self.json_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(self.json_path)


    def _append_csv(self, extra: dict) -> None:
        row = {"episode": self._episode, **extra}
        for name, result in self._results.items():
            row.update(result)  
        new_file = not self.csv_path.exists()

        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)


    def end_episode(self, **extra) -> None:
        """Call once per episode: overwrites the JSON and appends to the CSV."""
        self._episode += 1
        self._save_json()
        self._append_csv(extra)
