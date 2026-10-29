"""Training and evaluation for scikit-learn-compatible tabular models."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal

import numpy as np

from ..models import TabularModel
from .metrics import StreamingMetrics, compute_metrics


@contextmanager
def _disk_arrays(dataset, temp_dir=None):
    """Stage streamed observations in disk-backed arrays for model.fit()."""
    with TemporaryDirectory(prefix="lob_tabular_", dir=temp_dir) as directory:
        x_path, y_path = Path(directory) / "X.bin", Path(directory) / "y.bin"
        count, features, dtype = 0, None, None

        with x_path.open("wb") as xf, y_path.open("wb") as yf:
            for x, y, _ in dataset.stream():
                x, y = np.asarray(x, np.float32), np.asarray(y)
                if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
                    raise ValueError("Expected X=(B, F), y=(B,)")
                if features is None:
                    features, dtype = x.shape[1], y.dtype
                elif x.shape[1] != features or y.dtype != dtype:
                    raise ValueError("Inconsistent feature dimensions or label dtype")
                np.ascontiguousarray(x).tofile(xf)
                np.ascontiguousarray(y).tofile(yf)
                count += len(y)

        if not count:
            raise ValueError("Dataset produced no samples")

        x = np.memmap(x_path, dtype=np.float32, mode="r", shape=(count, features))
        y = np.memmap(y_path, dtype=dtype, mode="r", shape=(count,))
        try:
            yield x, y
        finally:
            del x, y


@dataclass
class TabularTrainerConfig:
    model: TabularModel
    task: str = "regression"
    verbose: bool = True
    data_mode: Literal["load", "stream"] = "load"
    temp_dir: str | None = None


class TabularTrainer:
    def __init__(self, config: TabularTrainerConfig):
        if config.data_mode not in ("load", "stream"):
            raise ValueError("data_mode must be load or stream")
        if config.task != config.model.task:
            raise ValueError("Trainer and model tasks differ")
        self.config, self.model, self.task = config, config.model, config.task

    def _check(self, dataset):
        if dataset.task != self.task:
            raise ValueError("Dataset and trainer tasks differ")

    def fit(self, train, val=None):
        self._check(train)
        if val is not None:
            self._check(val)
        start = perf_counter()
        with ExitStack() as stack:
            if self.config.data_mode == "load":
                x_train, y_train, _ = train.load()
                x_val, y_val, _ = val.load() if val is not None else (None, None, None)
            else:
                x_train, y_train = stack.enter_context(
                    _disk_arrays(train, self.config.temp_dir)
                )
                x_val, y_val = (
                    stack.enter_context(_disk_arrays(val, self.config.temp_dir))
                    if val is not None
                    else (None, None)
                )
            val_loss, best_iteration = self.model.fit(x_train, y_train, x_val, y_val)

        result = {"val_loss": val_loss, "best_iteration": best_iteration}
        if self.config.verbose:
            print(f"Training finished in {perf_counter() - start:.1f}s: {result}")
        return result

    def eval(self, dataset):
        self._check(dataset)
        if self.config.data_mode == "load":
            x, y, _ = dataset.load()
            predictions = self.model.predict(x)
            return {
                "metrics": compute_metrics(self.task, y, predictions),
                "n_samples": len(y),
            }

        metrics = StreamingMetrics(self.task)
        for x, y, _ in dataset.stream():
            metrics.update(y, self.model.predict(x))
        return {"metrics": metrics.result(), "n_samples": int(metrics.stats[0].item())}

    def save(self, path):
        self.model.save(path)
