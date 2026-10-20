from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal

import numpy as np

from models.tabular_model import TabularModel
from .metrics import StreamingMetrics, compute_metrics


@contextmanager
def _disk_arrays(dataset, temp_dir=None):
    """Write streamed batches to disk-backed arrays for standard model.fit()."""
    with TemporaryDirectory(prefix="lob_tabular_", dir=temp_dir) as directory:
        x_path = Path(directory) / "X.bin"
        y_path = Path(directory) / "y.bin"

        count = 0
        n_features = None
        y_dtype = None

        with x_path.open("wb") as x_file, y_path.open("wb") as y_file:
            for X, y, _ in dataset.stream():
                X = np.asarray(X, dtype=np.float32)
                y = np.asarray(y)

                if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
                    raise ValueError("Expected X=(batch, features), y=(batch,)")

                if n_features is None:
                    n_features = X.shape[1]
                    y_dtype = y.dtype
                elif X.shape[1] != n_features or y.dtype != y_dtype:
                    raise ValueError(
                        "Feature count or target dtype changed between batches"
                    )

                np.ascontiguousarray(X).tofile(x_file)
                np.ascontiguousarray(y).tofile(y_file)
                count += len(y)

        if count == 0:
            raise ValueError("Dataset produced no samples")

        X = np.memmap(x_path, dtype=np.float32, mode="r", shape=(count, n_features))
        y = np.memmap(y_path, dtype=y_dtype, mode="r", shape=(count,))
        try:
            yield X, y
        finally:
            del X, y


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
            raise ValueError("data_mode must be 'load' or 'stream'")

        self.config = config
        self.model = config.model
        self.task = config.task

    def fit(self, train, val=None):
        if self.config.verbose:
            print(f"Training {type(self.model).__name__}...")

        start = perf_counter()

        with ExitStack() as stack:
            if self.config.data_mode == "load":
                X_train, y_train, _ = train.load()
                if val is not None:
                    X_val, y_val, _ = val.load()
                else:
                    X_val = y_val = None
            else:
                X_train, y_train = stack.enter_context(
                    _disk_arrays(train, self.config.temp_dir)
                )
                if val is not None:
                    X_val, y_val = stack.enter_context(
                        _disk_arrays(val, self.config.temp_dir)
                    )
                else:
                    X_val = y_val = None

            val_loss, best_iteration = self.model.fit(X_train, y_train, X_val, y_val)

        result = {"val_loss": val_loss, "best_iteration": best_iteration}
        if self.config.verbose:
            print(f"Training finished in {perf_counter() - start:.1f}s: {result}")
        return result

    def eval(self, dataset):
        if self.config.verbose:
            print(f"Evaluating {type(self.model).__name__}...")

        start = perf_counter()

        if self.config.data_mode == "load":
            X, y, _ = dataset.load()
            predictions = self.model.predict(X)
            result = {
                "metrics": compute_metrics(self.task, y, predictions),
                "n_samples": len(y),
            }
        else:
            metrics = StreamingMetrics(self.task)
            for X, y, _ in dataset.stream():
                if len(y):
                    metrics.update(y, self.model.predict(X))
            result = {"metrics": metrics.result(), "n_samples": metrics.n}

        if self.config.verbose:
            print(f"Evaluation finished in {perf_counter() - start:.1f}s: {result}")
        return result

    def save(self, path):
        self.model.save(path)
