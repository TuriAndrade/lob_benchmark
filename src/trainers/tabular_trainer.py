from dataclasses import dataclass
from time import perf_counter

from models.tabular_model import TabularModel
from .metrics import compute_metrics


@dataclass
class TabularTrainerConfig:
    model: TabularModel
    task: str = "regression"
    verbose: bool = True


class TabularTrainer:
    def __init__(self, config: TabularTrainerConfig):
        self.config = config
        self.model = config.model
        self.task = config.task

    def fit(self, train, val=None):
        if self.config.verbose:
            print(f"Training {type(self.model).__name__}...")

        start = perf_counter()
        X_train, y_train, _ = train.load()

        if val is not None:
            X_val, y_val, _ = val.load()
        else:
            X_val = y_val = None

        val_loss, best_iteration = self.model.fit(X_train, y_train, X_val, y_val)

        result = {
            "val_loss": val_loss,
            "best_iteration": best_iteration,
        }

        if self.config.verbose:
            print(f"Training finished in {perf_counter() - start:.1f}s: {result}")

        return result

    def eval(self, dataset):
        if self.config.verbose:
            print(f"Evaluating {type(self.model).__name__}...")

        start = perf_counter()
        X, y, _ = dataset.load()
        predictions = self.model.predict(X)

        result = {
            "metrics": compute_metrics(self.task, y, predictions),
            "n_samples": len(y),
        }

        if self.config.verbose:
            print(f"Evaluation finished in {perf_counter() - start:.1f}s: {result}")

        return result

    def save(self, path):
        self.model.save(path)
