from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error


def _validation_loss(task, y_true, y_pred):
    if task == "regression":
        return float(mean_squared_error(y_true, y_pred))
    return float(log_loss(y_true, y_pred, labels=[0, 1, 2]))


class TabularModel(ABC):
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> tuple: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save(self, save_path) -> None: ...

    @staticmethod
    @abstractmethod
    def load(path) -> "TabularModel": ...
