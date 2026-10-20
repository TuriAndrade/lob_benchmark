import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression, Ridge
from .tabular_model import TabularModel, _validation_loss


class LinearModel(TabularModel):
    def __init__(self, task="regression", regularization=1.0, max_iter=1000):
        if task not in ("regression", "classification"):
            raise ValueError(f"Unknown task: {task}")
        if regularization <= 0:
            raise ValueError("regularization must be positive")

        self.task = task
        self.model = (
            Ridge(alpha=regularization)
            if task == "regression"
            else LogisticRegression(C=1.0 / regularization, max_iter=max_iter)
        )

    def fit(self, X, y, X_val=None, y_val=None):
        if self.task == "classification" and not np.array_equal(
            np.unique(y), [0, 1, 2]
        ):
            raise ValueError("Training labels must contain classes 0, 1, and 2")

        self.model.fit(X, y)

        val_loss = None
        if X_val is not None and y_val is not None:
            val_loss = _validation_loss(self.task, y_val, self.predict(X_val))

        return val_loss, None

    def predict(self, X):
        if self.task == "regression":
            return self.model.predict(X).reshape(-1)
        return self.model.predict_proba(X)

    def save(self, save_path):
        joblib.dump(self, save_path)

    @staticmethod
    def load(path):
        return joblib.load(path)
