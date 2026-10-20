import numpy as np
import lightgbm as lgb
import joblib
from .tabular_model import TabularModel, _validation_loss


class LightGBMModel(TabularModel):
    def __init__(
        self,
        task="regression",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        reg_lambda=0.0,
        early_stopping_rounds=50,
        n_jobs=-1,
    ):
        if task not in ("regression", "classification"):
            raise ValueError(f"Unknown task: {task}")

        self.task = task
        self.early_stopping_rounds = early_stopping_rounds

        params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            verbosity=-1,
        )

        self.model = (
            lgb.LGBMRegressor(objective="regression", **params)
            if task == "regression"
            else lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                **params,
            )
        )

    def fit(self, X, y, X_val=None, y_val=None):
        if self.task == "classification" and not np.array_equal(
            np.unique(y), [0, 1, 2]
        ):
            raise ValueError("Training labels must contain classes 0, 1, and 2")

        has_val = X_val is not None and y_val is not None
        use_early_stopping = has_val and self.early_stopping_rounds > 0

        kwargs = {}
        if has_val:
            kwargs["eval_set"] = [(X_val, y_val)]
            kwargs["eval_metric"] = (
                "l2" if self.task == "regression" else "multi_logloss"
            )

            if use_early_stopping:
                kwargs["callbacks"] = [
                    lgb.early_stopping(
                        self.early_stopping_rounds,
                        verbose=False,
                    )
                ]

        self.model.fit(X, y, **kwargs)

        best_iteration = (
            self.model.best_iteration_
            if use_early_stopping
            else self.model.n_estimators_
        )

        val_loss = None
        if has_val:
            val_loss = _validation_loss(self.task, y_val, self.predict(X_val))

        return val_loss, int(best_iteration)

    def predict(self, X):
        if self.task == "regression":
            return self.model.predict(X).reshape(-1)
        return self.model.predict_proba(X)

    def save(self, save_path):
        joblib.dump(self, save_path)

    @staticmethod
    def load(path):
        return joblib.load(path)
