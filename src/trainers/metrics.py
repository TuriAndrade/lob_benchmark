import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_metrics(task, y, predictions):
    if task == "regression":
        return {
            "mse": float(mean_squared_error(y, predictions)),
            "mae": float(mean_absolute_error(y, predictions)),
            "r2": float(r2_score(y, predictions)),
        }

    labels = np.argmax(predictions, axis=1)
    return {
        "log_loss": float(log_loss(y, predictions, labels=[0, 1, 2])),
        "accuracy": float(accuracy_score(y, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y, labels)),
        "macro_f1": float(
            f1_score(y, labels, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
    }


class StreamingMetrics:
    """Compute regression/classification metrics without storing all outputs."""

    def __init__(self, task):
        if task not in ("regression", "classification"):
            raise ValueError(f"Unknown task: {task}")

        self.task = task
        self.n = 0
        self.sse = self.sae = self.y_sum = self.y_sq_sum = 0.0
        self.loss_sum = 0.0
        self.confusion = np.zeros((3, 3), dtype=np.int64)

    def update(self, y, predictions):
        y = np.asarray(y)
        predictions = np.asarray(predictions)
        n = len(y)
        if n == 0:
            return

        self.n += n
        if self.task == "regression":
            actual = y.astype(np.float64)
            predicted = predictions.astype(np.float64).reshape(-1)
            if len(predicted) != n:
                raise ValueError("Incorrect regression prediction shape")

            error = actual - predicted
            self.sse += float(np.square(error).sum())
            self.sae += float(np.abs(error).sum())
            self.y_sum += float(actual.sum())
            self.y_sq_sum += float(np.square(actual).sum())
        else:
            if predictions.shape != (n, 3):
                raise ValueError(
                    "Classification predictions must have shape (batch, 3)"
                )

            classes = np.argmax(predictions, axis=1)
            self.confusion += np.bincount(
                y.astype(np.int64) * 3 + classes, minlength=9
            ).reshape(3, 3)
            self.loss_sum += float(log_loss(y, predictions, labels=[0, 1, 2])) * n

    def result(self):
        if self.n == 0:
            raise ValueError("Dataset produced no samples")

        if self.task == "regression":
            total_variance = self.y_sq_sum - self.y_sum**2 / self.n
            if self.n < 2:
                r2 = float("nan")
            elif total_variance <= 0:
                r2 = 1.0 if self.sse == 0 else 0.0
            else:
                r2 = 1.0 - self.sse / total_variance

            return {"mse": self.sse / self.n, "mae": self.sae / self.n, "r2": r2}

        true_positive = np.diag(self.confusion)
        support = self.confusion.sum(axis=1)
        predicted_support = self.confusion.sum(axis=0)
        recall = np.divide(
            true_positive,
            support,
            out=np.zeros(3, dtype=float),
            where=support > 0,
        )
        f1_denom = support + predicted_support
        f1 = np.divide(
            2 * true_positive,
            f1_denom,
            out=np.zeros(3, dtype=float),
            where=f1_denom > 0,
        )
        return {
            "log_loss": self.loss_sum / self.n,
            "accuracy": float(true_positive.sum() / self.n),
            "balanced_accuracy": float(recall[support > 0].mean()),
            "macro_f1": float(f1.mean()),
        }
