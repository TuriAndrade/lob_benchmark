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
