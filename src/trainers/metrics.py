"""Streaming, mergeable regression and three-class classification metrics."""

import torch


class StreamingMetrics:
    def __init__(self, task, device="cpu"):
        if task not in ("regression", "classification"):
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        # n, SSE, SAE, sum(y), sum(y²), log-loss sum, 3x3 confusion matrix
        self.stats = torch.zeros(15, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, y, predictions):
        y = torch.as_tensor(y, device=self.stats.device)
        predictions = torch.as_tensor(predictions, device=self.stats.device)
        if not y.numel():
            return
        n = len(y)
        self.stats[0] += n

        if self.task == "regression":
            values = y.double().reshape(-1)
            pred = predictions.double().reshape(-1)
            if pred.shape != values.shape:
                raise ValueError("Regression predictions must match targets")
            error = values - pred
            self.stats[1] += error.square().sum()
            self.stats[2] += error.abs().sum()
            self.stats[3] += values.sum()
            self.stats[4] += values.square().sum()
        else:
            if predictions.shape != (n, 3):
                raise ValueError("Classification predictions must have shape (N, 3)")
            truth = y.long().reshape(-1)
            probs = predictions.double().clamp(min=1e-15, max=1.0)
            probs = probs / probs.sum(dim=1, keepdim=True)
            self.stats[5] -= probs.gather(1, truth[:, None]).log().sum()
            predicted = probs.argmax(dim=1)
            self.stats[6:] += torch.bincount(
                truth * 3 + predicted, minlength=9
            ).double()

    def result(self):
        values = self.stats.cpu().tolist()
        n = values[0]
        if n <= 0:
            raise ValueError("No samples to evaluate")

        if self.task == "regression":
            sse, sae, y_sum, y_sq_sum = values[1:5]
            variance = max(0.0, y_sq_sum - y_sum * y_sum / n)
            r2 = (
                float("nan")
                if n < 2
                else (
                    1.0 - sse / variance if variance > 0 else (1.0 if sse == 0 else 0.0)
                )
            )
            return {"mse": sse / n, "mae": sae / n, "r2": r2}

        matrix = torch.tensor(values[6:], dtype=torch.float64).reshape(3, 3)
        tp = matrix.diag()
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        recall = torch.where(support > 0, tp / support.clamp_min(1), 0)
        f1 = 2 * tp / (support + predicted).clamp_min(1)
        return {
            "log_loss": values[5] / n,
            "accuracy": float(tp.sum() / n),
            "balanced_accuracy": float(recall[support > 0].mean()),
            "macro_f1": float(f1.mean()),
        }


def compute_metrics(task, y, predictions):
    metrics = StreamingMetrics(task)
    metrics.update(y, predictions)
    return metrics.result()
