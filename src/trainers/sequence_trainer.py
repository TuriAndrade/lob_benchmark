from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.sequence_model import SequenceModel
from .optimizer import AdamW
from .metrics import compute_metrics


@dataclass
class SequenceTrainerConfig:
    model: SequenceModel
    task: str = "regression"
    epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0

    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 0
    start_lr: float = 1e-6
    final_lr: float = 1e-6
    steps_per_epoch: int | None = None

    grad_clip: float | None = 1.0
    device: str = "auto"
    num_workers: int = 0
    verbose: bool = True


class SequenceTrainer:
    def __init__(self, config: SequenceTrainerConfig):
        self.config = config
        self.model = config.model
        self.task = config.task
        device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto"
            else config.device
        )
        self.device = torch.device(device)

        self.model.to(self.device)
        self.loss_fn = (
            nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()
        )

    def _loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.config.num_workers > 0,
        )

    def _run_epoch(self, loader, optimizer=None, lr_scheduler=None):
        training = optimizer is not None
        self.model.train(training)

        total_loss = 0.0
        total_samples = 0
        steps = 0

        batches = tqdm(
            loader,
            desc="Train" if training else "Validation",
            leave=False,
            disable=not self.config.verbose,
        )

        with torch.set_grad_enabled(training):
            for x, y, _ in batches:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if training:
                    optimizer.zero_grad(set_to_none=True)

                output = self.model(x)
                loss = self.loss_fn(output, y)

                if training:
                    loss.backward()

                    if self.config.grad_clip is not None:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )

                    if lr_scheduler is not None:
                        lr_scheduler.step()

                    optimizer.step()
                    steps += 1

                n = y.shape[0]
                total_loss += loss.detach().item() * n
                total_samples += n

        if total_samples == 0:
            raise ValueError("Dataset produced no samples")

        return total_loss / total_samples, steps

    def fit(self, train, val):
        cfg = self.config
        use_schedule = cfg.steps_per_epoch is not None

        if cfg.warmup_epochs and not use_schedule:
            raise ValueError("warmup_epochs requires steps_per_epoch")

        adamw = AdamW(
            models=self.model,
            ref_lr=cfg.lr,
            ref_wd=cfg.weight_decay,
            use_lr_scheduler=use_schedule,
            warmup_steps=cfg.warmup_epochs * (cfg.steps_per_epoch or 0),
            start_lr=cfg.start_lr,
            final_lr=cfg.final_lr,
            T_max=cfg.epochs * (cfg.steps_per_epoch or 0),
        )

        optimizer = adamw.get_optimizer()
        scheduler = adamw.get_lr_scheduler()

        train_loader = self._loader(train)
        val_loader = self._loader(val)

        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        stale_epochs = 0

        epochs = tqdm(
            range(1, cfg.epochs + 1),
            desc="Epochs",
            disable=not cfg.verbose,
        )

        for epoch in epochs:
            train_loss, steps = self._run_epoch(train_loader, optimizer, scheduler)

            if use_schedule and steps != cfg.steps_per_epoch:
                raise ValueError(f"Expected {cfg.steps_per_epoch} steps, got {steps}")

            with torch.inference_mode():
                val_loss, _ = self._run_epoch(val_loader)

            epochs.set_postfix(
                train_loss=f"{train_loss:.5f}",
                val_loss=f"{val_loss:.5f}",
            )

            if val_loss < best_loss - cfg.min_delta:
                best_loss = val_loss
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1

            if cfg.patience > 0 and stale_epochs >= cfg.patience:
                break

        self.model.load_state_dict(best_state)

        return {
            "val_loss": best_loss,
            "best_iteration": best_epoch,
        }

    @torch.inference_mode()
    def eval(self, dataset):
        self.model.eval()

        predictions = []
        targets = []

        for x, y, _ in tqdm(
            self._loader(dataset),
            desc="Evaluation",
            disable=not self.config.verbose,
        ):
            x = x.to(self.device, non_blocking=True)
            output = self.model(x)

            if self.task == "classification":
                output = output.softmax(dim=-1)

            predictions.append(output.cpu().numpy())
            targets.append(y.numpy())

        if not targets:
            raise ValueError("Dataset produced no samples")

        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        return {
            "metrics": compute_metrics(self.task, targets, predictions),
            "n_samples": len(targets),
        }

    def save(self, path):
        self.model.save(path)
