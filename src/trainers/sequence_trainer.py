"""Single-device, DDP and FSDP2 training for sequence models."""

from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import os
import shutil
from tempfile import mkdtemp

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)

from ..models import SequenceModel
from .distributed import prepare_model, setup_distributed
from .metrics import StreamingMetrics
from .optimizer import AdamW


@dataclass
class SequenceTrainerConfig:
    model: SequenceModel
    task: str = "regression"
    mode: str = "single"
    device: str = "auto"
    precision: str = "fp32"
    epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_schedule: str = "cosine"
    wd_schedule: str = "constant"
    warmup_epochs: int = 0
    start_lr: float = 1e-6
    final_lr: float = 1e-6
    final_wd: float = 0.0
    grad_accum_steps: int = 1
    grad_clip: float | None = 1.0
    num_workers: int = 0
    checkpoint_dir: str | None = None
    verbose: bool = True


class SequenceTrainer:
    def __init__(self, config: SequenceTrainerConfig):
        if config.task != config.model.task:
            raise ValueError("Trainer and model tasks differ")
        if config.epochs < 1 or config.grad_accum_steps < 1 or config.num_workers < 0:
            raise ValueError("Invalid epochs, accumulation, or worker count")
        if (
            config.patience < 0
            or config.warmup_epochs < 0
            or config.warmup_epochs > config.epochs
        ):
            raise ValueError("Invalid patience or warmup duration")
        if config.lr_schedule not in (
            "constant",
            "cosine",
        ) or config.wd_schedule not in ("constant", "cosine"):
            raise ValueError("Unknown LR or WD schedule")
        if (
            not all(
                isfinite(v)
                for v in (
                    config.lr,
                    config.weight_decay,
                    config.final_wd,
                    config.start_lr,
                    config.final_lr,
                    config.min_delta,
                )
            )
            or config.lr <= 0
            or min(
                config.weight_decay,
                config.final_wd,
                config.start_lr,
                config.final_lr,
                config.min_delta,
            )
            < 0
        ):
            raise ValueError("Invalid learning rate, weight decay or min_delta")
        if config.grad_clip is not None and (
            not isfinite(config.grad_clip) or config.grad_clip <= 0
        ):
            raise ValueError("grad_clip must be positive")

        self.config = config
        self.context = setup_distributed(config.mode, config.device)
        self.device = self.context.device
        self.raw_model = config.model
        self.model = prepare_model(config.model, self.context, config.precision)
        self.loss_fn = (
            nn.MSELoss() if config.task == "regression" else nn.CrossEntropyLoss()
        )

    def _check_dataset(self, dataset, training=False):
        if dataset.task != self.config.task:
            raise ValueError("Dataset and trainer tasks differ")
        if (dataset.rank, dataset.world_size) != (
            self.context.rank,
            self.context.world_size,
        ):
            raise ValueError("Dataset rank/world_size must match trainer")
        if not training and (dataset.shuffle or dataset.drop_last):
            raise ValueError("Evaluation requires shuffle=False and drop_last=False")
        if self.context.world_size > 1:
            local = torch.tensor(len(dataset), device=self.device)
            minimum, maximum = local.clone(), local.clone()
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            if minimum.item() != maximum.item():
                raise ValueError(
                    "Distributed ranks must yield the same number of batches"
                )

    def _loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=False,
        )

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.precision == "bf16" and self.context.mode != "fsdp",
        )

    def _epoch(self, loader, optimizer=None, lr_scheduler=None, wd_scheduler=None):
        training = optimizer is not None
        self.model.train(training)
        totals = torch.zeros(2, dtype=torch.float64, device=self.device)
        steps = 0
        group_samples = 0

        if training:
            optimizer.zero_grad(set_to_none=True)

        iterator = tqdm(
            loader,
            desc="Train" if training else "Validation",
            leave=False,
            disable=not (self.config.verbose and self.context.is_main),
        )

        with torch.enable_grad() if training else torch.inference_mode():
            for i, (x, y, _) in enumerate(iterator):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                n = len(y)

                if not n:
                    raise ValueError("Empty batch")

                if training:
                    start = (
                        i // self.config.grad_accum_steps
                    ) * self.config.grad_accum_steps
                    group_size = min(self.config.grad_accum_steps, len(loader) - start)
                    last = i - start + 1 == group_size
                    group_samples += n

                    sync = (
                        self.model.no_sync()
                        if self.context.mode == "ddp" and not last
                        else nullcontext()
                    )
                else:
                    sync = nullcontext()

                with sync, self._autocast():
                    output = self.model(x).float()
                    loss = self.loss_fn(output, y)

                    if training:
                        (loss * n).backward()

                totals[0] += loss.detach().double() * n
                totals[1] += n

                if training and last:
                    count = torch.tensor(
                        group_samples, dtype=torch.float64, device=self.device
                    )
                    self.context.reduce(count)

                    scale = self.context.world_size / count.item()
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)

                    group_samples = 0

                    if self.config.grad_clip is not None:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )

                    optimizer.step()

                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    if wd_scheduler is not None:
                        wd_scheduler.step()

                    optimizer.zero_grad(set_to_none=True)
                    steps += 1

        self.context.reduce(totals)

        if totals[1].item() == 0:
            raise ValueError("Dataset produced no samples")

        return (totals[0] / totals[1]).item(), steps

    def _checkpoint_root(self):
        if self.context.mode != "fsdp":
            return None
        if self.config.checkpoint_dir is None and self.context.world_size > int(
            os.environ.get("LOCAL_WORLD_SIZE", self.context.world_size)
        ):
            raise ValueError("Multi-node FSDP2 requires a shared checkpoint_dir")

        path = None
        if self.context.is_main:
            if self.config.checkpoint_dir is not None:
                Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            path = mkdtemp(prefix="lob_best_", dir=self.config.checkpoint_dir)

        paths = [path]
        dist.broadcast_object_list(paths, src=0)
        return Path(paths[0])

    def _save_best(self, epoch, previous, root):
        if self.context.mode != "fsdp":
            return {
                k: v.detach().cpu().clone()
                for k, v in self.raw_model.state_dict().items()
            }

        destination = root / str(epoch)
        dcp.save(
            {"model": get_model_state_dict(self.model)},
            checkpoint_id=destination,
        )
        self.context.barrier()

        if previous is not None and self.context.is_main:
            shutil.rmtree(previous)

        return destination

    def _restore_best(self, best):
        if self.context.mode != "fsdp":
            self.raw_model.load_state_dict(best)
            return

        state = {"model": get_model_state_dict(self.model)}
        dcp.load(state, checkpoint_id=best)
        set_model_state_dict(self.model, state["model"])

    def fit(self, train, val):
        self._check_dataset(train, training=True)
        self._check_dataset(val)

        cfg = self.config
        updates = (len(train) + cfg.grad_accum_steps - 1) // cfg.grad_accum_steps

        if updates < 1:
            raise ValueError("Training dataset has no batches")

        adamw = AdamW(
            models=self.model,
            ref_lr=cfg.lr,
            ref_wd=cfg.weight_decay,
            use_lr_scheduler=cfg.lr_schedule == "cosine",
            warmup_steps=cfg.warmup_epochs * updates,
            start_lr=cfg.start_lr,
            final_lr=cfg.final_lr,
            T_max=cfg.epochs * updates,
            use_wd_scheduler=cfg.wd_schedule == "cosine",
            final_wd=cfg.final_wd,
        )

        optimizer = adamw.get_optimizer()
        lr_scheduler = adamw.get_lr_scheduler()
        wd_scheduler = adamw.get_wd_scheduler()

        train_loader = self._loader(train)
        val_loader = self._loader(val)

        best_loss, best_epoch, stale, best = float("inf"), 0, 0, None
        root = self._checkpoint_root()

        try:
            for epoch in tqdm(
                range(1, cfg.epochs + 1),
                desc="Epochs",
                disable=not (cfg.verbose and self.context.is_main),
            ):
                train.set_epoch(epoch - 1)

                train_loss, steps = self._epoch(
                    train_loader, optimizer, lr_scheduler, wd_scheduler
                )
                if steps != updates:
                    raise RuntimeError(f"Expected {updates} updates, got {steps}")

                val_loss, _ = self._epoch(val_loader)

                if not isfinite(train_loss) or not isfinite(val_loss):
                    raise ValueError("Nonfinite training or validation loss")

                if cfg.verbose and self.context.is_main:
                    print(f"Epoch {epoch}: train={train_loss:.6f}, val={val_loss:.6f}")

                if val_loss < best_loss - cfg.min_delta:
                    best = self._save_best(epoch, best, root)
                    best_loss, best_epoch, stale = val_loss, epoch, 0
                else:
                    stale += 1

                if cfg.patience and stale >= cfg.patience:
                    break

            self.context.barrier()
            self._restore_best(best)
            self.context.barrier()
        finally:
            if root is not None:
                self.context.barrier()
                if self.context.is_main:
                    shutil.rmtree(root, ignore_errors=True)
                self.context.barrier()

        return {"val_loss": best_loss, "best_iteration": best_epoch}

    @torch.inference_mode()
    def eval(self, dataset):
        self._check_dataset(dataset)
        self.model.eval()

        metrics = StreamingMetrics(self.config.task, device=self.device)

        for x, y, _ in tqdm(
            self._loader(dataset),
            desc="Evaluation",
            disable=not (self.config.verbose and self.context.is_main),
        ):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with self._autocast():
                output = self.model(x).float()

            if self.config.task == "classification":
                output = output.softmax(dim=-1)

            metrics.update(y, output)

        self.context.reduce(metrics.stats)

        return {
            "metrics": metrics.result(),
            "n_samples": int(metrics.stats[0].item()),
        }

    def save(self, path):
        """Export a standalone model checkpoint; all ranks must call for FSDP2."""
        if self.context.mode == "fsdp":
            state = get_model_state_dict(
                self.model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
        else:
            state = {
                k: v.detach().cpu() for k, v in self.raw_model.state_dict().items()
            }

        if self.context.is_main:
            torch.save(
                {"config": self.raw_model.config, "state_dict": state},
                path,
            )

        self.context.barrier()

    def close(self):
        self.context.cleanup()
