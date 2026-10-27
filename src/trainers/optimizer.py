"""AdamW with optional warmup-cosine learning-rate and cosine weight-decay schedules."""

import math

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


class WarmupCosineLRScheduler:
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, T_max, final_lr=0.0):
        if T_max < 1 or not 0 <= warmup_steps <= T_max:
            raise ValueError("Require T_max >= 1 and 0 <= warmup_steps <= T_max")
        if min(start_lr, ref_lr, final_lr) < 0:
            raise ValueError("Learning rates must be nonnegative")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.T_max = T_max
        self._step = 0
        self._apply(self._rate(0))

    def _rate(self, step):
        if self.T_max == 1:
            return self.ref_lr

        if step < self.warmup_steps:
            if self.warmup_steps == 1:
                return self.start_lr
            return self.start_lr + (
                (self.ref_lr - self.start_lr) * step / (self.warmup_steps - 1)
            )

        remaining = self.T_max - self.warmup_steps
        progress = min(1.0, (step - self.warmup_steps) / max(1, remaining - 1))
        return self.final_lr + (
            (self.ref_lr - self.final_lr) * (1 + math.cos(math.pi * progress)) / 2
        )

    def _apply(self, rate):
        for group in self.optimizer.param_groups:
            group["lr"] = rate

    def step(self):
        self._step += 1
        rate = self._rate(min(self._step, self.T_max - 1))
        self._apply(rate)
        return rate


class CosineWDScheduler:
    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0):
        if T_max < 1 or min(ref_wd, final_wd) < 0:
            raise ValueError("Require T_max >= 1 and nonnegative weight decay")

        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0
        self._apply(ref_wd)

    def _apply(self, rate):
        for group in self.optimizer.param_groups:
            if not group.get("WD_exclude", False):
                group["weight_decay"] = rate

    def step(self):
        self._step += 1
        progress = min(1.0, self._step / max(1, self.T_max - 1))
        rate = self.final_wd + (
            (self.ref_wd - self.final_wd) * (1 + math.cos(math.pi * progress)) / 2
        )
        self._apply(rate)
        return rate


def _set_weight_decay(model, skip_list=(), skip_keywords=()):
    decay, no_decay = [], []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        excluded = (
            parameter.ndim == 1
            or name.endswith(".bias")
            or name in skip_list
            or any(keyword in name for keyword in skip_keywords)
        )
        (no_decay if excluded else decay).append(parameter)

    return [
        {"params": decay},
        {"params": no_decay, "WD_exclude": True, "weight_decay": 0.0},
    ]


class AdamW:
    def __init__(
        self,
        models: nn.Module | list[nn.Module] | tuple[nn.Module, ...],
        betas=(0.9, 0.999),
        eps=1e-8,
        use_lr_scheduler=False,
        warmup_steps=0,
        start_lr=1e-6,
        ref_lr=1e-4,
        final_lr=1e-6,
        T_max=0,
        use_wd_scheduler=False,
        ref_wd=1e-4,
        final_wd=0.0,
    ):
        if ref_lr <= 0 or ref_wd < 0:
            raise ValueError("ref_lr must be positive and ref_wd nonnegative")
        if (use_lr_scheduler or use_wd_scheduler) and T_max < 1:
            raise ValueError("Scheduled training requires T_max >= 1")

        models = models if isinstance(models, (list, tuple)) else [models]
        groups = []
        seen = set()

        for model in models:
            base = model.module if isinstance(model, DistributedDataParallel) else model
            skip = (
                set(base.no_weight_decay())
                if hasattr(base, "no_weight_decay")
                else set()
            )
            keywords = (
                tuple(base.no_weight_decay_keywords())
                if hasattr(base, "no_weight_decay_keywords")
                else ()
            )

            for group in _set_weight_decay(base, skip, keywords):
                for param in group["params"]:
                    if id(param) in seen:
                        raise ValueError("A parameter was supplied more than once")
                    seen.add(id(param))

                if group["params"]:
                    groups.append(group)

        if not groups:
            raise ValueError("No trainable parameters")

        self.optimizer = torch.optim.AdamW(
            groups,
            lr=ref_lr,
            weight_decay=ref_wd,
            betas=betas,
            eps=eps,
        )

        self.lr_scheduler = (
            WarmupCosineLRScheduler(
                self.optimizer, warmup_steps, start_lr, ref_lr, T_max, final_lr
            )
            if use_lr_scheduler
            else None
        )

        self.wd_scheduler = (
            CosineWDScheduler(self.optimizer, ref_wd, T_max, final_wd)
            if use_wd_scheduler
            else None
        )

    def get_optimizer(self):
        return self.optimizer

    def get_lr_scheduler(self):
        return self.lr_scheduler

    def get_wd_scheduler(self):
        return self.wd_scheduler
