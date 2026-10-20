import math
from typing import Tuple, Sequence, Any, Dict, List, Union, Optional

import torch
from torch import nn


# ------- Schedules -------
class WarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        start_lr: float,
        ref_lr: float,
        T_max: int,
        final_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.start_lr = float(start_lr)
        self.ref_lr = float(ref_lr)
        self.final_lr = float(final_lr)
        self.warmup_steps = int(warmup_steps)
        self.cosine_steps = int(T_max) - int(warmup_steps)
        self._step = 0.0

    def step(self) -> float:
        self._step += 1
        if self._step <= self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = float(self._step - self.warmup_steps) / float(
                max(1, self.cosine_steps)
            )
            new_lr = max(
                self.final_lr,
                self.final_lr
                + (self.ref_lr - self.final_lr)
                * 0.5
                * (1.0 + math.cos(math.pi * progress)),
            )

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        return float(new_lr)


class CosineWDScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        ref_wd: float,
        T_max: int,
        final_wd: float = 0.0,
    ):
        self.optimizer = optimizer
        self.ref_wd = float(ref_wd)
        self.final_wd = float(final_wd)
        self.T_max = int(T_max)
        self._step = 0.0

    def step(self) -> float:
        self._step += 1
        progress = float(self._step) / float(max(1, self.T_max))
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = float(new_wd)
        return float(new_wd)


def _check_keywords_in_name(name: str, keywords: Sequence[str] = ()) -> bool:
    return any(kw in name for kw in keywords)


def _set_weight_decay(
    model: nn.Module,
    skip_list: Sequence[str] = (),
    skip_keywords: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    has_decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or _check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [
        {"params": has_decay},
        {"params": no_decay, "WD_exclude": True, "weight_decay": 0.0},
    ]


# ------- AdamW + cosine warmup lr + cosine wd -------
class AdamW:
    def __init__(
        self,
        # --- optimizer core ---
        models: Union[nn.Module, Sequence[nn.Module]],
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        # --- LR scheduler ---
        use_lr_scheduler: bool = False,
        warmup_steps: int = 0,
        start_lr: float = 1e-6,
        ref_lr: float = 1e-4,
        final_lr: float = 1e-6,
        T_max: int = 0,  # total steps INCLUDING warmup
        # --- WD scheduler ---
        use_wd_scheduler: bool = False,
        ref_wd: float = 1e-4,
        final_wd: float = 0.0,
    ):
        self.models = models
        self.betas = betas
        self.eps = eps
        self.use_lr_scheduler = use_lr_scheduler
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.T_max = T_max
        self.use_wd_scheduler = use_wd_scheduler
        self.ref_wd = ref_wd
        self.final_wd = final_wd

        self.optimizer = torch.optim.AdamW(
            params=self._build_param_groups(),
            lr=self.start_lr if self.use_lr_scheduler else self.ref_lr,
            weight_decay=self.ref_wd,
            betas=self.betas,
            eps=self.eps,
        )

        self.lr_scheduler = None
        if self.use_lr_scheduler:
            if self.T_max <= 0:
                raise ValueError(
                    "use_lr_scheduler=True requires T_max > 0 (total training steps)."
                )
            self.lr_scheduler = WarmupCosineLRScheduler(
                optimizer=self.optimizer,
                warmup_steps=self.warmup_steps,
                start_lr=self.start_lr,
                ref_lr=self.ref_lr,
                final_lr=self.final_lr,
                T_max=self.T_max,
            )

        self.wd_scheduler = None
        if self.use_wd_scheduler:
            if self.T_max <= 0:
                raise ValueError(
                    "use_wd_scheduler=True requires T_max > 0 (total training steps)."
                )
            self.wd_scheduler = CosineWDScheduler(
                optimizer=self.optimizer,
                ref_wd=self.ref_wd,
                final_wd=self.final_wd,
                T_max=self.T_max,
            )

    def get_optimizer(self) -> torch.optim.AdamW:
        return self.optimizer

    def get_lr_scheduler(self) -> Optional[WarmupCosineLRScheduler]:
        return self.lr_scheduler

    def get_wd_scheduler(self) -> Optional[CosineWDScheduler]:
        return self.wd_scheduler

    def _build_param_groups(self) -> Any:
        models = (
            self.models if isinstance(self.models, (list, tuple)) else [self.models]
        )

        groups: List[Dict[str, Any]] = []

        for m in models:
            skip_list: Sequence[str] = ()
            skip_keywords: Sequence[str] = ()
            if hasattr(m, "no_weight_decay"):
                skip_list = m.no_weight_decay()
            if hasattr(m, "no_weight_decay_keywords"):
                skip_keywords = m.no_weight_decay_keywords()
            groups += _set_weight_decay(
                m, skip_list=skip_list, skip_keywords=skip_keywords
            )

        return groups
