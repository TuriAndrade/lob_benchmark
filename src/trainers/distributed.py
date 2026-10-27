"""Distributed setup and model preparation for sequence training."""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    mode: str
    device: torch.device
    rank: int = 0
    world_size: int = 1
    owns_group: bool = False

    @property
    def is_main(self):
        return self.rank == 0

    def reduce(self, values):
        if self.world_size > 1:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        return values

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def cleanup(self):
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()


def setup_distributed(mode, device="auto"):
    if mode not in ("single", "ddp", "fsdp"):
        raise ValueError("mode must be single, ddp or fsdp")

    if mode == "single":
        if dist.is_initialized() and dist.get_world_size() > 1:
            raise ValueError(
                "single mode cannot use an active distributed process group"
            )
        name = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if name == "auto":
            name = "cpu"
        return DistributedContext(mode, torch.device(name))

    if not dist.is_initialized() and not all(
        key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    ):
        raise RuntimeError("Launch distributed training with torchrun")

    local_rank = int(os.environ["LOCAL_RANK"])
    if device == "cpu":
        selected = torch.device("cpu")
        backend = "gloo"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use device='cpu' for CPU tests")
        if device != "auto" and torch.device(device).type != "cuda":
            raise ValueError("Distributed training requires CUDA or explicit CPU")
        torch.cuda.set_device(local_rank)
        selected = torch.device("cuda", local_rank)
        backend = "nccl"

    owned = not dist.is_initialized()
    if owned:
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(
        mode, selected, dist.get_rank(), dist.get_world_size(), owned
    )


def prepare_model(model, context, precision="fp32"):
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if precision == "bf16" and (
        context.device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("BF16 requires a CUDA device with BF16 support")

    if context.mode == "fsdp":
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16 if precision == "bf16" else None,
            reduce_dtype=torch.float32 if precision == "bf16" else None,
        )
        for block in model.fsdp_modules() if hasattr(model, "fsdp_modules") else ():
            fully_shard(block, mp_policy=policy)
        fully_shard(model, mp_policy=policy)
        return model

    model = model.to(context.device)
    if context.mode == "ddp":
        return DistributedDataParallel(
            model,
            device_ids=[context.device.index]
            if context.device.type == "cuda"
            else None,
        )
    return model
