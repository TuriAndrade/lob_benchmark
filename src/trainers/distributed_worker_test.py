"""Worker launched by sequence_trainer_test.py through torchrun."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import SequenceDataset
from models import TCNModel

from trainers import SequenceTrainer, SequenceTrainerConfig


def main():
    path, mode, shuffle, drop_last = sys.argv[1:]
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.manual_seed(42)

    options = dict(
        paths=[path],
        window_size=12,
        batch_size=8,
        rank=rank,
        world_size=world_size,
    )
    train = SequenceDataset(
        **options,
        n_samples=41,
        seed=42,
        shuffle=bool(int(shuffle)),
        drop_last=bool(int(drop_last)),
    )
    val = SequenceDataset(
        **options,
        n_samples=23,
        seed=123,
        shuffle=False,
        drop_last=False,
    )

    trainer = SequenceTrainer(
        SequenceTrainerConfig(
            model=TCNModel(
                input_dim=20,
                channels=8,
                num_layers=1,
                dropout=0.0,
            ),
            mode=mode,
            device="cpu",
            epochs=2,
            patience=0,
            grad_accum_steps=2,
            lr_schedule="constant",
            weight_decay=0.0,
            verbose=False,
        )
    )

    try:
        result = trainer.fit(train, val)
        evaluation = trainer.eval(val)

        assert np.isfinite(result["val_loss"])
        assert evaluation["n_samples"] == val.total == 23

        # All ranks must obtain identical globally reduced results.
        results = [None] * world_size
        dist.all_gather_object(results, (result["val_loss"], evaluation["metrics"]))
        assert all(item == results[0] for item in results)

        # Evaluation must cover every selected sample exactly once.
        local_ids = [
            int(event)
            for _, _, meta in trainer._loader(val)
            for event in meta["event_id"]
        ]
        all_ids = [None] * world_size
        dist.all_gather_object(all_ids, local_ids)
        combined = [event for ids in all_ids for event in ids]
        assert len(combined) == len(set(combined)) == val.total

        # DDP must leave identical parameters on every rank.
        if mode == "ddp":
            parameters = torch.cat(
                [p.detach().flatten() for p in trainer.raw_model.parameters()]
            )
            gathered = [torch.empty_like(parameters) for _ in range(world_size)]
            dist.all_gather(gathered, parameters)
            assert all(torch.allclose(p, gathered[0]) for p in gathered)

    finally:
        trainer.close()


if __name__ == "__main__":
    main()
