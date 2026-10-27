"""Single-device and distributed integration tests for the sequence trainer."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from data import SequenceDataset, _write_h5_fixture
from models import TCNModel

try:
    from .sequence_trainer import SequenceTrainer, SequenceTrainerConfig
except ImportError:
    from sequence_trainer import SequenceTrainer, SequenceTrainerConfig


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return _write_h5_fixture(
        tmp_path_factory.mktemp("sequence_trainer") / "data.h5",
        320,
        gaps=False,
    )


@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("drop_last", [False, True])
def test_single(data, task, shuffle, drop_last):
    torch.manual_seed(42)

    options = dict(window_size=12, batch_size=8, task=task, threshold_bps=2.0)
    train = SequenceDataset(
        [data],
        **options,
        n_samples=41,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    val = SequenceDataset(
        [data],
        **options,
        n_samples=23,
        shuffle=False,
        drop_last=False,
        seed=123,
    )

    model = TCNModel(
        input_dim=20,
        task=task,
        channels=8,
        num_layers=1,
        dropout=0.0,
    )
    trainer = SequenceTrainer(
        SequenceTrainerConfig(
            model=model,
            task=task,
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

        assert result["best_iteration"] in (1, 2)
        assert np.isfinite(result["val_loss"])
        assert evaluation["n_samples"] == val.total
        assert all(np.isfinite(v) for v in evaluation["metrics"].values())

        loss_key = "mse" if task == "regression" else "log_loss"
        assert evaluation["metrics"][loss_key] == pytest.approx(
            result["val_loss"], abs=1e-5
        )
    finally:
        trainer.close()


@pytest.mark.parametrize(
    "shuffle,drop_last",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_distributed_ddp(data, shuffle, drop_last):
    run_distributed(data, "ddp", shuffle, drop_last)


@pytest.mark.skipif(
    os.environ.get("LOB_TEST_FSDP_CPU") != "1",
    reason="Enable explicitly with LOB_TEST_FSDP_CPU=1",
)
def test_distributed_fsdp(data):
    run_distributed(data, "fsdp", True, False)


def run_distributed(data, mode, shuffle, drop_last):
    worker = Path(__file__).with_name("distributed_worker_test.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(SRC), env.get("PYTHONPATH", "")))
    )
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(worker),
        str(data),
        mode,
        str(int(shuffle)),
        str(int(drop_last)),
    ]

    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
