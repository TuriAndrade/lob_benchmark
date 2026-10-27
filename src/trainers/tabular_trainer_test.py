"""Integration tests for the tabular trainer."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import TabularDataset, _write_h5_fixture
from models import LightGBMModel, LinearModel

try:
    from .tabular_trainer import TabularTrainer, TabularTrainerConfig
except ImportError:
    from tabular_trainer import TabularTrainer, TabularTrainerConfig


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("tabular_trainer")
    return (
        _write_h5_fixture(root / "train.h5", 700, gaps=False),
        _write_h5_fixture(root / "val.h5", 350, gaps=False),
    )


@pytest.mark.parametrize("model_cls", [LinearModel, LightGBMModel])
@pytest.mark.parametrize("task", ["regression", "classification"])
@pytest.mark.parametrize("data_mode", ["load", "stream"])
def test_fit_eval(data, tmp_path, model_cls, task, data_mode):
    train_path, val_path = data
    options = {"task": task, "threshold_bps": 2.0}

    train = TabularDataset([train_path], **options)
    val = TabularDataset([val_path], **options)

    if model_cls is LightGBMModel:
        model = model_cls(
            task=task,
            n_estimators=20,
            early_stopping_rounds=0,
            n_jobs=1,
        )
    else:
        model = model_cls(task=task)

    trainer = TabularTrainer(
        TabularTrainerConfig(
            model=model,
            task=task,
            data_mode=data_mode,
            verbose=False,
            temp_dir=str(tmp_path),
        )
    )

    result = trainer.fit(train, val)
    evaluation = trainer.eval(val)

    assert np.isfinite(result["val_loss"])
    assert evaluation["n_samples"] == len(val.load()[1])
    assert all(np.isfinite(v) for v in evaluation["metrics"].values())

    path = tmp_path / "model.joblib"
    trainer.save(path)
    assert path.is_file()

    restored = model_cls.load(path)
    x, _, _ = val.load()
    np.testing.assert_allclose(
        restored.predict(x), model.predict(x), rtol=1e-6, atol=1e-6
    )
