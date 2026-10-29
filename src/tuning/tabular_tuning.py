"""Hyperparameter tuning for tabular models."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from ..data import TabularDataset
from ..trainers import TabularTrainer, TabularTrainerConfig

from .common import paths, prepare_dir, save_results, set_seed, summary


@dataclass
class TabularTuning:
    model_cls: type
    model_params: list[dict]
    data_path: str | Path
    save_dir: str | Path
    task: str = "regression"
    depth: int = 5
    horizon: int = 50
    threshold_bps: float = 0.25
    n_train_days: int = 4
    n_val_days: int = 1
    n_test_days: int = 1
    n_train_samples: int | None = None
    n_val_samples: int | None = None
    n_test_samples: int | None = None
    normalize: bool = False
    seed: int = 42
    read_batch_size: int = 65_536
    data_mode: str = "load"
    override: bool = False

    def run(self):
        if not self.model_params:
            raise ValueError("model_params must not be empty")

        files = paths(
            self.data_path,
            self.depth,
            self.horizon,
            self.n_train_days,
            self.n_val_days,
            self.n_test_days,
        )
        prepare_dir(self.save_dir, self.override)
        set_seed(self.seed)

        options = dict(
            horizon=self.horizon,
            task=self.task,
            threshold_bps=self.threshold_bps,
            normalize=self.normalize,
            seed=self.seed,
            read_batch_size=self.read_batch_size,
        )
        train = TabularDataset(files[0], n_samples=self.n_train_samples, **options)
        mean, std = train.get_stats()
        val = TabularDataset(
            files[1],
            n_samples=self.n_val_samples,
            mean=mean,
            std=std,
            **options,
        )

        trials = []
        best = best_result = best_trainer = None

        for i, params in enumerate(self.model_params, 1):
            print(f"Trial {i}/{len(self.model_params)}: {params}")
            set_seed(self.seed)

            trainer = TabularTrainer(
                TabularTrainerConfig(
                    model=self.model_cls(task=self.task, **params),
                    task=self.task,
                    data_mode=self.data_mode,
                    verbose=False,
                )
            )
            result = trainer.fit(train, val)

            if not isfinite(result["val_loss"]):
                raise ValueError(f"Nonfinite validation loss in trial {i}")

            trials.append({"model_params": params, **result})
            if best_result is None or result["val_loss"] < best_result["val_loss"]:
                best, best_result, best_trainer = params, result, trainer

        test = TabularDataset(
            files[2],
            n_samples=self.n_test_samples,
            mean=mean,
            std=std,
            **options,
        )
        evaluation = best_trainer.eval(test)
        best_trainer.save(Path(self.save_dir) / "model.joblib")

        results = summary(self, files, train, best_result, evaluation)
        results.update(best_model_params=best, data_mode=self.data_mode)
        save_results(self.save_dir, trials, results)

        print(f"Best: {best} | validation loss: {best_result['val_loss']:.6f}")
        return results
