"""Validation-based grid search for tabular models."""

from dataclasses import dataclass
from itertools import product
from math import isfinite
from pathlib import Path

from ..data import TabularDataset
from ..trainers import TabularTrainer, TabularTrainerConfig
from .utils import data_options, paths, set_seed


@dataclass
class TabularTuning:
    model_cls: type
    model_params: list[dict]
    trainer_params: list[dict]
    data_params: list[dict]
    data_path: str | Path
    task: str = "regression"
    target_name: str = "target_return_50_bps"
    threshold_bps: float = 0.0
    n_train_days: int = 4
    n_val_days: int = 1
    n_train_samples: int | None = None
    n_val_samples: int | None = None
    seed: int = 42
    read_batch_size: int = 65_536
    data_mode: str = "load"

    def run(self):
        if not all((self.model_params, self.trainer_params, self.data_params)):
            raise ValueError("All three parameter grids must be nonempty")

        if any("task" in params for params in self.model_params):
            raise ValueError("model_params cannot override task")

        if any({"model", "task"} & params.keys() for params in self.trainer_params):
            raise ValueError("trainer_params cannot override model or task")

        best_model = best_entry = None
        best_loss = float("inf")
        trials = []
        total = (
            len(self.data_params) * len(self.model_params) * len(self.trainer_params)
        )
        trial_number = 0

        for data_params in self.data_params:
            source_depth, options = data_options(data_params, sequence=False)
            train_paths, val_paths, _ = paths(
                self.data_path,
                self.n_train_days,
                self.n_val_days,
                n_test_days=0,
                source_depth=source_depth,
            )

            dataset_kwargs = dict(
                target_name=self.target_name,
                task=self.task,
                threshold_bps=self.threshold_bps,
                seed=self.seed,
                read_batch_size=self.read_batch_size,
                **options,
            )

            train = TabularDataset(
                train_paths,
                n_samples=self.n_train_samples,
                **dataset_kwargs,
            )
            mean, std = train.get_stats()
            val = TabularDataset(
                val_paths,
                n_samples=self.n_val_samples,
                mean=mean,
                std=std,
                **dataset_kwargs,
            )

            for model_params, trainer_params in product(
                self.model_params, self.trainer_params
            ):
                trial_number += 1
                set_seed(self.seed)
                print(
                    f"Trial {trial_number}/{total}: "
                    f"{model_params} / {trainer_params} / {data_params}"
                )

                model = self.model_cls(task=self.task, **model_params)
                settings = dict(
                    model=model,
                    task=self.task,
                    data_mode=self.data_mode,
                    verbose=False,
                )
                settings.update(trainer_params)

                trainer = TabularTrainer(TabularTrainerConfig(**settings))
                result = trainer.fit(train, val)
                loss = float(result["val_loss"])

                if not isfinite(loss):
                    raise ValueError(
                        f"Nonfinite validation loss in trial {trial_number}"
                    )

                entry = dict(
                    model_params=dict(model_params),
                    trainer_params=dict(trainer_params),
                    data_params=dict(data_params),
                )
                trials.append(
                    {
                        **entry,
                        **result,
                        "train_files": list(map(str, train_paths)),
                        "validation_files": list(map(str, val_paths)),
                    }
                )

                if loss < best_loss:
                    best_model, best_entry, best_loss = model, entry, loss

        return best_model, best_entry, trials
