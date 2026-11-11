"""Training and validation of a single sequence-model configuration."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from ..data import SequenceDataset
from ..trainers import SequenceTrainer, SequenceTrainerConfig
from .utils import data_options, paths, set_seed


@dataclass
class SequenceTraining:
    model_cls: type
    model_params: dict
    trainer_params: dict
    data_params: dict
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
    batch_size: int = 1024
    device: str = "auto"
    precision: str = "fp32"
    save_path: str | Path | None = None
    verbose: bool = False

    def run(self):
        if {"input_dim", "task"} & self.model_params.keys():
            raise ValueError("model_params cannot override input_dim or task")

        reserved = {"model", "task", "mode", "device", "precision", "checkpoint_dir"}
        if reserved & self.trainer_params.keys():
            raise ValueError(f"trainer_params cannot override {sorted(reserved)}")

        set_seed(self.seed)

        source_depth, options = data_options(self.data_params, sequence=True)
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
            batch_size=self.batch_size,
            rank=0,
            world_size=1,
            **options,
        )

        train = SequenceDataset(
            train_paths,
            n_samples=self.n_train_samples,
            shuffle=True,
            drop_last=True,
            **dataset_kwargs,
        )
        mean, std = train.get_stats()

        val = SequenceDataset(
            val_paths,
            n_samples=self.n_val_samples,
            mean=mean,
            std=std,
            shuffle=False,
            drop_last=False,
            **dataset_kwargs,
        )

        model = self.model_cls(
            input_dim=len(train.feature_names),
            task=self.task,
            **self.model_params,
        )

        settings = dict(
            model=model,
            task=self.task,
            mode="single",
            device=self.device,
            precision=self.precision,
            verbose=self.verbose,
        )
        settings.update(self.trainer_params)

        trainer = SequenceTrainer(SequenceTrainerConfig(**settings))

        try:
            result = trainer.fit(train, val)

            if not isfinite(float(result["val_loss"])):
                raise ValueError("Nonfinite validation loss")

            if self.save_path is not None:
                save_path = Path(self.save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                trainer.save(save_path)

            return {
                "model_params": dict(self.model_params),
                "trainer_params": dict(self.trainer_params),
                "data_params": dict(self.data_params),
                **result,
                "n_train_samples_actual": train.total,
                "n_val_samples_actual": val.total,
                "train_files": list(map(str, train_paths)),
                "validation_files": list(map(str, val_paths)),
                "model_path": (
                    str(self.save_path) if self.save_path is not None else None
                ),
            }

        finally:
            trainer.close()
