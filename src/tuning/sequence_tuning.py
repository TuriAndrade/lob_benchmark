"""Hyperparameter tuning for sequence models with distributed training."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

from data import SequenceDataset
from trainers import SequenceTrainer, SequenceTrainerConfig
from trainers.distributed import setup_distributed

from .common import paths, prepare_dir, save_results, set_seed, summary


@dataclass
class SequenceTuning:
    model_cls: type
    model_params: list[dict]
    trainer_params: list[dict]
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
    window_size: int = 100
    stride: int = 1
    batch_size: int = 1024
    include_engineered: bool = False
    shuffle: bool = True
    drop_last: bool = True
    mode: str = "single"
    device: str = "auto"
    precision: str = "fp32"
    override: bool = False

    def run(self):
        if not self.model_params or not self.trainer_params:
            raise ValueError("model_params and trainer_params must not be empty")
        if self.normalize and not self.include_engineered:
            raise ValueError("normalize requires include_engineered")

        reserved = {"model", "task", "mode", "device", "checkpoint_dir"}
        if any(reserved.intersection(p) for p in self.trainer_params):
            raise ValueError(f"trainer_params cannot set {sorted(reserved)}")

        context = setup_distributed(self.mode, self.device)
        try:
            files = paths(
                self.data_path,
                self.depth,
                self.horizon,
                self.n_train_days,
                self.n_val_days,
                self.n_test_days,
            )
            if Path(self.save_dir).exists() and not self.override:
                raise FileExistsError(
                    f"Output directory already exists: {self.save_dir}"
                )
            if context.is_main:
                prepare_dir(self.save_dir, self.override)
            context.barrier()

            set_seed(self.seed)
            options = dict(
                horizon=self.horizon,
                task=self.task,
                threshold_bps=self.threshold_bps,
                normalize=self.normalize,
                seed=self.seed,
                read_batch_size=self.read_batch_size,
                window_size=self.window_size,
                stride=self.stride,
                batch_size=self.batch_size,
                include_engineered=self.include_engineered,
                rank=context.rank,
                world_size=context.world_size,
            )

            train = SequenceDataset(
                files[0],
                n_samples=self.n_train_samples,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                **options,
            )
            mean, std = train.get_stats()
            val = SequenceDataset(
                files[1],
                n_samples=self.n_val_samples,
                mean=mean,
                std=std,
                shuffle=False,
                drop_last=False,
                **options,
            )

            trials = []
            best = best_result = None
            checkpoint = Path(self.save_dir) / "model.pt"
            n_trials = len(self.model_params) * len(self.trainer_params)

            for i, (model_params, trainer_params) in enumerate(
                product(self.model_params, self.trainer_params), 1
            ):
                if context.is_main:
                    print(f"Trial {i}/{n_trials}: {model_params} / {trainer_params}")

                set_seed(self.seed)
                model = self.model_cls(
                    input_dim=len(train.feature_columns),
                    task=self.task,
                    **model_params,
                )

                settings = dict(
                    task=self.task,
                    mode=self.mode,
                    device=self.device,
                    precision=self.precision,
                    checkpoint_dir=str(self.save_dir),
                    verbose=context.is_main,
                )
                settings.update(trainer_params)

                trainer = SequenceTrainer(
                    SequenceTrainerConfig(model=model, **settings)
                )
                try:
                    result = trainer.fit(train, val)
                    trials.append(
                        {
                            "model_params": model_params,
                            "trainer_params": trainer_params,
                            **result,
                        }
                    )

                    if (
                        best_result is None
                        or result["val_loss"] < best_result["val_loss"]
                    ):
                        trainer.save(checkpoint)
                        best = (model_params, trainer_params)
                        best_result = result
                finally:
                    trainer.close()
                    del trainer, model

            test = SequenceDataset(
                files[2],
                n_samples=self.n_test_samples,
                mean=mean,
                std=std,
                shuffle=False,
                drop_last=False,
                **options,
            )

            best_model, best_settings = best
            model = self.model_cls.load(checkpoint)
            trainer = SequenceTrainer(
                SequenceTrainerConfig(
                    model=model,
                    task=self.task,
                    mode=self.mode,
                    device=self.device,
                    precision=best_settings.get("precision", self.precision),
                    num_workers=best_settings.get("num_workers", 0),
                    verbose=context.is_main,
                )
            )
            try:
                evaluation = trainer.eval(test)
            finally:
                trainer.close()
                del trainer, model

            results = summary(self, files, train, best_result, evaluation)
            results.update(
                best_model_params=best_model,
                best_trainer_params=best_settings,
                window_size=self.window_size,
                stride=self.stride,
                batch_size=self.batch_size,
                include_engineered=self.include_engineered,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                mode=self.mode,
                device=self.device,
                precision=best_settings.get("precision", self.precision),
            )

            if context.is_main:
                save_results(self.save_dir, trials, results)
                print(f"Best: {best} | validation loss: {best_result['val_loss']:.6f}")

            context.barrier()
            return results
        finally:
            context.cleanup()
