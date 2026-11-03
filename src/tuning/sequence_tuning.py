"""Validation-based grid search for single-device, DDP, and FSDP sequence models."""

from dataclasses import dataclass
from itertools import product
from math import isfinite
from pathlib import Path

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from ..data import SequenceDataset
from ..trainers import SequenceTrainer, SequenceTrainerConfig, setup_distributed
from .utils import data_options, paths, set_seed


@dataclass
class SequenceTuning:
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
    batch_size: int = 1024
    mode: str = "single"
    device: str = "auto"
    precision: str = "fp32"
    checkpoint_dir: str | Path | None = None
    context: object | None = None

    def run(self):
        if not all((self.model_params, self.trainer_params, self.data_params)):
            raise ValueError("All three parameter grids must be nonempty")

        if any({"input_dim", "task"} & params.keys() for params in self.model_params):
            raise ValueError("model_params cannot override input_dim or task")

        reserved = {"model", "task", "mode", "device", "precision", "checkpoint_dir"}
        if any(reserved & params.keys() for params in self.trainer_params):
            raise ValueError(f"trainer_params cannot override {sorted(reserved)}")

        context = self.context or setup_distributed(self.mode, self.device)
        owns_context = self.context is None

        try:
            if context.mode != self.mode:
                raise ValueError("Distributed context and training mode differ")

            best_state = best_entry = None
            best_loss = float("inf")
            best_input_dim = None
            trials = []
            total = (
                len(self.data_params)
                * len(self.model_params)
                * len(self.trainer_params)
            )
            trial_number = 0

            for data_params in self.data_params:
                source_depth, options = data_options(data_params, sequence=True)
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
                    rank=context.rank,
                    world_size=context.world_size,
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

                for model_params, trainer_params in product(
                    self.model_params, self.trainer_params
                ):
                    trial_number += 1
                    if context.is_main:
                        print(
                            f"Trial {trial_number}/{total}: "
                            f"{model_params} / {trainer_params} / {data_params}"
                        )

                    set_seed(self.seed)
                    input_dim = len(train.feature_names)
                    model = self.model_cls(
                        input_dim=input_dim,
                        task=self.task,
                        **model_params,
                    )

                    settings = dict(
                        model=model,
                        task=self.task,
                        mode=self.mode,
                        device=self.device,
                        precision=self.precision,
                        checkpoint_dir=(
                            None
                            if self.checkpoint_dir is None
                            else str(self.checkpoint_dir)
                        ),
                        verbose=context.is_main,
                    )
                    settings.update(trainer_params)

                    trainer = SequenceTrainer(SequenceTrainerConfig(**settings))

                    try:
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
                                "n_train_samples_actual": train.total,
                                "n_val_samples_actual": val.total,
                                "train_files": list(map(str, train_paths)),
                                "validation_files": list(map(str, val_paths)),
                            }
                        )

                        if loss < best_loss:
                            state = get_model_state_dict(
                                trainer.model,
                                options=StateDictOptions(
                                    full_state_dict=True,
                                    cpu_offload=False,
                                ),
                            )
                            best_state = {
                                name: tensor.detach().cpu().clone()
                                for name, tensor in state.items()
                            }
                            best_input_dim = input_dim
                            best_entry = entry
                            best_loss = loss

                    finally:
                        trainer.close()
                        del trainer, model

            best_model = self.model_cls(
                input_dim=best_input_dim,
                task=self.task,
                **best_entry["model_params"],
            )
            best_model.load_state_dict(best_state)
            best_model.eval()

            return best_model, best_entry, trials

        finally:
            if owns_context:
                context.cleanup()
