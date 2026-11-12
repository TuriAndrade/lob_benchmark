"""Two-stage TCN hyperparameter tuning."""

from itertools import product
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

from ..data import SequenceDataset
from ..models import TCNModel
from ..trainers import SequenceTrainer, SequenceTrainerConfig
from .tuning_launcher import Trial, TuningLauncher
from .sequence_training import SequenceTraining
from .utils import (
    arguments,
    data_options,
    grid_search,
    launcher_kwargs,
    output_dir,
    paths,
    prepare_dir,
    save_results,
    summary,
    training_kwargs,
    trial_records,
)


def main():
    args = arguments(sequence=True)
    save_dir = prepare_dir(output_dir("tcn", args), args.override)

    common = training_kwargs(args)
    launcher_options = launcher_kwargs(args)

    base_model = dict(
        channels=64,
        num_layers=6,
        kernel_size=3,
        dropout=0.1,
    )
    base_trainer = dict(
        epochs=100,
        patience=7,
        weight_decay=1e-4,
        lr=3e-4,
    )

    # Stage 1: input window and feature selection.
    data_trials = [
        Trial(
            fn=SequenceTraining(
                model_cls=TCNModel,
                model_params=base_model,
                trainer_params=base_trainer,
                data_params=params,
                **common,
            ).run,
            estimated_vram=None,
        )
        for params in grid_search(
            {
                "window_size": [50, 100, 200],
                "stride": [1],
                "raw_feature_names": ["all"],
                "engineered_feature_names": [None, "all"],
                "normalize": [True],
            }
        )
    ]

    data_completed, data_failed = TuningLauncher(data_trials, **launcher_options).run()

    if not data_completed:
        raise RuntimeError("All TCN data-configuration trials failed")

    best_data = min(data_completed, key=lambda t: t.result["val_loss"]).result[
        "data_params"
    ]

    # Stage 2: architecture and optimization.
    with TemporaryDirectory(prefix="trials_", dir=save_dir) as directory:
        model_trials = [
            Trial(
                fn=SequenceTraining(
                    model_cls=TCNModel,
                    model_params=model_params,
                    trainer_params={**base_trainer, **lr_params},
                    data_params=best_data,
                    save_path=Path(directory) / f"{i}.pt",
                    **common,
                ).run,
                estimated_vram=None,
            )
            for i, (model_params, lr_params) in enumerate(
                product(
                    grid_search(
                        {
                            "channels": [32, 64],
                            "num_layers": [4, 5, 6],
                            "kernel_size": [3],
                            "dropout": [0.0, 0.1],
                        }
                    ),
                    grid_search(
                        {
                            "lr": [3e-4, 1e-3],
                        }
                    ),
                )
            )
        ]

        model_completed, model_failed = TuningLauncher(
            model_trials, **launcher_options
        ).run()

        if not model_completed:
            raise RuntimeError("All TCN architecture trials failed")

        best = min(model_completed, key=lambda t: t.result["val_loss"]).result

        checkpoint = save_dir / "model.pt"
        copy2(best["model_path"], checkpoint)

        trials = [
            {"stage": "data", **entry}
            for entry in trial_records(data_completed, data_failed)
        ] + [
            {"stage": "model", **entry}
            for entry in trial_records(model_completed, model_failed)
        ]

    # Final evaluation on held-out test days.
    best_params = {
        key: best[key] for key in ("model_params", "trainer_params", "data_params")
    }

    source_depth, options = data_options(best_params["data_params"], sequence=True)
    files = paths(
        args.data_path,
        args.n_train_days,
        args.n_val_days,
        args.n_test_days,
        source_depth=source_depth,
    )

    dataset_kwargs = dict(
        target_name=args.target_name,
        task=args.task,
        threshold_bps=args.threshold,
        seed=args.seed,
        read_batch_size=args.read_batch_size,
        batch_size=args.batch_size,
        rank=0,
        world_size=1,
        **options,
    )

    train = SequenceDataset(
        files[0],
        n_samples=args.n_train_samples,
        shuffle=True,
        drop_last=True,
        **dataset_kwargs,
    )
    mean, std = train.get_stats()
    del train

    test = SequenceDataset(
        files[2],
        n_samples=args.n_test_samples,
        mean=mean,
        std=std,
        shuffle=False,
        drop_last=False,
        **dataset_kwargs,
    )

    trainer = SequenceTrainer(
        SequenceTrainerConfig(
            model=TCNModel.load(checkpoint),
            task=args.task,
            device="auto",
            precision=args.precision,
            num_workers=best_params["trainer_params"].get("num_workers", 0),
            verbose=args.verbose,
        )
    )

    try:
        evaluation = trainer.eval(test)
    finally:
        trainer.close()

    results = summary(
        args,
        "tcn",
        files,
        best_params,
        evaluation,
        best_trial=best,
        normalization={
            "mean": None if mean is None else mean.tolist(),
            "std": None if std is None else std.tolist(),
        },
    )
    results["model_path"] = str(checkpoint)

    save_results(save_dir, trials, results)

    print(f"Best: {best_params}")
    print(f"Validation loss: {best['val_loss']:.6f}")
    print(f"Test metrics: {evaluation['metrics']}")

    return results


if __name__ == "__main__":
    main()
