"""Two-stage TCN hyperparameter tuning."""

from ..data import SequenceDataset
from ..models import TCNModel
from ..trainers import SequenceTrainer, SequenceTrainerConfig, setup_distributed

from .sequence_tuning import SequenceTuning
from .utils import (
    arguments,
    data_options,
    grid_search,
    output_dir,
    paths,
    prepare_dir,
    save_results,
    summary,
)


def main():
    args = arguments(sequence=True)
    save_dir = output_dir("tcn", args)

    if save_dir.exists() and not args.override:
        raise FileExistsError(f"Output directory already exists: {save_dir}")

    context = setup_distributed(args.training_mode)

    try:
        common = dict(
            model_cls=TCNModel,
            data_path=args.data_path,
            task=args.task,
            target_name=args.target_name,
            threshold_bps=args.threshold,
            n_train_days=args.n_train_days,
            n_val_days=args.n_val_days,
            n_train_samples=args.n_train_samples,
            n_val_samples=args.n_val_samples,
            seed=args.seed,
            read_batch_size=args.read_batch_size,
            batch_size=args.batch_size,
            mode=args.training_mode,
            precision=args.precision,
            context=context,
        )

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
        initial_model, data_best, data_trials = SequenceTuning(
            **common,
            model_params=[base_model],
            trainer_params=[base_trainer],
            data_params=list(
                grid_search(
                    {
                        "window_size": [50, 100, 200],
                        "stride": [1],
                        "raw_feature_names": ["all"],
                        "engineered_feature_names": [None, "all"],
                        "normalize": [True],
                    }
                )
            ),
        ).run()
        del initial_model

        # Stage 2: architecture and optimization.
        model, best_params, model_trials = SequenceTuning(
            **common,
            model_params=list(
                grid_search(
                    {
                        "channels": [32, 64],
                        "num_layers": [4, 5, 6],
                        "kernel_size": [3],
                        "dropout": [0.0, 0.1],
                    }
                )
            ),
            trainer_params=[
                {**base_trainer, **params}
                for params in grid_search(
                    {
                        "lr": [3e-4, 1e-3],
                    }
                )
            ],
            data_params=[data_best["data_params"]],
        ).run()

        # Final evaluation on held-out test days.
        source_depth, data_kwargs = data_options(
            best_params["data_params"], sequence=True
        )
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
            rank=context.rank,
            world_size=context.world_size,
            **data_kwargs,
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
                model=model,
                task=args.task,
                mode=args.training_mode,
                precision=args.precision,
                num_workers=best_params["trainer_params"].get("num_workers", 0),
                verbose=context.is_main,
            )
        )

        try:
            evaluation = trainer.eval(test)
            best_trial = min(model_trials, key=lambda trial: trial["val_loss"])
            trials = [{"stage": "data", **trial} for trial in data_trials] + [
                {"stage": "model", **trial} for trial in model_trials
            ]

            results = summary(
                args,
                "tcn",
                files,
                best_params,
                evaluation,
                best_trial=best_trial,
                normalization={
                    "mean": None if mean is None else mean.tolist(),
                    "std": None if std is None else std.tolist(),
                },
            )

            if context.is_main:
                prepare_dir(save_dir, args.override)
            context.barrier()

            trainer.save(save_dir / "model.pt")

            if context.is_main:
                save_results(save_dir, trials, results)
                print(f"Best: {best_params}")
                print(f"Validation loss: {best_trial['val_loss']:.6f}")
                print(f"Test metrics: {evaluation['metrics']}")

            context.barrier()
            return results

        finally:
            trainer.close()

    finally:
        context.cleanup()


if __name__ == "__main__":
    main()
