"""Linear model hyperparameter tuning."""

from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

from ..data import TabularDataset
from ..models import LinearModel
from ..trainers import TabularTrainer, TabularTrainerConfig
from .launcher import Trial, TuningLauncher
from .tabular_training import TabularTraining
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
    args = arguments()
    save_dir = prepare_dir(output_dir("linear", args), args.override)

    data_params = dict(
        raw_feature_names=None,
        engineered_feature_names="all",
        normalize=True,
    )

    # Hyperparameter tuning.
    with TemporaryDirectory(prefix="trials_", dir=save_dir) as directory:
        trials = [
            Trial(
                fn=TabularTraining(
                    model_cls=LinearModel,
                    model_params=params,
                    trainer_params={},
                    data_params=data_params,
                    save_path=Path(directory) / f"{i}.joblib",
                    **training_kwargs(args),
                ).run,
                estimated_vram=0,
            )
            for i, params in enumerate(
                grid_search(
                    {
                        "regularization": [
                            0.001,
                            0.01,
                            0.03,
                            0.1,
                            0.3,
                            1.0,
                            3.0,
                            10.0,
                        ],
                    }
                )
            )
        ]

        completed, failed = TuningLauncher(trials, **launcher_kwargs(args)).run()

        if not completed:
            raise RuntimeError("All linear model trials failed")

        best = min(completed, key=lambda t: t.result["val_loss"]).result

        checkpoint = save_dir / "model.joblib"
        copy2(best["model_path"], checkpoint)
        trial_rows = trial_records(completed, failed)

    # Final evaluation on held-out test days.
    best_params = {
        key: best[key] for key in ("model_params", "trainer_params", "data_params")
    }

    source_depth, options = data_options(best_params["data_params"], sequence=False)
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
        **options,
    )

    train = TabularDataset(
        files[0],
        n_samples=args.n_train_samples,
        **dataset_kwargs,
    )
    mean, std = train.get_stats()
    del train

    test = TabularDataset(
        files[2],
        n_samples=args.n_test_samples,
        mean=mean,
        std=std,
        **dataset_kwargs,
    )

    trainer = TabularTrainer(
        TabularTrainerConfig(
            model=LinearModel.load(checkpoint),
            task=args.task,
            data_mode=args.data_mode,
            verbose=args.verbose,
        )
    )
    evaluation = trainer.eval(test)

    results = summary(
        args,
        "linear",
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

    save_results(save_dir, trial_rows, results)

    print(f"Best: {best_params}")
    print(f"Validation loss: {best['val_loss']:.6f}")
    print(f"Test metrics: {evaluation['metrics']}")

    return results


if __name__ == "__main__":
    main()
