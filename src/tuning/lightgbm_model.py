"""LightGBM hyperparameter tuning."""

from ..data import TabularDataset
from ..models import LightGBMModel
from ..trainers import TabularTrainer, TabularTrainerConfig

from .tabular_tuning import TabularTuning
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
    args = arguments()
    save_dir = output_dir("lightgbm", args)

    if save_dir.exists() and not args.override:
        raise FileExistsError(f"Output directory already exists: {save_dir}")

    best_model, best_params, trials = TabularTuning(
        model_cls=LightGBMModel,
        model_params=list(
            grid_search(
                {
                    "learning_rate": [0.03, 0.1],
                    "num_leaves": [15, 31, 63],
                    "min_child_samples": [100, 500, 2000],
                }
            )
        ),
        trainer_params=[{}],
        data_params=[
            {
                "raw_feature_names": None,
                "engineered_feature_names": "all",
                "normalize": False,
            }
        ],
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
        data_mode=args.data_mode,
    ).run()

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

    test = TabularDataset(
        files[2],
        n_samples=args.n_test_samples,
        mean=mean,
        std=std,
        **dataset_kwargs,
    )

    trainer = TabularTrainer(
        TabularTrainerConfig(
            model=best_model,
            task=args.task,
            data_mode=args.data_mode,
            verbose=False,
        )
    )
    evaluation = trainer.eval(test)

    best_trial = min(trials, key=lambda trial: trial["val_loss"])
    results = summary(
        args,
        "lightgbm",
        files,
        best_params,
        evaluation,
        best_trial=best_trial,
        normalization={
            "mean": None if mean is None else mean.tolist(),
            "std": None if std is None else std.tolist(),
        },
    )

    prepare_dir(save_dir, args.override)
    trainer.save(save_dir / "model.joblib")
    save_results(save_dir, trials, results)

    print(f"Best: {best_params}")
    print(f"Validation loss: {best_trial['val_loss']:.6f}")
    print(f"Test metrics: {evaluation['metrics']}")

    return results


if __name__ == "__main__":
    main()
