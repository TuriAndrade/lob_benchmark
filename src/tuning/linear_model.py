from data import TabularDataset
from models import LinearModel
from trainers import TabularTrainer, TabularTrainerConfig
from .common import (
    arguments,
    grid_search,
    normalization,
    output_dir,
    paths,
    save_results,
    set_seed,
    summary,
)

GRID = {
    "regularization": [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
}


def main():
    args = arguments(normalize_default=True)
    directory = output_dir("linear", args)
    set_seed(args.seed)

    train_files, val_files, test_files = paths(args.depth, args.horizon)

    options = dict(
        depth=args.depth,
        horizon=args.horizon,
        task=args.task,
        threshold_bps=args.threshold,
        normalize=args.normalize,
        seed=args.seed,
    )

    train = TabularDataset(train_files, n_samples=args.train_samples, **options)
    mean, std = train.get_stats()

    val = TabularDataset(
        val_files,
        n_samples=args.val_samples,
        mean=mean,
        std=std,
        **options,
    )

    trials = []
    best = best_result = best_trainer = None

    for i, params in enumerate(grid_search(GRID), 1):
        print(f"\nTrial {i}: {params}")

        trainer = TabularTrainer(
            TabularTrainerConfig(
                model=LinearModel(task=args.task, **params),
                task=args.task,
                verbose=True,
            )
        )

        result = trainer.fit(train, val)
        trials.append({"params": params, **result})

        if best_result is None or result["val_loss"] < best_result["val_loss"]:
            best, best_result, best_trainer = params, result, trainer

    print(f"\nBest configuration: {best}")
    print(f"Validation loss: {best_result['val_loss']:.6f}")

    print("\nFinal test evaluation...")
    test = TabularDataset(
        test_files,
        n_samples=args.test_samples,
        mean=mean,
        std=std,
        **options,
    )
    evaluation = best_trainer.eval(test)

    best_trainer.save(directory / "model.joblib")

    results = summary(
        args,
        train_files,
        val_files,
        test_files,
        best,
        best_result,
        evaluation,
    )
    results["model"] = "linear"
    results["feature_columns"] = list(train.feature_columns)
    results["normalization"] = normalization(train)

    save_results(directory, trials, results)
    print(f"\nSaved results to {directory}")


if __name__ == "__main__":
    main()
