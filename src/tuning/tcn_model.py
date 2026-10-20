from data import SequenceDataset
from models import TCNModel
from trainers import SequenceTrainer, SequenceTrainerConfig
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
    "channels": [32, 64],
    "num_layers": [3, 4],
    "lr": [1e-4, 1e-3],
}

WINDOW_SIZE = 100
BATCH_SIZE = 1024
EPOCHS = 100
PATIENCE = 5
WEIGHT_DECAY = 1e-4


def main():
    args = arguments(normalize_default=False, sequence=True)
    directory = output_dir(
        "tcn",
        args,
        w=WINDOW_SIZE,
        engineered=int(args.include_engineered),
    )
    set_seed(args.seed)

    train_files, val_files, test_files = paths(args.depth, args.horizon)

    options = dict(
        depth=args.depth,
        horizon=args.horizon,
        task=args.task,
        threshold_bps=args.threshold,
        window_size=WINDOW_SIZE,
        batch_size=BATCH_SIZE,
        include_engineered=args.include_engineered,
        normalize=args.normalize,
        seed=args.seed,
    )

    train = SequenceDataset(
        train_files,
        n_samples=args.train_samples,
        **options,
    )
    mean, std = train.get_stats()

    val = SequenceDataset(
        val_files,
        n_samples=args.val_samples,
        mean=mean,
        std=std,
        **options,
    )

    input_dim = len(train.feature_columns)
    trials = []
    best = best_result = best_trainer = None

    for i, params in enumerate(grid_search(GRID), 1):
        print(f"\nTrial {i}: {params}")
        set_seed(args.seed)

        model_params = {k: v for k, v in params.items() if k != "lr"}

        trainer = SequenceTrainer(
            SequenceTrainerConfig(
                model=TCNModel(
                    input_dim=input_dim,
                    task=args.task,
                    **model_params,
                ),
                task=args.task,
                epochs=EPOCHS,
                patience=PATIENCE,
                lr=params["lr"],
                weight_decay=WEIGHT_DECAY,
                warmup_epochs=0,
                steps_per_epoch=None,
                verbose=True,
                device="cuda",
            )
        )

        result = trainer.fit(train, val)
        trials.append({"params": params, **result})

        if best_result is None or result["val_loss"] < best_result["val_loss"]:
            best, best_result, best_trainer = params, result, trainer

    print(f"\nBest configuration: {best}")
    print(f"Validation loss: {best_result['val_loss']:.6f}")
    print(f"Best epoch: {best_result['best_iteration']}")

    print("\nFinal test evaluation...")
    test = SequenceDataset(
        test_files,
        n_samples=args.test_samples,
        mean=mean,
        std=std,
        **options,
    )
    evaluation = best_trainer.eval(test)
    best_trainer.save(directory / "model.pt")

    results = summary(
        args,
        train_files,
        val_files,
        test_files,
        best,
        best_result,
        evaluation,
    )
    results.update(
        {
            "model": "tcn",
            "input_dim": input_dim,
            "window_size": WINDOW_SIZE,
            "batch_size": BATCH_SIZE,
            "include_engineered": args.include_engineered,
            "learning_rate": best["lr"],
            "weight_decay": WEIGHT_DECAY,
            "feature_columns": list(train.feature_columns),
            "normalization": normalization(train),
        }
    )

    save_results(directory, trials, results)
    print(f"\nSaved results to {directory}")


if __name__ == "__main__":
    main()
