import argparse
import json
import os
import random
import re
from itertools import product
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()


def paths(depth=5, horizon=50):
    root = Path(os.environ.get("DATA_DIR", "./")) / "tardis_book_snapshot" / "processed"
    groups = {}

    for path in root.rglob(f"*_depth{depth}_h*_tick*.parquet"):
        match = re.fullmatch(
            rf"(\d{{4}}-\d{{2}}-\d{{2}})_depth{depth}_h([\d-]+)_tick(.+)",
            path.stem,
        )
        if match is None:
            continue

        date, horizons, tick = match.groups()
        if horizon not in map(int, horizons.split("-")):
            continue

        key = (path.parent, horizons, tick)
        groups.setdefault(key, []).append((date, path))

    valid = [
        sorted(files)[:6]
        for files in groups.values()
        if len(files) >= 6 and len({date for date, _ in files}) == len(files)
    ]

    if len(valid) != 1:
        raise ValueError(
            f"Expected one dataset with at least 6 unique dates "
            f"for depth={depth}, horizon={horizon}; found {len(valid)}. "
            "Check DATA_DIR and preprocessing configurations."
        )

    files = [path for _, path in valid[0]]
    return files[:4], files[4:5], files[5:6]


def grid_search(grid):
    keys = list(grid)
    for values in product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def arguments(normalize_default=False, sequence=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
    )
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=normalize_default,
    )
    parser.add_argument(
        "--override",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    if sequence:
        parser.add_argument(
            "--include-engineered",
            action=argparse.BooleanOptionalAction,
            default=False,
        )

    args = parser.parse_args()

    if args.horizon < 1 or args.depth < 1:
        parser.error("horizon and depth must be positive")
    if args.threshold < 0:
        parser.error("threshold must be nonnegative")

    for name in ("train_samples", "val_samples", "test_samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")

    if sequence and args.normalize and not args.include_engineered:
        parser.error("--normalize requires --include-engineered for TCN")

    return args


def output_dir(model, args, override=False, **settings):
    name = f"{args.task}_h{args.horizon}_d{args.depth}"

    if args.task == "classification":
        name += f"_thr{args.threshold:g}"

    name += f"_norm{int(args.normalize)}_seed{args.seed}"

    for label, value in (
        ("tr", args.train_samples),
        ("va", args.val_samples),
        ("te", args.test_samples),
    ):
        if value is not None:
            name += f"_{label}{value}"

    for key, value in settings.items():
        name += f"_{key}{value}"

    directory = Path(os.environ.get("RESULTS_DIR", "./results")) / model / name

    if directory.exists() and not override:
        raise SystemExit(f"Output directory already exists: {directory}")

    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_results(directory, trials, results):
    for name, content in (
        ("trials.json", trials),
        ("results.json", results),
    ):
        (directory / name).write_text(
            json.dumps(content, indent=2, allow_nan=False),
            encoding="utf-8",
        )


def summary(args, train, val, test, best, best_result, evaluation):
    return {
        "task": args.task,
        "horizon": args.horizon,
        "threshold_bps": args.threshold if args.task == "classification" else None,
        "depth": args.depth,
        "seed": args.seed,
        "normalize": args.normalize,
        "train_samples_requested": args.train_samples,
        "val_samples_requested": args.val_samples,
        "test_samples_requested": args.test_samples,
        "train_files": [str(p) for p in train],
        "validation_files": [str(p) for p in val],
        "test_files": [str(p) for p in test],
        "best_params": best,
        "best_val_loss": best_result["val_loss"],
        "best_iteration": best_result["best_iteration"],
        "test_metrics": evaluation["metrics"],
        "test_n_samples": evaluation["n_samples"],
    }


def normalization(dataset):
    mean, std = dataset.get_stats()
    return {
        "mean": None if mean is None else mean.tolist(),
        "std": None if std is None else std.tolist(),
    }
