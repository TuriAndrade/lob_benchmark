"""Shared utilities for tuning experiments."""

import argparse
import json
import os
import random
import re
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()


def paths(data_path, depth=5, horizon=50, n_train_days=4, n_val_days=1, n_test_days=1):
    root = Path(data_path)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    counts = (n_train_days, n_val_days, n_test_days)
    if any(n < 1 for n in counts):
        raise ValueError("Day counts must be positive")

    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2})_depth(\d+)_h(\d+(?:-\d+)*)_tick(.+)\.(?:h5|hdf5)"
    )
    groups = {}

    for path in root.iterdir():
        match = pattern.fullmatch(path.name)
        if not path.is_file() or match is None:
            continue

        day, file_depth, horizons, tick = match.groups()
        if int(file_depth) != depth or horizon not in map(int, horizons.split("-")):
            continue

        date.fromisoformat(day)
        groups.setdefault((horizons, tick), []).append((day, path))

    valid = []
    needed = sum(counts)

    for files in groups.values():
        files.sort()
        if len({day for day, _ in files}) != len(files):
            raise ValueError("Duplicate dates found in input files")
        if len(files) >= needed:
            valid.append([path for _, path in files[:needed]])

    if len(valid) != 1:
        raise ValueError(
            f"Expected one compatible group with at least {needed} dates "
            f"in {root}; found {len(valid)}"
        )

    files = valid[0]
    a, b = n_train_days, n_train_days + n_val_days
    return files[:a], files[a:b], files[b:]


def grid_search(grid):
    for values in product(*(grid[key] for key in grid)):
        yield dict(zip(grid, values))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def arguments(normalize_default=False, sequence=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument(
        "--task", choices=["regression", "classification"], default="regression"
    )
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--n-train-days", type=int, default=4)
    parser.add_argument("--n-val-days", type=int, default=1)
    parser.add_argument("--n-test-days", type=int, default=1)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--val-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--read-batch-size", type=int, default=65_536)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=normalize_default,
    )
    parser.add_argument(
        "--override", action=argparse.BooleanOptionalAction, default=False
    )

    if sequence:
        parser.add_argument("--window-size", type=int, default=100)
        parser.add_argument("--stride", type=int, default=1)
        parser.add_argument("--batch-size", type=int, default=1024)
        parser.add_argument(
            "--include-engineered",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--shuffle", action=argparse.BooleanOptionalAction, default=True
        )
        parser.add_argument(
            "--drop-last", action=argparse.BooleanOptionalAction, default=True
        )
        parser.add_argument(
            "--mode", choices=["single", "ddp", "fsdp"], default="single"
        )
        parser.add_argument("--device", default="auto")
        parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    else:
        parser.add_argument("--data-mode", choices=["load", "stream"], default="load")

    args = parser.parse_args()

    positive = (
        "depth",
        "horizon",
        "n_train_days",
        "n_val_days",
        "n_test_days",
        "read_batch_size",
    )
    if sequence:
        positive += ("window_size", "stride", "batch_size")

    if any(getattr(args, name) < 1 for name in positive):
        parser.error(f"Must be positive: {', '.join(positive)}")

    if any(
        getattr(args, name) is not None and getattr(args, name) < 1
        for name in ("train_samples", "val_samples", "test_samples")
    ):
        parser.error("Sample counts must be positive")

    if not np.isfinite(args.threshold) or args.threshold < 0:
        parser.error("threshold must be finite and nonnegative")

    if sequence and args.normalize and not args.include_engineered:
        parser.error("--normalize requires --include-engineered")

    return args


def output_dir(model, args):
    if args.save_dir is not None:
        return args.save_dir

    name = (
        f"{args.task}_h{args.horizon}_d{args.depth}"
        f"_days{args.n_train_days}-{args.n_val_days}-{args.n_test_days}"
    )
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

    if hasattr(args, "window_size"):
        name += (
            f"_w{args.window_size}_s{args.stride}_b{args.batch_size}"
            f"_eng{int(args.include_engineered)}_sh{int(args.shuffle)}"
            f"_dl{int(args.drop_last)}_{args.mode}_{args.precision}"
        )
    else:
        name += f"_{args.data_mode}"

    return Path(os.environ.get("RESULTS_DIR", "./results")) / model / name


def prepare_dir(directory, override):
    directory = Path(directory)
    if directory.exists() and not override:
        raise FileExistsError(f"Output directory already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)


def save_results(directory, trials, results):
    for name, data in (("trials.json", trials), ("results.json", results)):
        (Path(directory) / name).write_text(
            json.dumps(data, indent=2, allow_nan=False),
            encoding="utf-8",
        )


def summary(config, files, train, best_result, evaluation):
    mean, std = train.get_stats()
    return {
        "model": config.model_cls.__name__.removesuffix("Model").lower(),
        "data_path": str(config.data_path),
        "task": config.task,
        "depth": config.depth,
        "horizon": config.horizon,
        "threshold_bps": (
            config.threshold_bps if config.task == "classification" else None
        ),
        "seed": config.seed,
        "normalize": config.normalize,
        "n_train_days": config.n_train_days,
        "n_val_days": config.n_val_days,
        "n_test_days": config.n_test_days,
        "train_samples_requested": config.n_train_samples,
        "val_samples_requested": config.n_val_samples,
        "test_samples_requested": config.n_test_samples,
        "train_files": [str(p) for p in files[0]],
        "validation_files": [str(p) for p in files[1]],
        "test_files": [str(p) for p in files[2]],
        "feature_columns": list(train.feature_columns),
        "normalization": {
            "mean": None if mean is None else mean.tolist(),
            "std": None if std is None else std.tolist(),
        },
        "best_val_loss": best_result["val_loss"],
        "best_iteration": best_result["best_iteration"],
        "test_metrics": evaluation["metrics"],
        "test_n_samples": evaluation["n_samples"],
    }


def tuning_kwargs(args, model):
    return dict(
        data_path=args.data_path,
        save_dir=output_dir(model, args),
        task=args.task,
        depth=args.depth,
        horizon=args.horizon,
        threshold_bps=args.threshold,
        n_train_days=args.n_train_days,
        n_val_days=args.n_val_days,
        n_test_days=args.n_test_days,
        n_train_samples=args.train_samples,
        n_val_samples=args.val_samples,
        n_test_samples=args.test_samples,
        normalize=args.normalize,
        seed=args.seed,
        read_batch_size=args.read_batch_size,
        override=args.override,
    )
