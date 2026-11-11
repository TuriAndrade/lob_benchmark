"""Shared helpers for LOB model tuning."""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from datetime import date
from functools import lru_cache
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import torch


def arguments(sequence=False):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--target-name", default="target_return_50_bps")
    parser.add_argument(
        "--task",
        choices=("regression", "classification"),
        default="regression",
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--n-train-days", type=int, default=8)
    parser.add_argument("--n-val-days", type=int, default=1)
    parser.add_argument("--n-test-days", type=int, default=1)
    parser.add_argument("--n-train-samples", type=int)
    parser.add_argument("--n-val-samples", type=int)
    parser.add_argument("--n-test-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--read-batch-size", type=int, default=65_536)
    parser.add_argument(
        "--override",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    if sequence:
        parser.add_argument(
            "--precision",
            choices=("fp32", "bf16"),
            default="fp32",
        )
        parser.add_argument("--batch-size", type=int, default=1024)
    else:
        parser.add_argument(
            "--data-mode",
            choices=("load", "stream"),
            default="load",
        )

    # Launcher arguments. Memory values are specified in GiB.
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-processes", type=int, default=4)
    parser.add_argument("--min-processes", type=int, default=1)
    parser.add_argument("--ram-offset", type=float, default=2)
    parser.add_argument("--vram-offset", type=float, default=1)
    parser.add_argument("--initial-ram", type=float, default=2)
    parser.add_argument("--initial-vram", type=float, default=2)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--memory-tolerance", type=float, default=0.0)
    parser.add_argument("--retry-factor", type=float, default=1.3)
    parser.add_argument("--max-retries", type=int, default=3)

    args = parser.parse_args()

    positive = (
        "n_train_days",
        "n_val_days",
        "n_test_days",
        "read_batch_size",
    )
    if sequence:
        positive += ("batch_size",)

    if any(getattr(args, name) < 1 for name in positive):
        parser.error("Day counts and batch sizes must be positive")

    if any(
        (value := getattr(args, f"n_{split}_samples")) is not None and value < 1
        for split in ("train", "val", "test")
    ):
        parser.error("Sample counts must be positive")

    if not args.target_name or not np.isfinite(args.threshold) or args.threshold < 0:
        parser.error("Invalid target name or classification threshold")

    for name in (
        "ram_offset",
        "vram_offset",
        "initial_ram",
        "initial_vram",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"{name} must be a nonnegative finite GiB value")

    return args


def training_kwargs(args):
    options = dict(
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
        verbose=args.verbose,
    )

    if hasattr(args, "batch_size"):
        options.update(
            batch_size=args.batch_size,
            precision=args.precision,
        )
    else:
        options["data_mode"] = args.data_mode

    return options


def launcher_kwargs(args):
    gib = 1024**3

    return dict(
        gpu_ids=args.gpu_ids,
        max_processes=args.max_processes,
        min_processes=args.min_processes,
        ram_offset=round(args.ram_offset * gib),
        vram_offset=round(args.vram_offset * gib),
        initial_ram=round(args.initial_ram * gib),
        initial_vram=round(args.initial_vram * gib),
        interval=args.interval,
        memory_tolerance=args.memory_tolerance,
        retry_factor=args.retry_factor,
        max_retries=args.max_retries,
        verbose=args.verbose,
    )


def trial_records(completed, failed):
    """Convert launched training trials into persistent result records."""
    records = []

    for status, trials in (("completed", completed), ("failed", failed)):
        for trial in trials:
            training = trial.fn.__self__

            record = dict(
                status=status,
                attempts=trial.attempts,
                model_params=training.model_params,
                trainer_params=training.trainer_params,
                data_params=training.data_params,
                estimated_ram=trial.estimated_ram,
                estimated_vram=trial.estimated_vram,
            )

            if status == "completed":
                record.update(
                    {
                        key: value
                        for key, value in trial.result.items()
                        if key != "model_path"
                    }
                )
            else:
                record["error"] = trial.error

            records.append(record)

    return records


@lru_cache(maxsize=32)
def paths(
    data_path,
    n_train_days=4,
    n_val_days=1,
    n_test_days=1,
    *,
    source_depth=None,
):
    """Select chronological days, retaining all HDF5 parts per date."""
    root = Path(data_path)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    if n_train_days < 1 or n_val_days < 1 or n_test_days < 0:
        raise ValueError("Invalid train/validation/test day counts")

    groups = defaultdict(lambda: defaultdict(list))

    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".h5", ".hdf5"):
            continue

        with h5py.File(path, "r") as h5:
            depth = int(h5.attrs["source_depth"])
            day = h5.attrs["date"]
            if isinstance(day, bytes):
                day = day.decode()
            day = date.fromisoformat(str(day))

        groups[depth][day].append(path)

    if not groups:
        raise ValueError(f"No HDF5 files found in {root}")

    if source_depth is None:
        if len(groups) != 1:
            raise ValueError(
                f"Multiple source depths found: {sorted(groups)}. Specify source_depth."
            )
        source_depth = next(iter(groups))

    if source_depth not in groups:
        raise ValueError(
            f"Source depth {source_depth} not found; available: {sorted(groups)}"
        )

    dated_files = groups[source_depth]
    needed = n_train_days + n_val_days + n_test_days

    if len(dated_files) < needed:
        raise ValueError(
            f"Requested {needed} days at source depth {source_depth}; "
            f"found {len(dated_files)}"
        )

    dates = sorted(dated_files)[:needed]
    train_end = n_train_days
    val_end = n_train_days + n_val_days

    def flatten(days):
        return [path for day in days for path in dated_files[day]]

    return (
        flatten(dates[:train_end]),
        flatten(dates[train_end:val_end]),
        flatten(dates[val_end:]),
    )


def data_options(params, *, sequence):
    """Separate file selection from dataset parameters."""
    allowed = {
        "source_depth",
        "raw_feature_names",
        "engineered_feature_names",
        "normalize",
    }
    if sequence:
        allowed |= {"window_size", "stride", "include_engineered"}

    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"Unknown data parameters: {sorted(unknown)}")

    options = {
        "raw_feature_names": "all" if sequence else None,
        "engineered_feature_names": None if sequence else "all",
        "normalize": False,
    }
    options.update(
        {
            key: value
            for key, value in params.items()
            if key not in ("source_depth", "include_engineered")
        }
    )

    if "include_engineered" in params:
        if "engineered_feature_names" in params:
            raise ValueError(
                "Specify either include_engineered or engineered_feature_names"
            )
        options["engineered_feature_names"] = (
            "all" if params["include_engineered"] else None
        )

    return params.get("source_depth"), options


def grid_search(grid):
    for values in product(*(grid[key] for key in grid)):
        yield dict(zip(grid, values))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def output_dir(model, args):
    if args.save_dir is not None:
        return args.save_dir

    target = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.target_name)
    name = (
        f"{args.task}_{target}"
        f"_days{args.n_train_days}-{args.n_val_days}-{args.n_test_days}"
    )

    if args.task == "classification":
        name += f"_thr{args.threshold:g}"

    for split in ("train", "val", "test"):
        count = getattr(args, f"n_{split}_samples")
        if count is not None:
            name += f"_{split}{count}"

    name += f"_seed{args.seed}"

    if hasattr(args, "batch_size"):
        name += f"_{args.precision}_b{args.batch_size}"
    else:
        name += f"_{args.data_mode}"

    return Path(os.environ.get("RESULTS_DIR", "./results")) / model / name


def prepare_dir(directory, override=False):
    path = Path(directory)
    if path.exists() and not override:
        raise FileExistsError(f"Output directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_results(directory, trials, results):
    root = Path(directory)

    for name, value in (
        ("trials.json", trials),
        ("results.json", results),
    ):
        (root / name).write_text(
            json.dumps(
                value,
                indent=2,
                allow_nan=False,
                default=_json_default,
            ),
            encoding="utf-8",
        )


def summary(
    args,
    model,
    files,
    best_entry,
    evaluation,
    *,
    best_trial=None,
    normalization=None,
):
    return {
        "model": model,
        "data_path": str(args.data_path),
        "task": args.task,
        "target_name": args.target_name,
        "threshold_bps": (args.threshold if args.task == "classification" else None),
        "seed": args.seed,
        "read_batch_size": args.read_batch_size,
        "n_train_days": args.n_train_days,
        "n_val_days": args.n_val_days,
        "n_test_days": args.n_test_days,
        "n_train_samples": args.n_train_samples,
        "n_val_samples": args.n_val_samples,
        "n_test_samples": args.n_test_samples,
        "precision": getattr(args, "precision", None),
        "batch_size": getattr(args, "batch_size", None),
        "data_mode": getattr(args, "data_mode", None),
        "train_files": list(map(str, files[0])),
        "validation_files": list(map(str, files[1])),
        "test_files": list(map(str, files[2])),
        "best_params": best_entry,
        "best_val_loss": (None if best_trial is None else best_trial["val_loss"]),
        "best_iteration": (
            None if best_trial is None else best_trial["best_iteration"]
        ),
        "normalization": normalization,
        "test_metrics": evaluation["metrics"],
        "test_n_samples": evaluation["n_samples"],
    }
