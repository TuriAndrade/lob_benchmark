"""HDF5 datasets for tabular and sequence LOB prediction."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class _File:
    path: Path
    rows: int
    horizon_index: int


def _names(attrs, key):
    return tuple(x.decode() if isinstance(x, bytes) else str(x) for x in attrs[key])


def _inspect_files(paths, horizon):
    files = []
    reference = None

    for path in map(Path, paths):
        with h5py.File(path, "r") as h5:
            attrs = h5["attributes"].attrs
            raw = _names(attrs, "raw_feature_names")
            engineered = _names(attrs, "engineered_feature_names")
            horizons = tuple(int(h) for h in attrs["horizons"])
            if horizon not in horizons:
                raise ValueError(f"Horizon {horizon} is not available in {path}")

            config = (raw, engineered) + tuple(
                tuple(np.atleast_1d(attrs[k]).tolist())
                for k in (
                    "depth",
                    "depth_levels",
                    "return_lags",
                    "ofi_windows",
                    "volatility_window",
                    "event_rate_window",
                )
            )
            if reference is None:
                reference = config
            elif config != reference:
                raise ValueError(f"Incompatible feature configuration: {path}")

            n = h5["raw_features"].shape[0]
            shapes = {
                "raw_features": ((n, len(raw)), np.float32),
                "engineered_features": ((n, len(engineered)), np.float32),
                "targets": ((n, len(horizons)), np.float32),
                "valid_book": ((n,), np.bool_),
                "metadata/event_id": ((n,), np.int64),
                "metadata/local_timestamp_us": ((n,), np.int64),
                "metadata/target_end_event": ((n, len(horizons)), np.int64),
                "metadata/target_end_timestamp_us": ((n, len(horizons)), np.int64),
            }
            for name, (shape, dtype) in shapes.items():
                if h5[name].shape != shape or h5[name].dtype != np.dtype(dtype):
                    raise ValueError(f"Invalid dataset {name} in {path}")
            if n == 0:
                raise ValueError(f"Empty HDF5 file: {path}")
            files.append(_File(path, n, horizons.index(horizon)))

    if not files:
        raise ValueError("At least one HDF5 file is required")
    return tuple(files), reference[0], reference[1]


def _compute_stats(files, read_batch_size):
    """Mean/std of finite engineered rows with valid books (training files only)."""
    count = 0
    mean = m2 = None

    for file in files:
        with h5py.File(file.path, "r") as h5:
            features = h5["engineered_features"]
            for start in range(0, file.rows, read_batch_size):
                stop = min(start + read_batch_size, file.rows)
                x = features[start:stop]
                valid = h5["valid_book"][start:stop] & np.isfinite(x).all(axis=1)
                x = x[valid].astype(np.float64)
                if not len(x):
                    continue

                n = len(x)
                avg = x.mean(axis=0)
                if mean is None:
                    mean = avg
                    m2 = ((x - avg) ** 2).sum(axis=0)
                else:
                    delta = avg - mean
                    mean += delta * n / (count + n)
                    m2 += ((x - avg) ** 2).sum(axis=0) + delta**2 * count * n / (
                        count + n
                    )
                count += n

    if not count:
        raise ValueError("No valid engineered observations for normalization")
    return mean.astype(np.float32), np.maximum(np.sqrt(m2 / count), 1e-6).astype(
        np.float32
    )


def _normalization(normalize, mean, std, count, files, read_batch_size):
    if not normalize:
        return None, None
    if (mean is None) != (std is None):
        raise ValueError("Provide both mean and std, or neither")
    if mean is None:
        mean, std = _compute_stats(files, read_batch_size)
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    if (
        mean.shape != (count,)
        or std.shape != (count,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("Invalid normalization statistics")
    return mean, std


def _sample_indices(indices, n_samples, seed):
    """Sample uniformly without replacement across all files."""
    if n_samples is None:
        return indices
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    counts = [len(x) for x in indices]
    total = sum(counts)
    if n_samples >= total:
        return indices

    selected = np.sort(
        np.random.default_rng(seed).choice(total, n_samples, replace=False)
    )
    offsets = np.cumsum([0, *counts], dtype=np.int64)
    sampled = []
    for i, values in enumerate(indices):
        lo, hi = np.searchsorted(selected, offsets[i : i + 2])
        sampled.append(values[selected[lo:hi] - offsets[i]])
    return sampled


def _labels(y, task, threshold_bps):
    if task == "regression":
        return y
    return np.where(y < -threshold_bps, 0, np.where(y > threshold_bps, 2, 1)).astype(
        np.int64
    )


def _metadata(h5, file_id, horizon_index, start, stop, indices):
    """Read endpoint metadata, preserving only the selected rows."""
    meta = h5["metadata"]
    return {
        "source_id": np.full(len(indices), file_id, dtype=np.int64),
        "event_id": meta["event_id"][start:stop][indices],
        "local_timestamp_us": meta["local_timestamp_us"][start:stop][indices],
        "target_end_event": meta["target_end_event"][start:stop, horizon_index][
            indices
        ],
        "target_end_timestamp_us": meta["target_end_timestamp_us"][
            start:stop, horizon_index
        ][indices],
    }


def _validate_task(task, threshold_bps, read_batch_size):
    if task not in ("regression", "classification"):
        raise ValueError("task must be 'regression' or 'classification'")
    if not np.isfinite(threshold_bps) or threshold_bps < 0:
        raise ValueError("threshold_bps must be finite and nonnegative")
    if not isinstance(read_batch_size, int) or read_batch_size < 1:
        raise ValueError("read_batch_size must be a positive integer")


class TabularDataset:
    """Sequential HDF5 reads for classical machine-learning models."""

    def __init__(
        self,
        paths,
        horizon=50,
        task="regression",
        threshold_bps=0.25,
        n_samples=None,
        normalize=False,
        mean=None,
        std=None,
        seed=42,
        read_batch_size=65_536,
    ):
        _validate_task(task, threshold_bps, read_batch_size)
        self.files, _, self.feature_columns = _inspect_files(paths, horizon)
        self.paths = tuple(f.path for f in self.files)
        self.task = task
        self.threshold_bps = threshold_bps
        self.read_batch_size = read_batch_size
        self.normalize = normalize
        self.mean, self.std = _normalization(
            normalize, mean, std, len(self.feature_columns), self.files, read_batch_size
        )

        self.selected = None
        if n_samples is not None:
            counts = [self._count_valid(file) for file in self.files]
            if not isinstance(n_samples, int) or n_samples < 1:
                raise ValueError("n_samples must be a positive integer")
            total = sum(counts)
            if not total:
                raise ValueError("No eligible tabular observations")
            if n_samples < total:
                chosen = np.sort(
                    np.random.default_rng(seed).choice(total, n_samples, replace=False)
                )
                offsets = np.cumsum([0, *counts], dtype=np.int64)
                self.selected = [
                    chosen[
                        np.searchsorted(chosen, offsets[i]) : np.searchsorted(
                            chosen, offsets[i + 1]
                        )
                    ]
                    - offsets[i]
                    for i in range(len(counts))
                ]

    def _count_valid(self, file):
        total = 0
        with h5py.File(file.path, "r") as h5:
            for start in range(0, file.rows, self.read_batch_size):
                stop = min(start + self.read_batch_size, file.rows)
                x = h5["engineered_features"][start:stop]
                y = h5["targets"][start:stop, file.horizon_index]
                valid = h5["valid_book"][start:stop]
                total += int(
                    np.count_nonzero(
                        valid & np.isfinite(y) & np.isfinite(x).all(axis=1)
                    )
                )
        return total

    def get_stats(self):
        return self.mean, self.std

    def stream(self):
        for file_id, file in enumerate(self.files):
            selected = None if self.selected is None else self.selected[file_id]
            if selected is not None and len(selected) == 0:
                continue
            seen = 0
            with h5py.File(file.path, "r") as h5:
                for start in range(0, file.rows, self.read_batch_size):
                    stop = min(start + self.read_batch_size, file.rows)
                    x = h5["engineered_features"][start:stop]
                    y = h5["targets"][start:stop, file.horizon_index]
                    valid = h5["valid_book"][start:stop]
                    indices = np.flatnonzero(
                        valid & np.isfinite(y) & np.isfinite(x).all(axis=1)
                    )

                    if selected is not None:
                        count = len(indices)
                        lo, hi = np.searchsorted(selected, [seen, seen + count])
                        indices = indices[selected[lo:hi] - seen]
                        seen += count
                    if not len(indices):
                        continue

                    xb = x[indices]
                    if self.normalize:
                        xb = (xb - self.mean) / self.std
                    yield (
                        xb,
                        _labels(y[indices], self.task, self.threshold_bps),
                        _metadata(
                            h5, file_id, file.horizon_index, start, stop, indices
                        ),
                    )

    def load(self):
        batches = list(self.stream())
        if not batches:
            raise ValueError("No eligible tabular observations")
        return (
            np.concatenate([b[0] for b in batches]),
            np.concatenate([b[1] for b in batches]),
            {k: np.concatenate([b[2][k] for b in batches]) for k in batches[0][2]},
        )


class SequenceDataset(IterableDataset):
    """Indexed, locality-preserving batches of HDF5 sequence windows."""

    def __init__(
        self,
        paths,
        horizon=50,
        task="regression",
        threshold_bps=0.25,
        window_size=100,
        stride=1,
        include_engineered=False,
        batch_size=256,
        n_samples=None,
        normalize=False,
        mean=None,
        std=None,
        seed=42,
        read_batch_size=65_536,
        shuffle=True,
        drop_last=True,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        _validate_task(task, threshold_bps, read_batch_size)
        if any(
            not isinstance(v, int) or v < 1
            for v in (window_size, stride, batch_size, world_size)
        ):
            raise ValueError(
                "window_size, stride, batch_size and world_size must be positive integers"
            )
        if not isinstance(rank, int) or not 0 <= rank < world_size:
            raise ValueError("rank must be between 0 and world_size - 1")

        self.files, self.raw_columns, self.engineered_columns = _inspect_files(
            paths, horizon
        )
        self.paths = tuple(f.path for f in self.files)
        self.include_engineered = include_engineered
        self.feature_columns = self.raw_columns + (
            self.engineered_columns if include_engineered else ()
        )
        self.task = task
        self.threshold_bps = threshold_bps
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.read_batch_size = read_batch_size
        self.seed = seed
        self.epoch = 0
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.normalize = normalize and include_engineered
        self.mean, self.std = _normalization(
            self.normalize,
            mean,
            std,
            len(self.engineered_columns),
            self.files,
            read_batch_size,
        )

        self.endpoints = _sample_indices(
            [self._valid_endpoints(file) for file in self.files], n_samples, seed
        )
        self.offsets = np.cumsum([0, *(len(a) for a in self.endpoints)], dtype=np.int64)
        self.total = int(self.offsets[-1])
        if self.total == 0:
            raise ValueError("No eligible sequence endpoints")

        self.available_batches = (
            self.total // batch_size
            if drop_last
            else (self.total + batch_size - 1) // batch_size
        )
        self.n_batches = self.available_batches
        if drop_last:
            self.n_batches -= self.n_batches % world_size
        elif world_size > 1:
            if self.total < world_size:
                raise ValueError(
                    "Distributed execution needs at least one example per rank"
                )
            self.rank_start = rank * (self.total // world_size) + min(
                rank, self.total % world_size
            )
            self.rank_count = self.total // world_size + (
                rank < self.total % world_size
            )
            self.n_batches = (
                (self.total + world_size - 1) // world_size + batch_size - 1
            ) // batch_size

            if self.rank_count < self.n_batches:
                raise ValueError(
                    "Insufficient samples per rank for nonempty distributed batches"
                )
        if self.n_batches == 0:
            raise ValueError(
                "Not enough eligible endpoints to form distributed batches"
            )

    def _valid_endpoints(self, file):
        """Scan once; retain only endpoints whose complete input window is valid."""
        W = self.window_size
        tail = np.empty(0, bool)
        selected = []
        dtype = np.int32 if file.rows <= np.iinfo(np.int32).max else np.int64

        with h5py.File(file.path, "r") as h5:
            for start in range(0, file.rows, self.read_batch_size):
                stop = min(start + self.read_batch_size, file.rows)
                valid = h5["valid_book"][start:stop]
                valid &= np.isfinite(h5["raw_features"][start:stop]).all(axis=1)
                if self.include_engineered:
                    valid &= np.isfinite(h5["engineered_features"][start:stop]).all(
                        axis=1
                    )

                combined = np.concatenate((tail, valid))
                lengths = len(tail) + np.arange(stop - start)
                positions = start + np.arange(stop - start)
                enough = positions >= W - 1
                complete = np.zeros(stop - start, dtype=bool)
                if enough.any():
                    invalid = np.r_[0, np.cumsum(~combined, dtype=np.int64)]
                    begins = lengths[enough] - W + 1
                    complete[enough] = invalid[lengths[enough] + 1] == invalid[begins]

                target = h5["targets"][start:stop, file.horizon_index]
                eligible = (
                    complete
                    & np.isfinite(target)
                    & ((positions - W + 1) % self.stride == 0)
                )
                ids = positions[eligible].astype(dtype)
                if len(ids):
                    selected.append(ids)
                if W > 1:
                    tail = combined[-(W - 1) :].copy()

        return np.concatenate(selected) if selected else np.empty(0, dtype=dtype)

    def get_stats(self):
        return self.mean, self.std

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if not self.drop_last and self.world_size > 1:
            return self.n_batches
        return len(range(self.rank, self.n_batches, self.world_size))

    def _read_file_batch(self, h5, file_id, endpoints):
        """Read each contiguous interval once; reuse it for overlapping windows."""
        W = self.window_size
        size = len(endpoints)
        x = np.empty((size, W, len(self.feature_columns)), np.float32)
        y = np.empty(size, np.float32)
        metadata = {
            name: np.empty(size, np.int64)
            for name in (
                "source_id",
                "event_id",
                "local_timestamp_us",
                "target_end_event",
                "target_end_timestamp_us",
            )
        }

        splits = np.r_[0, np.flatnonzero(np.diff(endpoints) > W) + 1, size]
        horizon = self.files[file_id].horizon_index
        for left, right in zip(splits[:-1], splits[1:]):
            group = endpoints[left:right]
            first, last = int(group[0]), int(group[-1])
            start, stop = first - W + 1, last + 1

            raw = h5["raw_features"][start:stop]
            if self.include_engineered:
                features = np.concatenate(
                    (raw, h5["engineered_features"][start:stop]), axis=1
                )
            else:
                features = raw

            windows = np.lib.stride_tricks.sliding_window_view(features, W, axis=0)
            x[left:right] = windows[group - first].transpose(0, 2, 1)
            local = group - start
            y[left:right] = h5["targets"][start:stop, horizon][local]
            values = _metadata(h5, file_id, horizon, start, stop, local)
            for name, column in values.items():
                metadata[name][left:right] = column

        if self.normalize:
            raw_count = len(self.raw_columns)
            x[:, :, raw_count:] = (x[:, :, raw_count:] - self.mean) / self.std
        return x, _labels(y, self.task, self.threshold_bps), metadata

    def _read_batch(self, first, stop, opened):
        xs, ys, metas = [], [], []

        while first < stop:
            file_id = int(np.searchsorted(self.offsets[1:], first, side="right"))
            file_stop = min(stop, int(self.offsets[file_id + 1]))
            lo, hi = first - self.offsets[file_id], file_stop - self.offsets[file_id]

            if file_id not in opened:
                if len(opened) == 8:
                    opened.popitem(last=False)[1].close()
                opened[file_id] = h5py.File(self.files[file_id].path, "r")
            opened.move_to_end(file_id)
            x, y, metadata = self._read_file_batch(
                opened[file_id], file_id, self.endpoints[file_id][lo:hi]
            )
            xs.append(x)
            ys.append(y)
            metas.append(metadata)
            first = file_stop

        if len(xs) == 1:
            return xs[0], ys[0], metas[0]
        return (
            np.concatenate(xs),
            np.concatenate(ys),
            {k: np.concatenate([m[k] for m in metas]) for k in metas[0]},
        )

    def __iter__(self):
        worker = get_worker_info()
        worker_id, n_workers = (
            (0, 1) if worker is None else (worker.id, worker.num_workers)
        )

        if not self.drop_last and self.world_size > 1:
            sizes = np.full(
                self.n_batches, self.rank_count // self.n_batches, dtype=np.int64
            )
            sizes[: self.rank_count % self.n_batches] += 1
            starts = self.rank_start + np.r_[0, np.cumsum(sizes[:-1])]

            batch_ids = np.arange(self.n_batches)
            if self.shuffle:
                batch_ids = np.random.default_rng(
                    self.seed + self.epoch + self.rank
                ).permutation(batch_ids)

            batch_ids = batch_ids[worker_id::n_workers]
            ranges = zip(starts[batch_ids], (starts + sizes)[batch_ids])
        else:
            batch_ids = np.arange(self.available_batches, dtype=np.int64)
            if self.shuffle:
                batch_ids = np.random.default_rng(self.seed + self.epoch).permutation(
                    batch_ids
                )

            batch_ids = batch_ids[: self.n_batches][self.rank :: self.world_size]
            ranges = (
                (
                    int(i * self.batch_size),
                    int(min((i + 1) * self.batch_size, self.total)),
                )
                for i in batch_ids[worker_id::n_workers]
            )

        opened = OrderedDict()
        try:
            for first, stop in ranges:
                x, y, metadata = self._read_batch(first, stop, opened)
                yield (
                    torch.from_numpy(x),
                    torch.from_numpy(y),
                    {k: torch.from_numpy(v) for k, v in metadata.items()},
                )
        finally:
            for h5 in opened.values():
                h5.close()
