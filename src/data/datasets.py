"""Source-independent HDF5 datasets for tabular and sequence prediction."""

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
    raw_indices: tuple[int, ...]
    engineered_indices: tuple[int, ...]
    target_index: int


def _names(attrs, key):
    return tuple(x.decode() if isinstance(x, bytes) else str(x) for x in attrs[key])


def _selection(request, available, group):
    if request is None:
        return ()
    if isinstance(request, str):
        if request != "all":
            raise ValueError(f"{group} must be 'all', None, or a sequence of names")
        return available
    names = tuple(request)
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(
        names
    ):
        raise ValueError(f"{group} must contain unique feature names")
    return names


def _inspect_files(paths, raw_request, engineered_request, target_name):
    if isinstance(paths, (str, Path)):
        paths = (paths,)
    if not isinstance(target_name, str) or not target_name:
        raise ValueError("target_name must be a nonempty string")

    files = []
    raw_names = engineered_names = None
    metadata = {"timestamp_us", "target_end_timestamp_us"}

    for path in map(Path, paths):
        with h5py.File(path, "r") as h5:
            raw = _names(h5.attrs, "raw_feature_names")
            engineered = _names(h5.attrs, "engineered_feature_names")
            targets = _names(h5.attrs, "target_names")
            if any(
                len(set(names)) != len(names) for names in (raw, engineered, targets)
            ):
                raise ValueError(f"Duplicate feature or target names in {path}")

            if raw_names is None:
                raw_names = _selection(raw_request, raw, "raw_feature_names")
                engineered_names = _selection(
                    engineered_request, engineered, "engineered_feature_names"
                )
                if not raw_names and not engineered_names:
                    raise ValueError("Select at least one raw or engineered feature")

            indices = []
            for requested, available, group in (
                (raw_names, raw, "raw_features"),
                (engineered_names, engineered, "engineered_features"),
            ):
                mapping = {name: i for i, name in enumerate(available)}
                missing = set(requested) - mapping.keys()
                if missing:
                    raise ValueError(f"Missing {group} in {path}: {sorted(missing)}")
                indices.append(tuple(mapping[name] for name in requested))

            if target_name not in targets:
                raise ValueError(f"Missing target {target_name!r} in {path}")
            n = h5["valid_book"].shape[0]
            for key, shape, dtype in (
                ("raw_features", (n, len(raw)), np.float32),
                ("engineered_features", (n, len(engineered)), np.float32),
                ("targets", (n, len(targets)), np.float32),
                ("valid_book", (n,), np.bool_),
            ):
                if h5[key].shape != shape or h5[key].dtype != np.dtype(dtype):
                    raise ValueError(f"Invalid {key} in {path}")
            if n == 0:
                raise ValueError(f"Empty HDF5 file: {path}")

            available_metadata = set()
            if "metadata" in h5:
                if "timestamp_us" in h5["metadata"] and h5[
                    "metadata/timestamp_us"
                ].shape == (n,):
                    available_metadata.add("timestamp_us")
                if "target_end_timestamp_us" in h5["metadata"] and h5[
                    "metadata/target_end_timestamp_us"
                ].shape == (n, len(targets)):
                    available_metadata.add("target_end_timestamp_us")
            metadata &= available_metadata
            files.append(_File(path, n, *indices, targets.index(target_name)))

    if not files:
        raise ValueError("At least one HDF5 file is required")
    return tuple(files), raw_names, engineered_names, tuple(sorted(metadata))


def _read_features(h5, file, start, stop):
    arrays = []
    for key, indices in (
        ("raw_features", file.raw_indices),
        ("engineered_features", file.engineered_indices),
    ):
        if indices:
            arrays.append(h5[key][start:stop][:, indices])
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1)


def _compute_stats(files, batch_size):
    count = 0
    mean = m2 = None
    for file in files:
        with h5py.File(file.path, "r") as h5:
            for start in range(0, file.rows, batch_size):
                stop = min(start + batch_size, file.rows)
                x = _read_features(h5, file, start, stop)
                x = x[h5["valid_book"][start:stop] & np.isfinite(x).all(axis=1)]
                if not len(x):
                    continue
                x = x.astype(np.float64)
                n = len(x)
                avg = x.mean(axis=0)
                variance_sum = np.square(x - avg).sum(axis=0)
                if mean is None:
                    mean, m2 = avg, variance_sum
                else:
                    delta = avg - mean
                    m2 += variance_sum + delta**2 * count * n / (count + n)
                    mean += delta * n / (count + n)
                count += n
    if count == 0:
        raise ValueError("No valid observations for normalization")
    return mean.astype(np.float32), np.maximum(np.sqrt(m2 / count), 1e-6).astype(
        np.float32
    )


def _normalization(normalize, mean, std, files, count, batch_size):
    if not normalize:
        return None, None
    if (mean is None) != (std is None):
        raise ValueError("Provide both mean and std, or neither")
    if mean is None:
        mean, std = _compute_stats(files, batch_size)
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
    if n_samples is None:
        return indices
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    counts = np.array([len(x) for x in indices], dtype=np.int64)
    total = int(counts.sum())
    if n_samples >= total:
        return indices
    selected = np.sort(
        np.random.default_rng(seed).choice(total, n_samples, replace=False)
    )
    offsets = np.r_[0, np.cumsum(counts)]
    result = []
    for i, values in enumerate(indices):
        lo, hi = np.searchsorted(selected, offsets[i : i + 2])
        result.append(values[selected[lo:hi] - offsets[i]])
    return result


def _labels(y, task, threshold_bps):
    if task == "regression":
        return y
    return np.where(y < -threshold_bps, 0, np.where(y > threshold_bps, 2, 1)).astype(
        np.int64
    )


def _metadata(h5, file_id, file, names, start, stop, indices):
    result = {"source_id": np.full(len(indices), file_id, dtype=np.int64)}
    for name in names:
        ds = h5[f"metadata/{name}"]
        values = (
            ds[start:stop]
            if name == "timestamp_us"
            else ds[start:stop, file.target_index]
        )
        result[name] = values[indices]
    return result


def _validate(task, threshold_bps, read_batch_size):
    if task not in ("regression", "classification"):
        raise ValueError("task must be 'regression' or 'classification'")
    if not np.isfinite(threshold_bps) or threshold_bps < 0:
        raise ValueError("threshold_bps must be finite and nonnegative")
    if type(read_batch_size) is not int or read_batch_size < 1:
        raise ValueError("read_batch_size must be a positive integer")


class TabularDataset:
    """Stream selected HDF5 features for classical machine-learning models."""

    def __init__(
        self,
        paths,
        raw_feature_names=None,
        engineered_feature_names="all",
        target_name="target_return_50_bps",
        task="regression",
        threshold_bps=0.25,
        n_samples=None,
        normalize=False,
        mean=None,
        std=None,
        seed=42,
        read_batch_size=65_536,
    ):
        _validate(task, threshold_bps, read_batch_size)
        self.files, raw, engineered, self.metadata_names = _inspect_files(
            paths, raw_feature_names, engineered_feature_names, target_name
        )
        self.paths = tuple(f.path for f in self.files)
        self.raw_feature_names, self.engineered_feature_names = raw, engineered
        self.feature_names = raw + engineered
        self.target_name, self.task = target_name, task
        self.threshold_bps = threshold_bps
        self.read_batch_size = read_batch_size
        self.normalize = normalize
        self.mean, self.std = _normalization(
            normalize, mean, std, self.files, len(self.feature_names), read_batch_size
        )

        self.selected = None
        if n_samples is not None:
            if type(n_samples) is not int or n_samples < 1:
                raise ValueError("n_samples must be a positive integer")
            counts = [self._count_valid(file) for file in self.files]
            if sum(counts) == 0:
                raise ValueError("No eligible tabular observations")
            if n_samples < sum(counts):
                choices = np.sort(
                    np.random.default_rng(seed).choice(
                        sum(counts), n_samples, replace=False
                    )
                )
                offsets = np.r_[0, np.cumsum(counts)]
                self.selected = []
                for i in range(len(counts)):
                    lo, hi = np.searchsorted(choices, offsets[i : i + 2])
                    self.selected.append(choices[lo:hi] - offsets[i])

    def _count_valid(self, file):
        total = 0
        with h5py.File(file.path, "r") as h5:
            for start in range(0, file.rows, self.read_batch_size):
                stop = min(start + self.read_batch_size, file.rows)
                x = _read_features(h5, file, start, stop)
                y = h5["targets"][start:stop, file.target_index]
                total += np.count_nonzero(
                    h5["valid_book"][start:stop]
                    & np.isfinite(y)
                    & np.isfinite(x).all(axis=1)
                )
        return int(total)

    def get_stats(self):
        return self.mean, self.std

    def stream(self):
        for file_id, file in enumerate(self.files):
            chosen = None if self.selected is None else self.selected[file_id]
            if chosen is not None and not len(chosen):
                continue
            seen = 0
            with h5py.File(file.path, "r") as h5:
                for start in range(0, file.rows, self.read_batch_size):
                    stop = min(start + self.read_batch_size, file.rows)
                    x = _read_features(h5, file, start, stop)
                    y = h5["targets"][start:stop, file.target_index]
                    ids = np.flatnonzero(
                        h5["valid_book"][start:stop]
                        & np.isfinite(y)
                        & np.isfinite(x).all(axis=1)
                    )
                    if chosen is not None:
                        count = len(ids)
                        lo, hi = np.searchsorted(chosen, [seen, seen + count])
                        ids = ids[chosen[lo:hi] - seen]
                        seen += count
                    if not len(ids):
                        continue
                    xb = x[ids]
                    if self.normalize:
                        xb = (xb - self.mean) / self.std
                    yield (
                        xb,
                        _labels(y[ids], self.task, self.threshold_bps),
                        _metadata(
                            h5, file_id, file, self.metadata_names, start, stop, ids
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
    """Yield locality-preserving, prebatched sequence windows for PyTorch."""

    def __init__(
        self,
        paths,
        raw_feature_names="all",
        engineered_feature_names=None,
        target_name="target_return_50_bps",
        task="regression",
        threshold_bps=0.25,
        window_size=100,
        stride=1,
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
        _validate(task, threshold_bps, read_batch_size)
        if any(
            type(v) is not int or v < 1
            for v in (window_size, stride, batch_size, world_size)
        ):
            raise ValueError(
                "window_size, stride, batch_size, world_size must be positive integers"
            )
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        self.files, raw, engineered, self.metadata_names = _inspect_files(
            paths, raw_feature_names, engineered_feature_names, target_name
        )
        self.paths = tuple(f.path for f in self.files)
        self.raw_feature_names, self.engineered_feature_names = raw, engineered
        self.feature_names = raw + engineered
        self.target_name, self.task = target_name, task
        self.threshold_bps = threshold_bps
        self.window_size, self.stride, self.batch_size = window_size, stride, batch_size
        self.read_batch_size = read_batch_size
        self.seed, self.epoch, self.shuffle = seed, 0, shuffle
        self.drop_last, self.rank, self.world_size = drop_last, rank, world_size
        self.normalize = normalize
        self.mean, self.std = _normalization(
            normalize, mean, std, self.files, len(self.feature_names), read_batch_size
        )
        self.endpoints = _sample_indices(
            [self._valid_endpoints(f) for f in self.files], n_samples, seed
        )
        self.offsets = np.r_[
            0, np.cumsum([len(a) for a in self.endpoints], dtype=np.int64)
        ]
        self.total = int(self.offsets[-1])
        if not self.total:
            raise ValueError("No eligible sequence endpoints")

        if drop_last:
            self.n_batches = (self.total // batch_size) // world_size
            self.global_batches = self.n_batches * world_size
        elif world_size == 1:
            self.n_batches = (self.total + batch_size - 1) // batch_size
            self.global_batches = self.n_batches
        else:
            base, extra = divmod(self.total, world_size)
            if base == 0:
                raise ValueError("Not enough samples for distributed evaluation")
            self.rank_start = rank * base + min(rank, extra)
            self.rank_count = base + (rank < extra)
            self.n_batches = (base + (extra > 0) + batch_size - 1) // batch_size
            if base < self.n_batches:
                raise ValueError(
                    "Insufficient samples for equal distributed evaluation steps"
                )
        if not self.n_batches:
            raise ValueError("Not enough samples to form batches")

    def _valid_endpoints(self, file):
        W = self.window_size
        tail = np.empty(0, dtype=bool)
        selected = []
        dtype = np.int32 if file.rows <= np.iinfo(np.int32).max else np.int64
        with h5py.File(file.path, "r") as h5:
            for start in range(0, file.rows, self.read_batch_size):
                stop = min(start + self.read_batch_size, file.rows)
                x = _read_features(h5, file, start, stop)
                valid = h5["valid_book"][start:stop] & np.isfinite(x).all(axis=1)
                combined = np.r_[tail, valid]
                prefix = np.r_[0, np.cumsum(~combined)]
                positions = np.arange(start, stop)
                enough = positions >= W - 1
                complete = np.zeros(len(valid), dtype=bool)
                local = np.flatnonzero(enough)
                if len(local):
                    ends = len(tail) + local
                    complete[local] = prefix[ends + 1] == prefix[ends - W + 1]
                y = h5["targets"][start:stop, file.target_index]
                ok = (
                    complete & np.isfinite(y) & ((positions - W + 1) % self.stride == 0)
                )
                if np.any(ok):
                    selected.append(positions[ok].astype(dtype))
                if W > 1:
                    tail = combined[-(W - 1) :].copy()
        return np.concatenate(selected) if selected else np.empty(0, dtype=dtype)

    def get_stats(self):
        return self.mean, self.std

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n_batches

    def _read_file_batch(self, h5, file_id, endpoints):
        file = self.files[file_id]
        W, count = self.window_size, len(endpoints)
        x = np.empty((count, W, len(self.feature_names)), dtype=np.float32)
        y = np.empty(count, dtype=np.float32)
        metadata = {"source_id": np.full(count, file_id, dtype=np.int64)}
        metadata.update(
            {k: np.empty(count, dtype=np.int64) for k in self.metadata_names}
        )
        splits = np.r_[0, np.flatnonzero(np.diff(endpoints) > W) + 1, count]
        for left, right in zip(splits[:-1], splits[1:]):
            group = endpoints[left:right]
            first, last = int(group[0]), int(group[-1])
            start, stop = first - W + 1, last + 1
            features = _read_features(h5, file, start, stop)
            windows = np.lib.stride_tricks.sliding_window_view(features, W, axis=0)
            x[left:right] = windows[group - first].transpose(0, 2, 1)
            loc = group - start
            y[left:right] = h5["targets"][start:stop, file.target_index][loc]
            for key, values in _metadata(
                h5, file_id, file, self.metadata_names, start, stop, loc
            ).items():
                metadata[key][left:right] = values
        if self.normalize:
            x -= self.mean
            x /= self.std
        return x, _labels(y, self.task, self.threshold_bps), metadata

    def _read_batch(self, first, stop, opened):
        xs, ys, metas = [], [], []
        while first < stop:
            file_id = int(np.searchsorted(self.offsets[1:], first, side="right"))
            file_stop = min(stop, int(self.offsets[file_id + 1]))
            lo, hi = first - self.offsets[file_id], file_stop - self.offsets[file_id]
            if file_id not in opened:
                if len(opened) >= 8:
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
            count = self.rank_count
            sizes = np.full(self.n_batches, count // self.n_batches, dtype=np.int64)
            sizes[: count % self.n_batches] += 1
            starts = self.rank_start + np.r_[0, np.cumsum(sizes[:-1])]
            batch_ids = np.arange(self.n_batches)
            if self.shuffle:
                batch_ids = np.random.default_rng(
                    self.seed + self.epoch + self.rank
                ).permutation(batch_ids)
            batch_ids = batch_ids[worker_id::n_workers]
            ranges = zip(starts[batch_ids], (starts + sizes)[batch_ids])
        else:
            batch_ids = np.arange(self.global_batches)
            if self.shuffle:
                batch_ids = np.random.default_rng(self.seed + self.epoch).permutation(
                    batch_ids
                )
            batch_ids = batch_ids[self.rank :: self.world_size][worker_id::n_workers]
            ranges = (
                (
                    int(i * self.batch_size),
                    int(min((i + 1) * self.batch_size, self.total)),
                )
                for i in batch_ids
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
