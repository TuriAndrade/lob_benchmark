from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

try:
    from .schema import BookSnapshotSchema
except ImportError:
    from schema import BookSnapshotSchema


def _features(schema, include_raw=False, include_engineered=True):
    raw = (
        (
            schema.prices("bid")
            + schema.prices("ask")
            + schema.sizes("bid")
            + schema.sizes("ask")
        )
        if include_raw
        else []
    )

    excluded = {schema.mid_price, schema.spread, schema.microprice_delta}
    engineered = (
        [c for c in schema.feature_columns if c not in excluded]
        if include_engineered
        else []
    )

    return raw, engineered


def _transform(values, names, schema, mid=None):
    """Deterministic transformations; does not fit statistics."""
    x = values.astype(np.float32, copy=True)

    prices = set(schema.prices("bid") + schema.prices("ask"))
    positive = (
        set(schema.sizes("bid") + schema.sizes("ask"))
        | {
            schema.depth_name(side, k)
            for side in ("bid", "ask")
            for k in schema.depth_levels
        }
        | {schema.interarrival_ms, schema.event_rate(schema.EVENT_RATE_WINDOW)}
    )
    signed = {schema.ofi} | {schema.ofi_sum(k) for k in schema.OFI_WINDOWS}

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for j, name in enumerate(names):
            if name in prices:
                x[:, j] = 10_000 * (x[:, j] - mid) / mid
            elif name in positive:
                x[:, j] = np.log1p(x[:, j])
            elif name in signed:
                v = x[:, j]
                x[:, j] = np.sign(v) * np.log1p(np.abs(v))

    return x


def _read(path, columns, batch_size):
    parquet = pq.ParquetFile(str(path))
    yield from parquet.iter_batches(
        batch_size=batch_size,
        columns=columns,
        use_threads=True,
    )


def _array(batch, name, dtype=np.float32):
    return np.asarray(
        batch.column(name).to_numpy(zero_copy_only=False),
        dtype=dtype,
    )


def _stats(dataset):
    """Mean/std over valid underlying observations, not overlapping windows."""
    count = 0
    mean = np.zeros(len(dataset.engineered), np.float64)
    m2 = np.zeros_like(mean)

    if not dataset.engineered:
        return None, None

    columns = [*dataset.engineered, dataset.schema.valid_book]

    for path in dataset.paths:
        for batch in _read(path, columns, dataset.read_batch_size):
            x = np.column_stack([_array(batch, c) for c in dataset.engineered])
            x = _transform(x, dataset.engineered, dataset.schema)

            valid = _array(batch, dataset.schema.valid_book, bool)
            valid &= np.isfinite(x).all(axis=1)
            x = x[valid].astype(np.float64)

            if not len(x):
                continue

            n = len(x)
            avg = x.mean(axis=0)
            delta = avg - mean

            mean += delta * n / (count + n)
            m2 += ((x - avg) ** 2).sum(axis=0) + (delta**2 * count * n / (count + n))
            count += n

    if count == 0:
        raise ValueError("No valid observations for normalization")

    std = np.sqrt(m2 / count)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def _sample_indices(counts, n_samples, seed):
    """Uniform sample of eligible endpoints across all files."""
    total = sum(counts)

    if n_samples is None or n_samples >= total:
        return None

    selected = np.sort(
        np.random.default_rng(seed).choice(total, size=n_samples, replace=False)
    )

    offsets = np.cumsum([0, *counts])
    return [
        selected[
            np.searchsorted(selected, offsets[i]) : np.searchsorted(
                selected, offsets[i + 1]
            )
        ]
        - offsets[i]
        for i in range(len(counts))
    ]


class TabularDataset:
    def __init__(
        self,
        paths,
        depth=5,
        n_samples=None,
        horizon=50,
        task="regression",
        threshold_bps=0.25,
        normalize=False,
        mean=None,
        std=None,
        seed=42,
        read_batch_size=65_536,
    ):
        self.paths = tuple(map(Path, paths))
        self.schema = BookSnapshotSchema(depth)
        self.horizon = horizon
        self.task = task
        self.threshold_bps = threshold_bps
        self.read_batch_size = read_batch_size

        _, self.engineered = _features(self.schema, include_raw=False, include_engineered=True)
        self.feature_columns = tuple(self.engineered)

        self.mean = None if mean is None else np.asarray(mean, np.float32)
        self.std = None if std is None else np.asarray(std, np.float32)

        if normalize:
            if self.mean is None or self.std is None:
                self.mean, self.std = _stats(self)

        self.normalize = normalize

        self.selected = None
        if n_samples is not None:
            counts = [
                sum(self._iterate_file(i, count_only=True))
                for i in range(len(self.paths))
            ]
            self.selected = _sample_indices(counts, n_samples, seed)

    def get_stats(self):
        return self.mean, self.std

    def _iterate_file(self, file_id, count_only=False):
        s = self.schema
        target = s.target_return_bps(self.horizon)
        meta_names = (
            s.event_id,
            s.local_timestamp_us,
            s.target_end_event(self.horizon),
            s.target_end_timestamp_us(self.horizon),
        )
        columns = [*self.engineered, s.valid_book, target, *meta_names]

        selected = None if self.selected is None else self.selected[file_id]
        seen = 0

        for batch in _read(self.paths[file_id], columns, self.read_batch_size):
            x = np.column_stack([_array(batch, c) for c in self.engineered])
            x = _transform(x, self.engineered, s)
            y = _array(batch, target)

            valid = _array(batch, s.valid_book, bool)
            valid &= np.isfinite(x).all(axis=1) & np.isfinite(y)
            indices = np.flatnonzero(valid)

            if count_only:
                yield len(indices)
                continue

            if selected is not None:
                lo, hi = np.searchsorted(selected, [seen, seen + len(indices)])
                indices = indices[selected[lo:hi] - seen]
                seen += int(valid.sum())

            if not len(indices):
                continue

            x, y = x[indices], y[indices]

            if self.normalize:
                x = (x - self.mean) / self.std

            if self.task == "classification":
                y = np.where(
                    y < -self.threshold_bps,
                    0,
                    np.where(y > self.threshold_bps, 2, 1),
                ).astype(np.int64)

            metadata = {"source_id": np.full(len(indices), file_id)}
            metadata.update(
                {name: _array(batch, name, np.int64)[indices] for name in meta_names}
            )

            yield x, y, metadata

    def stream(self):
        for i in range(len(self.paths)):
            yield from self._iterate_file(i)

    def load(self):
        batches = list(self.stream())

        if not batches:
            raise ValueError("No valid samples")

        return (
            np.concatenate([b[0] for b in batches]),
            np.concatenate([b[1] for b in batches]),
            {k: np.concatenate([b[2][k] for b in batches]) for k in batches[0][2]},
        )


class SequenceDataset(IterableDataset):
    def __init__(
        self,
        paths,
        depth=5,
        n_samples=None,
        horizon=50,
        task="regression",
        threshold_bps=0.25,
        window_size=100,
        stride=1,
        include_engineered=False,
        normalize=False,
        mean=None,
        std=None,
        batch_size=256,
        seed=42,
        read_batch_size=65_536,
    ):
        super().__init__()

        self.paths = tuple(map(Path, paths))
        self.schema = BookSnapshotSchema(depth)
        self.horizon = horizon
        self.task = task
        self.threshold_bps = threshold_bps
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.read_batch_size = read_batch_size

        self.raw, self.engineered = _features(
            self.schema,
            include_raw=True,
            include_engineered=include_engineered,
        )
        self.feature_columns = tuple(self.raw + self.engineered)

        self.mean = None if mean is None else np.asarray(mean, np.float32)
        self.std = None if std is None else np.asarray(std, np.float32)

        if normalize and self.engineered:
            if self.mean is None or self.std is None:
                self.mean, self.std = _stats(self)

        self.normalize = normalize

        self.selected = None
        if n_samples is not None:
            counts = [
                sum(self._iterate_file(i, count_only=True))
                for i in range(len(self.paths))
            ]
            self.selected = _sample_indices(counts, n_samples, seed)

    def get_stats(self):
        return self.mean, self.std

    def _iterate_file(self, file_id, count_only=False):
        s = self.schema
        target = s.target_return_bps(self.horizon)
        meta_names = (
            s.event_id,
            s.local_timestamp_us,
            s.target_end_event(self.horizon),
            s.target_end_timestamp_us(self.horizon),
        )

        columns = list(
            dict.fromkeys(
                [
                    *self.feature_columns,
                    s.mid_price,
                    s.valid_book,
                    target,
                    *meta_names,
                ]
            )
        )

        selected = None if self.selected is None else self.selected[file_id]
        seen = offset = 0
        length = self.window_size

        tail_x = np.empty((0, len(self.feature_columns)), np.float32)
        tail_valid = np.empty(0, bool)

        for batch in _read(self.paths[file_id], columns, self.read_batch_size):
            raw = np.column_stack([_array(batch, c) for c in self.raw])
            mid = _array(batch, s.mid_price)
            raw = _transform(raw, self.raw, s, mid)

            if self.engineered:
                engineered = np.column_stack(
                    [_array(batch, c) for c in self.engineered]
                )
                engineered = _transform(engineered, self.engineered, s)

                if self.normalize:
                    engineered = (engineered - self.mean) / self.std

                x = np.concatenate([raw, engineered], axis=1)
            else:
                x = raw

            y = _array(batch, target)

            valid = _array(batch, s.valid_book, bool)
            valid &= np.isfinite(x).all(axis=1)

            combined_x = np.concatenate([tail_x, x])
            combined_valid = np.concatenate([tail_valid, valid])

            n = len(x)
            ends = len(tail_x) + np.arange(n)
            starts = ends - length + 1
            enough = starts >= 0

            bad = np.r_[0, np.cumsum(~combined_valid)]
            complete = np.zeros(n, bool)
            complete[enough] = bad[ends[enough] + 1] == bad[starts[enough]]

            positions = offset + np.arange(n)
            eligible = (
                complete
                & np.isfinite(y)
                & (positions >= length - 1)
                & ((positions - length + 1) % self.stride == 0)
            )
            indices = np.flatnonzero(eligible)

            if count_only:
                yield len(indices)
            else:
                if selected is not None:
                    lo, hi = np.searchsorted(selected, [seen, seen + len(indices)])
                    indices = indices[selected[lo:hi] - seen]
                    seen += int(eligible.sum())

                if len(indices):
                    windows = np.lib.stride_tricks.sliding_window_view(
                        combined_x, length, axis=0
                    )

                    metadata = {
                        name: _array(batch, name, np.int64) for name in meta_names
                    }

                    for start in range(0, len(indices), self.batch_size):
                        ids = indices[start : start + self.batch_size]
                        begins = ids + len(tail_x) - length + 1

                        xb = np.ascontiguousarray(windows[begins].transpose(0, 2, 1))
                        yb = y[ids]

                        if self.task == "classification":
                            yb = np.where(
                                yb < -self.threshold_bps,
                                0,
                                np.where(yb > self.threshold_bps, 2, 1),
                            ).astype(np.int64)

                        meta = {
                            "source_id": np.full(len(ids), file_id, np.int64),
                            **{name: values[ids] for name, values in metadata.items()},
                        }

                        yield xb, yb, meta

            if length > 1:
                tail_x = combined_x[-(length - 1) :].copy()
                tail_valid = combined_valid[-(length - 1) :].copy()

            offset += n

    def stream(self):
        for i in range(len(self.paths)):
            yield from self._iterate_file(i)

    def __iter__(self):
        worker = get_worker_info()

        files = (
            range(len(self.paths))
            if worker is None
            else range(worker.id, len(self.paths), worker.num_workers)
        )

        for i in files:
            for x, y, metadata in self._iterate_file(i):
                yield (
                    torch.from_numpy(x),
                    torch.from_numpy(y),
                    {k: torch.from_numpy(v) for k, v in metadata.items()},
                )
