"""Correctness and batch-throughput tests for HDF5 LOB datasets.

Run with: pytest -q -s src/data/datasets_test.py
Optional real-file benchmark: LOB_BENCHMARK_PATH=/path/to/file.h5 pytest -q -s ...
"""

import os
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

try:
    from .datasets import TabularDataset, SequenceDataset
    from .schema import BookSnapshotSchema
except ImportError:  # Also supports running the test file directly.
    from datasets import TabularDataset, SequenceDataset
    from schema import BookSnapshotSchema


HORIZON = 50
WINDOW = 20
READ_SIZE = 31  # Deliberately smaller than the window plus several batches.


def _write_h5_fixture(path, n, date="2020-09-01", gaps=True):
    """Small deterministic fixture with the exact converter layout."""
    schema = BookSnapshotSchema(
        depth=5,
        exchange="binance-futures",
        symbol="BTCUSDT",
        date=date,
        source="tardis",
    )
    rng = np.random.default_rng(123)
    row = np.arange(n, dtype=np.int64)
    mid = 100.0 + row * 0.002 + 0.08 * np.sin(row / 19.0)
    timestamps = 1_700_000_000_000_000 + row * 2_000

    raw = rng.normal(0, 0.4, (n, len(schema.raw_feature_columns))).astype(np.float32)
    raw[:, :5] -= 1.0  # Pre-transformed bid price offsets (bps).
    raw[:, 5:10] += 1.0  # Pre-transformed ask price offsets (bps).
    raw[:, 10:] = np.log1p(np.abs(raw[:, 10:]) * 10)
    engineered = rng.normal(0, 1, (n, len(schema.engineered_feature_columns))).astype(
        np.float32
    )

    targets = np.full((n, len(schema.horizons)), np.nan, dtype=np.float32)
    end_events = np.full(targets.shape, -1, dtype=np.int64)
    end_times = np.full(targets.shape, -1, dtype=np.int64)
    for col, h in enumerate(schema.horizons):
        if h < n:
            targets[:-h, col] = (10_000 * (mid[h:] / mid[:-h] - 1)).astype(np.float32)
            end_events[:-h, col] = row[h:]
            end_times[:-h, col] = timestamps[h:]

    valid = np.ones(n, dtype=bool)
    if gaps:
        for i in (8, 30, 127, 256, 511):
            if i < n:
                valid[i] = False
        for i, col, value in ((43, 1, np.nan), (194, 3, np.inf)):
            if i < n:
                engineered[i, col] = value
        if 75 < n:
            raw[75, 0] = np.nan
        if 180 < n:
            targets[180, schema.horizons.index(HORIZON)] = np.nan

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        attrs = h5.create_group("attributes").attrs
        for name, value in schema.attributes.items():
            if isinstance(value, list):
                attrs[name] = np.asarray(value, dtype=h5py.string_dtype("utf-8"))
            elif isinstance(value, tuple):
                attrs[name] = np.asarray(value, dtype=np.int32)
            else:
                attrs[name] = value

        arrays = {
            "raw_features": raw,
            "engineered_features": engineered,
            "targets": targets,
            "valid_book": valid,
            "metadata/event_id": row,
            "metadata/timestamp_us": timestamps,
            "metadata/local_timestamp_us": timestamps + 120,
            "metadata/mid_price": mid.astype(np.float32),
            "metadata/spread": np.full(n, 0.02, np.float32),
            "metadata/microprice_delta": np.zeros(n, np.float32),
            "metadata/target_end_event": end_events,
            "metadata/target_end_timestamp_us": end_times,
        }
        for name, values in arrays.items():
            h5.create_dataset(name, data=values, chunks=True)
    return path


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return _write_h5_fixture(tmp_path_factory.mktemp("lob") / "day1.h5", 750)


@pytest.fixture(scope="module")
def two_files(sample, tmp_path_factory):
    second = _write_h5_fixture(
        tmp_path_factory.mktemp("lob") / "day2.h5", 390, "2020-09-02"
    )
    return [sample, second]


def _reference(
    path, horizon=HORIZON, include_engineered=False, window=WINDOW, stride=3
):
    """Independent, straightforward eligibility reference (test-sized files)."""
    with h5py.File(path) as h5:
        h = list(h5["attributes"].attrs["horizons"]).index(horizon)
        raw = h5["raw_features"][:]
        eng = h5["engineered_features"][:]
        target = h5["targets"][:, h]
        row_valid = h5["valid_book"][:] & np.isfinite(raw).all(axis=1)
        if include_engineered:
            row_valid &= np.isfinite(eng).all(axis=1)
        endpoints = np.asarray(
            [
                i
                for i in range(window - 1, len(target))
                if (i - window + 1) % stride == 0
                and np.isfinite(target[i])
                and row_valid[i - window + 1 : i + 1].all()
            ],
            dtype=np.int64,
        )
    return endpoints


def _collect_sequence(ds, workers=0):
    batches = list(DataLoader(ds, batch_size=None, num_workers=workers))
    if not batches:
        return None
    x = torch.cat([b[0] for b in batches]).numpy()
    y = torch.cat([b[1] for b in batches]).numpy()
    metadata = {
        name: torch.cat([b[2][name] for b in batches]).numpy() for name in batches[0][2]
    }
    return x, y, metadata, len(batches)


def test_tabular_features_targets_metadata(sample):
    ds = TabularDataset([sample], horizon=HORIZON, read_batch_size=READ_SIZE)
    x, y, meta = ds.load()
    with h5py.File(sample) as h5:
        h = list(h5["attributes"].attrs["horizons"]).index(HORIZON)
        features = h5["engineered_features"][:]
        targets = h5["targets"][:, h]
        expected = np.flatnonzero(
            h5["valid_book"][:]
            & np.isfinite(features).all(axis=1)
            & np.isfinite(targets)
        )
        np.testing.assert_array_equal(meta["event_id"], expected)
        np.testing.assert_array_equal(x, features[expected])
        np.testing.assert_array_equal(y, targets[expected])
        np.testing.assert_array_equal(
            meta["target_end_event"], h5["metadata/target_end_event"][:, h][expected]
        )
        np.testing.assert_array_equal(
            meta["local_timestamp_us"], h5["metadata/local_timestamp_us"][:][expected]
        )
    assert x.shape[1] == 21
    assert x.dtype == y.dtype == np.float32
    assert set(meta) == {
        "source_id",
        "event_id",
        "local_timestamp_us",
        "target_end_event",
        "target_end_timestamp_us",
    }
    assert ds.get_stats() == (None, None)


def test_tabular_stream_and_load(sample):
    ds = TabularDataset([sample], read_batch_size=READ_SIZE)
    full = ds.load()
    batches = list(ds.stream())
    np.testing.assert_array_equal(full[0], np.concatenate([b[0] for b in batches]))
    np.testing.assert_array_equal(full[1], np.concatenate([b[1] for b in batches]))
    assert all(0 < len(b[1]) <= READ_SIZE for b in batches)


def test_tabular_sampling_multiple_files(two_files):
    opts = dict(
        paths=two_files,
        horizon=HORIZON,
        n_samples=110,
        seed=42,
        read_batch_size=READ_SIZE,
    )
    a = TabularDataset(**opts).load()
    b = TabularDataset(**opts).load()
    assert len(a[1]) == 110
    for idx in (0, 1):
        np.testing.assert_array_equal(a[idx], b[idx])
    assert (
        np.unique(
            np.stack((a[2]["source_id"], a[2]["event_id"]), axis=1), axis=0
        ).shape[0]
        == 110
    )
    assert set(a[2]["source_id"]) == {0, 1}


def test_tabular_classification(sample):
    opts = dict(paths=[sample], n_samples=100, seed=19)
    y = TabularDataset(**opts).load()[1]
    classification = TabularDataset(
        **opts, task="classification", threshold_bps=2.0
    ).load()[1]
    np.testing.assert_array_equal(
        classification, np.where(y < -2, 0, np.where(y > 2, 2, 1))
    )
    assert classification.dtype == np.int64


def test_tabular_normalization(sample):
    opts = dict(paths=[sample], n_samples=120)
    original = TabularDataset(**opts).load()[0]
    normalized = TabularDataset(**opts, normalize=True)
    mean, std = normalized.get_stats()
    np.testing.assert_allclose(
        normalized.load()[0], (original - mean) / std, rtol=1e-5, atol=1e-5
    )
    reused = TabularDataset(**opts, normalize=True, mean=mean, std=std)
    np.testing.assert_array_equal(reused.load()[0], normalized.load()[0])
    assert mean.shape == std.shape == (21,)


def test_sequence_windows_and_stride(sample):
    ds = SequenceDataset(
        [sample],
        window_size=WINDOW,
        stride=3,
        batch_size=16,
        drop_last=False,
        shuffle=False,
        read_batch_size=READ_SIZE,
    )
    x, y, meta, count = _collect_sequence(ds)
    endpoints = _reference(sample)
    np.testing.assert_array_equal(meta["event_id"], endpoints)
    assert x.shape == (len(endpoints), WINDOW, 20)
    assert count == len(ds)
    with h5py.File(sample) as h5:
        raw = h5["raw_features"][:]
        expected = np.stack([raw[t - WINDOW + 1 : t + 1] for t in endpoints])
        np.testing.assert_array_equal(x, expected)
        np.testing.assert_array_equal(y, h5["targets"][:, 1][endpoints])
        np.testing.assert_array_equal(
            meta["target_end_event"], h5["metadata/target_end_event"][:, 1][endpoints]
        )


def test_sequence_combined_features_and_validity(sample):
    ds = SequenceDataset(
        [sample],
        window_size=WINDOW,
        stride=3,
        include_engineered=True,
        batch_size=23,
        drop_last=False,
        shuffle=False,
        read_batch_size=READ_SIZE,
    )
    x, _, meta, _ = _collect_sequence(ds)
    endpoints = _reference(sample, include_engineered=True)
    np.testing.assert_array_equal(meta["event_id"], endpoints)
    assert x.shape[1:] == (WINDOW, 41)
    with h5py.File(sample) as h5:
        raw = h5["raw_features"][:]
        eng = h5["engineered_features"][:]
        for j in (0, len(endpoints) // 2, len(endpoints) - 1):
            t = endpoints[j]
            np.testing.assert_array_equal(x[j, :, :20], raw[t - WINDOW + 1 : t + 1])
            np.testing.assert_array_equal(x[j, :, 20:], eng[t - WINDOW + 1 : t + 1])
    # Engineered NaNs do not invalidate raw-only sequences.
    raw_only = _reference(sample)
    assert len(raw_only) > len(endpoints)


def test_sequence_sampling_and_classification(two_files):
    opts = dict(
        paths=two_files,
        window_size=WINDOW,
        stride=2,
        n_samples=93,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        read_batch_size=READ_SIZE,
    )
    a = _collect_sequence(SequenceDataset(**opts))
    b = _collect_sequence(SequenceDataset(**opts))
    assert len(a[1]) == 93
    np.testing.assert_array_equal(a[2]["event_id"], b[2]["event_id"])
    assert set(a[2]["source_id"]) == {0, 1}
    classes = _collect_sequence(
        SequenceDataset(**opts, task="classification", threshold_bps=1.5)
    )[1]
    np.testing.assert_array_equal(
        classes, np.where(a[1] < -1.5, 0, np.where(a[1] > 1.5, 2, 1))
    )


def test_sequence_normalization(sample):
    opts = dict(
        paths=[sample],
        window_size=WINDOW,
        stride=3,
        include_engineered=True,
        n_samples=48,
        batch_size=16,
        drop_last=False,
        shuffle=False,
    )
    original = _collect_sequence(SequenceDataset(**opts))[0]
    ds = SequenceDataset(**opts, normalize=True)
    mean, std = ds.get_stats()
    scaled = _collect_sequence(ds)[0]
    np.testing.assert_array_equal(original[:, :, :20], scaled[:, :, :20])
    np.testing.assert_allclose(
        scaled[:, :, 20:], (original[:, :, 20:] - mean) / std, rtol=1e-5, atol=1e-5
    )
    reused = SequenceDataset(**opts, normalize=True, mean=mean, std=std)
    np.testing.assert_array_equal(_collect_sequence(reused)[0], scaled)


def test_workers_have_no_duplicates(sample):
    opts = dict(
        paths=[sample],
        window_size=WINDOW,
        stride=2,
        batch_size=13,
        drop_last=False,
        shuffle=False,
        read_batch_size=READ_SIZE,
    )
    ds = SequenceDataset(**opts)
    single = _collect_sequence(ds, workers=0)
    multi = _collect_sequence(ds, workers=2)
    assert single[3] == multi[3] == len(ds)
    np.testing.assert_array_equal(
        np.sort(single[2]["event_id"]), np.sort(multi[2]["event_id"])
    )
    assert len(np.unique(multi[2]["event_id"])) == len(multi[1])


def test_distributed_train_equal_batches_no_overlap(two_files):
    opts = dict(
        paths=two_files,
        window_size=WINDOW,
        stride=2,
        batch_size=17,
        shuffle=True,
        drop_last=True,
        world_size=3,
        seed=33,
    )
    datasets = [SequenceDataset(**opts, rank=i) for i in range(3)]
    for epoch in (0, 1):
        ids = []
        for ds in datasets:
            ds.set_epoch(epoch)
            batch = _collect_sequence(ds)
            assert batch[3] == len(ds)
            ids.append(set(zip(batch[2]["source_id"], batch[2]["event_id"])))
        assert all(len(datasets[i]) == len(datasets[0]) for i in range(3))
        assert all(not ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3))
        assert sum(map(len, ids)) == 3 * len(datasets[0]) * 17


def test_distributed_eval_exact_without_duplicates(two_files):
    opts = dict(
        paths=two_files,
        window_size=WINDOW,
        stride=3,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        world_size=3,
    )
    all_ids = []
    counts = []
    for rank in range(3):
        ds = SequenceDataset(**opts, rank=rank)
        batch = _collect_sequence(ds, workers=2)
        assert batch[3] == len(ds)
        counts.append(batch[3])
        all_ids.extend(zip(batch[2]["source_id"], batch[2]["event_id"]))
    expected = sum(len(_reference(path)) for path in two_files)
    assert counts[0] == counts[1] == counts[2]
    assert len(all_ids) == len(set(all_ids)) == expected


def test_window_does_not_cross_files(two_files):
    ds = SequenceDataset(
        two_files,
        window_size=WINDOW,
        stride=1,
        batch_size=32,
        drop_last=False,
        shuffle=False,
    )
    batch = _collect_sequence(ds)
    with h5py.File(two_files[1]) as h5:
        raw = h5["raw_features"][:]
    ids = np.flatnonzero(batch[2]["source_id"] == 1)
    assert len(ids)
    first = ids[0]
    event = batch[2]["event_id"][first]
    np.testing.assert_array_equal(batch[0][first], raw[event - WINDOW + 1 : event + 1])
    assert event >= WINDOW - 1


def test_future_feature_leakage(sample, tmp_path):
    cutoff = 380
    altered = tmp_path / "future.h5"
    with h5py.File(sample) as source, h5py.File(altered, "w") as target:
        for name in source:
            source.copy(name, target)
        target["raw_features"][cutoff:] = 123.0
        target["engineered_features"][cutoff:] = 123.0

    for dataset_cls, opts in (
        (TabularDataset, {}),
        (
            SequenceDataset,
            dict(
                window_size=WINDOW,
                stride=3,
                batch_size=12,
                drop_last=False,
                shuffle=False,
            ),
        ),
    ):
        if dataset_cls is TabularDataset:
            original = dataset_cls([sample], **opts).load()
            changed = dataset_cls([altered], **opts).load()
        else:
            original = _collect_sequence(dataset_cls([sample], **opts))
            changed = _collect_sequence(dataset_cls([altered], **opts))
        before1 = original[2]["event_id"] < cutoff
        before2 = changed[2]["event_id"] < cutoff
        np.testing.assert_array_equal(
            original[2]["event_id"][before1], changed[2]["event_id"][before2]
        )
        np.testing.assert_array_equal(original[0][before1], changed[0][before2])


def test_invalid_options(sample):
    with pytest.raises(ValueError, match="Horizon"):
        TabularDataset([sample], horizon=999)
    with pytest.raises(ValueError, match="threshold"):
        TabularDataset([sample], threshold_bps=-1)
    with pytest.raises(ValueError, match="positive"):
        SequenceDataset([sample], window_size=0)
    with pytest.raises(ValueError, match="rank"):
        SequenceDataset([sample], rank=2, world_size=2)


@pytest.fixture(scope="module")
def benchmark_path(tmp_path_factory):
    """Benchmark real data if explicitly provided; otherwise a larger synthetic HDF5."""
    real = os.environ.get("LOB_BENCHMARK_PATH")
    if real:
        path = Path(real).expanduser()
        if not path.is_file():
            pytest.fail(f"LOB_BENCHMARK_PATH does not exist: {path}")
        return path, "REAL HDF5"
    return _write_h5_fixture(
        tmp_path_factory.mktemp("lob_bench") / "benchmark.h5", 60_000, gaps=False
    ), "SYNTHETIC HDF5"


def _time_batches(iterator, n=25, warmup=3):
    """Time actual batch materialization, excluding initialization and warmup."""
    iterator = iter(iterator)
    for _ in range(warmup):
        next(iterator)
    times = []
    samples = 0
    for _ in range(n):
        start = perf_counter()
        x, _, _ = next(iterator)
        times.append(perf_counter() - start)
        samples += len(x)
    times = np.asarray(times)
    return {
        "median_ms": float(np.median(times) * 1000),
        "mean_ms": float(np.mean(times) * 1000),
        "p90_ms": float(np.percentile(times, 90) * 1000),
        "samples_s": float(samples / times.sum()),
        "samples": samples,
    }


def test_batch_streaming_timing(benchmark_path):
    """Run with pytest -s to display observed per-batch times; no speed threshold."""
    path, label = benchmark_path
    with h5py.File(path) as h5:
        available = tuple(int(v) for v in h5["attributes"].attrs["horizons"])
    horizon = HORIZON if HORIZON in available else available[0]
    batch_size = 256
    read_size = batch_size  # Comparable tabular and sequence output batch sizes.

    results = {}
    for mode in ("tabular", "sequence_raw", "sequence_combined"):
        start = perf_counter()
        if mode == "tabular":
            ds = TabularDataset([path], horizon=horizon, read_batch_size=read_size)
            iterable = ds.stream()
        else:
            ds = SequenceDataset(
                [path],
                horizon=horizon,
                window_size=100,
                include_engineered=(mode == "sequence_combined"),
                batch_size=batch_size,
                read_batch_size=65_536,
                shuffle=False,
                drop_last=True,
            )
            iterable = ds
        initialization = perf_counter() - start
        result = _time_batches(iterable)
        results[mode] = result
        print(
            f"\n[{label}] {mode:17s} "
            f"init={initialization:.3f}s, "
            f"median={result['median_ms']:.3f}ms/batch, "
            f"mean={result['mean_ms']:.3f}ms, "
            f"p90={result['p90_ms']:.3f}ms, "
            f"rate={result['samples_s']:,.0f} samples/s "
            f"({result['samples']} samples)"
        )
    assert all(
        np.isfinite(r["median_ms"]) and r["samples"] > 0 for r in results.values()
    )
