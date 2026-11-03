"""Correctness and throughput tests for the source-independent HDF5 loaders.

Run: python -m pytest -q -s src/data/datasets_test.py
Optional: LOB_BENCHMARK_PATH=/path/to/data.h5 python -m pytest -q -s src/data/datasets_test.py
"""

import os
import shutil
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from .datasets import SequenceDataset, TabularDataset


RAW = (
    "ask_price_1",
    "ask_size_1",
    "bid_price_1",
    "bid_size_1",
    "ask_price_2",
    "ask_size_2",
    "bid_price_2",
    "bid_size_2",
)
ENGINEERED = (
    "spread_bps",
    "imbalance_1",
    "return_10_bps",
    "ofi_best_sum_10",
    "volatility_50_bps",
    "log_update_rate_50",
)
TARGETS = ("target_return_10_bps", "target_return_50_bps", "target_return_100_bps")
TARGET = TARGETS[1]
WINDOW = 20
READ_SIZE = 13  # Smaller than a window: tests history spanning HDF5 reads.


def _names(h5, key):
    return tuple(v.decode() if isinstance(v, bytes) else str(v) for v in h5.attrs[key])


def _write_h5_fixture(
    path, n=750, *, offset=0, permuted=False, metadata=True, gaps=True
):
    """Write only the four required datasets and three required attributes."""
    rng = np.random.default_rng(123 + offset)
    row = np.arange(n)
    mid = 100 + offset + 0.004 * row + 0.1 * np.sin(row / 19)
    times = 1_700_000_000_000_000 + offset * 1_000_000 + row * 100_000

    raw = rng.normal(size=(n, len(RAW))).astype(np.float32)
    raw[:, 0] = mid + 0.01
    raw[:, 2] = mid - 0.01
    raw[:, 4] = mid + 0.02
    raw[:, 6] = mid - 0.02
    raw[:, [1, 3, 5, 7]] = np.abs(raw[:, [1, 3, 5, 7]]) + 0.01
    engineered = rng.normal(size=(n, len(ENGINEERED))).astype(np.float32)

    targets = np.full((n, len(TARGETS)), np.nan, np.float32)
    target_times = np.full((n, len(TARGETS)), -1, np.int64)
    for j, horizon in enumerate((10, 50, 100)):
        if n > horizon:
            targets[:-horizon, j] = 1e4 * (mid[horizon:] / mid[:-horizon] - 1)
            target_times[:-horizon, j] = times[horizon:]

    valid = np.ones(n, bool)
    if gaps:
        valid[[i for i in (8, 30, 127, 256, 511) if i < n]] = False
        if n > 43:
            engineered[43, 1] = np.nan
        if n > 194:
            engineered[194, 4] = np.inf
        if n > 75:
            raw[75, 0] = np.nan
        if n > 180:
            targets[180, 1] = np.nan
        if n > 190:
            targets[190, 0] = np.nan  # Other target: must not affect TARGET.

    raw_names, engineered_names, target_names = RAW, ENGINEERED, TARGETS
    if permuted:
        raw_order = [6, 3, 0, 7, 1, 5, 2, 4]
        engineered_order = [4, 0, 5, 2, 1, 3]
        target_order = [2, 0, 1]
        raw, raw_names = raw[:, raw_order], tuple(RAW[i] for i in raw_order)
        engineered = np.column_stack(
            (engineered[:, engineered_order], np.zeros(n, np.float32))
        )
        engineered_names = tuple(ENGINEERED[i] for i in engineered_order) + (
            "extra_unused",
        )
        targets, target_names = (
            targets[:, target_order],
            tuple(TARGETS[i] for i in target_order),
        )
        target_times = target_times[:, target_order]

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for key, values in (
            ("raw_feature_names", raw_names),
            ("engineered_feature_names", engineered_names),
            ("target_names", target_names),
        ):
            h5.attrs[key] = np.asarray(values, dtype=h5py.string_dtype("utf-8"))
        for key, values in (
            ("raw_features", raw),
            ("engineered_features", engineered),
            ("targets", targets),
            ("valid_book", valid),
        ):
            h5.create_dataset(key, data=values, chunks=True)
        if metadata:
            h5.create_dataset("metadata/timestamp_us", data=times)
            h5.create_dataset("metadata/target_end_timestamp_us", data=target_times)
            h5.create_dataset("metadata/update_id", data=row)
    return path


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    return _write_h5_fixture(tmp_path_factory.mktemp("lob") / "day1.h5")


@pytest.fixture(scope="module")
def two_files(sample, tmp_path_factory):
    second = _write_h5_fixture(
        tmp_path_factory.mktemp("lob") / "day2.h5",
        n=430,
        offset=100,
        permuted=True,
    )
    return [sample, second]


def _reference(path, raw_names="all", engineered_names="all", target=TARGET):
    """Independent, in-memory reference used only on small test fixtures."""
    with h5py.File(path) as h5:
        groups = []
        for key, request, attr in (
            ("raw_features", raw_names, "raw_feature_names"),
            ("engineered_features", engineered_names, "engineered_feature_names"),
        ):
            if request is None:
                continue
            available = _names(h5, attr)
            requested = (
                available
                if isinstance(request, str) and request == "all"
                else tuple(request)
            )
            groups.append(h5[key][:][:, [available.index(name) for name in requested]])
        x = np.concatenate(groups, axis=1)
        y = h5["targets"][:, _names(h5, "target_names").index(target)]
        valid = h5["valid_book"][:] & np.isfinite(x).all(axis=1)
    return x, y, valid


def _endpoints(
    path, raw="all", engineered="all", target=TARGET, window=WINDOW, stride=3
):
    x, y, valid = _reference(path, raw, engineered, target)
    return np.asarray(
        [
            t
            for t in range(window - 1, len(y))
            if (t - window + 1) % stride == 0
            and np.isfinite(y[t])
            and valid[t - window + 1 : t + 1].all()
        ],
        dtype=np.int64,
    )


def _collect(dataset, workers=0):
    batches = list(DataLoader(dataset, batch_size=None, num_workers=workers))
    assert batches, "Expected a nonempty dataset"
    return (
        torch.cat([b[0] for b in batches]).numpy(),
        torch.cat([b[1] for b in batches]).numpy(),
        {k: torch.cat([b[2][k] for b in batches]).numpy() for k in batches[0][2]},
        len(batches),
    )


def _ids(metadata):
    return set(zip(metadata["source_id"].tolist(), metadata["timestamp_us"].tolist()))


def test_tabular_defaults_and_metadata(sample):
    ds = TabularDataset(sample, read_batch_size=READ_SIZE)
    x, y, meta = ds.load()
    raw, target, valid = _reference(sample, None, "all")
    ids = np.flatnonzero(valid & np.isfinite(target))
    np.testing.assert_array_equal(x, raw[ids])
    np.testing.assert_array_equal(y, target[ids])
    assert x.shape == (len(ids), len(ENGINEERED))
    assert x.dtype == y.dtype == np.float32
    assert ds.feature_names == ENGINEERED
    assert ds.get_stats() == (None, None)
    assert set(meta) == {"source_id", "timestamp_us", "target_end_timestamp_us"}
    np.testing.assert_array_equal(meta["source_id"], 0)
    with h5py.File(sample) as h5:
        np.testing.assert_array_equal(
            meta["timestamp_us"], h5["metadata/timestamp_us"][:][ids]
        )
        np.testing.assert_array_equal(
            meta["target_end_timestamp_us"],
            h5["metadata/target_end_timestamp_us"][:, 1][ids],
        )


@pytest.mark.parametrize(
    "raw,engineered",
    [
        ("all", None),
        (None, "all"),
        (("bid_price_1", "ask_size_2"), ["imbalance_1", "spread_bps"]),
        (np.array(["ask_size_1", "bid_price_2"]), None),
        ([], ["ofi_best_sum_10"]),
    ],
)
def test_tabular_feature_selection(sample, raw, engineered):
    ds = TabularDataset(
        sample,
        raw_feature_names=raw,
        engineered_feature_names=engineered,
        read_batch_size=READ_SIZE,
    )
    x, y, _ = ds.load()
    expected_x, expected_y, valid = _reference(sample, raw, engineered)
    ids = valid & np.isfinite(expected_y)
    np.testing.assert_array_equal(x, expected_x[ids])
    np.testing.assert_array_equal(y, expected_y[ids])


def test_target_by_name_and_classification(sample):
    params = dict(
        paths=sample,
        raw_feature_names=None,
        engineered_feature_names=["spread_bps"],
        target_name=TARGETS[0],
        n_samples=90,
        seed=7,
    )
    continuous = TabularDataset(**params).load()[1]
    classes = TabularDataset(**params, task="classification", threshold_bps=1.5).load()[
        1
    ]
    np.testing.assert_array_equal(
        classes, np.where(continuous < -1.5, 0, np.where(continuous > 1.5, 2, 1))
    )
    assert classes.dtype == np.int64


def test_per_file_column_order_and_extra_columns(two_files):
    raw = ["bid_price_2", "ask_price_1"]
    engineered = ("imbalance_1", "volatility_50_bps", "spread_bps")
    ds = TabularDataset(
        two_files,
        raw_feature_names=raw,
        engineered_feature_names=engineered,
        read_batch_size=READ_SIZE,
    )
    x, y, meta = ds.load()
    offset = 0
    for file_id, path in enumerate(two_files):
        expected, target, valid = _reference(path, raw, engineered)
        ids = valid & np.isfinite(target)
        count = int(ids.sum())
        np.testing.assert_array_equal(x[offset : offset + count], expected[ids])
        np.testing.assert_array_equal(y[offset : offset + count], target[ids])
        np.testing.assert_array_equal(
            meta["source_id"][offset : offset + count], file_id
        )
        offset += count
    assert offset == len(y)
    assert ds.feature_names == tuple(raw) + engineered


def test_tabular_stream_sampling_and_reproducibility(two_files):
    params = dict(paths=two_files, n_samples=110, seed=42, read_batch_size=READ_SIZE)
    a = TabularDataset(**params)
    b = TabularDataset(**params)
    full = a.load()
    batches = list(a.stream())
    assert len(full[1]) == 110
    assert all(0 < len(x) <= READ_SIZE for x, _, _ in batches)
    np.testing.assert_array_equal(
        full[0], np.concatenate([batch[0] for batch in batches])
    )
    np.testing.assert_array_equal(full[1], b.load()[1])
    assert len(_ids(full[2])) == 110
    assert set(full[2]["source_id"]) == {0, 1}


def test_normalization_all_features_and_reuse(two_files):
    params = dict(
        paths=two_files,
        raw_feature_names="all",
        engineered_feature_names="all",
        read_batch_size=READ_SIZE,
    )
    train = TabularDataset(**params, normalize=True)
    mean, std = train.get_stats()
    assert mean.shape == std.shape == (len(RAW) + len(ENGINEERED),)
    values = []
    for path in two_files:
        x, _, valid = _reference(path, RAW, ENGINEERED)
        values.append(x[valid])
    all_valid = np.concatenate(values).astype(np.float64)
    np.testing.assert_allclose(mean, all_valid.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(std, np.maximum(all_valid.std(axis=0), 1e-6), rtol=1e-6)
    original = TabularDataset(**params).load()[0]
    np.testing.assert_allclose(
        train.load()[0], (original - mean) / std, rtol=1e-5, atol=1e-5
    )
    reused = TabularDataset(**params, normalize=True, mean=mean, std=std)
    np.testing.assert_array_equal(reused.load()[0], train.load()[0])


def test_normalization_only_selected_columns(sample):
    params = dict(
        paths=sample,
        raw_feature_names=["bid_size_1"],
        engineered_feature_names=["spread_bps", "imbalance_1"],
    )
    ds = TabularDataset(**params, normalize=True, read_batch_size=READ_SIZE)
    mean, std = ds.get_stats()
    assert mean.shape == std.shape == (3,)
    original = TabularDataset(**params).load()[0]
    np.testing.assert_allclose(
        ds.load()[0], (original - mean) / std, rtol=1e-5, atol=1e-5
    )


def test_optional_metadata(sample, tmp_path):
    other = _write_h5_fixture(tmp_path / "no_meta.h5", 280, offset=200, metadata=False)
    for cls, options in (
        (TabularDataset, {}),
        (
            SequenceDataset,
            dict(window_size=WINDOW, batch_size=16, shuffle=False, drop_last=False),
        ),
    ):
        ds = cls([sample, other], **options)
        meta = ds.load()[2] if cls is TabularDataset else _collect(ds)[2]
        assert set(meta) == {"source_id"}
        assert set(meta["source_id"]) == {0, 1}


def test_sequence_windows_vs_reference(sample):
    params = dict(
        paths=sample,
        engineered_feature_names="all",
        window_size=WINDOW,
        stride=3,
        batch_size=16,
        read_batch_size=READ_SIZE,
        shuffle=False,
        drop_last=False,
    )
    ds = SequenceDataset(**params)
    x, y, meta, batches = _collect(ds)
    ends = _endpoints(sample)
    raw, targets, _ = _reference(sample)
    np.testing.assert_array_equal(
        meta["timestamp_us"], 1_700_000_000_000_000 + ends * 100_000
    )
    np.testing.assert_array_equal(
        x, np.stack([raw[t - WINDOW + 1 : t + 1] for t in ends])
    )
    np.testing.assert_array_equal(y, targets[ends])
    assert x.shape == (len(ends), WINDOW, len(RAW) + len(ENGINEERED))
    assert batches == len(ds)


def test_sequence_selected_groups_and_unused_invalid_features(sample):
    raw = ("bid_price_1", "ask_size_1")
    engineered = ("spread_bps",)
    params = dict(
        paths=sample,
        raw_feature_names=raw,
        engineered_feature_names=engineered,
        window_size=WINDOW,
        stride=2,
        batch_size=23,
        shuffle=False,
        drop_last=False,
        read_batch_size=READ_SIZE,
    )
    x, y, meta, _ = _collect(SequenceDataset(**params))
    ends = _endpoints(sample, raw, engineered, stride=2)
    selected, target, _ = _reference(sample, raw, engineered)
    np.testing.assert_array_equal(
        x, np.stack([selected[t - WINDOW + 1 : t + 1] for t in ends])
    )
    np.testing.assert_array_equal(y, target[ends])
    np.testing.assert_array_equal(
        meta["timestamp_us"], 1_700_000_000_000_000 + ends * 100_000
    )
    raw_ends = _endpoints(sample, "all", None, stride=2)
    all_ends = _endpoints(sample, "all", "all", stride=2)
    assert len(raw_ends) > len(all_ends)


def test_sequence_sampling_classification_multiple_files(two_files):
    params = dict(
        paths=two_files,
        raw_feature_names=None,
        engineered_feature_names="all",
        window_size=WINDOW,
        stride=2,
        n_samples=93,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        seed=17,
        read_batch_size=READ_SIZE,
    )
    a = _collect(SequenceDataset(**params))
    b = _collect(SequenceDataset(**params))
    assert len(a[1]) == 93
    np.testing.assert_array_equal(a[2]["source_id"], b[2]["source_id"])
    np.testing.assert_array_equal(a[2]["timestamp_us"], b[2]["timestamp_us"])
    assert set(a[2]["source_id"]) == {0, 1}
    classes = _collect(
        SequenceDataset(**params, task="classification", threshold_bps=1.5)
    )[1]
    np.testing.assert_array_equal(
        classes, np.where(a[1] < -1.5, 0, np.where(a[1] > 1.5, 2, 1))
    )


def test_sequence_normalization_all_features(sample):
    params = dict(
        paths=sample,
        window_size=WINDOW,
        stride=3,
        n_samples=48,
        batch_size=16,
        shuffle=False,
        drop_last=False,
    )
    original = _collect(SequenceDataset(**params))[0]
    ds = SequenceDataset(**params, normalize=True)
    mean, std = ds.get_stats()
    scaled = _collect(ds)[0]
    np.testing.assert_allclose(scaled, (original - mean) / std, rtol=1e-5, atol=1e-5)
    reused = SequenceDataset(**params, normalize=True, mean=mean, std=std)
    np.testing.assert_array_equal(_collect(reused)[0], scaled)


def test_sequence_workers_no_duplicates(sample):
    ds = SequenceDataset(
        sample,
        window_size=WINDOW,
        stride=2,
        batch_size=13,
        shuffle=False,
        drop_last=False,
        read_batch_size=READ_SIZE,
    )
    single = _collect(ds)
    multi = _collect(ds, workers=2)
    assert single[3] == multi[3] == len(ds)
    np.testing.assert_array_equal(
        np.sort(single[2]["timestamp_us"]), np.sort(multi[2]["timestamp_us"])
    )
    assert len(np.unique(multi[2]["timestamp_us"])) == len(multi[1])


def test_distributed_training_equal_batches_no_overlap(two_files):
    params = dict(
        paths=two_files,
        window_size=WINDOW,
        stride=2,
        batch_size=17,
        shuffle=True,
        drop_last=True,
        world_size=3,
        seed=33,
    )
    datasets = [SequenceDataset(**params, rank=i) for i in range(3)]
    for epoch in (0, 1):
        ids = []
        for ds in datasets:
            ds.set_epoch(epoch)
            result = _collect(ds)
            assert result[3] == len(ds)
            assert all(len(b[1]) == 17 for b in ds)
            ids.append(_ids(result[2]))
        assert len({len(ds) for ds in datasets}) == 1
        assert all(not ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3))
        assert sum(map(len, ids)) == sum(len(ds) * 17 for ds in datasets)


def test_distributed_evaluation_exact_no_duplicates(two_files):
    params = dict(
        paths=two_files,
        engineered_feature_names="all",
        window_size=WINDOW,
        stride=3,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        world_size=3,
    )
    found = set()
    counts = []
    for rank in range(3):
        ds = SequenceDataset(**params, rank=rank)
        result = _collect(ds, workers=2)
        assert result[3] == len(ds)
        counts.append(result[3])
        ids = _ids(result[2])
        assert not ids & found
        found.update(ids)
    expected = sum(len(_endpoints(path)) for path in two_files)
    assert len(set(counts)) == 1
    assert len(found) == expected


def test_windows_never_cross_files(two_files):
    ds = SequenceDataset(
        two_files,
        engineered_feature_names="all",
        window_size=WINDOW,
        batch_size=32,
        shuffle=False,
        drop_last=False,
    )
    x, _, meta, _ = _collect(ds)
    second = np.flatnonzero(meta["source_id"] == 1)
    assert len(second)
    with h5py.File(two_files[1]) as h5:
        timestamps = h5["metadata/timestamp_us"][:]
    endpoint = int(np.searchsorted(timestamps, meta["timestamp_us"][second[0]]))
    assert endpoint >= WINDOW - 1
    expected, _, _ = _reference(two_files[1], RAW, ENGINEERED)
    np.testing.assert_array_equal(
        x[second[0]], expected[endpoint - WINDOW + 1 : endpoint + 1]
    )


def test_no_future_feature_leakage(sample, tmp_path):
    cutoff = 380
    altered = tmp_path / "future.h5"
    shutil.copyfile(sample, altered)
    with h5py.File(altered, "r+") as h5:
        h5["raw_features"][cutoff:] = 123.0
        h5["engineered_features"][cutoff:] = 123.0

    for cls, options in (
        (TabularDataset, {}),
        (
            SequenceDataset,
            dict(
                window_size=WINDOW,
                stride=3,
                batch_size=16,
                shuffle=False,
                drop_last=False,
            ),
        ),
    ):
        before = cls(sample, **options)
        after = cls(altered, **options)
        a = before.load() if cls is TabularDataset else _collect(before)
        b = after.load() if cls is TabularDataset else _collect(after)
        old = a[2]["timestamp_us"] < 1_700_000_000_000_000 + cutoff * 100_000
        new = b[2]["timestamp_us"] < 1_700_000_000_000_000 + cutoff * 100_000
        np.testing.assert_array_equal(
            a[2]["timestamp_us"][old], b[2]["timestamp_us"][new]
        )
        np.testing.assert_array_equal(a[0][old], b[0][new])


@pytest.mark.parametrize(
    "options",
    [
        dict(raw_feature_names=None, engineered_feature_names=None),
        dict(raw_feature_names=[], engineered_feature_names=[]),
        dict(raw_feature_names="unknown"),
        dict(raw_feature_names=["missing"]),
        dict(engineered_feature_names=["spread_bps", "spread_bps"]),
        dict(target_name="missing_target"),
        dict(target_name=[TARGET]),
        dict(normalize=True, mean=np.zeros(2), std=np.ones(2)),
        dict(normalize=True, mean=np.zeros(14)),
        dict(threshold_bps=-1),
        dict(task="other"),
        dict(read_batch_size=0),
    ],
)
def test_invalid_options(sample, options):
    with pytest.raises(ValueError):
        TabularDataset(sample, **options)
    with pytest.raises(ValueError):
        SequenceDataset(sample, **options)


def test_invalid_sequence_distribution(sample):
    for kwargs in (
        dict(window_size=0),
        dict(stride=0),
        dict(batch_size=0),
        dict(world_size=0),
        dict(rank=2, world_size=2),
    ):
        with pytest.raises(ValueError):
            SequenceDataset(sample, **kwargs)


def test_incompatible_hdf5(sample, tmp_path):
    invalid = tmp_path / "invalid.h5"
    shutil.copyfile(sample, invalid)
    with h5py.File(invalid, "r+") as h5:
        del h5.attrs["target_names"]
    with pytest.raises(KeyError):
        TabularDataset(invalid)


@pytest.fixture(scope="module")
def benchmark_path(tmp_path_factory):
    path = os.environ.get("LOB_BENCHMARK_PATH")
    if path:
        result = Path(path).expanduser()
        if not result.is_file():
            pytest.fail(f"LOB_BENCHMARK_PATH does not exist: {result}")
        return result, "REAL HDF5"
    return _write_h5_fixture(
        tmp_path_factory.mktemp("lob_bench") / "benchmark.h5", n=16_000, gaps=False
    ), "SYNTHETIC HDF5"


def _time_batches(iterator, n=12, warmup=2):
    iterator = iter(iterator)
    for _ in range(warmup):
        next(iterator)
    times, samples = [], 0
    for _ in range(n):
        start = perf_counter()
        x, _, _ = next(iterator)
        times.append(perf_counter() - start)
        samples += len(x)
    times = np.asarray(times)
    return float(np.median(times) * 1e3), float(samples / times.sum())


def test_batch_streaming_timing(benchmark_path):
    """Diagnostics only: no environment-dependent performance threshold."""
    path, label = benchmark_path
    with h5py.File(path) as h5:
        target = _names(h5, "target_names")[0]
    for group in ("tabular", "sequence_raw", "sequence_all"):
        start = perf_counter()
        if group == "tabular":
            ds = TabularDataset(path, target_name=target, read_batch_size=256)
            batches = ds.stream()
        else:
            ds = SequenceDataset(
                path,
                target_name=target,
                window_size=100,
                batch_size=256,
                raw_feature_names="all",
                engineered_feature_names=None if group == "sequence_raw" else "all",
                shuffle=False,
                drop_last=True,
            )
            batches = iter(ds)
        initialization = perf_counter() - start
        median, rate = _time_batches(batches)
        print(
            f"\n[{label}] {group:13s} init={initialization:.3f}s "
            f"median={median:.3f}ms/batch throughput={rate:,.0f} samples/s"
        )
        assert np.isfinite(median) and rate > 0
