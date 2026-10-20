import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from schema import BookSnapshotSchema
from datasets import TabularDataset, SequenceDataset

load_dotenv()

ROOT = (
    Path(os.environ.get("DATA_DIR", "./"))
    / "tardis_book_snapshot"
    / "processed"
    / "binance-futures"
    / "BTCUSDT"
)

DEPTH = 5
HORIZON = 50
ROWS = 2_000
WINDOW = 20


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    files = sorted(ROOT.glob(f"*_depth{DEPTH}_h*_tick*.parquet"))
    files = [
        path
        for path in files
        if str(HORIZON) in path.stem.split("_h")[-1].split("_tick")[0].split("-")
    ]

    if not files:
        pytest.fail(f"No processed Parquet file found in {ROOT}")

    parquet = pq.ParquetFile(files[0])
    batch = next(parquet.iter_batches(batch_size=ROWS))
    table = pa.Table.from_batches([batch])

    path = tmp_path_factory.mktemp("lob") / "sample.parquet"
    pq.write_table(table, path, row_group_size=128)

    return path


@pytest.fixture(scope="module")
def schema():
    return BookSnapshotSchema(DEPTH)


@pytest.fixture(scope="module")
def reference(sample):
    table = pq.read_table(sample)
    return {
        name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names
    }


@pytest.fixture
def tabular(sample):
    return TabularDataset(
        [sample],
        depth=DEPTH,
        horizon=HORIZON,
        read_batch_size=128,
    )


@pytest.fixture
def sequence(sample):
    return SequenceDataset(
        [sample],
        depth=DEPTH,
        horizon=HORIZON,
        window_size=WINDOW,
        stride=3,
        batch_size=16,
        read_batch_size=128,
    )


def test_tabular_features(tabular, schema):
    X, y, meta = tabular.load()

    excluded = {
        schema.mid_price,
        schema.spread,
        schema.microprice_delta,
    }

    assert set(tabular.feature_columns) == (set(schema.feature_columns) - excluded)
    assert X.shape[1] == 21
    assert len(X) == len(y) == len(meta["source_id"])
    assert X.dtype == y.dtype == np.float32
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert tabular.get_stats() == (None, None)


def test_tabular_values(tabular, schema, reference):
    X, y, meta = tabular.load()

    for j, event in enumerate(meta[schema.event_id]):
        assert y[j] == pytest.approx(
            reference[schema.target_return_bps(HORIZON)][event],
            abs=1e-4,
        )

        spread_idx = tabular.feature_columns.index(schema.spread_bps)
        assert X[j, spread_idx] == pytest.approx(
            reference[schema.spread_bps][event],
            abs=1e-4,
        )

        depth_idx = tabular.feature_columns.index(schema.depth_name("bid", 1))
        assert X[j, depth_idx] == pytest.approx(
            np.log1p(reference[schema.depth_name("bid", 1)][event]),
            abs=1e-4,
        )


def test_tabular_stream(tabular):
    X, y, meta = tabular.load()
    batches = list(tabular.stream())

    np.testing.assert_array_equal(X, np.concatenate([b[0] for b in batches]))
    np.testing.assert_array_equal(y, np.concatenate([b[1] for b in batches]))


def test_tabular_classification(sample):
    kwargs = dict(
        paths=[sample],
        depth=DEPTH,
        horizon=HORIZON,
        n_samples=100,
        seed=42,
        read_batch_size=128,
    )

    _, returns, _ = TabularDataset(**kwargs).load()
    _, labels, _ = TabularDataset(
        **kwargs,
        task="classification",
        threshold_bps=0.25,
    ).load()

    expected = np.where(
        returns < -0.25,
        0,
        np.where(returns > 0.25, 2, 1),
    )

    np.testing.assert_array_equal(labels, expected)
    assert labels.dtype == np.int64


def test_tabular_sampling(sample, schema):
    kwargs = dict(
        paths=[sample],
        depth=DEPTH,
        horizon=HORIZON,
        n_samples=100,
        seed=42,
        read_batch_size=128,
    )

    first = TabularDataset(**kwargs).load()
    second = TabularDataset(**kwargs).load()

    assert len(first[1]) == 100
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(
        first[2][schema.event_id],
        second[2][schema.event_id],
    )


def test_tabular_normalization(sample):
    kwargs = dict(
        paths=[sample],
        depth=DEPTH,
        horizon=HORIZON,
        n_samples=100,
        read_batch_size=128,
    )

    X, _, _ = TabularDataset(**kwargs).load()

    normalized = TabularDataset(**kwargs, normalize=True)
    mean, std = normalized.get_stats()
    scaled, _, _ = normalized.load()

    assert mean.shape == std.shape == (21,)
    assert np.all(std > 0)

    np.testing.assert_allclose(scaled, (X - mean) / std, atol=1e-4, rtol=1e-4)

    reused = TabularDataset(**kwargs, normalize=True, mean=mean, std=std)

    np.testing.assert_allclose(reused.load()[0], scaled, atol=1e-5, rtol=1e-5)


def test_sequence_features(sequence, sample):
    raw = sequence
    combined = SequenceDataset(
        [sample],
        depth=DEPTH,
        horizon=HORIZON,
        window_size=WINDOW,
        include_engineered=True,
        batch_size=16,
        read_batch_size=128,
    )

    assert len(raw.feature_columns) == 20
    assert len(combined.feature_columns) == 41
    assert combined.feature_columns[:20] == tuple(raw.raw)


def test_sequence_windows(sequence, schema, reference):
    batches = list(sequence.stream())

    X = np.concatenate([b[0] for b in batches])
    y = np.concatenate([b[1] for b in batches])
    events = np.concatenate([b[2][schema.event_id] for b in batches])

    assert X.shape[1:] == (WINDOW, 20)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert np.all((events - WINDOW + 1) % 3 == 0)

    bid_idx = sequence.feature_columns.index(schema.price("bid", 0))
    size_idx = sequence.feature_columns.index(schema.size("bid", 0))

    for j, event in enumerate(events):
        history = np.arange(event - WINDOW + 1, event + 1)

        mid = reference[schema.mid_price][history]
        bid = reference[schema.price("bid", 0)][history]
        size = reference[schema.size("bid", 0)][history]

        np.testing.assert_allclose(
            X[j, :, bid_idx],
            10_000 * (bid - mid) / mid,
            atol=1e-3,
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            X[j, :, size_idx],
            np.log1p(size),
            atol=1e-5,
            rtol=1e-5,
        )
        assert y[j] == pytest.approx(
            reference[schema.target_return_bps(HORIZON)][event],
            abs=1e-4,
        )


def test_sequence_classification(sample):
    kwargs = dict(
        paths=[sample],
        depth=DEPTH,
        horizon=HORIZON,
        window_size=WINDOW,
        stride=3,
        n_samples=30,
        batch_size=16,
        read_batch_size=128,
    )

    regression = SequenceDataset(**kwargs)
    classification = SequenceDataset(
        **kwargs,
        task="classification",
        threshold_bps=0.25,
    )

    y = np.concatenate([b[1] for b in regression.stream()])
    labels = np.concatenate([b[1] for b in classification.stream()])

    np.testing.assert_array_equal(
        labels,
        np.where(y < -0.25, 0, np.where(y > 0.25, 2, 1)),
    )


def test_sequence_normalization(sample):
    kwargs = dict(
        paths=[sample],
        depth=DEPTH,
        horizon=HORIZON,
        window_size=WINDOW,
        stride=3,
        include_engineered=True,
        n_samples=30,
        batch_size=16,
        read_batch_size=128,
    )

    original = SequenceDataset(**kwargs)
    normalized = SequenceDataset(**kwargs, normalize=True)

    mean, std = normalized.get_stats()

    X = np.concatenate([b[0] for b in original.stream()])
    scaled = np.concatenate([b[0] for b in normalized.stream()])

    assert mean.shape == std.shape == (21,)

    np.testing.assert_array_equal(
        scaled[:, :, :20],
        X[:, :, :20],
    )
    np.testing.assert_allclose(
        scaled[:, :, 20:],
        (X[:, :, 20:] - mean) / std,
        atol=1e-4,
        rtol=1e-4,
    )


def test_pytorch_dataloader(sequence):
    expected = list(sequence.stream())

    loader = DataLoader(
        sequence,
        batch_size=None,
        num_workers=0,
    )
    actual = list(loader)

    torch.testing.assert_close(
        torch.cat([b[0] for b in actual]),
        torch.from_numpy(np.concatenate([b[0] for b in expected])),
    )
    torch.testing.assert_close(
        torch.cat([b[1] for b in actual]),
        torch.from_numpy(np.concatenate([b[1] for b in expected])),
    )


def test_target_alignment(schema, reference):
    events = reference[schema.event_id]
    mid = reference[schema.mid_price]
    returns = reference[schema.target_return_bps(HORIZON)]
    end_events = reference[schema.target_end_event(HORIZON)]

    lookup = {int(e): i for i, e in enumerate(events)}
    checked = 0

    for i, end_event in enumerate(end_events):
        if not np.isfinite(returns[i]) or int(end_event) not in lookup:
            continue

        j = lookup[int(end_event)]

        assert j == i + HORIZON

        expected = 1e4 * (mid[j] / mid[i] - 1)
        assert returns[i] == pytest.approx(expected, rel=1e-4, abs=1e-4)
        checked += 1

    assert checked > 0


@pytest.mark.parametrize("kind", ["tabular", "sequence"])
def test_future_feature_leakage(sample, schema, tmp_path, kind):
    cutoff = ROWS // 2
    table = pq.read_table(sample)

    kwargs = dict(
        depth=DEPTH,
        horizon=HORIZON,
        read_batch_size=128,
    )

    if kind == "tabular":
        cls = TabularDataset
    else:
        cls = SequenceDataset
        kwargs.update(
            window_size=WINDOW,
            stride=3,
            batch_size=16,
        )

    original = cls([sample], **kwargs)
    modified = table

    for name in original.feature_columns:
        values = table[name].to_numpy(zero_copy_only=False).copy()
        values[cutoff:] = 999.0

        modified = modified.set_column(
            modified.schema.get_field_index(name),
            name,
            pa.array(values),
        )

    path = tmp_path / "future_modified.parquet"
    pq.write_table(modified, path)

    def collect(ds):
        if kind == "tabular":
            X, _, meta = ds.load()
            return X, meta[schema.event_id]

        batches = list(ds.stream())
        return (
            np.concatenate([b[0] for b in batches]),
            np.concatenate([b[2][schema.event_id] for b in batches]),
        )

    X1, events1 = collect(original)
    X2, events2 = collect(cls([path], **kwargs))

    before1 = events1 < cutoff
    before2 = events2 < cutoff

    np.testing.assert_array_equal(events1[before1], events2[before2])
    np.testing.assert_array_equal(X1[before1], X2[before2])
