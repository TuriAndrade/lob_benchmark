from dotenv import load_dotenv

load_dotenv()

import argparse
import gzip
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import h5py
import numpy as np
import polars as pl

try:
    from .features import BookSnapshotFeatures as F
    from .schema import BookSnapshotSchema
except ImportError:
    from features import BookSnapshotFeatures as F
    from schema import BookSnapshotSchema


class TardisBookSnapshot:
    def __init__(
        self,
        data_dir: str | Path | None = None,
        horizons: tuple[int, ...] = (10, 50, 100),
        tick_size: float | None = None,
        depth_levels: tuple[int, ...] = (1, 3, 5, 10, 25),
        return_lags: tuple[int, ...] = (1, 10, 50),
        ofi_windows: tuple[int, ...] = (10, 50, 100),
        volatility_window: int = 50,
        event_rate_window: int = 50,
    ):
        self.root = (
            Path(data_dir or os.environ.get("DATA_DIR", "./")) / "tardis_book_snapshot"
        )
        self.options = {
            "horizons": tuple(sorted(horizons)),
            "tick_size": tick_size,
            "depth_levels": tuple(depth_levels),
            "return_lags": tuple(return_lags),
            "ofi_windows": tuple(ofi_windows),
            "volatility_window": volatility_window,
            "event_rate_window": event_rate_window,
        }

    def _paths(self, url: str) -> tuple[Path, Path, BookSnapshotSchema]:
        parts = urlparse(url).path.strip("/").split("/")

        if (
            len(parts) != 7
            or parts[0] != "v1"
            or not parts[2].startswith("book_snapshot_")
            or not parts[2].removeprefix("book_snapshot_").isdigit()
            or not parts[-1].endswith(".csv.gz")
        ):
            raise ValueError(f"Invalid Tardis snapshot URL: {url}")

        _, exchange, kind, year, month, day, filename = parts
        symbol = filename.removesuffix(".csv.gz")
        depth = int(kind.rsplit("_", 1)[1])
        date = f"{year}-{month}-{day}"

        schema = BookSnapshotSchema(
            depth=depth,
            exchange=exchange,
            symbol=symbol,
            date=date,
            source="tardis",
            **self.options,
        )

        horizons = "-".join(map(str, schema.horizons))
        tick = schema.tick_size if schema.tick_size is not None else 0
        folder = Path(exchange) / symbol

        raw = self.root / "raw" / folder / f"{date}_depth{depth}.csv.gz"
        processed = (
            self.root
            / "processed"
            / folder
            / f"{date}_depth{depth}_h{horizons}_tick{tick}.h5"
        )
        return raw, processed, schema

    @staticmethod
    def _valid_gzip(path: Path) -> bool:
        try:
            with gzip.open(path, "rb") as source:
                while source.read(8 * 1024 * 1024):
                    pass
            return True
        except (OSError, EOFError):
            return False

    @classmethod
    def download(cls, url: str, path: Path) -> Path:
        if path.exists():
            if cls._valid_gzip(path):
                return path
            path.unlink()

        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")

        try:
            with urlopen(url, timeout=60) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
            if not cls._valid_gzip(part):
                raise ValueError(f"Corrupted gzip download: {url}")
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)

        return path

    @staticmethod
    def _rename_map(schema: BookSnapshotSchema) -> dict[str, str]:
        names = {
            "timestamp": schema.timestamp_us,
            "local_timestamp": schema.local_timestamp_us,
        }
        for side, plural in (("bid", "bids"), ("ask", "asks")):
            for i in range(schema.depth):
                names[f"{plural}[{i}].price"] = schema.price(side, i)
                names[f"{plural}[{i}].amount"] = schema.size(side, i)
        return names

    @classmethod
    def _frame(cls, csv: Path, schema: BookSnapshotSchema) -> pl.LazyFrame:
        names = cls._rename_map(schema)
        types = {
            original: pl.Int64
            if original in ("timestamp", "local_timestamp")
            else pl.Float64
            for original in names
        }
        lf = pl.scan_csv(csv, schema_overrides=types, row_index_name=schema.event_id)

        missing = names.keys() - set(lf.collect_schema().names())
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        lf = lf.rename(names).with_columns(pl.col(schema.event_id).cast(pl.Int64))

        a, b = pl.col(schema.price("ask", 0)), pl.col(schema.price("bid", 0))
        aq, bq = pl.col(schema.size("ask", 0)), pl.col(schema.size("bid", 0))
        valid = (a > b) & (b > 0) & (aq >= 0) & (bq >= 0)
        lf = lf.with_columns(valid.fill_null(False).alias(schema.valid_book))

        def safe(expr: pl.Expr) -> pl.Expr:
            return pl.when(pl.col(schema.valid_book)).then(expr).otherwise(None)

        features = [
            safe(F.mid_price(a, b)).alias(schema.mid_price),
            safe(F.spread(a, b)).alias(schema.spread),
            pl.when(pl.col(schema.valid_book) & pl.col(schema.valid_book).shift(1))
            .then(F.ofi(b, a, bq, aq))
            .otherwise(None)
            .alias(schema.ofi),
            safe(F.microprice(a, b, aq, bq) - F.mid_price(a, b)).alias(
                schema.microprice_delta
            ),
        ]

        for k in schema.depth_levels:
            if k > schema.depth:
                continue
            bids = [pl.col(c) for c in schema.sizes("bid")[:k]]
            asks = [pl.col(c) for c in schema.sizes("ask")[:k]]
            features.extend(
                (
                    F.depth(bids).alias(schema.depth_name("bid", k)),
                    F.depth(asks).alias(schema.depth_name("ask", k)),
                    F.imbalance(bids, asks).alias(schema.imbalance(k)),
                )
            )

        lf = lf.with_columns(features)
        mid = pl.col(schema.mid_price)

        lf = lf.with_columns(
            [
                (pl.col(schema.spread) / mid * 10_000).alias(schema.spread_bps),
                (pl.col(schema.microprice_delta) / mid * 10_000).alias(
                    schema.microprice_delta_bps
                ),
                (
                    (
                        pl.col(schema.local_timestamp_us)
                        - pl.col(schema.local_timestamp_us).shift(1)
                    )
                    / 1000
                ).alias(schema.interarrival_ms),
                F.event_rate(
                    pl.col(schema.local_timestamp_us), schema.event_rate_window
                ).alias(schema.event_rate(schema.event_rate_window)),
                *(
                    F.return_bps(mid, lag).alias(schema.return_bps(lag))
                    for lag in schema.return_lags
                ),
                F.volatility_bps(mid, schema.volatility_window).alias(
                    schema.volatility_bps(schema.volatility_window)
                ),
                *(
                    pl.col(schema.ofi)
                    .rolling_sum(window_size=window)
                    .alias(schema.ofi_sum(window))
                    for window in schema.ofi_windows
                ),
            ]
        )

        targets = []
        for h in schema.horizons:
            future = mid.shift(-h)
            targets.extend(
                (
                    ((future / mid - 1) * 10_000).alias(schema.target_return_bps(h)),
                    pl.col(schema.event_id).shift(-h).alias(schema.target_end_event(h)),
                    pl.col(schema.local_timestamp_us)
                    .shift(-h)
                    .alias(schema.target_end_timestamp_us(h)),
                )
            )

        lf = lf.with_columns(targets)
        columns = (
            schema.raw_feature_columns
            + schema.engineered_feature_columns
            + schema.metadata_columns
            + schema.target_columns
            + [schema.valid_book]
        )
        return lf.select(columns)

    @staticmethod
    def _floats(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
        return frame.select(
            [pl.col(name).cast(pl.Float64).fill_null(float("nan")) for name in columns]
        ).to_numpy()

    @staticmethod
    def _integers(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
        return frame.select(
            [pl.col(name).cast(pl.Int64).fill_null(-1) for name in columns]
        ).to_numpy()

    @classmethod
    def _arrays(
        cls, frame: pl.DataFrame, schema: BookSnapshotSchema
    ) -> dict[str, np.ndarray]:
        raw = cls._floats(frame, schema.raw_feature_columns)
        engineered = cls._floats(frame, schema.engineered_feature_columns)
        mid = cls._floats(frame, [schema.mid_price])[:, 0]
        n_prices = 2 * schema.depth

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            raw[:, :n_prices] = (
                10_000 * (raw[:, :n_prices] - mid[:, None]) / mid[:, None]
            )
            raw[:, n_prices:] = np.log1p(raw[:, n_prices:])

            positive = {
                schema.depth_name(side, k)
                for side in ("bid", "ask")
                for k in schema.depth_levels
                if k <= schema.depth
            } | {
                schema.interarrival_ms,
                schema.event_rate(schema.event_rate_window),
            }
            signed = {schema.ofi} | {schema.ofi_sum(k) for k in schema.ofi_windows}
            for i, name in enumerate(schema.engineered_feature_columns):
                if name in positive:
                    engineered[:, i] = np.log1p(engineered[:, i])
                elif name in signed:
                    x = engineered[:, i]
                    engineered[:, i] = np.sign(x) * np.log1p(np.abs(x))

        return {
            "raw_features": raw.astype(np.float32),
            "engineered_features": engineered.astype(np.float32),
            "targets": cls._floats(frame, schema.target_columns).astype(np.float32),
            "valid_book": frame.get_column(schema.valid_book).to_numpy().astype(bool),
            "metadata/event_id": cls._integers(frame, [schema.event_id])[:, 0],
            "metadata/timestamp_us": cls._integers(frame, [schema.timestamp_us])[:, 0],
            "metadata/local_timestamp_us": cls._integers(
                frame, [schema.local_timestamp_us]
            )[:, 0],
            "metadata/mid_price": mid.astype(np.float32),
            "metadata/spread": cls._floats(frame, [schema.spread])[:, 0].astype(
                np.float32
            ),
            "metadata/microprice_delta": cls._floats(frame, [schema.microprice_delta])[
                :, 0
            ].astype(np.float32),
            "metadata/target_end_event": cls._integers(
                frame, [schema.target_end_event(h) for h in schema.horizons]
            ),
            "metadata/target_end_timestamp_us": cls._integers(
                frame, [schema.target_end_timestamp_us(h) for h in schema.horizons]
            ),
        }

    @staticmethod
    def _write_attributes(h5: h5py.File, schema: BookSnapshotSchema):
        attrs = h5.attrs
        for name, value in schema.attributes.items():
            if isinstance(value, list):
                attrs[name] = np.asarray(value, dtype=h5py.string_dtype("utf-8"))
            elif isinstance(value, tuple):
                attrs[name] = np.asarray(value, dtype=np.int32)
            else:
                attrs[name] = value

    @staticmethod
    def _check_output(path: Path, schema: BookSnapshotSchema):
        with h5py.File(path, "r") as h5:
            attrs = h5.attrs
            for name, expected in schema.attributes.items():
                if name not in attrs:
                    raise ValueError(f"Missing HDF5 attribute: {name}")
                actual = attrs[name]
                if isinstance(expected, (tuple, list)):
                    actual = [
                        x.decode() if isinstance(x, bytes) else x
                        for x in np.atleast_1d(actual)
                    ]
                    if actual != list(expected):
                        raise ValueError(f"HDF5 attribute mismatch: {name}")
                else:
                    if isinstance(actual, bytes):
                        actual = actual.decode()
                    if actual != expected:
                        raise ValueError(f"HDF5 attribute mismatch: {name}")

            n = h5["raw_features"].shape[0]
            dimensions = {
                "raw_features": (n, len(schema.raw_feature_columns)),
                "engineered_features": (n, len(schema.engineered_feature_columns)),
                "targets": (n, len(schema.horizons)),
                "valid_book": (n,),
                "metadata/event_id": (n,),
                "metadata/timestamp_us": (n,),
                "metadata/local_timestamp_us": (n,),
                "metadata/mid_price": (n,),
                "metadata/spread": (n,),
                "metadata/microprice_delta": (n,),
                "metadata/target_end_event": (n, len(schema.horizons)),
                "metadata/target_end_timestamp_us": (n, len(schema.horizons)),
            }
            for name, shape in dimensions.items():
                if h5[name].shape != shape:
                    raise ValueError(f"HDF5 dataset shape mismatch: {name}")
                if name == "valid_book":
                    expected_dtype = np.bool_
                elif name in (
                    "raw_features",
                    "engineered_features",
                    "targets",
                    "metadata/mid_price",
                    "metadata/spread",
                    "metadata/microprice_delta",
                ):
                    expected_dtype = np.float32
                else:
                    expected_dtype = np.int64
                if h5[name].dtype != np.dtype(expected_dtype):
                    raise ValueError(f"HDF5 dataset dtype mismatch: {name}")
            if n == 0:
                raise ValueError("Empty HDF5 file")

    def process(
        self,
        source: Path,
        output: Path,
        schema: BookSnapshotSchema,
        batch_size: int = 65_536,
    ) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        part = output.with_suffix(".h5.part")

        try:
            with tempfile.TemporaryDirectory(dir=source.parent) as tmp:
                csv = Path(tmp) / "snapshot.csv"
                with gzip.open(source, "rb") as packed, csv.open("wb") as unpacked:
                    shutil.copyfileobj(packed, unpacked, length=8 * 1024 * 1024)

                with h5py.File(part, "w") as h5:
                    self._write_attributes(h5, schema)
                    total = 0

                    for batch in self._frame(csv, schema).collect_batches(
                        chunk_size=batch_size, maintain_order=True, engine="streaming"
                    ):
                        if batch.is_empty():
                            continue
                        arrays = self._arrays(batch, schema)
                        stop = total + batch.height

                        for name, values in arrays.items():
                            if name not in h5:
                                shape = (0,) + values.shape[1:]
                                h5.create_dataset(
                                    name,
                                    shape=shape,
                                    maxshape=(None,) + values.shape[1:],
                                    chunks=(4096,) + values.shape[1:],
                                    dtype=values.dtype,
                                )
                            ds = h5[name]
                            ds.resize((stop,) + ds.shape[1:])
                            ds[total:stop] = values
                        total = stop

                    if total == 0:
                        raise ValueError(f"No snapshots in {source}")

            self._check_output(part, schema)
            part.replace(output)
        finally:
            part.unlink(missing_ok=True)

        return output

    def prepare(
        self,
        urls: list[str],
        overwrite: bool = False,
    ) -> list[Path]:
        outputs = []
        for url in urls:
            raw, processed, schema = self._paths(url)
            if processed.exists() and not overwrite:
                try:
                    self._check_output(processed, schema)
                except (OSError, KeyError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid or incompatible {processed}; use --overwrite"
                    ) from exc
            else:
                self.process(self.download(url, raw), processed, schema)
            outputs.append(processed)
        return outputs


if __name__ == "__main__":
    DEFAULT_URLS = [
        f"https://datasets.tardis.dev/v1/binance-futures/book_snapshot_5/{year}/{month:02d}/01/BTCUSDT.csv.gz"
        for year, month in [
            (2020, 9),
            (2020, 10),
            (2020, 11),
            (2020, 12),
            (2021, 1),
            (2021, 2),
        ]
    ]

    parser = argparse.ArgumentParser(
        description="Download and prepare Tardis order book snapshots"
    )
    parser.add_argument("urls", nargs="*", default=DEFAULT_URLS)
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--tick-size", type=float, default=None)
    parser.add_argument(
        "--depth-levels", type=int, nargs="+", default=[1, 3, 5, 10, 25]
    )
    parser.add_argument("--return-lags", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument("--ofi-windows", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--volatility-window", type=int, default=50)
    parser.add_argument("--event-rate-window", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = TardisBookSnapshot(
        horizons=tuple(args.horizons),
        tick_size=args.tick_size,
        depth_levels=tuple(args.depth_levels),
        return_lags=tuple(args.return_lags),
        ofi_windows=tuple(args.ofi_windows),
        volatility_window=args.volatility_window,
        event_rate_window=args.event_rate_window,
    )
    for path in dataset.prepare(args.urls, overwrite=args.overwrite):
        print(path)
