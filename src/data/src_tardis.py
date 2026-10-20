from dotenv import load_dotenv

load_dotenv()

import argparse
import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import polars as pl

try:
    from .features import BookSnapshotFeatures as F
    from .schema import BookSnapshotSchema
except ImportError:  # Direct script execution
    from features import BookSnapshotFeatures as F
    from schema import BookSnapshotSchema


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


class TardisBookSnapshot:
    def __init__(
        self,
        data_dir: str | Path | None = None,
        horizons: tuple[int, ...] = (10, 50, 100),
        tick_size: float | None = None,
    ):
        if (
            not horizons
            or any(h < 1 for h in horizons)
            or len(set(horizons)) != len(horizons)
        ):
            raise ValueError("horizons must contain unique positive integers")
        if tick_size is not None and tick_size <= 0:
            raise ValueError("tick_size must be positive")

        self.root = (
            Path(data_dir or os.environ.get("DATA_DIR", "./")) / "tardis_book_snapshot"
        )
        self.horizons = tuple(sorted(horizons))
        self.tick_size = tick_size

    def _paths(self, url: str) -> tuple[Path, Path, int]:
        parts = urlparse(url).path.strip("/").split("/")

        if (
            len(parts) != 7
            or parts[0] != "v1"
            or parts[2] not in ("book_snapshot_5", "book_snapshot_25")
            or not parts[-1].endswith(".csv.gz")
        ):
            raise ValueError(f"Invalid Tardis snapshot URL: {url}")

        _, exchange, kind, year, month, day, filename = parts

        symbol = filename.removesuffix(".csv.gz")
        depth = int(kind.rsplit("_", 1)[1])
        date = f"{year}-{month}-{day}"

        horizons = "-".join(map(str, sorted(self.horizons)))
        tick = self.tick_size if self.tick_size is not None else 0

        folder = Path(exchange) / symbol

        raw = self.root / "raw" / folder / f"{date}_depth{depth}.csv.gz"

        processed = (
            self.root
            / "processed"
            / folder
            / f"{date}_depth{depth}_h{horizons}_tick{tick}.parquet"
        )

        return raw, processed, depth

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
        except Exception:
            part.unlink(missing_ok=True)
            raise

        return path

    @staticmethod
    def _rename_map(schema: BookSnapshotSchema) -> dict[str, str]:
        names = {
            "exchange": schema.exchange,
            "symbol": schema.symbol,
            "timestamp": schema.timestamp_us,
            "local_timestamp": schema.local_timestamp_us,
        }

        for side, plural in (("bid", "bids"), ("ask", "asks")):
            for i in range(schema.depth):
                names[f"{plural}[{i}].price"] = schema.price(side, i)
                names[f"{plural}[{i}].amount"] = schema.size(side, i)

        return names

    def _frame(self, csv: Path, depth: int) -> pl.LazyFrame:
        schema = BookSnapshotSchema(depth)
        names = self._rename_map(schema)

        lf = pl.scan_csv(
            csv,
            schema_overrides={src: schema.dtypes[dst] for src, dst in names.items()},
            row_index_name=schema.event_id,
        )

        missing = names.keys() - set(lf.collect_schema().names())
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        lf = lf.rename(names).with_columns(pl.col(schema.event_id).cast(pl.Int64))

        a, b = pl.col(schema.price("ask", 0)), pl.col(schema.price("bid", 0))
        aq, bq = pl.col(schema.size("ask", 0)), pl.col(schema.size("bid", 0))

        valid = (a > b) & (b > 0) & (aq >= 0) & (bq >= 0)
        lf = lf.with_columns(valid.fill_null(False).alias(schema.valid_book))

        safe = (
            lambda expr: pl.when(pl.col(schema.valid_book)).then(expr).otherwise(None)
        )

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
            bids = [pl.col(c) for c in schema.sizes("bid")[:k]]
            asks = [pl.col(c) for c in schema.sizes("ask")[:k]]

            features += [
                F.depth(bids).alias(schema.depth_name("bid", k)),
                F.depth(asks).alias(schema.depth_name("ask", k)),
                F.imbalance(bids, asks).alias(schema.imbalance(k)),
            ]

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
                    pl.col(schema.local_timestamp_us),
                    schema.EVENT_RATE_WINDOW,
                ).alias(schema.event_rate(schema.EVENT_RATE_WINDOW)),
                *[
                    F.return_bps(mid, n).alias(schema.return_bps(n))
                    for n in schema.RETURN_LAGS
                ],
                F.volatility_bps(mid, schema.VOLATILITY_WINDOW).alias(
                    schema.volatility_bps(schema.VOLATILITY_WINDOW)
                ),
                *[
                    pl.col(schema.ofi)
                    .rolling_sum(window_size=n)
                    .alias(schema.ofi_sum(n))
                    for n in schema.OFI_WINDOWS
                ],
            ]
        )

        targets = []

        for h in self.horizons:
            future = mid.shift(-h)

            targets.extend(
                [
                    ((future / mid - 1) * 10_000).alias(schema.target_return_bps(h)),
                    pl.col(schema.event_id).shift(-h).alias(schema.target_end_event(h)),
                    pl.col(schema.local_timestamp_us)
                    .shift(-h)
                    .alias(schema.target_end_timestamp_us(h)),
                ]
            )

            if self.tick_size is not None:
                targets.append(
                    ((future - mid) / self.tick_size).alias(schema.target_move_ticks(h))
                )

        lf = lf.with_columns(targets)

        return lf

    def process(self, source: Path, output: Path, depth: int) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_suffix(".parquet.part")

        try:
            with tempfile.TemporaryDirectory(dir=source.parent) as tmp:
                csv = Path(tmp) / "snapshot.csv"

                with gzip.open(source, "rb") as packed, csv.open("wb") as unpacked:
                    shutil.copyfileobj(packed, unpacked, length=8 * 1024 * 1024)

                self._frame(csv, depth).sink_parquet(
                    temporary_output,
                    compression="zstd",
                    row_group_size=131_072,
                    engine="streaming",
                )

            temporary_output.replace(output)
        finally:
            temporary_output.unlink(missing_ok=True)

        return output

    def prepare(
        self,
        urls: list[str] = DEFAULT_URLS,
        overwrite: bool = False,
    ) -> list[Path]:
        outputs = []

        for url in urls:
            raw, processed, depth = self._paths(url)
            metadata = processed.with_suffix(".json")
            options = {
                "url": url,
                "depth": depth,
                "horizons": list(self.horizons),
                "tick_size": self.tick_size,
            }

            if processed.exists() and not overwrite:
                if not metadata.exists() or json.loads(metadata.read_text()) != options:
                    raise ValueError(
                        f"Preprocessing options changed for {processed}; "
                        "use --overwrite"
                    )
            else:
                self.process(self.download(url, raw), processed, depth)
                metadata.write_text(json.dumps(options, indent=2))

            outputs.append(processed)

        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare Tardis order book snapshots"
    )
    parser.add_argument("urls", nargs="*", default=DEFAULT_URLS)
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--tick-size", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    dataset = TardisBookSnapshot(
        horizons=args.horizons,
        tick_size=args.tick_size,
    )

    for path in dataset.prepare(args.urls, overwrite=args.overwrite):
        print(path)
