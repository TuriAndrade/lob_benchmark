"""Stream historical Bybit order books into continuous HDF5 sequences."""

import json
import os
import shutil
from datetime import date as Date
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from zipfile import ZipFile

import h5py
import numpy as np
import orjson
from sortedcontainers import SortedDict

from .features import BybitFeatures


class BybitProcessor:
    def __init__(
        self,
        data_dir: str | Path | None = None,
        features: BybitFeatures | None = None,
        batch_size: int = 8192,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.root = Path(data_dir or os.environ.get("DATA_DIR", ".")) / "bybit"
        self.features = features or BybitFeatures()
        self.batch_size = batch_size

    @staticmethod
    def download(url: str, path: Path) -> Path:
        if path.is_file():
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        try:
            with urlopen(url, timeout=90) as src, part.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            with ZipFile(part):
                pass
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
        return path

    @staticmethod
    def _append(h5: h5py.File, name: str, values: np.ndarray) -> None:
        if name not in h5:
            shape = (0, *values.shape[1:])
            h5.create_dataset(
                name,
                shape=shape,
                maxshape=(None, *shape[1:]),
                dtype=values.dtype,
                chunks=(min(8192, max(1, len(values))), *shape[1:]),
            )
        ds = h5[name]
        old = len(ds)
        ds.resize(old + len(values), axis=0)
        ds[old:] = values

    def _write_attributes(
        self, h5: h5py.File, symbol: str, category: str, date: str, source_depth: int
    ):
        for key, value in self.features.attributes.items():
            h5.attrs[key] = (
                np.asarray(value, dtype=h5py.string_dtype("utf-8"))
                if key.endswith("_names")
                else np.asarray(value, dtype=np.int32)
                if isinstance(value, tuple)
                else value
            )
        h5.attrs.update(
            source="bybit",
            category=category,
            symbol=symbol,
            date=date,
            source_depth=source_depth,
        )
        h5.require_group("metadata")

    def _finish(self, h5: h5py.File) -> None:
        """Populate future-return targets without loading full price histories."""
        n = len(h5["raw_features"])
        h = self.features.horizons

        targets = h5.create_dataset(
            "targets",
            (n, len(h)),
            dtype="f4",
            chunks=(min(8192, n), len(h)),
            fillvalue=np.nan,
        )
        end_event = h5.create_dataset(
            "metadata/target_end_event",
            (n, len(h)),
            dtype="i8",
            chunks=(min(8192, n), len(h)),
            fillvalue=-1,
        )
        end_time = h5.create_dataset(
            "metadata/target_end_timestamp_us",
            (n, len(h)),
            dtype="i8",
            chunks=(min(8192, n), len(h)),
            fillvalue=-1,
        )

        mid = h5["metadata/mid_price"]
        times = h5["metadata/timestamp_us"]
        valid = h5["valid_book"]

        for j, horizon in enumerate(h):
            for start in range(0, n - horizon, self.batch_size):
                stop = min(n - horizon, start + self.batch_size)
                now = mid[start:stop]
                future = mid[start + horizon : stop + horizon]

                flags = valid[start : stop + horizon].astype(np.int64)
                prefix = np.r_[0, np.cumsum(~flags.astype(bool))]
                ok = prefix[horizon + 1 :] - prefix[: -(horizon + 1)] == 0
                ok &= np.isfinite(now) & np.isfinite(future) & (now > 0)

                result = np.full(stop - start, np.nan, dtype=np.float32)
                result[ok] = (1e4 * (future[ok] / now[ok] - 1)).astype(np.float32)
                targets[start:stop, j] = result

                indices = np.arange(start + horizon, stop + horizon, dtype=np.int64)
                indices[~ok] = -1
                end_event[start:stop, j] = indices

                t = times[start + horizon : stop + horizon]
                end_time[start:stop, j] = np.where(ok, t, -1)

    def process(
        self,
        source: str | Path,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
    ) -> list[Path]:
        """Process one daily ZIP into one HDF5 per uninterrupted snapshot run."""
        source, output_dir = Path(source), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = source.name.removesuffix(".data.zip")
        date = stem.split("_", 1)[0]
        try:
            day = Date.fromisoformat(date)
            if day.isoformat() != date:
                raise ValueError
        except ValueError:
            raise ValueError(f"Invalid archive date: {date!r}") from None

        day_end_us = (
            day.toordinal() - Date(1970, 1, 1).toordinal() + 1
        ) * 86_400_000_000

        manifest = output_dir / f"{stem}_depth{self.features.depth}.json"
        previous = json.loads(manifest.read_text()) if manifest.exists() else []

        if manifest.exists() and not overwrite:
            paths = [output_dir / name for name in previous]
            if paths and all(p.exists() for p in paths):
                with h5py.File(paths[0]) as h5:
                    for key, expected in self.features.attributes.items():
                        actual = h5.attrs[key]
                        if isinstance(expected, tuple):
                            actual = tuple(
                                x.decode() if isinstance(x, bytes) else x
                                for x in actual
                            )
                        if actual != expected:
                            raise ValueError(
                                f"Incompatible HDF5 attribute {key}; use overwrite=True"
                            )
                return paths

        paths: list[Path] = []
        active = None
        buffers = None
        history = None
        book = None
        last_u = last_seq = last_ts = None
        symbol = category = None
        source_depth = None

        def flush():
            nonlocal history
            if not buffers["timestamp_us"]:
                return

            keys = (
                "bid_prices",
                "ask_prices",
                "bid_sizes",
                "ask_sizes",
                "timestamp_us",
            )
            current = {k: np.asarray(buffers[k]) for k in keys}
            if history is not None:
                inputs = {k: np.concatenate((history[k], current[k])) for k in keys}
            else:
                inputs = current

            raw, engineered, mid, valid = self.features.extract(**inputs)
            count = len(current["timestamp_us"])

            self._append(active, "raw_features", raw[-count:])
            self._append(active, "engineered_features", engineered[-count:])
            self._append(active, "valid_book", valid[-count:])

            for name in ("timestamp_us", "system_timestamp_us", "update_id", "seq"):
                self._append(
                    active,
                    f"metadata/{name}",
                    np.asarray(buffers[name], dtype=np.int64),
                )
            self._append(active, "metadata/mid_price", mid[-count:])

            keep = min(self.features.required_history, len(inputs["timestamp_us"]))
            history = {k: x[-keep:].copy() for k, x in inputs.items()}
            for entries in buffers.values():
                entries.clear()

        def close():
            nonlocal active, buffers, history
            if active is None:
                return
            try:
                flush()
                self._finish(active)
                if len(active["raw_features"]) == 0:
                    raise ValueError("Empty reconstructed segment")
                active.flush()
            finally:
                active.close()
                active = None

            temp = paths[-1].with_suffix(".h5.part")
            temp.replace(paths[-1])
            buffers = history = None

        try:
            with ZipFile(source) as archive:
                members = [
                    entry
                    for entry in archive.infolist()
                    if entry.filename.endswith(".data") and not entry.is_dir()
                ]
                if len(members) != 1:
                    raise ValueError(f"Expected exactly one .data member in {source}")

                with archive.open(members[0]) as stream:
                    for line_number, line in enumerate(stream, 1):
                        if not line.strip():
                            continue

                        try:
                            message = orjson.loads(line)
                            data = message["data"]
                            topic = message["topic"].split(".")
                            depth = int(topic[1])
                            current_symbol = data["s"]

                            if (
                                len(topic) != 3
                                or topic[0] != "orderbook"
                                or topic[2] != current_symbol
                            ):
                                raise ValueError("Inconsistent order book topic")
                            if depth < self.features.depth:
                                raise ValueError(
                                    "Source depth is smaller than feature depth"
                                )

                            timestamp = int(message.get("cts", message["ts"])) * 1000
                            system_timestamp = int(message["ts"]) * 1000
                            u, seq = int(data["u"]), int(data["seq"])
                            kind = message["type"]
                            bids, asks = data["b"], data["a"]
                        except (KeyError, TypeError, IndexError, ValueError) as exc:
                            raise ValueError(
                                f"Invalid Bybit message at line {line_number}"
                            ) from exc

                        if system_timestamp >= day_end_us:
                            break

                        if symbol is None:
                            symbol, source_depth = current_symbol, depth
                            category = source.parent.parent.name
                        elif (current_symbol, depth) != (symbol, source_depth):
                            raise ValueError(
                                f"Mixed symbols/depths at line {line_number}"
                            )

                        if kind == "snapshot":
                            close()
                            book = (SortedDict(), SortedDict())
                            last_u = last_seq = last_ts = None

                            path = output_dir / (
                                f"{date}_{symbol}_ob{source_depth}_"
                                f"depth{self.features.depth}_part{len(paths):03d}.h5"
                            )
                            if path.exists() and not overwrite:
                                raise FileExistsError(path)

                            paths.append(path)
                            part = path.with_suffix(".h5.part")
                            active = h5py.File(part, "w")
                            self._write_attributes(
                                active, symbol, category, date, source_depth
                            )
                            buffers = {
                                k: []
                                for k in (
                                    "bid_prices",
                                    "ask_prices",
                                    "bid_sizes",
                                    "ask_sizes",
                                    "timestamp_us",
                                    "system_timestamp_us",
                                    "update_id",
                                    "seq",
                                )
                            }

                        elif kind != "delta":
                            raise ValueError(
                                f"Unsupported message type {kind!r} "
                                f"at line {line_number}"
                            )
                        elif book is None:
                            raise ValueError("Delta before first snapshot")
                        elif u == last_u and seq == last_seq:
                            continue
                        elif u < last_u or seq < last_seq or timestamp < last_ts:
                            raise ValueError(
                                f"Out-of-order delta at line {line_number}"
                            )

                        for side, updates in zip(book, (bids, asks)):
                            for price_text, size_text in updates:
                                price, size = float(price_text), float(size_text)
                                if (
                                    not np.isfinite(price)
                                    or price <= 0
                                    or not np.isfinite(size)
                                    or size < 0
                                ):
                                    raise ValueError(
                                        f"Invalid book value at line {line_number}"
                                    )
                                if size == 0:
                                    side.pop(price, None)
                                else:
                                    side[price] = size

                        bid_book, ask_book = book
                        while len(bid_book) > source_depth:
                            bid_book.popitem(0)
                        while len(ask_book) > source_depth:
                            ask_book.popitem(-1)

                        L = self.features.depth
                        bid_levels = [
                            bid_book.peekitem(-i - 1)
                            for i in range(min(L, len(bid_book)))
                        ]
                        ask_levels = [
                            ask_book.peekitem(i) for i in range(min(L, len(ask_book)))
                        ]

                        for name, levels, idx in (
                            ("bid_prices", bid_levels, 0),
                            ("bid_sizes", bid_levels, 1),
                            ("ask_prices", ask_levels, 0),
                            ("ask_sizes", ask_levels, 1),
                        ):
                            buffers[name].append(
                                [row[idx] for row in levels]
                                + [np.nan] * (L - len(levels))
                            )

                        for name, value in (
                            ("timestamp_us", timestamp),
                            ("system_timestamp_us", system_timestamp),
                            ("update_id", u),
                            ("seq", seq),
                        ):
                            buffers[name].append(value)

                        last_u, last_seq, last_ts = u, seq, timestamp
                        if len(buffers["timestamp_us"]) >= self.batch_size:
                            flush()

            if not paths:
                raise ValueError(f"No snapshots found in {source}")

            close()

            tmp_manifest = manifest.with_suffix(".json.part")
            tmp_manifest.write_text(json.dumps([p.name for p in paths]))
            tmp_manifest.replace(manifest)

            if overwrite:
                for name in set(previous) - {p.name for p in paths}:
                    (output_dir / name).unlink(missing_ok=True)

            return paths
        finally:
            if active is not None:
                active.close()
            for path in paths:
                path.with_suffix(".h5.part").unlink(missing_ok=True)

    def prepare(
        self,
        paths: list[str | Path],
        overwrite: bool = False,
    ) -> list[Path]:
        """Download (if needed) and process URLs or local daily ZIP archives."""
        outputs = []
        for item in paths:
            local = (
                Path(item)
                if not str(item).startswith(("http://", "https://"))
                else None
            )
            parts = Path(urlparse(str(item)).path).parts
            raw = local if local is not None else self.root / "raw" / Path(*parts[-3:])
            if local is None:
                raw = self.download(str(item), raw)
            outputs.extend(
                self.process(
                    raw,
                    self.root / "processed",
                    overwrite=overwrite,
                )
            )
        return outputs
