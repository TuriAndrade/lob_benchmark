"""Causal features for reconstructed, fixed-depth Bybit order books."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BybitFeatures:
    depth: int = 10
    depth_levels: tuple[int, ...] = (1, 5, 10)
    return_lags: tuple[int, ...] = (1, 10, 50)
    windows: tuple[int, ...] = (10, 50, 100)
    horizons: tuple[int, ...] = (10, 50, 100)

    def __post_init__(self):
        if type(self.depth) is not int or self.depth < 1:
            raise ValueError("depth must be a positive integer")
        for name in ("depth_levels", "return_lags", "windows", "horizons"):
            values = getattr(self, name)
            if (
                not values
                or any(type(v) is not int or v < 1 for v in values)
                or tuple(sorted(set(values))) != values
            ):
                raise ValueError(
                    f"{name} must contain strictly increasing positive integers"
                )
        if self.depth_levels[-1] > self.depth:
            raise ValueError("depth_levels cannot exceed depth")
        if self.windows[0] < 2:
            raise ValueError("windows must be at least 2")

    @staticmethod
    def price(side, level):
        return f"{side}_price_{level}"

    @staticmethod
    def size(side, level):
        return f"{side}_size_{level}"

    @staticmethod
    def imbalance(depth):
        return f"imbalance_{depth}"

    @staticmethod
    def flow(side, level):
        return f"{side}_flow_{level}"

    @staticmethod
    def return_bps(lag):
        return f"return_{lag}_bps"

    @staticmethod
    def target_return_bps(horizon):
        return f"target_return_{horizon}_bps"

    @property
    def raw_feature_names(self):
        return tuple(
            f"{side}_{field}_{level}"
            for level in range(1, self.depth + 1)
            for side, field in (
                ("ask", "price"),
                ("ask", "size"),
                ("bid", "price"),
                ("bid", "size"),
            )
        )

    @property
    def engineered_feature_names(self):
        names = ["spread_bps", "microprice_delta_bps"]
        for k in self.depth_levels:
            names.extend(
                (
                    f"log_bid_notional_{k}",
                    f"log_ask_notional_{k}",
                    self.imbalance(k),
                )
            )
        names.extend(("bid_distance_bps", "ask_distance_bps"))

        for i in range(1, self.depth + 1):
            names.extend((self.flow("bid", i), self.flow("ask", i)))

        for w in self.windows:
            names.extend((f"ofi_best_sum_{w}", f"ofi_all_sum_{w}"))

        names.extend(self.return_bps(k) for k in self.return_lags)

        for w in self.windows:
            names.append(f"volatility_{w}_bps")
        for w in self.windows:
            names.append(f"spread_mean_{w}_bps")
        for k in self.depth_levels:
            names.extend(f"imbalance_{k}_mean_{w}" for w in self.windows)

        names.append("log_interarrival_ms")
        for w in self.windows:
            names.extend((f"log_update_rate_{w}", f"mid_change_rate_{w}"))
        return tuple(names)

    @property
    def target_names(self):
        return tuple(self.target_return_bps(h) for h in self.horizons)

    @property
    def metadata_names(self):
        return (
            "timestamp_us",
            "system_timestamp_us",
            "update_id",
            "seq",
            "mid_price",
            "target_end_event",
            "target_end_timestamp_us",
        )

    @property
    def attributes(self):
        return {
            "raw_feature_names": self.raw_feature_names,
            "engineered_feature_names": self.engineered_feature_names,
            "target_names": self.target_names,
            "depth": self.depth,
            "depth_levels": self.depth_levels,
            "return_lags": self.return_lags,
            "windows": self.windows,
            "horizons": self.horizons,
        }

    @property
    def required_history(self):
        return max(self.return_lags[-1], self.windows[-1])

    @staticmethod
    def _rolling_sum(x, w):
        result = np.full(len(x), np.nan)
        if len(x) < w:
            return result
        finite = np.isfinite(x)
        sums = np.r_[0.0, np.cumsum(np.where(finite, x, 0.0))]
        counts = np.r_[0, np.cumsum(finite)]
        result[w - 1 :] = np.where(
            counts[w:] - counts[:-w] == w, sums[w:] - sums[:-w], np.nan
        )
        return result

    @classmethod
    def _rolling_std(cls, x, w):
        s = cls._rolling_sum(x, w)
        s2 = cls._rolling_sum(x * x, w)
        return np.sqrt(np.maximum((s2 - s * s / w) / (w - 1), 0))

    def extract(self, bid_prices, ask_prices, bid_sizes, ask_sizes, timestamp_us):
        """Return raw, engineered, mid_price, valid_book.

        Book arrays are (N, depth), sorted best-first. Timestamps are integer
        Unix microseconds. Price * size must represent quote notional.
        Each input batch must belong to a single uninterrupted sequence.
        """
        bp, ap, bq, aq = (
            np.asarray(x, dtype=np.float64)
            for x in (bid_prices, ask_prices, bid_sizes, ask_sizes)
        )
        if (
            bp.ndim != 2
            or bp.shape[1] != self.depth
            or any(x.shape != bp.shape for x in (ap, bq, aq))
        ):
            raise ValueError("book arrays must have shape (N, depth)")

        n = len(bp)
        ts = np.asarray(timestamp_us)
        if ts.shape != (n,) or ts.dtype.kind not in "iu":
            raise ValueError("timestamp_us must be a 1D integer array of length N")
        ts = ts.astype(np.int64, copy=False)
        if (np.diff(ts) < 0).any():
            raise ValueError("timestamps must be nondecreasing")

        valid = (
            np.isfinite(bp).all(1)
            & np.isfinite(ap).all(1)
            & np.isfinite(bq).all(1)
            & np.isfinite(aq).all(1)
            & (bp > 0).all(1)
            & (ap > 0).all(1)
            & (bq > 0).all(1)
            & (aq > 0).all(1)
            & (np.diff(bp, axis=1) < 0).all(1)
            & (np.diff(ap, axis=1) > 0).all(1)
            & (bp[:, 0] < ap[:, 0])
        )
        raw = np.stack((ap, aq, bp, bq), axis=-1).reshape(n, 4 * self.depth)
        raw[~valid] = np.nan

        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            mid = (bp[:, 0] + ap[:, 0]) / 2
            mid[~valid] = np.nan
            spread = 1e4 * (ap[:, 0] - bp[:, 0]) / mid
            micro = (ap[:, 0] * bq[:, 0] + bp[:, 0] * aq[:, 0]) / (bq[:, 0] + aq[:, 0])
            f = {
                "spread_bps": spread,
                "microprice_delta_bps": 1e4 * (micro - mid) / mid,
            }

            bn, an = bp * bq, ap * aq
            bc, ac = np.cumsum(bn, axis=1), np.cumsum(an, axis=1)
            for k in self.depth_levels:
                bd, ad = bc[:, k - 1], ac[:, k - 1]
                f[f"log_bid_notional_{k}"] = np.log1p(bd)
                f[f"log_ask_notional_{k}"] = np.log1p(ad)
                f[self.imbalance(k)] = (bd - ad) / (bd + ad)

            f["bid_distance_bps"] = (
                1e4 * (bn * (mid[:, None] - bp)).sum(1) / (bc[:, -1] * mid)
            )
            f["ask_distance_bps"] = (
                1e4 * (an * (ap - mid[:, None])).sum(1) / (ac[:, -1] * mid)
            )

            bid_flow = np.full((n, self.depth), np.nan)
            ask_flow = np.full((n, self.depth), np.nan)
            if n > 1:
                bid_flow[1:] = np.where(
                    bp[1:] > bp[:-1],
                    bq[1:],
                    np.where(bp[1:] < bp[:-1], -bq[:-1], bq[1:] - bq[:-1]),
                )
                ask_flow[1:] = np.where(
                    ap[1:] < ap[:-1],
                    aq[1:],
                    np.where(ap[1:] > ap[:-1], -aq[:-1], aq[1:] - aq[:-1]),
                )
                paired = valid[1:] & valid[:-1]
                bid_flow[1:][~paired] = np.nan
                ask_flow[1:][~paired] = np.nan

            for i in range(self.depth):
                f[self.flow("bid", i + 1)] = bid_flow[:, i]
                f[self.flow("ask", i + 1)] = ask_flow[:, i]

            net = bid_flow - ask_flow
            best, all_levels = net[:, 0], net.sum(axis=1)
            for w in self.windows:
                f[f"ofi_best_sum_{w}"] = self._rolling_sum(best, w)
                f[f"ofi_all_sum_{w}"] = self._rolling_sum(all_levels, w)

            for lag in self.return_lags:
                r = np.full(n, np.nan)
                if n > lag:
                    r[lag:] = 1e4 * (mid[lag:] / mid[:-lag] - 1)
                f[self.return_bps(lag)] = r

            log_return = np.full(n, np.nan)
            if n > 1:
                log_return[1:] = np.log(mid[1:] / mid[:-1])

            for w in self.windows:
                f[f"volatility_{w}_bps"] = 1e4 * self._rolling_std(log_return, w)
                f[f"spread_mean_{w}_bps"] = self._rolling_sum(spread, w) / w

            for k in self.depth_levels:
                imbalance = f[self.imbalance(k)]
                for w in self.windows:
                    f[f"imbalance_{k}_mean_{w}"] = self._rolling_sum(imbalance, w) / w

            interarrival = np.full(n, np.nan)
            mid_changed = np.full(n, np.nan)
            if n > 1:
                paired = valid[1:] & valid[:-1]
                interarrival[1:] = np.where(
                    paired,
                    np.log1p(np.diff(ts).astype(np.float64) / 1e3),
                    np.nan,
                )
                mid_changed[1:] = np.where(
                    paired, (mid[1:] != mid[:-1]).astype(float), np.nan
                )
            f["log_interarrival_ms"] = interarrival

            for w in self.windows:
                change_sum = self._rolling_sum(mid_changed, w)
                rate = np.full(n, np.nan)
                if n > w:
                    elapsed = (ts[w:] - ts[:-w]).astype(np.float64)
                    np.divide(w * 1e6, elapsed, out=rate[w:], where=elapsed > 0)
                rate[~np.isfinite(change_sum)] = np.nan
                f[f"log_update_rate_{w}"] = np.log1p(rate)
                f[f"mid_change_rate_{w}"] = change_sum / w

            engineered = np.column_stack(
                [f[name] for name in self.engineered_feature_names]
            )

        engineered[~valid] = np.nan
        return (
            raw.astype(np.float32),
            engineered.astype(np.float32),
            mid,
            valid,
        )
