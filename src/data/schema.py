from dataclasses import dataclass
from datetime import date as Date
from math import isfinite
from typing import ClassVar


@dataclass(frozen=True)
class LOBSnapshotSchema:
    depth: int
    exchange: str
    symbol: str
    date: str
    source: str

    depth_levels: tuple[int, ...] = (1, 3, 5, 10, 25)
    return_lags: tuple[int, ...] = (1, 10, 50)
    ofi_windows: tuple[int, ...] = (10, 50, 100)
    volatility_window: int = 50
    event_rate_window: int = 50
    horizons: tuple[int, ...] = (10, 50, 100)
    tick_size: float | None = None

    event_id: ClassVar[str] = "event_id"
    timestamp_us: ClassVar[str] = "timestamp_us"
    local_timestamp_us: ClassVar[str] = "local_timestamp_us"
    valid_book: ClassVar[str] = "valid_book"

    mid_price: ClassVar[str] = "mid_price"
    spread: ClassVar[str] = "spread"
    spread_bps: ClassVar[str] = "spread_bps"
    microprice_delta: ClassVar[str] = "microprice_delta"
    microprice_delta_bps: ClassVar[str] = "microprice_delta_bps"
    ofi: ClassVar[str] = "ofi"
    interarrival_ms: ClassVar[str] = "interarrival_ms"

    def __post_init__(self):
        if not isinstance(self.depth, int) or self.depth < 1:
            raise ValueError("depth must be a positive integer")

        if not self.exchange or not self.symbol or not self.source:
            raise ValueError("exchange, symbol and source must be nonempty")

        try:
            if Date.fromisoformat(self.date).isoformat() != self.date:
                raise ValueError
        except ValueError:
            raise ValueError("date must use YYYY-MM-DD format") from None

        for name in ("depth_levels", "return_lags", "ofi_windows", "horizons"):
            values = getattr(self, name)
            if not values or any(not isinstance(v, int) or v < 1 for v in values):
                raise ValueError(f"{name} must contain positive integers")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique values")

        for name in ("volatility_window", "event_rate_window"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        if self.tick_size is not None and (
            not isfinite(self.tick_size) or self.tick_size <= 0
        ):
            raise ValueError("tick_size must be positive and finite")

    def price(self, side: str, level: int) -> str:
        return f"{side}_price_{level}"

    def size(self, side: str, level: int) -> str:
        return f"{side}_size_{level}"

    def prices(self, side: str) -> list[str]:
        return [self.price(side, i) for i in range(self.depth)]

    def sizes(self, side: str) -> list[str]:
        return [self.size(side, i) for i in range(self.depth)]

    def depth_name(self, side: str, levels: int) -> str:
        return f"{side}_depth_{levels}"

    def imbalance(self, levels: int) -> str:
        return f"imbalance_{levels}"

    def ofi_sum(self, window: int) -> str:
        return f"ofi_sum_{window}"

    def return_bps(self, lag: int) -> str:
        return f"return_{lag}_bps"

    def volatility_bps(self, window: int) -> str:
        return f"volatility_{window}_bps"

    def event_rate(self, window: int) -> str:
        return f"event_rate_{window}"

    def target_return_bps(self, horizon: int) -> str:
        return f"target_return_{horizon}_bps"

    def target_end_event(self, horizon: int) -> str:
        return f"target_end_event_{horizon}"

    def target_end_timestamp_us(self, horizon: int) -> str:
        return f"target_end_timestamp_{horizon}_us"

    @property
    def raw_feature_columns(self) -> list[str]:
        return (
            self.prices("bid")
            + self.prices("ask")
            + self.sizes("bid")
            + self.sizes("ask")
        )

    @property
    def engineered_feature_columns(self) -> list[str]:
        columns = [self.ofi]

        for k in self.depth_levels:
            if k <= self.depth:
                columns.extend(
                    (
                        self.depth_name("bid", k),
                        self.depth_name("ask", k),
                        self.imbalance(k),
                    )
                )

        columns.extend(
            (
                self.spread_bps,
                self.microprice_delta_bps,
                self.interarrival_ms,
                self.event_rate(self.event_rate_window),
            )
        )
        columns.extend(self.return_bps(n) for n in self.return_lags)
        columns.append(self.volatility_bps(self.volatility_window))
        columns.extend(self.ofi_sum(n) for n in self.ofi_windows)

        return columns

    @property
    def metadata_columns(self) -> list[str]:
        columns = [
            self.event_id,
            self.timestamp_us,
            self.local_timestamp_us,
            self.mid_price,
            self.spread,
            self.microprice_delta,
        ]

        for h in self.horizons:
            columns.extend((self.target_end_event(h), self.target_end_timestamp_us(h)))

        return columns

    @property
    def target_columns(self) -> list[str]:
        return [self.target_return_bps(h) for h in self.horizons]

    @property
    def attributes(self) -> dict:
        return {
            "depth": self.depth,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "date": self.date,
            "source": self.source,
            "depth_levels": self.depth_levels,
            "return_lags": self.return_lags,
            "ofi_windows": self.ofi_windows,
            "volatility_window": self.volatility_window,
            "event_rate_window": self.event_rate_window,
            "horizons": self.horizons,
            "tick_size": self.tick_size if self.tick_size is not None else 0.0,
            "raw_feature_names": self.raw_feature_columns,
            "engineered_feature_names": self.engineered_feature_columns,
        }
