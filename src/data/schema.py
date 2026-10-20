from dataclasses import dataclass
from typing import ClassVar

import polars as pl


@dataclass(frozen=True)
class BookSnapshotSchema:
    depth: int

    event_id: ClassVar[str] = "event_id"
    exchange: ClassVar[str] = "exchange"
    symbol: ClassVar[str] = "symbol"
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

    DEPTH_LEVELS: ClassVar[tuple[int, ...]] = (1, 3, 5, 10, 25)
    RETURN_LAGS: ClassVar[tuple[int, ...]] = (1, 10, 50)
    OFI_WINDOWS: ClassVar[tuple[int, ...]] = (10, 50, 100)
    VOLATILITY_WINDOW: ClassVar[int] = 50
    EVENT_RATE_WINDOW: ClassVar[int] = 50

    def __post_init__(self):
        if self.depth < 1:
            raise ValueError("depth must be positive")

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

    def target_move_ticks(self, horizon: int) -> str:
        return f"target_move_{horizon}_ticks"

    def target_end_event(self, horizon: int) -> str:
        return f"target_end_event_{horizon}"

    def target_end_timestamp_us(self, horizon: int) -> str:
        return f"target_end_timestamp_{horizon}_us"

    @property
    def depth_levels(self) -> tuple[int, ...]:
        return tuple(k for k in self.DEPTH_LEVELS if k <= self.depth)

    @property
    def dtypes(self) -> dict[str, pl.DataType]:
        columns = {
            self.exchange: pl.String,
            self.symbol: pl.String,
            self.timestamp_us: pl.Int64,
            self.local_timestamp_us: pl.Int64,
        }
        for side in ("bid", "ask"):
            columns.update(
                dict.fromkeys(self.prices(side) + self.sizes(side), pl.Float64)
            )
        return columns

    @property
    def feature_columns(self) -> list[str]:
        columns = [self.mid_price, self.spread, self.ofi, self.microprice_delta]

        for k in self.depth_levels:
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
                self.event_rate(self.EVENT_RATE_WINDOW),
            )
        )
        columns.extend(self.return_bps(n) for n in self.RETURN_LAGS)
        columns.append(self.volatility_bps(self.VOLATILITY_WINDOW))
        columns.extend(self.ofi_sum(n) for n in self.OFI_WINDOWS)

        return columns

    def target_columns(
        self,
        horizons: tuple[int, ...],
        include_ticks: bool = False,
    ) -> list[str]:
        columns = []

        for h in horizons:
            columns.extend(
                (
                    self.target_return_bps(h),
                    self.target_end_event(h),
                    self.target_end_timestamp_us(h),
                )
            )
            if include_ticks:
                columns.append(self.target_move_ticks(h))

        return columns
