import polars as pl


class BookSnapshotFeatures:
    """Causal market microstructure features expressed as Polars expressions."""

    @staticmethod
    def ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
        return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)

    @staticmethod
    def mid_price(ask: pl.Expr, bid: pl.Expr) -> pl.Expr:
        return (ask + bid) / 2

    @staticmethod
    def spread(ask: pl.Expr, bid: pl.Expr) -> pl.Expr:
        return ask - bid

    @staticmethod
    def depth(sizes: list[pl.Expr]) -> pl.Expr:
        return pl.sum_horizontal([x.fill_null(0) for x in sizes])

    @staticmethod
    def imbalance(bids: list[pl.Expr], asks: list[pl.Expr]) -> pl.Expr:
        bid, ask = BookSnapshotFeatures.depth(bids), BookSnapshotFeatures.depth(asks)
        return BookSnapshotFeatures.ratio(bid - ask, bid + ask)

    @staticmethod
    def microprice(
        ask: pl.Expr, bid: pl.Expr, ask_size: pl.Expr, bid_size: pl.Expr
    ) -> pl.Expr:
        return BookSnapshotFeatures.ratio(
            ask * bid_size + bid * ask_size, bid_size + ask_size
        )

    @staticmethod
    def ofi(
        bid: pl.Expr, ask: pl.Expr, bid_size: pl.Expr, ask_size: pl.Expr
    ) -> pl.Expr:
        prev_bid, prev_ask = bid.shift(1), ask.shift(1)
        prev_bid_size, prev_ask_size = bid_size.shift(1), ask_size.shift(1)

        buy = (
            pl.when(bid > prev_bid)
            .then(bid_size)
            .when(bid < prev_bid)
            .then(-prev_bid_size)
            .otherwise(bid_size - prev_bid_size)
        )
        sell = (
            pl.when(ask < prev_ask)
            .then(ask_size)
            .when(ask > prev_ask)
            .then(-prev_ask_size)
            .otherwise(ask_size - prev_ask_size)
        )
        return buy - sell

    @staticmethod
    def return_bps(mid: pl.Expr, lag: int = 1) -> pl.Expr:
        return (mid / mid.shift(lag) - 1) * 10_000

    @staticmethod
    def volatility_bps(mid: pl.Expr, window: int) -> pl.Expr:
        return (mid.log() - mid.shift(1).log()).rolling_std(window_size=window) * 10_000

    @staticmethod
    def event_rate(timestamp_us: pl.Expr, window: int) -> pl.Expr:
        return BookSnapshotFeatures.ratio(
            pl.lit(window * 1_000_000), timestamp_us - timestamp_us.shift(window)
        )
