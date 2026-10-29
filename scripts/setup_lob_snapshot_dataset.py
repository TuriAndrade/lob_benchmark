from dotenv import load_dotenv

load_dotenv()

import argparse
from src.data import lob_snapshot_providers


def main():
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
        description="Download and prepare order book snapshots",
        allow_abbrev=False,
    )
    parser.add_argument("urls", nargs="*", default=DEFAULT_URLS)
    parser.add_argument("--provider", type=str, choices=("tardis",), default="tardis")
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

    dataset = lob_snapshot_providers[args.provider](
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


if __name__ == "__main__":
    main()
