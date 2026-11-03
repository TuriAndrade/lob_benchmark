from dotenv import load_dotenv

load_dotenv()

import argparse
from datetime import date, timedelta

from src.data.bybit import BybitFeatures
from src.data.bybit import BybitProcessor


def main():
    start = date.today() - timedelta(days=10)
    DEFAULT_URLS = [
        f"https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/"
        f"{(start + timedelta(days=i)).isoformat()}_BTCUSDT_ob200.data.zip"
        for i in range(10)
    ]

    parser = argparse.ArgumentParser(
        description="Download and preprocess Bybit limit order books",
        allow_abbrev=False,
    )
    parser.add_argument("urls", nargs="*", default=DEFAULT_URLS)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--depth-levels", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--return-lags", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument("--windows", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    features = BybitFeatures(
        depth=args.depth,
        depth_levels=tuple(args.depth_levels),
        return_lags=tuple(args.return_lags),
        windows=tuple(args.windows),
        horizons=tuple(args.horizons),
    )

    processor = BybitProcessor(
        features=features,
        batch_size=args.batch_size,
    )

    for path in processor.prepare(args.urls, overwrite=args.overwrite):
        print(path)


if __name__ == "__main__":
    main()
