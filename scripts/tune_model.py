from dotenv import load_dotenv

load_dotenv()

import argparse
import sys

from src.tuning import models


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=models)

    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0], *remaining]
    models[args.model]()


if __name__ == "__main__":
    main()
