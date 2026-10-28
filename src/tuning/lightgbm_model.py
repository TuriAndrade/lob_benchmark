"""LightGBM tuning configuration."""

from models import LightGBMModel

from .common import arguments, grid_search, tuning_kwargs
from .tabular_tuning import TabularTuning


def main():
    args = arguments()

    TabularTuning(
        model_cls=LightGBMModel,
        model_params=list(
            grid_search(
                {
                    "learning_rate": [0.01, 0.03, 0.1],
                    "num_leaves": [15, 31],
                    "min_child_samples": [20, 100],
                }
            )
        ),
        **tuning_kwargs(args, "lightgbm"),
        data_mode=args.data_mode,
    ).run()


if __name__ == "__main__":
    main()
