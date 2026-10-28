"""Linear model tuning configuration."""

from models import LinearModel

from .common import arguments, grid_search, tuning_kwargs
from .tabular_tuning import TabularTuning


def main():
    args = arguments(normalize_default=True)

    TabularTuning(
        model_cls=LinearModel,
        model_params=list(
            grid_search(
                {
                    "regularization": [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                }
            )
        ),
        **tuning_kwargs(args, "linear"),
        data_mode=args.data_mode,
    ).run()


if __name__ == "__main__":
    main()
