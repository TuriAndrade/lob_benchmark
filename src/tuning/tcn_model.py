"""TCN tuning configuration."""

from ..models import TCNModel

from .common import arguments, grid_search, tuning_kwargs
from .sequence_tuning import SequenceTuning


def main():
    args = arguments(sequence=True)

    SequenceTuning(
        model_cls=TCNModel,
        model_params=list(
            grid_search(
                {
                    "channels": [32, 64],
                    "num_layers": [3, 4],
                }
            )
        ),
        trainer_params=[
            dict(epochs=2, patience=5, weight_decay=1e-4, **params)
            for params in grid_search({"lr": [1e-4, 1e-3]})
        ],
        **tuning_kwargs(args, "tcn"),
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        include_engineered=args.include_engineered,
        shuffle=args.shuffle,
        drop_last=args.drop_last,
        mode=args.mode,
        device=args.device,
        precision=args.precision,
    ).run()


if __name__ == "__main__":
    main()
