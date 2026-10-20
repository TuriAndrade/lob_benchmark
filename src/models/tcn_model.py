import torch
import torch.nn.functional as F
from torch import nn

from .sequence_model import SequenceModel


class _TemporalBlock(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, dilation, dropout):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.skip = (
            nn.Conv1d(in_channels, channels, kernel_size=1)
            if in_channels != channels
            else nn.Identity()
        )

        self.padding = (kernel_size - 1) * dilation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.skip(x)

        x = F.pad(x, (self.padding, 0))
        x = self.dropout(F.relu(self.conv1(x)))

        x = F.pad(x, (self.padding, 0))
        x = self.dropout(F.relu(self.conv2(x)))

        return F.relu(x + residual)


class TCNModel(SequenceModel):
    def __init__(
        self,
        input_dim,
        task="regression",
        channels=64,
        num_layers=4,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()

        if task not in ("regression", "classification"):
            raise ValueError(f"Unknown task: {task}")
        if min(input_dim, channels, num_layers, kernel_size) < 1:
            raise ValueError("Model dimensions and kernel_size must be positive")

        self.config = {
            "input_dim": input_dim,
            "task": task,
            "channels": channels,
            "num_layers": num_layers,
            "kernel_size": kernel_size,
            "dropout": dropout,
        }

        self.task = task
        self.network = nn.Sequential(
            *[
                _TemporalBlock(
                    in_channels=input_dim if i == 0 else channels,
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.head = nn.Linear(channels, 1 if task == "regression" else 3)

    def forward(self, x):
        x = self.network(x.transpose(1, 2))
        out = self.head(x[:, :, -1])
        return out.squeeze(-1) if self.task == "regression" else out

    def save(self, save_path):
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            save_path,
        )

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = TCNModel(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
