from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn


class SequenceModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @torch.inference_mode()
    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device
            x = torch.as_tensor(X, dtype=torch.float32, device=device)
            output = self(x)

            if self.task == "classification":
                output = output.softmax(dim=-1)
            else:
                output = output.reshape(-1)

            return output.cpu().numpy()
        finally:
            self.train(was_training)

    @abstractmethod
    def save(self, save_path) -> None: ...

    @staticmethod
    @abstractmethod
    def load(path) -> "SequenceModel": ...
