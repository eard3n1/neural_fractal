import math
import torch
import torch.nn as nn


class FractalNet(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_frequencies: int) -> None:
        super().__init__()

        self.num_frequencies = num_frequencies
        self.scale = 1.0

        input_dim = 2 + 4 * num_frequencies

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 3))
        self.model = nn.Sequential(*layers)

    def fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for i in range(self.num_frequencies):
            freq = 2.0 ** i * self.scale
            features.append(torch.sin(2 * math.pi * freq * x))
            features.append(torch.cos(2 * math.pi * freq * x))
        return torch.cat(features, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fourier_features(x)
        return self.model(x)