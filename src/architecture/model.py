import torch.nn as nn
from .fourier import FourierFeatures

class FractalNet(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, num_frequencies=6):
        super().__init__()

        self.encoder = FourierFeatures(num_frequencies=num_frequencies, scale=1.0)
        input_dim = 2 + 4 * num_frequencies

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 3))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return self.model(x)