import torch
import torch.nn as nn
import math

class FourierFeatures(nn.Module):
    def __init__(self, num_frequencies=6, scale=1.0):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.scale = scale

    def forward(self, x):
        features = [x]
        for i in range(self.num_frequencies):
            freq = 2.0 ** i * self.scale
            features.append(torch.sin(2 * math.pi * freq * x))
            features.append(torch.cos(2 * math.pi * freq * x))
        return torch.cat(features, dim =- 1)