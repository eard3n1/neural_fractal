import torch
from torch.utils.data import Dataset

class FractalDataset(Dataset):
    def __init__(self, num_samples=100000, scale=2.0):
        self.num_samples = num_samples
        self.scale = scale
        self.data = self.generate_points()

    def generate_points(self):
        return (torch.rand(self.num_samples, 2) - 0.5) * 2 * self.scale

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]