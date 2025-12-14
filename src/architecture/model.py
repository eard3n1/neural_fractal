import torch.nn as nn

class FractalNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=512, num_layers=4):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.skip = nn.Linear(2, hidden_dim)
        layers.append(nn.Linear(hidden_dim, 3))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        skip_out = self.skip(x)
        hidden_out = self.model[:-1](x)
        hidden_out = hidden_out + skip_out
        return self.model[-1](hidden_out)