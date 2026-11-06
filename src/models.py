import torch
import torch.nn as nn


class Pinn2D(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 3, hidden_dim: int = 64, num_layers: int = 6) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)