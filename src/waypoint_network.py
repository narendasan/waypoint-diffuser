import torch
import torch.nn as nn

class WaypointMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
