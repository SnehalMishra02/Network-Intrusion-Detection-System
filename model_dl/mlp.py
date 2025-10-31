# model_dl/mlp.py
import torch
import torch.nn as nn

class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),      nn.BatchNorm1d(32),  nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(32, n_classes)
        )
    def forward(self, x):
        return self.net(x)
