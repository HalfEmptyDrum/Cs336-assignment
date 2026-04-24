import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=(2 / (in_features + out_features)) ** 0.5)

    def forward(self, x: torch.Tensor):
        return einsum(self.weight, x, "i j, ... j -> ... i")