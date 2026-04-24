import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmax(nn.Module):
    def forward(self, v: torch.Tensor, dim: int):
        v = v - v.max(dim=dim, keepdim=True).values
        v_exp = v.exp()
        return v_exp / v_exp.sum(dim=dim, keepdim=True)
