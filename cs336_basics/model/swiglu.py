import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, repeat
from cs336_basics.model.linear import Linear
from cs336_basics.model.silu import SiLU


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x):
        a1 = SiLU().forward(self.w1(x))
        a2 = self.w3(x)
        a3 = einsum(a1, a2, "... d_model, ... d_model -> ... d_model")
        return self.w2(a3)
        
    
   