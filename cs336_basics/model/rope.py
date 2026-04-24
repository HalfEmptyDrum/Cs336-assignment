import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.device = device
        
        k = torch.arange(1, d_k // 2 +1, device=device)
        angles = 1 / theta ** ((2*k - 2) / d_k)
        
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        
        angles = einsum(positions, angles, 'pos, theta -> pos theta')
        
        
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())
        
        

    def __call__(self, x: torch.Tensor, token_positions: torch.Tensor):
        return self.forward(x, token_positions)
    

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        
        x1 = x[..., 0::2] # (B, ..., seq_len, d_k//2)
        x2 = x[..., 1::2] # (B, ..., seq_len, d_k//2)
        
        cos = self.cos_cached[token_positions] # (..., seq_len, d_k//2
        sin = self.sin_cached[token_positions] # (..., seq_len, d_k//2)
        
        rotated_1 = cos * x1 - sin * x2
        rotated_2 = sin * x1 + cos * x2
        
        out = torch.stack((rotated_1, rotated_2), dim=-1)
        return out.flatten(-2)
    
        

