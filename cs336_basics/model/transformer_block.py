import torch
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.model.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.model.swiglu import PositionWiseFeedForward
from cs336_basics.model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff


        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        
        
    def forward(self, x: torch.Tensor):
        # x : (batch_size, seq_len, d_model)
        seq_len = x.shape[-2]
        positions = torch.arange(seq_len, device=x.device)
        x = self.attn(self.ln1(x), positions) + x
        
        x = self.ffn(self.ln2(x)) + x
        
        return x
        
        
        