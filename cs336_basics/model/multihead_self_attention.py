import torch
import torch.nn as nn

from einops import einsum, rearrange

from cs336_basics.model.linear import Linear
from cs336_basics.model.rope import RotaryPositionalEmbedding
from cs336_basics.model.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, theta: float = None, max_seq_len: int = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        d_k = d_model // num_heads
        d_h = d_k

        self.d_k = d_k

        self.weight_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.weight_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.weight_v = Linear(d_model, d_model, device=device, dtype=dtype)

        self.weight_o = Linear(d_model, d_model, device=device, dtype=dtype)

        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=device)

        self.scaled_dot_attn = ScaledDotProductAttention()

        if max_seq_len is None:
            self.mask = None
        else:
            self.register_buffer('mask', torch.triu(torch.ones(max_seq_len, max_seq_len, device=device)).T.type(torch.bool))
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        q = self.weight_q(x)
        k = self.weight_k(x)
        
        v = self.weight_v(x)
        
        
        q = rearrange(q, "... seq_length (num_head d_k) -> ... num_head seq_length d_k", num_head=self.num_heads)
        k = rearrange(k, "... seq_length (num_head d_k) -> ... num_head seq_length d_k", num_head=self.num_heads)
        
        v = rearrange(v, "... seq_length (num_head d_h) -> ... num_head seq_length d_h", num_head=self.num_heads)
        
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        seq_length = q.shape[-2]
        
        if self.mask is not None:
            mask = self.mask[:seq_length, :seq_length]
        else:
            mask = torch.triu(torch.ones(seq_length, seq_length)).T.type(torch.bool)
        
        result = self.scaled_dot_attn(q, k, v, mask)
        
        result = rearrange(result, "... num_head seq_length d_h -> ... seq_length (num_head d_h)")
        
        result = self.weight_o(result)
        
        return result
        
        
        

    
    
        
    