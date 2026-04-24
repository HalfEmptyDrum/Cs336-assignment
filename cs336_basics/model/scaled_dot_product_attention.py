import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einsum

from cs336_basics.model.softmax import Softmax

"""
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """

class ScaledDotProductAttention(nn.Module):
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        d = torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32))
        scores = (d**-1) * einsum(Q, K, "... i j, ... k j -> ... i k")
        scores = scores.masked_fill(~mask, -torch.inf)
        weights = F.softmax(scores, dim=-1)
        result = einsum(weights, V, "... n m, ... m d_v -> ... n d_v")
        return result