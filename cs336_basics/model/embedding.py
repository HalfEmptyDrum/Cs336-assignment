import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.E = nn.Parameter(torch.randn((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.E)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.E[token_ids]