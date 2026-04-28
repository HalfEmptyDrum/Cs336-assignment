import torch
import torch.nn as nn

from cs336_basics.model.embedding import Embedding
from cs336_basics.model.transformer_block import TransformerBlock
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.linear import Linear
from cs336_basics.model.softmax import Softmax


class TransformerLanguageModel(nn.Module):
    
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        
        max_seq_len = context_length
        
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope_theta, max_seq_len, device=device, dtype=dtype) for _ in range(num_layers)]
        )
        
        # Learnable scalar mixing weights for the embedding skip-connection at each layer.
        # Initialized to 0 so the model starts behaving exactly like the original.
        self.embed_skip_weights = nn.Parameter(torch.zeros(num_layers, device=device, dtype=dtype))
        
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.softmax = Softmax()
        
    def forward(self, x):
        # x : (batch_size, sequence_length)
        
        embeds = self.token_embeddings(x)  # (batch, seq, d_model)
        x = embeds
        
        for i, layer in enumerate(self.layers):
            x = layer(x + self.embed_skip_weights[i] * embeds)
        
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x