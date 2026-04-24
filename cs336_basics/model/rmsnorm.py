import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, repeat

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gain = nn.Parameter(torch.randn(d_model, device=device, dtype=dtype))
        
        
    def _RMS(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = einsum(x, x, "... i, ... i -> ...") # (batch_size, sequence_length)
        return torch.sqrt((self.d_model ** -1) * x_norm + self.eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        
        x = x.to(torch.float32)
        
        rms_norm_inv = self._RMS(x) ** -1
        
        result = einsum(rms_norm_inv, self.gain, "batch seq_len, d_model -> batch seq_len d_model")
        
        result = result * x

        return result.to(in_dtype)

        
        
        

    
    
    
# if __name__ == "__main__":
#     norm = RMSNorm(3, eps=1e-2)
#     x = torch.randn(64, 4, 3)
#     print(norm(x).shape)