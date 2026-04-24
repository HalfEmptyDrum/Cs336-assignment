
import torch
from einops import rearrange

import numpy as np

def data_loading(x, batch_size: int, context_length: int, device=None):
    # x is a numpy memmap (uint16). Don't move it to GPU.
    n = len(x)
    starts = np.random.randint(0, n - context_length, size=batch_size)
    idx = starts[:, None] + np.arange(context_length)[None, :]   # (B, T)

    # Index in numpy, cast to int64 (torch long), then send to GPU.
    x_batch = torch.from_numpy(x[idx].astype(np.int64)).to(device)
    y_batch = torch.from_numpy(x[idx + 1].astype(np.int64)).to(device)
    return x_batch, y_batch

    
    
    
if __name__ == "__main__":
    B = 4
    m = 3
    x = torch.tensor([1,2,3,4,5,6,7,8,9,10])
    