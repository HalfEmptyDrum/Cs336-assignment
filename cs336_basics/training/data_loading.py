
import torch
from einops import rearrange

def data_loading(x: torch.Tensor, batch_size: int, context_length: int, device = None):
    
    x = torch.tensor(x, device=device)
    
    starting_indices = torch.randint(low=0, high=x.shape[0]-context_length, size=(batch_size,), device=device)
    
    idx_train = rearrange(starting_indices, "n -> n 1") + torch.arange(context_length, device=device)

    return x[idx_train], x[idx_train + 1]

    
    
    
if __name__ == "__main__":
    B = 4
    m = 3
    x = torch.tensor([1,2,3,4,5,6,7,8,9,10])
    