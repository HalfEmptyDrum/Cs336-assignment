import torch
import os
import typing

def save_checkpoint(model, optimizer, iteration, path):
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }, path)
    
def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ckpt = torch.load(src)
    if model is not None:
        model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    iteration = ckpt['iteration']
    return iteration
    




