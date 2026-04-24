import torch
from torch import optim

from collections.abc import Callable, Iterable
from typing import Optional
import math


class SGD(optim.Optimizer):
    
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f'Invalid learning rate: {lr}')
        defaults = {"lr" : lr}
        super().__init__(params, defaults)
        

        
    
    # should make one update of the parameters. During the training loop this will be called
    # after the backward pass, so we have access to the gradients on the last batch.
    # This method should iterate through each parameter p and modify them in place, i.e.
    # setting p.data, which holds the tensor associated with that parameter based on the gradient
    # p.grad (if it exists), the tensor representing the gradient of the loss w.r.t. that parameter.
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr'] # Get the learning rate.
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                t = state.get('t', 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update in-place.
                state['t'] = t + 1 # Increment iteration number.
                
        return loss
        
        
if __name__ == '__main__':
    for lr in [1e1, 1e2, 1e3]:
        torch.manual_seed(42)
        weights = torch.nn.Parameter(5 * torch.randn((10,10)))
        opt = SGD([weights], lr=lr)

        for t in range(10):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.
        print()
