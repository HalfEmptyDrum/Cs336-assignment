import torch

from collections.abc import Callable, Iterable
from typing import Optional
import math



class AdamW(torch.optim.Optimizer):
    
    
    def __init__(self, params, lr, weight_decay, betas, eps=1e-8):
        if lr < 0:
            raise ValueError(f'Learning rate is below 0!: lr = {lr}')
        defaults = {'lr' : lr, 'betas' : betas, 'weight_decay' : weight_decay, 'eps' : eps}
        super().__init__(params, defaults)
        
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr'] # Get the learning rate.
            beta = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get('t', 1) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                
                lr_t = lr * math.sqrt(1-(beta[1]**t)) / (1-(beta[0]**t))
                p.data -= lr * weight_decay * p.data
                
                m = state.get('m', 0) # Get first momentum estimate
                v = state.get('v', 0) # Get second momentum estimate.
                
                m = beta[0] * m + (1-beta[0]) * grad
                v = beta[1] * v + (1-beta[1]) * grad**2
                
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v
        
        
        
        
        return loss
    
    
if __name__ == '__main__':
    for lr in [0.1]:
        torch.manual_seed(42)
        weights = torch.nn.Parameter(5 * torch.randn((10,10)))
        opt = AdamW([weights], lr=lr, beta = (0.9, 0.999), lamb = 0.8)

        for t in range(100):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.
        print()
