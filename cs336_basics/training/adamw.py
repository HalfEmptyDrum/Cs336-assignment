import torch

from collections.abc import Callable, Iterable
from typing import Optional
import math



import torch
import math
from collections.abc import Callable
from typing import Optional


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps=1e-8):
        if lr < 0:
            raise ValueError(f'Learning rate is below 0!: lr = {lr}')
        defaults = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay, 'eps': eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                m, v = state['m'], state['v']
                state['t'] += 1
                t = state['t']

                # m = beta1*m + (1-beta1)*grad
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v = beta2*v + (1-beta2)*grad^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # decoupled weight decay
                p.data.mul_(1 - lr * weight_decay)

                # p -= lr_t * m / (sqrt(v) + eps)
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-lr_t)

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
