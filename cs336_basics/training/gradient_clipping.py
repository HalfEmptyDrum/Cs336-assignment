import torch


def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    # Filter out frozen parameters (p.grad is None)
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    # Global L2 norm across all gradients: sqrt(sum of squared norms)
    total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))

    # Only scale down, never up
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)       # in-place, modifies p.grad directly
    
    
    
    
if __name__ == '__main__':

    torch.manual_seed(42)
    weights = torch.nn.Parameter(torch.randn((10,10)))
    
    print(weights)
    print()
    print(gradient_clipping(weights, 2))
        