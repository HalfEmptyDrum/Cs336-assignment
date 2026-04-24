import torch
import torch.nn as nn


def cross_entropy(inputs, targets):
    # inputs: (batch_size, vocab_size)
    # targets: (batch_size, )
    """
    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.
    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    batch_size, vocab_size = inputs.shape
    # softmax over last axis:
    # -log (softmax(o_i)[x_{i+1}])
    # o_i[x_{i+1}] - log(\sum_{a=1}^{vocab_size} exp(o_i[a]))
    # inputs = inputs - inputs.max(dim=1, keepdim=True).values
    
    inputs = inputs - inputs.max(dim=1, keepdim=True).values

    inputs_exp = inputs.exp()

    
    o_i = inputs[torch.arange(batch_size), targets]

    return (-o_i+torch.log(inputs_exp.sum(dim=1))).mean()
    
    
if __name__ == '__main__':
    inputs = torch.tensor([[1.5, -1.2, 3.4], [-4, 3, 2]])
    targets = torch.tensor([0, 1])
    print(cross_entropy(inputs, targets))
    
   
