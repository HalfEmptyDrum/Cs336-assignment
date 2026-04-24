import torch

from cs336_basics.model.transformer_lm import TransformerLanguageModel


from cs336_basics.training.checkpointing import load_checkpoint

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ckpt = torch.load(src)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    iteration = ckpt['iteration']
    return iteration


def generate():
    d_model = 512
    d_ff = 1344
    rope_theta = 10000
    context_length = 256
    vocab_size = 10000

    device = 'cuda'
    num_heads = 16
    num_layers = 4
    
    language_model = TransformerLanguageModel(
        vocab_size=vocab_size, context_length=context_length,
        num_layers=num_layers, d_model=d_model, d_ff=d_ff,
        num_heads=num_heads, rope_theta=rope_theta, device=device,
    )
    language_model = torch.compile(language_model, backend="aot_eager")
    
    

    
    
    
    
    
if __name__== "__main__":
    generate()