import torch

from cs336_basics.model.transformer_lm import TransformerLanguageModel
from cs336_basics.tokenizer.encoder import Tokenizer

import os

from einops import rearrange


from cs336_basics.training.checkpointing import load_checkpoint

from cs336_basics.model.softmax import Softmax


@torch.no_grad()
def generate():
    d_model = 512
    d_ff = 1344
    rope_theta = 10000
    context_length = 256
    vocab_size = 10000

   
    batch_size = 64
    training_steps = 5000
    val_interval = 500        # how often to run a quick validation pass
    val_batches = 50          # how many batches to average over

    device = 'cuda'
    num_heads = 16
    num_layers = 4
    
    special_tokens = ['<|endoftext|>']

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "latest.pt")

    vocab_path = "vocab/vocab_full.json"
    merges_path = "merges/merges_full.json"
    
    language_model = TransformerLanguageModel(
        vocab_size=vocab_size, context_length=context_length,
        num_layers=num_layers, d_model=d_model, d_ff=d_ff,
        num_heads=num_heads, rope_theta=rope_theta, device=device,
    )
    
    load_checkpoint("checkpoints/latest.pt", model=language_model, optimizer=None)
    
    language_model = torch.compile(language_model, backend="aot_eager")
    
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    
    start_tokens = tokenizer.encode("<|endoftext|>")
    
    x = torch.tensor([start_tokens], dtype=torch.long, device=device) # shape (1,1)
    
    max_tokens_generated = 250
    
    
    for _ in range(max_tokens_generated):
        logits = language_model(x) # (1, T, V)
        
        next_logits = logits[0, -1, :] # only the last position matters.
        
        probs = torch.softmax(next_logits, dim=-1)
        
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        x = torch.cat([x, next_token], dim=1)
        
        if x.shape[1] > language_model.context_length:
            x = x[:, -language_model.context_length:]
        
        
        
    print(tokenizer.decode(x))

    
    
    
    
    
if __name__== "__main__":
    generate()