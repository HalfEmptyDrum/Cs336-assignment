import torch

from cs336_basics.model.transformer_lm import TransformerLanguageModel
from cs336_basics.tokenizer.encoder import Tokenizer

import os



from cs336_basics.training.checkpointing import load_checkpoint



def generate():
    d_model = 512
    d_ff = 1344
    rope_theta = 10000
    context_length = 256
    vocab_size = 10000

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
    

    
    
    
    
    
if __name__== "__main__":
    generate()