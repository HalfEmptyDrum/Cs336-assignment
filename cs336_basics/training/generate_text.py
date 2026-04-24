import torch

from cs336_basics.model.transformer_lm import TransformerLanguageModel
from cs336_basics.tokenizer.encoder import Tokenizer
from cs336_basics.training.checkpointing import load_checkpoint

import os


@torch.no_grad()
def generate():
    # model / config
    d_model = 512
    d_ff = 1344
    rope_theta = 10000
    context_length = 256
    vocab_size = 10000

    device = 'cuda'
    num_heads = 16
    num_layers = 4

    special_tokens = ['<|endoftext|>']

    ckpt_path = os.path.join("checkpoints", "latest.pt")
    vocab_path = "vocab/vocab_full.json"
    merges_path = "merges/merges_full.json"

    # build model and load weights
    language_model = TransformerLanguageModel(
        vocab_size=vocab_size, context_length=context_length,
        num_layers=num_layers, d_model=d_model, d_ff=d_ff,
        num_heads=num_heads, rope_theta=rope_theta, device=device,
    )
    load_checkpoint(ckpt_path, model=language_model, optimizer=None)
    language_model.eval()

    # tokenizer
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    # prime with <|endoftext|> so the model starts a fresh story
    start_tokens = tokenizer.encode("<|endoftext|>")
    x = torch.tensor([start_tokens], dtype=torch.long, device=device)  # (1, T)

    max_tokens_generated = 250

    for _ in range(max_tokens_generated):
        # truncate to context window before the forward pass
        x_cond = x if x.shape[1] <= context_length else x[:, -context_length:]

        logits = language_model(x_cond)                    # (1, T, V)
        next_logits = logits[0, -1, :]                     # (V,)
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).view(1, 1)
        x = torch.cat([x, next_token], dim=1)

    generated_ids = x[0].tolist()
    print(tokenizer.decode(generated_ids))


if __name__ == "__main__":
    generate()