import argparse
import os

import torch
import yaml

from cs336_basics.model.transformer_lm import TransformerLanguageModel
from cs336_basics.tokenizer.encoder import Tokenizer
from cs336_basics.training.checkpointing import load_checkpoint


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def generate(cfg: dict):
    paths = cfg["paths"]
    tok_cfg = cfg["tokenizer"]
    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]

    device = gen_cfg["device"]
    context_length = model_cfg["context_length"]

    ckpt_path = os.path.join(paths["checkpoint_dir"], paths["checkpoint_name"])
    
    temperature = gen_cfg["temperature"]

    # build model and load weights
    language_model = TransformerLanguageModel(
        vocab_size=tok_cfg["vocab_size"],
        context_length=context_length,
        num_layers=model_cfg["num_layers"],
        d_model=model_cfg["d_model"],
        d_ff=model_cfg["d_ff"],
        num_heads=model_cfg["num_heads"],
        rope_theta=model_cfg["rope_theta"],
        device=device,
    )
    load_checkpoint(ckpt_path, model=language_model, optimizer=None)
    language_model.eval()

    # tokenizer
    tokenizer = Tokenizer.from_files(
        paths["vocab"], paths["merges"], tok_cfg["special_tokens"]
    )

    # prime with the configured prompt (e.g. "<|endoftext|>")
    start_tokens = tokenizer.encode(gen_cfg["prompt"])
    x = torch.tensor([start_tokens], dtype=torch.long, device=device)  # (1, T)

    max_new_tokens = gen_cfg["max_new_tokens"]
    
    end_token = tokenizer.encode(gen_cfg['end_token'])

    for _ in range(max_new_tokens):
        x_cond = x if x.shape[1] <= context_length else x[:, -context_length:]

        logits = language_model(x_cond)                # (1, T, V)
        next_logits = logits[0, -1, :]                 # (V,)

        if temperature == 0:
            next_token = torch.argmax(next_logits, dim=-1)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).view(1, 1)
        
        x = torch.cat([x, next_token], dim=1)
        
        if next_token == end_token:
            break

    generated_ids = x[0].tolist()
    print(tokenizer.decode(generated_ids))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="cs336_basics/training/configs/default.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    generate(cfg)