import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from einops import rearrange

from cs336_basics.tokenizer.encoder import Tokenizer
from cs336_basics.model.transformer_lm import TransformerLanguageModel
from cs336_basics.training.adamw import AdamW
from cs336_basics.training.data_loading import data_loading
from cs336_basics.training.cross_entropy import cross_entropy
from cs336_basics.training.checkpointing import save_checkpoint

from cs336_basics.training.preprocessing import tokenize_to_bin


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, token_ids, batch_size, context_length, device, num_batches=50):
    """Compute average cross-entropy loss over `num_batches` random batches."""
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = data_loading(token_ids, batch_size, context_length, device)
        y_pred = model(x)
        y_pred = rearrange(y_pred, "B T V -> (B T) V")
        y = rearrange(y, "B T -> (B T)")
        loss = cross_entropy(y_pred, y)
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def train(cfg: dict):
    print("train()")

    paths = cfg["paths"]
    tok_cfg = cfg["tokenizer"]
    model_cfg = cfg["model"]
    opt_cfg = cfg["optimizer"]
    tr_cfg = cfg["training"]

    device = tr_cfg["device"]
    batch_size = tr_cfg["batch_size"]
    context_length = model_cfg["context_length"]

    os.makedirs(paths["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(paths["checkpoint_dir"], paths["checkpoint_name"])

    print("creating the tokenizer...")
    tokenizer = Tokenizer.from_files(
        paths["vocab"], paths["merges"], tok_cfg["special_tokens"]
    )

    train_bin_path = Path(paths["train_bin"])
    val_bin_path = Path(paths["val_bin"])
    tokenize_to_bin(tokenizer, paths["train_text"], train_bin_path)
    tokenize_to_bin(tokenizer, paths["val_text"], val_bin_path)

    train_ids = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
    val_ids = np.memmap(val_bin_path, dtype=np.uint16, mode="r")
    print(f"loaded {len(train_ids):,} train tokens, {len(val_ids):,} val tokens")

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
    language_model = torch.compile(language_model, backend="aot_eager")

    optimizer = AdamW(
        language_model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
    )

    training_steps = tr_cfg["training_steps"]
    val_interval = tr_cfg["val_interval"]
    val_batches = tr_cfg["val_batches"]
    ckpt_interval = tr_cfg["ckpt_interval"]

    print("starting training...")
    for it in range(training_steps):
        x, y = data_loading(train_ids, batch_size, context_length, device)

        optimizer.zero_grad()
        y_pred = language_model(x)
        y_pred = rearrange(y_pred, "B T V -> (B T) V")
        y = rearrange(y, "B T -> (B T)")
        loss = cross_entropy(y_pred, y)
        loss.backward()
        optimizer.step()


        if it % 100 == 0:
            print(f"iteration {it}, train loss = {loss.item():.4f}")

        if it % val_interval == 0 and it > 0:
            val_loss = evaluate(
                language_model, val_ids, batch_size, context_length,
                device, num_batches=val_batches,
            )
            print(f"  [val @ step {it}] loss = {val_loss:.4f}")

        if it % ckpt_interval == 0:
            save_checkpoint(language_model, optimizer, it, ckpt_path)

    print(f"final training loss = {loss.item():.4f}")

    final_val_loss = evaluate(
        language_model, val_ids, batch_size, context_length,
        device, num_batches=tr_cfg["final_val_batches"],
    )
    print(f"final validation loss = {final_val_loss:.4f}")

    save_checkpoint(language_model, optimizer, training_steps, ckpt_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="cs336_basics/training/configs/default.yaml",
                   help="Path to YAML config file.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)