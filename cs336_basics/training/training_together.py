import torch
from cs336_basics.tokenizer.tokenizer import train_bpe
from cs336_basics.tokenizer.encoder import Tokenizer
from cs336_basics.tokenizer.save_bpe import save_bpe
from cs336_basics.model.transformer_lm import TransformerLanguageModel
from cs336_basics.training.adamw import AdamW
from cs336_basics.training.data_loading import data_loading
from cs336_basics.training.cross_entropy import cross_entropy
from cs336_basics.training.learning_rate_schedule import lr_cosine_schedule
from cs336_basics.training.checkpointing import save_checkpoint

from einops import rearrange
import numpy as np
from pathlib import Path
import os


def tokenize_to_bin(tokenizer, text_filepath: str, bin_path: Path):
    """Tokenize a text file to a uint16 binary on disk. Skips if bin already exists."""
    if bin_path.exists():
        return
    print(f"tokenizing {text_filepath} -> {bin_path} ...")
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with open(text_filepath, "r", encoding="utf-8") as f, open(bin_path, "wb") as out:
        buf = []
        for tid in tokenizer.encode_iterable(f):
            buf.append(tid)
            if len(buf) >= 1_000_000:
                np.asarray(buf, dtype=np.uint16).tofile(out)
                buf.clear()
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(out)
    print(f"done tokenizing {bin_path}.")


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


def train():
    print("train()")

    d_model = 512
    d_ff = 1344
    rope_theta = 10000
    context_length = 256
    vocab_size = 10000

    device = 'cuda'
    num_heads = 16
    num_layers = 4
    batch_size = 256
    training_steps = 5000
    val_interval = 500        # how often to run a quick validation pass
    val_batches = 50          # how many batches to average over
    special_tokens = ['<|endoftext|>']

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "latest.pt")

    vocab_path = "vocab/vocab_full.json"
    merges_path = "merges/merges_full.json"
    train_text_path = "TinyStoriesV2-GPT4-train.txt"
    val_text_path = "TinyStoriesV2-GPT4-valid.txt"
    train_bin_path = Path("tokens/train.bin")
    val_bin_path = Path("tokens/val.bin")

    print("creating the tokenizer...")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    # tokenize both files (skipped if already on disk)
    tokenize_to_bin(tokenizer, train_text_path, train_bin_path)
    tokenize_to_bin(tokenizer, val_text_path, val_bin_path)

    train_ids = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
    val_ids = np.memmap(val_bin_path, dtype=np.uint16, mode="r")
    print(f"loaded {len(train_ids):,} train tokens, {len(val_ids):,} val tokens")

    language_model = TransformerLanguageModel(
        vocab_size=vocab_size, context_length=context_length,
        num_layers=num_layers, d_model=d_model, d_ff=d_ff,
        num_heads=num_heads, rope_theta=rope_theta, device=device,
    )
    language_model = torch.compile(language_model, backend="aot_eager")

    optimizer = AdamW(language_model.parameters(), lr=0.01,
                      weight_decay=0.9, betas=(0.99, 0.999))

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

        if it % 50 == 0:
            print(f"iteration {it}, train loss = {loss.item():.4f}")

        if it % val_interval == 0 and it > 0:
            val_loss = evaluate(
                language_model, val_ids, batch_size, context_length,
                device, num_batches=val_batches,
            )
            print(f"  [val @ step {it}] loss = {val_loss:.4f}")

        if it % 1000 == 0:
            save_checkpoint(language_model, optimizer, it, ckpt_path)

    print(f"final training loss = {loss.item():.4f}")

    # final, more thorough validation pass
    final_val_loss = evaluate(
        language_model, val_ids, batch_size, context_length,
        device, num_batches=200,
    )
    print(f"final validation loss = {final_val_loss:.4f}")

    save_checkpoint(language_model, optimizer, training_steps, ckpt_path)


if __name__ == "__main__":
    train()