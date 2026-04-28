import argparse
import math
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


from cs336_basics.training.learning_rate_schedule import lr_cosine_schedule

from cs336_basics.training.preprocessing import tokenize_to_bin

from cs336_basics.tokenizer.tokenizer import train_bpe
from cs336_basics.tokenizer.save_bpe import save_bpe

import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def train(cfg: dict,
          lr_override: float | None = None,
          steps_override: int | None = None,
          sweep_subdir: str | None = None,
          experiment: str | None = None) -> Path:
    """Train one model. Returns the path to the run log CSV.

    Args:
        cfg: parsed YAML config.
        lr_override: if set, overrides cfg.optimizer.lr (used by sweeps).
        steps_override: if set, overrides cfg.training.training_steps.
        sweep_subdir: if set, log file goes under <log_dir>/<sweep_subdir>/
            so all runs in a sweep share a directory. Ignored when
            `experiment` is also set — the experiment folder is used instead.
        experiment: if set, all logs go under <log_dir>/<experiment>/
            (created if needed). Filenames already include lr+timestamp so
            multiple runs/sweeps under the same name won't collide.
    """
    print("train()")

    paths = cfg["paths"]
    tok_cfg = cfg["tokenizer"]
    model_cfg = cfg["model"]
    opt_cfg = cfg["optimizer"]
    tr_cfg = cfg["training"]
    
    schedule_cfg = cfg["scheduler"]

    device = tr_cfg["device"]
    batch_size = tr_cfg["batch_size"]
    context_length = model_cfg["context_length"]

    lr = lr_override if lr_override is not None else opt_cfg["lr"]
    training_steps = (steps_override if steps_override is not None
                      else tr_cfg["training_steps"])

    os.makedirs(paths["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(paths["checkpoint_dir"], paths["checkpoint_name"])

    # --- set up the run log file ---
    log_dir = Path(paths.get("log_dir", "logs"))
    if experiment is not None:
        log_dir = log_dir / experiment
    elif sweep_subdir is not None:
        log_dir = log_dir / sweep_subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_lr{lr:.2e}_{ts}"
    log_path = log_dir / f"{run_name}.csv"
    log_file = open(log_path, "w", newline="", buffering=1)  # line-buffered
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "split", "loss", "lr"])
    print(f"logging to {log_path}")

    print("creating the tokenizer...")

    vocab_path = Path(paths["vocab"])
    merges_path = Path(paths["merges"])

    if not vocab_path.exists() or not merges_path.exists():
        print("vocab/merges not found, training tokenizer from scratch...")
        vocab, merges = train_bpe(
            input_path=paths["train_text"],
            vocab_size=tok_cfg["vocab_size"],
            special_tokens=tok_cfg["special_tokens"],
        )
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        merges_path.parent.mkdir(parents=True, exist_ok=True)
        save_bpe(vocab, merges, vocab_path, merges_path)
        print("done training tokenizer!")

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
        lr=lr,
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
    )

    val_interval = tr_cfg["val_interval"]
    val_batches = tr_cfg["val_batches"]
    ckpt_interval = tr_cfg["ckpt_interval"]
    
    a_max = schedule_cfg["a_max"]
    a_min = schedule_cfg["a_min"]
    warmup_steps = schedule_cfg["warmup_steps"]

    print(f"starting training (lr={lr:.2e}, steps={training_steps})...")
    diverged = False
    loss_val = float("nan")
    try:
        for it in range(training_steps):
            x, y = data_loading(train_ids, batch_size, context_length, device)

            lr = lr_cosine_schedule(it, a_max=a_max, a_min=a_min, T_w=warmup_steps, T_c=training_steps)
            
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            optimizer.zero_grad()
            y_pred = language_model(x)
            y_pred = rearrange(y_pred, "B T V -> (B T) V")
            y = rearrange(y, "B T -> (B T)")

            loss = cross_entropy(y_pred, y)
            loss_val = loss.item()

            # bail out if the run diverged — avoids polluting plots with inf/NaN
            if not math.isfinite(loss_val):
                print(f"  diverged at step {it} (loss={loss_val}); stopping run.")
                log_writer.writerow([it, "train", f"{loss_val}", lr])
                diverged = True
                break

            loss.backward()
            optimizer.step()

            # log every training step
            log_writer.writerow([it, "train", f"{loss_val:.6f}", lr])

            if it % 10 == 0:
                print(f"iteration {it}, train loss = {loss_val:.4f}")

            if it % val_interval == 0 and it > 0:
                val_loss = evaluate(
                    language_model, val_ids, batch_size, context_length,
                    device, num_batches=val_batches,
                )
                print(f"  [val @ step {it}] loss = {val_loss:.4f}")
                log_writer.writerow([it, "val", f"{val_loss:.6f}", lr])

            if it % ckpt_interval == 0:
                save_checkpoint(language_model, optimizer, it, ckpt_path)

        if not diverged:
            print(f"final training loss = {loss_val:.4f}")
            final_val_loss = evaluate(
                language_model, val_ids, batch_size, context_length,
                device, num_batches=tr_cfg["final_val_batches"],
            )
            print(f"final validation loss = {final_val_loss:.4f}")
            log_writer.writerow([training_steps, "val", f"{final_val_loss:.6f}", lr])
            save_checkpoint(language_model, optimizer, training_steps, ckpt_path)
    finally:
        log_file.close()
        print(f"run log saved to {log_path}")

    return log_path


def make_lr_schedule(start: float = 0.3, decay: float = 0.7,
                     num_steps: int = 10) -> list[float]:
    """Geometric LR schedule: [start, start*decay, start*decay^2, ...].

    Defaults span ~1.5 orders of magnitude (0.3 down to ~0.012). Override
    `start` lower (e.g. 1e-2) and/or `decay` smaller (e.g. 0.5) for a wider
    range.
    """
    return [start * (decay ** i) for i in range(num_steps)]


def lr_sweep(cfg: dict,
             lrs: list[float] | None = None,
             steps_override: int | None = None,
             start: float = 0.3,
             decay: float = 0.7,
             num_steps: int = 10,
             experiment: str | None = None) -> list[tuple[float, Path]]:
    """Run `train()` for each LR; return [(lr, log_path), ...].

    If `lrs` is None, generates a geometric schedule via `make_lr_schedule`.
    If `experiment` is set, all runs go under logs/<experiment>/. Otherwise
    they go under a timestamped sweep subdirectory.
    """
    if lrs is None:
        lrs = make_lr_schedule(start, decay, num_steps)
    print(f"sweep LRs: {[f'{lr:.2e}' for lr in lrs]}")

    sweep_tag = (None if experiment is not None
                 else datetime.now().strftime("sweep_%Y%m%d_%H%M%S"))
    runs: list[tuple[float, Path]] = []
    for lr in lrs:
        print(f"\n=== LR sweep: lr={lr:.2e} ===")
        log_path = train(cfg, lr_override=lr,
                         steps_override=steps_override,
                         sweep_subdir=sweep_tag,
                         experiment=experiment)
        runs.append((lr, log_path))
    return runs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c",
                   default="cs336_basics/training/configs/default.yaml",
                   help="Path to YAML config file.")
    p.add_argument("--name", "-n", type=str, default=None,
                   help="Experiment name. All logs go to logs/<name>/. "
                        "Folder is created if it doesn't exist.")
    p.add_argument("--sweep", action="store_true",
                   help="Run an LR sweep with exponential stepping "
                        "instead of a single training run.")
    p.add_argument("--sweep-start", type=float, default=0.3,
                   help="Starting LR for the sweep (default: 0.3).")
    p.add_argument("--sweep-decay", type=float, default=0.7,
                   help="Multiplicative decay factor per step (default: 0.7).")
    p.add_argument("--sweep-num", type=int, default=10,
                   help="Number of LR values to try (default: 10).")
    p.add_argument("--sweep-steps", type=int, default=None,
                   help="Override training_steps for sweep runs "
                        "(sweeps are usually short).")
    return p.parse_args()


def plot_run(log_path, save_path=None, smooth_window=50):
    """Plot train/val loss curves from a single run log CSV."""
    log_path = Path(log_path)
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            loss = float(row["loss"])
            if not math.isfinite(loss):
                continue
            if row["split"] == "train":
                train_steps.append(step)
                train_losses.append(loss)
            else:
                val_steps.append(step)
                val_losses.append(loss)

    fig, ax = plt.subplots(figsize=(9, 5))

    if train_steps:
        ax.plot(train_steps, train_losses, label="train (raw)",
                alpha=0.25, linewidth=0.8)
        if smooth_window > 1 and len(train_losses) >= smooth_window:
            t = torch.tensor(train_losses, dtype=torch.float32)
            kernel = torch.ones(smooth_window) / smooth_window
            smoothed = torch.nn.functional.conv1d(
                t.view(1, 1, -1), kernel.view(1, 1, -1)
            ).view(-1).numpy()
            offset = smooth_window - 1
            ax.plot(train_steps[offset:], smoothed,
                    label=f"train (avg-{smooth_window})", linewidth=1.6)

    if val_steps:
        ax.plot(val_steps, val_losses, label="val",
                marker="o", linewidth=1.6, color="tab:red")

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"run: {log_path.stem}")
    ax.grid(alpha=0.3)
    ax.legend()

    if save_path is None:
        save_path = log_path.with_suffix(".png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot to {save_path}")
    return save_path


def plot_sweep(runs: list[tuple[float, Path]],
               save_path: Path | str,
               split: str = "val",
               smooth_window: int = 50):
    """Plot one curve per LR from a sweep.

    Args:
        runs: list of (lr, log_path) tuples (e.g. as returned by lr_sweep).
        save_path: where to write the PNG.
        split: "train" or "val". Train is smoothed since it's noisy.
        smooth_window: rolling-mean window for the train split.
    """
    save_path = Path(save_path)
    runs = sorted(runs, key=lambda x: x[0])
    cmap = cm.viridis
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (lr, log_path) in enumerate(runs):
        steps, losses = [], []
        with open(log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                loss = float(row["loss"])
                if not math.isfinite(loss):
                    continue
                steps.append(int(row["step"]))
                losses.append(loss)
        if not steps:
            continue

        color = cmap(i / max(1, len(runs) - 1))

        if split == "train" and smooth_window > 1 and len(losses) >= smooth_window:
            t = torch.tensor(losses, dtype=torch.float32)
            kernel = torch.ones(smooth_window) / smooth_window
            smoothed = torch.nn.functional.conv1d(
                t.view(1, 1, -1), kernel.view(1, 1, -1)
            ).view(-1).numpy()
            offset = smooth_window - 1
            ax.plot(steps[offset:], smoothed,
                    label=f"lr={lr:.2e}", color=color, linewidth=1.4)
        else:
            ax.plot(steps, losses,
                    label=f"lr={lr:.2e}", color=color,
                    marker="o" if split == "val" else None, linewidth=1.4)

    ax.set_xlabel("step")
    ax.set_ylabel(f"{split} loss")
    ax.set_yscale("log")  # losses span orders of magnitude across LRs
    ax.set_title(f"LR sweep ({split})")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved sweep plot to {save_path}")
    return save_path


def plot_lr_vs_loss(runs: list[tuple[float, Path]],
                    save_path: Path | str,
                    split: str = "val",
                    at_step: int | None = None):
    """One point per LR: x = LR (log), y = loss at `at_step` (or final).

    Classic LR-range-test picture. Use this to pick the best LR at a glance.
    Diverged runs are skipped (no finite loss at the requested step).

    Args:
        runs: list of (lr, log_path) tuples (e.g. from lr_sweep).
        save_path: where to write the PNG.
        split: "train" or "val".
        at_step: if given, take the loss at the latest logged step <= at_step.
            If None, use the final logged loss for that split.
    """
    save_path = Path(save_path)
    runs = sorted(runs, key=lambda x: x[0])
    xs, ys = [], []

    for lr, log_path in runs:
        steps, losses = [], []
        with open(log_path) as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                loss = float(row["loss"])
                if not math.isfinite(loss):
                    continue
                steps.append(int(row["step"]))
                losses.append(loss)
        if not losses:
            continue

        if at_step is None:
            y = losses[-1]
        else:
            valid = [i for i, s in enumerate(steps) if s <= at_step]
            if not valid:
                continue
            y = losses[max(valid)]

        xs.append(lr)
        ys.append(y)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, marker="o", linewidth=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    label = f"{split} loss" + (f" @ step {at_step}" if at_step else " (final)")
    ax.set_ylabel(label)
    ax.set_title("LR vs loss")
    ax.grid(alpha=0.3, which="both")

    if xs and ys:
        best_i = min(range(len(ys)), key=lambda i: ys[i])
        ax.axvline(xs[best_i], color="tab:red", linestyle="--", alpha=0.5,
                   label=f"best lr={xs[best_i]:.2e}")
        ax.legend()

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved lr-vs-loss plot to {save_path}")
    return save_path


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    if args.sweep:
        runs = lr_sweep(cfg,
                        steps_override=args.sweep_steps,
                        start=args.sweep_start,
                        decay=args.sweep_decay,
                        num_steps=args.sweep_num,
                        experiment=args.name)
        sweep_dir = runs[0][1].parent
        plot_sweep(runs, sweep_dir / "sweep_val.png", split="val")
        plot_sweep(runs, sweep_dir / "sweep_train.png", split="train")
        plot_lr_vs_loss(runs, sweep_dir / "lr_vs_loss.png", split="val")
    else:
        print("here")
        log_path = train(cfg, experiment=args.name)
        plot_run(log_path)