"""Delete stale tokenizer + tokenized-data artifacts.

Reads paths from the same YAML config as the trainer so the two stay in sync.
Run from the project root, e.g.:

    python clean_artifacts.py
    python clean_artifacts.py -c cs336_basics/training/configs/other.yaml
"""

import argparse
from pathlib import Path

import yaml

ARTIFACT_KEYS = ["vocab", "merges", "train_bin", "val_bin"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", "-c",
        default="cs336_basics/training/configs/default.yaml",
        help="Path to the training YAML config.",
    )
    p.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Show what would be deleted without deleting.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]

    for key in ARTIFACT_KEYS:
        if key not in paths:
            print(f"  [skip] '{key}' not in config")
            continue

        path = Path(paths[key])
        if not path.exists():
            print(f"  [skip] {path} (not found)")
            continue

        if args.dry_run:
            print(f"  [dry-run] would delete {path}")
        else:
            path.unlink()
            print(f"  [deleted] {path}")


if __name__ == "__main__":
    main()