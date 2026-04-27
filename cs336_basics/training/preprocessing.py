from pathlib import Path
import numpy as np


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