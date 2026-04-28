"""Download TinyStories V2 (GPT-4) train and validation files."""

import sys
from pathlib import Path
from urllib.request import urlopen, Request

BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
FILES = [
    "TinyStoriesV2-GPT4-train.txt",
    "TinyStoriesV2-GPT4-valid.txt",
]
OUT_DIR = Path("data")


def download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream-download `url` to `dest` with a simple progress indicator."""
    if dest.exists():
        print(f"✓ {dest.name} already exists ({dest.stat().st_size / 1e9:.2f} GB), skipping")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        print(f"↓ Downloading {dest.name} ({total / 1e9:.2f} GB)")

        with open(tmp, "wb") as f:
            while chunk := resp.read(chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                    sys.stdout.write(
                        f"\r  [{bar}] {pct:5.1f}%  "
                        f"{downloaded / 1e9:.2f} / {total / 1e9:.2f} GB"
                    )
                    sys.stdout.flush()

    tmp.rename(dest)
    print(f"\n✓ Saved to {dest}")


def main() -> None:
    for filename in FILES:
        download(f"{BASE_URL}/{filename}", OUT_DIR / filename)
    print("\nDone.")


if __name__ == "__main__":
    main()