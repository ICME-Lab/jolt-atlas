#!/usr/bin/env python3

from pathlib import Path
import argparse
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "models" / "qwen3-0.6b"
BASE = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main"
FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
]


def download(out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dst = out / name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"{dst} already exists, skipping")
        return

    url = f"{BASE}/{name}"
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    print(f"Downloading {url} -> {dst}")
    with urlopen(url) as response, tmp.open("wb") as file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
    tmp.rename(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()
    for name in FILES:
        download(args.out, name)


if __name__ == "__main__":
    main()
