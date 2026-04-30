#!/usr/bin/env python3

from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "atlas-onnx-tracer" / "models" / "qwen"
BASE = "https://huggingface.co/Qwen/Qwen2-0.5B/resolve/main"
FILES = ["config.json", "model.safetensors"]


def download(name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"{dst} already exists, skipping")
        return

    url = f"{BASE}/{name}"
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    print(f"Downloading {url} -> {dst}")
    with urlopen(url) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dst)


def main() -> None:
    for name in FILES:
        download(name)


if __name__ == "__main__":
    main()
