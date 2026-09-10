#!/usr/bin/env python3
"""Download a WikiText-2 sample for the `tab:accuracy` bench.

Pulls the WikiText-2 raw test split, concatenates non-empty lines into a plain-text file capped
at MAX_CHARS. Output (gitignored): jolt-atlas-core/examples/bench/data/wikitext-2-test.txt
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "jolt-atlas-core" / "examples" / "bench" / "data" / "wikitext-2-test.txt"

MAX_CHARS = 200_000


def ensure_packages():
    print("Ensuring required Python packages are installed …")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "datasets"],
    )


def download():
    if OUT_PATH.exists():
        print(f"{OUT_PATH} already exists, skipping.")
        return

    from datasets import load_dataset

    print("Downloading WikiText-2 (raw, test split) …")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chars = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in ds:
            line = row["text"].strip()
            if not line or line.startswith("="):  # skip section headers/blank rows
                continue
            f.write(line + "\n")
            chars += len(line) + 1
            if chars >= MAX_CHARS:
                break

    print(f"Wrote {chars} chars → {OUT_PATH}")


def main():
    ensure_packages()
    download()
    print(f"\n✅  WikiText-2 sample ready at {OUT_PATH}")


if __name__ == "__main__":
    main()
