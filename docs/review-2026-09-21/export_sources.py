#!/usr/bin/env python3
"""Record exact Git trees and small patches over the preserved source archives."""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import tarfile

root = Path(__file__).resolve().parent
repo = root / "repo"
for name, (branch, original) in json.loads((root / "archive-heads.json").read_text()).items():
    head = subprocess.check_output(["git", "rev-parse", "review/" + branch], cwd=repo, text=True).strip()
    (root / f"{name}.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", original, head], cwd=repo))
    data = subprocess.check_output(["git", "archive", head], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(data)) as bundle:
        files = {m.name: hashlib.sha256(bundle.extractfile(m).read()).hexdigest()
                 for m in bundle.getmembers() if m.isfile()}
    (root / f"{name}.manifest.json").write_text(json.dumps(dict(head=head, files=files), indent=2) + "\n")
    print(name, head[:8], len(files), flush=True)
