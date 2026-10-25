#!/usr/bin/env python3
"""Download the BAAI/bge-small-en-v1.5 ONNX model using Hugging Face Optimum.

The script:
  1. Installs required pip packages (optimum, onnxruntime) if missing.
  2. Exports BAAI/bge-small-en-v1.5 to ONNX via `optimum-cli`.
  3. Renames model.onnx -> network.onnx (convention used by other models).

Output directory: models/bge-small-en-v1.5/
"""

import subprocess
import sys
from pathlib import Path

MODEL_ID = "BAAI/bge-small-en-v1.5"
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "bge-small-en-v1.5"


def ensure_packages():
    """Install optimum[exporters] and onnxruntime if not already present."""
    pkgs = ["optimum[exporters]", "optimum[onnxruntime]"]
    print("Ensuring required Python packages are installed ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *pkgs],
    )


def export_model():
    """Export BAAI/bge-small-en-v1.5 to ONNX using optimum-cli."""
    model_onnx = MODEL_DIR / "model.onnx"
    if model_onnx.exists():
        print(f"model.onnx already exists at {MODEL_DIR}, skipping export.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {MODEL_ID} to ONNX -> {MODEL_DIR} ...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "optimum.exporters.onnx",
            "--model",
            MODEL_ID,
            "--opset",
            "14",
            str(MODEL_DIR),
        ],
    )

    if not model_onnx.exists():
        sys.exit(f"ERROR: Export finished but {model_onnx} not found.")

    print("Export complete.")


def normalize_graph():
    """Keep ``token_embeddings`` as the sole graph output.

    The export has two outputs: ``token_embeddings`` (the encoder's last
    hidden state) and ``sentence_embedding`` (pooled + L2-normalized). The
    Atlas prover seeds an opening only for ``outputs()[0]``, so the second
    output would never be opened; pooling + normalization are a cheap public
    post-step. We keep ``token_embeddings`` and dead-node eliminate the
    pooling / L2-norm subgraph (whose axis-1 ``Gather`` the pipeline does not
    lower anyway).

    The old tracer also needed ``Div(x, Sqrt(v))`` rewritten here; the
    atlas-onnx-import pipeline lowers that pattern itself, so no arithmetic
    surgery remains.
    """
    import onnx

    path = MODEL_DIR / "model.onnx"
    model = onnx.load(str(path))
    g = model.graph

    nodes = list(g.node)

    # -- Single output = token_embeddings, then dead-node elimination --------
    keep_output = "token_embeddings"
    pruned_outputs = 0
    if any(o.name == keep_output for o in g.output) and len(g.output) > 1:
        for o in [o for o in g.output if o.name != keep_output]:
            g.output.remove(o)
            pruned_outputs += 1

    producer = {out: n for n in nodes for out in n.output}
    live = set()
    frontier = [o.name for o in g.output]
    while frontier:
        name = frontier.pop()
        n = producer.get(name)
        if n is None or id(n) in live:
            continue
        live.add(id(n))
        frontier.extend(n.input)
    kept = [n for n in nodes if id(n) in live]
    removed_nodes = len(nodes) - len(kept)

    del g.node[:]
    g.node.extend(kept)
    onnx.save(model, str(path))
    print(
        f"Graph normalized: {pruned_outputs} output(s) pruned, "
        f"{removed_nodes} dead node(s) removed."
    )


def rename_network():
    """Rename model.onnx -> network.onnx for compatibility."""
    src = MODEL_DIR / "model.onnx"
    dst = MODEL_DIR / "network.onnx"
    if dst.exists():
        print("network.onnx already exists, skipping rename.")
        return
    if not src.exists():
        sys.exit(f"ERROR: {src} not found, cannot rename.")
    print("Renaming model.onnx -> network.onnx ...")
    src.rename(dst)
    print("Done.")


def main():
    ensure_packages()
    export_model()
    normalize_graph()
    rename_network()
    print(f"\nOK: {MODEL_ID} ONNX model is ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
