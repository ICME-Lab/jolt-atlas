"""Helpers shared by the model download scripts and the tiny-fixture generator.

Every ONNX model in the zoo is produced by ``python -m optimum.exporters.onnx``
with the versions pinned in ``requirements-models.txt``; the helpers here keep
the post-processing (output pruning, ``model.onnx`` -> ``network.onnx``)
identical between a real download and its tiny stand-in.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
REQUIREMENTS = SCRIPTS_DIR / "requirements-models.txt"


def ensure_packages():
    """Install the pinned export toolchain (no-op when already satisfied)."""
    print(f"Ensuring pinned packages from {REQUIREMENTS.name} ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "-r", str(REQUIREMENTS)],
    )


def export_onnx(model: str, out_dir: Path, *extra_args: str):
    """Run the optimum ONNX exporter, skipping when ``model.onnx`` exists."""
    model_onnx = out_dir / "model.onnx"
    if model_onnx.exists() or (out_dir / "network.onnx").exists():
        print(f"ONNX already present in {out_dir}, skipping export.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {model} to ONNX -> {out_dir} ...")
    subprocess.check_call(
        [sys.executable, "-m", "optimum.exporters.onnx", "--model", model, *extra_args, str(out_dir)],
    )
    if not model_onnx.exists():
        sys.exit(f"ERROR: Export finished but {model_onnx} not found.")
    print("Export complete.")


def keep_single_output(path: Path, keep_output: str):
    """Keep ``keep_output`` as the sole graph output and drop dead nodes.

    Used for BERT-style encoders whose export has two outputs
    (``token_embeddings`` / ``last_hidden_state`` plus a pooled
    ``sentence_embedding``). The Atlas prover seeds an opening only for
    ``outputs()[0]``, so the second output would never be opened; pooling +
    normalization are a cheap public post-step. A no-op when the graph already
    has a single output or ``keep_output`` is absent.
    """
    import onnx

    model = onnx.load(str(path))
    g = model.graph
    nodes = list(g.node)

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
    print(f"Graph normalized: {pruned_outputs} output(s) pruned, {removed_nodes} dead node(s) removed.")


def rename_network(model_dir: Path):
    """Rename ``model.onnx`` -> ``network.onnx`` (the zoo convention)."""
    src = model_dir / "model.onnx"
    dst = model_dir / "network.onnx"
    if dst.exists():
        print("network.onnx already exists, skipping rename.")
        return
    if not src.exists():
        sys.exit(f"ERROR: {src} not found, cannot rename.")
    print("Renaming model.onnx -> network.onnx ...")
    src.rename(dst)
