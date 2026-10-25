#!/usr/bin/env python3
"""Generate the tiny stand-in zoo models: models/{gpt2,qwen2,bge}-tiny/.

Each is a *random-init* Hugging Face model with the production architecture
(GPT-2, Qwen2, BERT) exported by the same ``optimum`` CLI and flags as the
corresponding ``download_*.py`` script, so the ONNX op surface and the
importer passes it triggers match the real model; only the dims are tiny.
They exist so that CI proves "a GPT-2 / Qwen2 / BERT graph" on every PR in
seconds instead of minutes.

Regenerating changes the weights (and, with a different toolchain, possibly
the graph), which invalidates the pinned goldens in
``atlas-onnx-import/tests/golden_zoo.rs``. Run from the pinned toolchain:

    python -m venv .venv && .venv/bin/pip install -r scripts/requirements-models.txt
    .venv/bin/python scripts/gen_tiny_models.py [gpt2-tiny|qwen2-tiny|bge-tiny ...]
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

from onnx_export_utils import REPO_ROOT, export_onnx, keep_single_output, rename_network

MODELS_DIR = REPO_ROOT / "models"
SEED = 0xA71A5

# Dims are chosen to keep each real model's *character*, not its size:
#   gpt2-tiny : non-pow2 n_embd / heads / vocab (real: 768 / 12 / 50257),
#               gelu_new (tanh GELU), native LayerNormalization (opset 18),
#               fused QKV Split, Gemm, with-past export (KV inputs elided).
#   qwen2-tiny: the real GQA layout, 14 q-heads over 2 kv-heads (ratio 7, not
#               a power of two -> importer `repeat_to_gather` + the
#               outer-product einsum `amk,ank->amn`; 2 kv-heads matter: a
#               single kv-head gives a one-row gather dictionary, which the
#               prover cannot handle yet); RoPE Cos/Sin over scaled position_ids;
#               RMSNorm eps 1e-6; SiLU; tied embeddings; non-pow2 hidden /
#               intermediate (real: 896 / 4864). Vocab kept < 65_536 so the
#               checked-in file stays < 1 MB; GatherLarge is covered by the
#               single-op table and by the real-Qwen heavy CI job.
#   bge-tiny  : Erf GELU, decomposed LayerNorm with eps 1e-12
#               (`rsqrt_eps_floor`), token_type gather, opset 14.
SPECS = {
    "gpt2-tiny": dict(
        kind="gpt2",
        config=dict(
            n_layer=2,
            n_embd=48,
            n_head=3,
            n_inner=192,
            vocab_size=1000,
            n_positions=64,
            n_ctx=64,
            activation_function="gelu_new",
            layer_norm_epsilon=1e-5,
            use_cache=True,
            bos_token_id=999,
            eos_token_id=999,
        ),
        export=["--task", "text-generation-with-past", "--opset", "18"],
    ),
    "qwen2-tiny": dict(
        kind="qwen2",
        config=dict(
            num_hidden_layers=2,
            hidden_size=56,
            num_attention_heads=14,
            num_key_value_heads=2,
            intermediate_size=152,
            vocab_size=1000,
            max_position_embeddings=64,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
            hidden_act="silu",
            tie_word_embeddings=True,
            use_cache=False,
            bos_token_id=999,
            eos_token_id=999,
        ),
        export=["--task", "text-generation", "--opset", "18"],
    ),
    "bge-tiny": dict(
        kind="bert",
        config=dict(
            num_hidden_layers=2,
            hidden_size=48,
            num_attention_heads=3,
            intermediate_size=192,
            vocab_size=1000,
            type_vocab_size=2,
            max_position_embeddings=64,
            layer_norm_eps=1e-12,
            hidden_act="gelu",
            pad_token_id=0,
        ),
        export=["--task", "feature-extraction", "--opset", "14"],
        keep_output="last_hidden_state",
    ),
}

KEEP_FILES = {"network.onnx", "config.json"}


def build_model(kind: str, config: dict):
    import torch
    from transformers import (
        BertConfig,
        BertModel,
        GPT2Config,
        GPT2LMHeadModel,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    torch.manual_seed(SEED)
    if kind == "gpt2":
        return GPT2LMHeadModel(GPT2Config(**config))
    if kind == "qwen2":
        return Qwen2ForCausalLM(Qwen2Config(**config))
    if kind == "bert":
        return BertModel(BertConfig(**config))
    raise ValueError(kind)


def generate(name: str):
    spec = SPECS[name]
    out_dir = MODELS_DIR / name
    if out_dir.exists():
        shutil.rmtree(out_dir)

    model = build_model(spec["kind"], spec["config"]).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] {n_params} parameters")

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp, safe_serialization=True)
        # A minimal tokenizer config lets optimum resolve the task without a
        # real tokenizer; the tiny models are driven with synthetic ids.
        export_onnx(tmp, out_dir, *spec["export"])

    if keep := spec.get("keep_output"):
        keep_single_output(out_dir / "model.onnx", keep)
    rename_network(out_dir)

    for p in out_dir.iterdir():
        if p.name not in KEEP_FILES:
            p.unlink() if p.is_file() else shutil.rmtree(p)
    (out_dir / "GENERATED.json").write_text(
        json.dumps({"generator": "scripts/gen_tiny_models.py", "seed": SEED, **spec}, indent=2) + "\n"
    )
    size = (out_dir / "network.onnx").stat().st_size
    print(f"[{name}] network.onnx {size / 1e6:.2f} MB")


def main():
    names = sys.argv[1:] or list(SPECS)
    for name in names:
        if name not in SPECS:
            sys.exit(f"unknown model {name!r}; choose from {list(SPECS)}")
        generate(name)


if __name__ == "__main__":
    main()
