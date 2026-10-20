#!/usr/bin/env python3
"""Generate the synthetic test models in models/.

Replaces the old pile of single-op fixtures (tanh/, erf/, gather/, five
sum_* variants, ...) with:

- models/op_gauntlet/network.onnx — one small transformer-ish model that
  exercises the entire lowerable ONNX surface in a single graph: embedding
  gather, decomposed LayerNorm (mean/var form), attention with a runtime
  boolean mask (Where + IsNaN guard), both GELU forms (erf and tanh/x^3),
  an RMSNorm pattern, Split/Concat, Expand, every ReduceSum axis
  configuration the prover's sum normalization distinguishes (1-D, 2-D
  axis 0, 2-D axis 1, 3-D trailing axis), sin/cos/sigmoid, a float->int->
  float Cast round trip (descale), and constant-divisor Div rewrites.
  All dims are powers of two, so the model imports, traces, proves, and
  verifies.

- models/reject_slice/network.onnx and models/reject_concat/network.onnx —
  tiny negative fixtures whose padded execution cannot be made coherent
  (a non-power-of-two slice extent; concat extents whose paddings do not
  add up). The importer must reject both with typed errors.

Requires the `onnx` package (the repo-root .venv has it):
    .venv/bin/python scripts/gen_test_models.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"

OPSET = 13
RNG = np.random.default_rng(0xA71A5)

# Gauntlet dimensions (all powers of two; small enough to prove quickly).
SEQ = 16
DIM = 32
HEADS = 4
HEAD_DIM = DIM // HEADS
VOCAB = 64
FFN = 64


def w(name, shape, scale=0.25):
    """A small random float initializer."""
    data = RNG.uniform(-scale, scale, size=shape).astype(np.float32)
    return numpy_helper.from_array(data, name=name)


def consts_i64(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def const_f32(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)


def save(graph, dirname):
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="gen_test_models",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    out = MODELS / dirname
    out.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out / "network.onnx"))
    print(f"wrote {out / 'network.onnx'}")


def op_gauntlet():
    nodes = []
    inits = []

    def n(op, inputs, outputs, **attrs):
        nodes.append(helper.make_node(op, inputs, outputs, **attrs))
        return outputs[0]

    # ── Inputs ────────────────────────────────────────────────────────────
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, SEQ])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, [1, SEQ]
    )

    # ── Embeddings ────────────────────────────────────────────────────────
    inits += [w("wte", (VOCAB, DIM)), w("pos_emb", (1, SEQ, DIM), scale=0.1)]
    tok = n("Gather", ["wte", "input_ids"], ["tok_emb"], axis=0)
    x0 = n("Add", [tok, "pos_emb"], ["x0"])

    # ── Decomposed LayerNorm (mean/variance form) ─────────────────────────
    inits += [consts_i64("axes_last", [-1]), const_f32("eps", [1e-5]),
              const_f32("two", [2.0]), w("ln1_g", (DIM,), 0.5), w("ln1_b", (DIM,), 0.1)]
    mean = n("ReduceMean", ["x0"], ["ln_mean"], axes=[-1], keepdims=1)
    centered = n("Sub", ["x0", mean], ["ln_centered"])
    sq = n("Pow", [centered, "two"], ["ln_sq"])
    var = n("ReduceMean", [sq], ["ln_var"], axes=[-1], keepdims=1)
    var_eps = n("Add", [var, "eps"], ["ln_var_eps"])
    std = n("Sqrt", [var_eps], ["ln_std"])
    normed = n("Div", [centered, std], ["ln_normed"])
    scaled = n("Mul", [normed, "ln1_g"], ["ln_scaled"])
    ln1 = n("Add", [scaled, "ln1_b"], ["ln1"])

    # ── Attention ─────────────────────────────────────────────────────────
    for name in ["wq", "wk", "wv", "wo"]:
        inits.append(w(name, (DIM, DIM)))
        inits.append(w(name + "_b", (DIM,), 0.05))
    inits += [consts_i64("qkv_shape", [1, SEQ, HEADS, HEAD_DIM]),
              consts_i64("merge_shape", [1, SEQ, DIM]),
              const_f32("inv_sqrt_d", [1.0]),  # divisor applied via Div below
              const_f32("sqrt_d", [float(np.sqrt(HEAD_DIM))]),
              const_f32("neg_big", [-3.4e38]),
              const_f32("zero", [0.0]),
              consts_i64("mask_unsq_axes", [1, 2])]

    def proj(name, src):
        mm = n("MatMul", [src, name], [f"{name}_mm"])
        add = n("Add", [mm, name + "_b"], [f"{name}_proj"])
        rs = n("Reshape", [add, "qkv_shape"], [f"{name}_rs"])
        return n("Transpose", [rs], [f"{name}_t"], perm=[0, 2, 1, 3])

    q = proj("wq", ln1)
    k = proj("wk", ln1)
    v = proj("wv", ln1)

    kt = n("Transpose", [k], ["k_T"], perm=[0, 1, 3, 2])
    scores = n("MatMul", [q, kt], ["scores"])
    scaled_scores = n("Div", [scores, "sqrt_d"], ["scores_scaled"])

    # Runtime boolean mask: raw 0/1 ints -> bool cond for Where.
    mask_u = n("Unsqueeze", ["attention_mask", "mask_unsq_axes"], ["mask_u"])
    mask_b = n("Cast", [mask_u], ["mask_b"], to=TensorProto.BOOL)
    masked = n("Where", [mask_b, scaled_scores, "neg_big"], ["scores_masked"])
    probs = n("Softmax", [masked], ["probs"], axis=-1)

    # NaN guard (the GPT-2 export pattern): Where(IsNaN(p), 0, p).
    is_nan = n("IsNaN", [probs], ["probs_nan"])
    probs_clean = n("Where", [is_nan, "zero", probs], ["probs_clean"])

    ctx = n("MatMul", [probs_clean, v], ["ctx"])
    ctx_t = n("Transpose", [ctx], ["ctx_t"], perm=[0, 2, 1, 3])
    ctx_m = n("Reshape", [ctx_t, "merge_shape"], ["ctx_m"])
    attn_mm = n("MatMul", [ctx_m, "wo"], ["attn_mm"])
    attn = n("Add", [attn_mm, "wo_b"], ["attn_out"])
    x1 = n("Add", ["x0", attn], ["x1"])

    # ── RMSNorm pattern ───────────────────────────────────────────────────
    inits += [w("rms_g", (DIM,), 0.5)]
    rms_sq = n("Pow", ["x1", "two"], ["rms_sq"])
    rms_mean = n("ReduceMean", [rms_sq], ["rms_mean"], axes=[-1], keepdims=1)
    rms_eps = n("Add", [rms_mean, "eps"], ["rms_eps"])
    rms_std = n("Sqrt", [rms_eps], ["rms_std"])
    rms_n = n("Div", ["x1", rms_std], ["rms_n"])
    x2 = n("Mul", [rms_n, "rms_g"], ["x2"])

    # ── FFN: up-project, split, GELU both ways, concat, down-project ──────
    inits += [w("w_up", (DIM, FFN)), w("w_up_b", (FFN,), 0.05), w("w_down", (FFN, DIM)),
              const_f32("half", [0.5]), const_f32("one", [1.0]),
              const_f32("inv_sqrt2", [float(1 / np.sqrt(2))]),
              const_f32("three", [3.0]),
              const_f32("gelu_c0", [float(np.sqrt(2 / np.pi))]),
              const_f32("gelu_c1", [0.044715])]
    up_mm = n("MatMul", ["x2", "w_up"], ["up_mm"])
    up = n("Add", [up_mm, "w_up_b"], ["up"])
    nodes.append(helper.make_node("Split", ["up"], ["h_erf", "h_tanh"], axis=2))

    # GELU (erf form): 0.5·h·(1 + erf(h/√2))
    e_in = n("Mul", ["h_erf", "inv_sqrt2"], ["gelu_e_in"])
    e_erf = n("Erf", [e_in], ["gelu_e_erf"])
    e_1 = n("Add", [e_erf, "one"], ["gelu_e_1"])
    e_h = n("Mul", ["h_erf", e_1], ["gelu_e_h"])
    g_erf = n("Mul", [e_h, "half"], ["gelu_erf"])

    # GELU (tanh form): 0.5·h·(1 + tanh(c0·(h + c1·h³)))
    t_cube = n("Pow", ["h_tanh", "three"], ["gelu_t_cube"])
    t_c1 = n("Mul", [t_cube, "gelu_c1"], ["gelu_t_c1"])
    t_sum = n("Add", ["h_tanh", t_c1], ["gelu_t_sum"])
    t_arg = n("Mul", [t_sum, "gelu_c0"], ["gelu_t_arg"])
    t_tanh = n("Tanh", [t_arg], ["gelu_t_tanh"])
    t_1 = n("Add", [t_tanh, "one"], ["gelu_t_1"])
    t_h = n("Mul", ["h_tanh", t_1], ["gelu_t_h"])
    g_tanh = n("Mul", [t_h, "half"], ["gelu_tanh"])

    ffn_cat = n("Concat", [g_erf, g_tanh], ["ffn_cat"], axis=2)
    ffn_out = n("MatMul", [ffn_cat, "w_down"], ["ffn_out"])
    x3 = n("Add", [x2, ffn_out], ["x3"])

    # ── Trig / sigmoid spice ──────────────────────────────────────────────
    s_sig = n("Sigmoid", ["x3"], ["spice_sig"])
    s_sin = n("Sin", ["x3"], ["spice_sin"])
    s_cos = n("Cos", ["x3"], ["spice_cos"])
    s_mix = n("Mul", [s_sig, s_sin], ["spice_mix"])
    s_neg = n("Neg", [s_cos], ["spice_neg"])
    s_sq = n("Pow", [s_neg, "two"], ["spice_sq"])  # standalone fused Square
    s_mix2 = n("Add", [s_mix, s_sq], ["spice_mix2"])
    x4 = n("Add", ["x3", s_mix2], ["x4"])

    # ── Reduction coverage: every sum-axis configuration ──────────────────
    inits += [consts_i64("sq_axes0", [0]), consts_i64("flat_shape", [SEQ * DIM]),
              consts_i64("exp_shape", [1, SEQ, DIM]),
              consts_i64("ax2", [2]), consts_i64("ax0", [0]), consts_i64("ax1", [1])]
    # 3-D, trailing axis (keepdims) -> [1, SEQ, 1]
    red3 = n("ReduceSum", ["x4", "ax2"], ["red3"], keepdims=1)
    # 2-D views: squeeze the batch dim.
    x4_2d = n("Squeeze", ["x4", "sq_axes0"], ["x4_2d"])  # [SEQ, DIM]
    red2a = n("ReduceSum", [x4_2d, "ax0"], ["red2a"], keepdims=1)  # [1, DIM]
    red2b = n("ReduceSum", [x4_2d, "ax1"], ["red2b"], keepdims=1)  # [SEQ, 1]
    # 1-D: flatten then sum to a scalar-ish [1].
    x4_1d = n("Reshape", [x4_2d, "flat_shape"], ["x4_1d"])
    red1 = n("ReduceSum", [x4_1d, "ax0"], ["red1"], keepdims=1)  # [1]

    # Cast round trip (descale + identity) on the scalar branch.
    red1_i = n("Cast", [red1], ["red1_i"], to=TensorProto.INT64)
    red1_f = n("Cast", [red1_i], ["red1_f"], to=TensorProto.FLOAT)

    # Merge the reduction branches back (Expand covers atlas.Broadcast).
    red3_e = n("Expand", [red3, "exp_shape"], ["red3_e"])
    mix_a = n("Add", ["x4", red3_e], ["mix_a"])
    red2a_3d = n("Unsqueeze", [red2a, "ax0"], ["red2a_3d"])  # [1, 1, DIM]
    mix_b = n("Add", [mix_a, red2a_3d], ["mix_b"])
    red2b_3d = n("Unsqueeze", [red2b, "ax0"], ["red2b_3d"])  # [1, SEQ, 1]
    mix_c = n("Add", [mix_b, red2b_3d], ["mix_c"])
    mix_d = n("Add", [mix_c, red1_f], ["mix_d"])

    # ── Head ──────────────────────────────────────────────────────────────
    inits += [w("w_head", (DIM, VOCAB))]
    logits = n("MatMul", [mix_d, "w_head"], ["logits"])

    graph = helper.make_graph(
        nodes,
        "op_gauntlet",
        [input_ids, attention_mask],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, SEQ, VOCAB])],
        initializer=inits,
    )
    save(graph, "op_gauntlet")


def reject_slice():
    """Non-power-of-two slice extent: padded output dims cannot match."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])
    inits = [
        consts_i64("starts", [1]),
        consts_i64("ends", [4]),
        consts_i64("axes", [1]),
    ]
    nodes = [helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["y"])]
    graph = helper.make_graph(
        nodes,
        "reject_slice",
        [x],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=inits,
    )
    save(graph, "reject_slice")


def reject_concat():
    """Concat extents whose independent paddings don't sum to the padded
    output (16 + 64 = 80, but pow2(80) = 128)."""
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 16])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 64])
    nodes = [helper.make_node("Concat", ["a", "b"], ["y"], axis=1)]
    graph = helper.make_graph(
        nodes,
        "reject_concat",
        [a, b],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 80])],
    )
    save(graph, "reject_concat")


if __name__ == "__main__":
    op_gauntlet()
    reject_slice()
    reject_concat()
