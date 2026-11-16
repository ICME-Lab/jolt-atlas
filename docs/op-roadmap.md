# Roadmap: op coverage

Living document. It ranks the ops worth adding next and says which proving technique each one
needs. Nothing here is scheduled; an op moves from this page into a spec when someone picks it
up. Last updated 2026-11-15.

`docs/roadmap.md` tracks the proof-system toolkit program. This page is independent of it.

## Starting point

The trace IR has 33 op variants (`atlas_trace::Op`). The single-input nonlinear ones are ReLU,
Clamp, Tanh, Sigmoid, Erf, Sin, Cos and Rsqrt, plus the softmax exponential. Three gaps follow
from that set:

- **No fused activations.** GELU and SiLU reach the prover as chains: the Tanh, Erf or Sigmoid
  lookup followed by multiplies and adds, each with its own proof.
- **No standalone Exp, Log, Sqrt or Reciprocal.** The importer accepts `Sqrt` and `Reciprocal`
  only where it can rewrite them to `atlas.Rsqrt` (`Reciprocal(Sqrt(v))`, `Div(x, Sqrt(v))`)
  or fold them as constants.
- **No convolution, pooling or runtime comparison ops.** `Less`, `Greater` and `Equal` exist in
  the importer only for constant folding and mask handling.

One experiment informs the ranking. `atlas-lookup-tables/src/piecewise_linear.rs` (commit
`267e942ec`, read by no op) holds piecewise-linear tables for tanh, erf, sigmoid, sin and cos:
the function's value at every 256-input segment boundary, interpolated and kept unrounded.
Measured at scale 14:

| Table | Max error | Against |
|---|---|---|
| tanh | 1.24 ulp | the semantics table |
| erf | 1.31 ulp | the semantics table |
| sigmoid | 1.03 ulp | the semantics table |
| sin | 0.99 ulp | real sine (the current sin table is up to 95 ulp off) |

The technique works for any function that is smooth with second derivative at most about 1 on
a bounded domain. It does not yet plug into the prefix-suffix sumcheck; see "Open decisions".

## Tier 1: fused activations (done)

GELU (tanh and erf forms) and SiLU are one op, `SmoothRelu` (`specs/fused-gelu.md`). Each
satisfies `f(x) − f(−x) = x`, so `f(x) − ReLU(x)` is even and vanishes away from zero, and

```text
f(x) = ReLU(x) + H[clamp(x)]
```

with `H` a plain rounded table that is zero at both ends of its domain. The linear tail needs
no table of its own: it is the ReLU read. The importer fuses the exported subgraphs
(`smooth_relu_fusion`); GPT-2, BERT/bge and Qwen2 lower to it.

Not yet covered: other gated activations (Mish, HardSwish, Swish with a slope other than one).

## Tier 2: the mantissa family

Log, Sqrt and Reciprocal are the most important ops we cannot express at all today.

| Op | Needed for |
|---|---|
| Log | LogSoftmax and cross-entropy (proving a perplexity or a loss), log-mel audio front ends, tabular and finance models |
| Sqrt | Standalone norms and distances outside the `Rsqrt` rewrite patterns |
| Reciprocal | Division by a runtime value outside the `Rsqrt` rewrite patterns |

Uniform segments fail for all three because curvature blows up near zero. The fix is one shared
table shape: split the input into its leading-one position `k` and a mantissa `m` in `[1, 2)`,
and interpolate on the mantissa only. For example `log(x) = k·ln 2 + log(m)`. Building this
shape once covers the three ops, and it would show whether a table can compete with the
current `Rsqrt` relation.

Exp belongs with this tier in importance but not in technique. Its negative side fits the
uniform table; its positive side grows without bound and would need a multiplicative split
(`e^(a+b) = e^a·e^b`); softmax reads one table and does not use one.

## Tier 3: more single-input activations

Cheap once tier 1 lands, since each is a new knot function for the same table.

| Op | Models | Notes |
|---|---|---|
| Softplus | Mamba and other state-space models, variance heads | Linear tail. |
| Mish | YOLOv4 | Same shape as SiLU. |
| ELU, SELU, CELU | Older MLPs, tabular models | Saturates on one side, linear on the other. |
| HardSigmoid | MobileNetV3, EfficientNet-Lite | Already piecewise-linear, so the table is exact. |
| HardSwish | MobileNetV3, EfficientNet-Lite | Piecewise-quadratic, so approximated like the smooth ones. |
| LeakyReLU | YOLOv3/v4, GANs | Exact with one knot at zero; the slope's rounding is the only subtlety. |

## Tier 4: not tables

These matter for vision models and for proving sampling, but they are arithmetic or comparison
relations and the table work above does not help with them.

- Conv (1-D and 2-D), and with it BatchNorm folding and GroupNorm.
- MaxPool, AveragePool, GlobalAveragePool, Resize.
- Max, Min, ArgMax, TopK, and runtime `Less`/`Greater`/`Equal`.

## What adding an op requires

From `CLAUDE.md` and `docs/pull-requests.md`:

- The table or kernel is generated in `atlas_trace::semantics`, in integer arithmetic, under a
  `SemVersion`. The trig knots in `piecewise_linear.rs` are rounded in `atlas-lookup-tables`
  and would have to move there.
- Replacing an existing op's table changes its outputs, so it is a new `SemVersion` with
  re-pinned golden hashes. The piecewise-linear tanh, erf and sigmoid differ from today's
  tables on 7–16% of inputs after rounding.
- Arithmetic is exact or absent: a kernel whose result is a sum, difference, negation or
  rescaled quotient narrows it with `atlas_trace::semantics::fixed::narrow_i32` and returns
  `KernelError::WitnessOverflow` outside `i32`, never a clamped value. Its
  `OpSpec::checks_output` is `true`, so the verifier range-checks the output in the packed
  buckets (`global_clamp`), and its relation's doc states the no-wrap bound.
- A row in `onnx_proof/testing/op_surface.rs`, a probe model in `testing/models.rs` so the
  soundness sweeps cover it, and a relation that follows `docs/pull-requests.md` §4.
- A spec in `specs/` for anything over 500 non-trivial lines.

## Open decisions

- **Vector-valued prefix state.** The piecewise-linear tables implement `JoltLookupTable` but
  not `PrefixSuffixDecompositionTrait`: mid-sumcheck they need a partially folded knot vector
  (up to 257 field elements for tanh) where `PrefixCheckpoints` holds one scalar per prefix.
  Every tier 2–3 table depends on the prefix-suffix sumcheck accepting such state.
- **Rounding the output.** The tables emit 8 extra fractional bits. Bringing the value back to
  model scale needs a remainder range check per lookup, which gives back part of the saving.
