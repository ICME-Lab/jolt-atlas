# Structural / data-movement and linear ops in jolt-atlas: current proving cost, cheapest treatment, proposals

Conventions used in these notes

- **[CODE]** = verified by reading the working tree on branch `dev` (2026-11-09); cited as `path:line`. Paths are relative to `/Users/kg/swe/icme/official/jolt-atlas/`. "verifier/" abbreviates `jolt-atlas-verifier/src/onnx_proof/`, "core/" abbreviates `jolt-atlas-core/src/onnx_proof/`.
- **[TRACE]** = measured from the checked-in Chrome trace `trace-1792493663983876.json` (repo root, dated Oct 20). It is a single traced Qwen2-0.5B-like run (2172 nodes, 24 softmax layers) on the **old** stack (Dory PCS, tract importer). Treat op counts and timings as indicative only; the stack has since moved to Akita and the tract-free importer. No cargo command was run for these notes.
- **[LIT]** = literature claim with URL. **[EST]** = my own estimate or inference, not measured.
- n = log2 of a tensor's padded element count; N or k = number of claims on one tensor.

Working-tree state: the 34 uncommitted files (`git diff --stat`: +440/−346) move the prover-only trait methods out of the verifier crate, and change no protocol. `EngineRelation::witnesses` became a new core trait `RelationWitnesses` and `TieRelation::claims` became core trait `TieClaims` (`jolt-atlas-core/src/onnx_proof/relation.rs:62-80`; removed from `jolt-atlas-verifier/src/onnx_proof/relation.rs:71,108`). Each op's `witnesses`/`claims` impl moved from `verifier/ops/*.rs` to `core/ops/*.rs` (e.g. concat's `extend_input_to_max_domain`). This matches the untracked `specs/verifier-audit-simplification.md` ("Trace-free verifier crate"). Everything below describes that working tree.

---

## Q0 (objective a + b). What does each structural/linear op cost today, and what is the cheapest possible treatment?

### Takeaway
Seven ops are already zero-round "ties" (Input, Constant, Identity, Neg, IsNan, Broadcast, MoveAxis), but Reshape, Slice and Concat each run a full n-round degree-2 sumcheck with an O(2^n) verifier, even though in a compiled (padded) model Reshape is the identity on the flat array and aligned Slice/Concat are pure variable-fixing. Add/Sub/Sum are not "linear" in cost: every non-scalar one pays a saturating-clamp range proof (8 dense committed one-hot chunks per output element plus slack chunks), which is where their real cost sits.

### Cited Findings

Global structure
- [CODE] Intermediate tensors are never committed. Node outputs are `VirtualPoly::NodeOutput`, and claims flow output-to-input in reverse topological order (`core/prover.rs:173-185`). The only committed polynomials are one-hot lookup/range-check chunks and a few advice columns (`common/src/lib.rs:35ff`; `polynomial_map`, `core/prover.rs:295-324`).
- [CODE] Per node: first an N-to-1 evaluation reduction on the node's output claims, then the op's own protocol (`core/ops/mod.rs:231-240`, `ReductionFlow::Default`).
- [CODE] A "tie" is a zero-round relation. The prover appends the produced openings; the verifier checks `input_claim == output_expression` (`verifier/relation.rs:106-144`; prover `core/relation.rs:84-95`).
- [CODE] A "dense engine relation" is a sumcheck whose summand is public tables times witness polynomials. The verifier derives each `RelationDerived::Table(i)` by **materialising the whole 2^rounds table and evaluating its MLE** (`atlas-sumcheck/src/relation/mod.rs:310-311`). So the verifier is O(2^n) field operations for every Reshape/Slice/Concat node.
- [CODE] Production models are compiled with `pad_pow2: true` (`atlas-onnx-import/src/lib.rs:53,76`; `atlas-trace/src/model.rs:47-48,242`). `pad_graph` rounds every node's `output_dims` up to powers of two and lets each op rewrite or reject itself via `apply_padding` (`atlas-trace/src/model.rs:157-183`). The "raw dims" the relations read from the model are therefore already powers of two in production; `next_power_of_two()` calls in the relations are no-ops except in tests built with `pad_pow2 = false` (`core/testing/model_builder.rs:379-395`).

Per-op table (all [CODE] unless marked)

| Op | Today | Sumcheck | Commits | Verifier | Cheapest treatment [EST] |
|---|---|---|---|---|---|
| Input | tie: `out(r) = public_input~(r)` (`verifier/ops/input.rs:18-37`, `constant.rs:113-119`) | none | none | O(2^n) MLE evaluation of the input | already minimal |
| Constant | tie: `out(r) = const~(r)` (`verifier/ops/constant.rs:32-45,113-119`) | none | none | O(2^n) per constant. [TRACE] `Constant::verify` = 2.89 s of 9.37 s verify (31%), top two 719 ms and 583 ms | Free for the prover. Verifier cost is removable only by committing weights (preprocessing commitment plus one opening); a design choice, not a protocol flaw |
| Identity | tie `out(r) = in(r)`, claim copied from the accumulator, no MLE evaluation (`verifier/ops/neg.rs:248-323`, `core/ops/neg.rs` `TieClaims`) | none | none | O(1) | Free; better still, elide the node in the importer (`elide` already drops ONNX Identity, `atlas-onnx-import/src/passes.rs:89-117`; later passes re-introduce some, `passes.rs:1417,1461`) |
| Neg | tie `out(r) = −in(r)` (same relation, SIGN = −1) | none | none | O(1) | Free. Caveat in code: `saturating_neg(i32::MIN)` breaks completeness, not soundness (`verifier/ops/neg.rs:47-50`) |
| IsNan | tie `out(r) = 0`; the operand is still opened at r and the prover does an O(2^n) `evaluate_mle` for a claim the relation never reads (`verifier/ops/is_nan.rs:24-33,63-72`; `core/ops/is_nan.rs`) | none | none | O(1) | Should not exist as a node: output is the constant zero tensor. Today it also adds one extra claim on its operand, which triggers an eval reduction there |
| Broadcast | tie `out(r) = ones~(r_bc) · in(r_in)`; r is split by axis (`verifier/ops/broadcast.rs:53-133,196-218`). The prover evaluates the operand MLE at `r_in` (`core/ops/broadcast.rs`) | none | none | Builds an all-ones tensor of the broadcast size and evaluates its MLE (`broadcast.rs:123-131,159-163`) | Free. With power-of-two dims `ones~ ≡ 1`, so the derived factor and the tensor build are unnecessary |
| MoveAxis | tie `out(r) = in(π⁻¹(r))`, permuting per-axis coordinate groups (`verifier/ops/moveaxis.rs:27-134`). The claim is copied, no MLE evaluation (`core/ops/moveaxis.rs`) | none | none | O(n) | Already free. Could be elided entirely by composing the permutation into the consumer |
| Reshape | dense relation `out(r₀) = Σ_i selector(i)·in(i)` (`verifier/ops/reshape.rs:96-150`) | n rounds, degree 2 | none | O(2^n): rebuilds the selector (`reshape.rs:48-88`) | **Free tie** `out(r) = in(r)` in any compiled model (see Q-padding below) |
| Slice | dense relation `out(r₀) = Σ_i selector(i)·in(i)` over the **input** domain (`verifier/ops/slice.rs:37-112`) | n_in rounds, degree 2 | none | O(2^n_in) | **Free tie** when `start` is a multiple of the extent: fix the high axis bits. Otherwise one sumcheck with an O(n)-evaluable shift selector |
| Concat | dense relation over the largest operand's domain, `Σ_i selector_i(r′)·in_i(r′[..n_i])`; smaller operands are lifted by repetition (`verifier/ops/concat.rs:37-157`; `core/ops/concat.rs` `extend_input_to_max_domain`) | max_i n_i rounds, degree 2, k product terms | none | O(k·2^n) | **Free tie** when every operand sits at a dyadic-aligned offset: `out(r) = Σ_i eq(r_axis_hi, prefix_i)·in_i(r_rest)` |
| GatherSmall / GatherLarge | read sumcheck `ra(ρ‖r_idx)·(dict(ρ‖r_word) + γ·id(ρ))`, log2(dict rows) rounds, degree 2 (`verifier/ops/gather/mod.rs:42-129`), plus one-hot checks: Small = hamming-booleanity + booleanity batch, then hamming weight (`gather/small.rs:25-63`); Large = ra-virtualisation + booleanity (`gather/large.rs:21-57`) | yes, 2–3 sumcheck stages | `GatherRa(node)` (small) or `GatherRaD(node, d)` chunks (large) (`small.rs:13-15`, `large.rs:12-18`) | succinct apart from the dictionary constant | A genuine lookup; not free in general. Free only for constant indices (see Q-proposals, P8) |
| Add / Sub | (1) clamp "lookup" proving `out = SatClamp(acc)`, deferred into packed buckets; (2) zero-round accumulation tie `acc(r) = left(r) ± right(r)` (`verifier/ops/add.rs:29-52,84-148,186-209`). Scalar nodes skip (1) | the tie itself: none. The clamp: one "split" sumcheck plus one booleanity sumcheck **per bucket**, shared across all clamped nodes of that width (`verifier/global_clamp.rs:1-18`, `clamp_split.rs:1-52`) | 8 dense 4-bit output chunks + w/4 mostly-empty slack chunks per bucket; Add's bound is `2·I32_MAG` (`atlas-trace/src/op/linear.rs:55-57`), which `ClampWidth::for_bound` maps to W40 (`atlas-trace/src/model.rs` test `width_rounding_matches_old_crate`), so 10 slack chunks | succinct | The tie is already free. The clamp is the entire cost and can only be removed by changing semantics or by static range analysis (Q-proposals P3/P4) |
| Sum | (1) same clamp as Add; (2) axis-reduction sumcheck `acc(r) = Σ_x operand(x, r)`: log2(axis extent) rounds, **degree 1** (`verifier/ops/sum/mod.rs:28-96`, `sum/axis.rs:34-95`) | yes, short | clamp bucket chunks only | succinct | The axis sumcheck is near-minimal (a degree-1 sumcheck over only the summed axis). The clamp is the cost |
| Iff | eq-weighted relation `eq(r₀,r′)·(mask·a + (1−mask)·b)`, n rounds, degree 3 (`verifier/ops/iff.rs:352-409`) | yes | none | succinct | Needs a sumcheck when the mask is a runtime tensor. Free when the mask is a constant (then linear in a, b) |
| And | `ProductRelation<And>`: boolean AND as a product (`verifier/ops/and.rs:9-12`), n rounds, degree = factors + 1 = 3 (`verifier/ops/mul.rs:116-122`) | yes | none | succinct | Genuinely multiplicative; a sumcheck is needed unless an operand is constant |

Cost of the dense structural sumchecks
- [CODE] The engine prover for a dense relation materialises every witness and every selector table as full field-element `MultilinearPolynomial`s (`core/relation.rs:139-150`), then runs `dense_message` per round and `bind_parallel` on every wire (`core/relation.rs:167-176,213-221`). For Concat that is k witness polys + k selector tables, all of size 2^max_n.
- [TRACE] Old Qwen run, prover: `Reshape::prove` 0.232 s over 147 nodes, `Slice::prove` 0.137 s over 96, `Concat::prove` 0.104 s over 49: 0.47 s in total, against `ONNXProof::iop` 17.06 s and `ONNXProof::prove` 50.3 s. Verifier: Reshape 0.123 s, Concat 0.064 s, Slice 0.056 s (0.24 s of 9.37 s).
- [TRACE] Tie ops in the same run: `MoveAxis::prove` 0.2 ms over 48 nodes, `Neg::prove` 6 ms over 48, `Broadcast::prove` 23 ms over 484, `IsNan::prove` 2 ms over 24, `Add::prove` 155 ms over 241 (tie + bookkeeping; the clamp itself is in `deferred_lookups::prove_all`, 9.7 s for all lookups and clamps together).

Op frequencies
- [TRACE] Lowered op counts in the old Qwen run (2172 nodes): Broadcast 484, Constant 322, Add 241, Einsum 218, Reshape 147, Rsqrt-related 98, Slice 96, Concat 49, MeanOfSquares 49, MoveAxis 48, Neg 48, Iff 48, IsNan 24, Softmax 24, Sigmoid 24, Input 3, Identity 2, And 1, GatherLarge 1, GatherSmall 1, ScalarConstDiv 1. A further 824 spans are named just `prove` (ops whose `#[tracing::instrument]` has no name: Mul/Sub/Square/etc.), so their split is unknown.
- Raw ONNX node histograms of the files currently in `models/` (counted with the repo's `.venv` `onnx` package; **pre-lowering**, so most Constant/Shape/Gather/Unsqueeze/Concat nodes are shape arithmetic that constant folding removes):
  - gpt2 (2649 nodes): Constant 1024, Unsqueeze 268, Shape 250, Gather 190, Concat 173, Reshape 150, Mul 74, Slice 73, Add 64, Transpose 60, Squeeze 49, Gemm 48, Sqrt 36, Cast 32, Where 25, LayerNormalization 25, MatMul 25, Split 12, Div 12, Softmax 12, IsNaN 12, Pow 12, Tanh 12, Range 3, ConstantOfShape 2, And 2, LessOrEqual 1, Flatten 1, Equal 1, Expand 1.
  - gpt2-tiny (553): Constant 204, Unsqueeze 58, Shape 50, Gather 40, Concat 33, Reshape 30, Identity 14, Add 14, Mul 14, Slice 13, Cast 12, Transpose 10, Squeeze 9, Gemm 8, Sqrt 6, Where 5, LayerNormalization 5, MatMul 5, Softmax 2, IsNaN 2, Split 2, …
  - qwen2-tiny (626): Constant 221, Unsqueeze 78, Mul 39, Shape 38, Gather 33, Cast 30, Concat 24, Add 22, Reshape 20, MatMul 20, Transpose 11, Where 10, Slice 10, Div 9, Identity 8, Expand 6, Neg 4, Softmax 2, IsNaN 2, Sigmoid 2, Cos 1, Sin 1, …
  - bge-tiny (277): Constant 74, Add 32, Identity 19, Unsqueeze 16, MatMul 16, Mul 14, Sqrt 11, ReduceMean 10, Concat 9, Reshape 9, Transpose 8, Sub 6, Slice 3, …
  - bge-small-en-v1.5 (1248): Constant 344, Add 172, MatMul 96, Mul 74, Unsqueeze 66, Sqrt 61, ReduceMean 50, Div 49, Concat 49, Reshape 49, Transpose 48, Sub 26, Slice 13, Softmax 12, Erf 12, …
  - op_gauntlet (87): Add 20, Mul 10, MatMul 9, Reshape 5, Transpose 5, …

Timings recorded in docs
- [CODE/doc] README GPT-2 on Akita, M3: commit phase 25.2 s, IOP 5.28 s, opening reduction + PCS openings 6.72 s, prove 40.6 s, verify 2.21 s (`README.md:69-75`). "The commitment phase dominates (24.4 s), and two of the six 2^24-variable one-hot groups take about 10 s each" (`docs/upstream-inventory.md:156-158`).
- [CODE/doc] The op disposition list matches the table above: ties = Input, Constant, Identity, Neg, IsNan, Broadcast, MoveAxis; accumulation ties = Add, Sub; dense engine relations = Reshape, Slice, Concat, Sum (`docs/upstream-inventory.md:117-121`).

### Inferences
- [EST] On the IOP side the structural sumchecks are small (about 1% of prove time and 3% of IOP time in the old Qwen trace). Making them free is worth doing mainly for (i) verifier succinctness: today the verifier does O(tensor size) field work per Reshape/Slice/Concat; (ii) proof size: each removes an n-round degree-2 sumcheck proof; (iii) code deletion: three relations, three selector builders and the dense-`Table` verifier path would lose their main users.
- [EST] The dominant cost attributable to "linear" ops is the Add/Sub/Sum saturating clamp, because it lands in the commitment phase (62% of GPT-2 prove time per the README table), not in the IOP. Each clamped output element contributes 8 dense one-hot chunk entries to a bucket. I could not attribute commit seconds to Add specifically (see Gaps).

### Gaps
- Current lowered op histograms for the zoo could not be produced: that needs `cargo run … --example dump_lowered_ops`, which was out of bounds, and no dump output is checked in. The raw-ONNX counts above overstate structural ops; the old-trace counts are for one model on the old importer.
- No per-node clamp attribution exists in the trace (clamps are bucketed and deferred), so the share of commit time caused by Add/Sub/Sum versus Einsum/Mul clamps is unknown.
- The old trace does not record tensor sizes per span, so per-element costs could not be derived.

---

## Q1. Concat with non-power-of-two or unequal operand sizes under per-dimension padding: what does the protocol do, what is minimal, and when is a sumcheck genuinely needed? (Includes the reshape/slice padding analysis.)

### Takeaway
The compiler already rejects every concat, slice and reshape whose padding would land in the interior, so every instance that reaches the prover is either exactly free (reshape: identity; aligned slice/concat: fix selector bits) or a shift by a non-aligned offset. A sumcheck is genuinely needed only for the non-aligned shift case, and even there the selector is a structured "shifted eq" the verifier can evaluate in O(n) rather than O(2^n).

### Cited Findings

What the compiler admits
- [CODE] Concat: every operand except the last must be a power of two on the concat axis, and the padded operand extents must sum to the padded output extent; otherwise compile fails (`atlas-trace/src/op/structural.rs:389-419`; test `concat_interior_padding_rejected`, `atlas-trace/src/model.rs:351-378`). `models/reject_concat` and `models/reject_slice` exist as rejection fixtures.
- [CODE] Slice: the extent `end − start` must be a power of two; `start` is **not** required to be aligned (`structural.rs:330-341`).
- [CODE] Reshape: after dropping size-1 axes, the reshape is factored into balanced groups and every non-leading axis of a multi-axis group must already be a power of two on both sides; the padded element counts must match (`structural.rs:24-85,191-212`). The code comment states the consequence: "the original values occupy the same flat positions in the padded corner as in the padded target, with padding confined to the tail" (`structural.rs:192-198`). Tests: `[1,7,8]→[7,8]` accepted, `[2,7]→[14]` rejected (`model.rs:381-406`).
- [CODE] The importer avoids the rejected patterns upstream: `slice_through_projection` rewrites `Slice(x·W+b)` into a projection against sliced weights "and it is what keeps slice extents like 768 out of the padded graph" (`atlas-onnx-import/src/passes.rs:958-964`); `repeat_to_gather` replaces a Broadcast+Reshape that merges a non-power-of-two repeat axis by a gather (`passes.rs:1267-1278`).

What the current Concat protocol does
- [CODE] One dense sumcheck over the largest operand's domain. For operand i it builds `selector_i` of length 2^max_n holding `eq(r_out, output_cell)` at each live input cell (`verifier/ops/concat.rs:183-231`), and lifts the operand's witness to 2^max_n by repeating each value (`core/ops/concat.rs`). The final claim is `Σ_i selector_i(r′)·in_i(r′[..n_i])`; each operand is opened at a **prefix** of the same sumcheck point (`concat.rs:138-145`). Unequal operand sizes are thus handled by lifting, and non-power-of-two sizes by leaving padding cells of the selector at zero.

### Inferences
- [EST] **Reshape is exactly free in a compiled model.** Given the coherence check, the padded input and padded output are the same flat vector of length 2^n in row-major order, so `out~(r) = in~(r)` with the same coordinate order: a tie with `rounds = 0`, the same shape as `Identity`. The n-round sumcheck, the selector build on both sides and the O(2^n) verifier work can all go. Precise freeness condition = exactly `check_pad_coherent_reshape`: padding breaks freeness when a non-power-of-two axis is merged as a non-leading member of a group (e.g. `[2,7]→[14]`: padded `[2,8]` puts a zero at flat index 7, while padded `[16]` wants element 7 there). Those cases are a non-affine index map and would need a sumcheck with a mixed-radix selector; the compiler rejects them today.
  - Soundness caveat: the tie must only be used when the model was compiled with padding (or when the coherence predicate holds on raw dims). With `pad_pow2 = false` test models the selector relation is still needed, or the verifier must re-run the coherence predicate and fall back.
- [EST] **Aligned Slice is free.** If the extent is L = 2^ℓ on an axis of padded size 2^a and `start = j·L`, then `out~(r_before, r_axis[ℓ bits], r_after) = in~(r_before, bits(j), r_axis, r_after)`: one claim on the operand at a point that fixes a−ℓ coordinates to the constant bits of j. Zero rounds, no selector. Variable order needs care: the fixed bits are the high-order bits of the axis group in the big-endian layout used by `coord_to_linear`.
- [EST] **Aligned Concat is free.** If operand i occupies `[o_i, o_i + L_i)` on the axis with `L_i = 2^ℓ_i` and `o_i` a multiple of `L_i`, then `out~(r) = Σ_i eq(r_axis_hi(i), bits(o_i / L_i)) · in_i~(r_before, r_axis_lo(i), r_after)`. One claim on `out` becomes one claim per operand with no sumcheck; the verifier computes k small eq factors (O(a) each). For two equal halves this is exactly `out(r, b) = (1−b)·A(r) + b·B(r)`. The compiler's rule (all but the last operand power-of-two, padded extents summing to a power of two) gives alignment in the common cases (`[4]+[3→4]`, `[8]+[4]+[4]`, `[4]+[4]+[8]`), but **not always**: `[4]+[8]+[4]` passes the check with the middle operand at offset 4, which is not a multiple of 8. That case needs either a sumcheck or a split of the misaligned operand into aligned dyadic pieces.
  - Unlike the current protocol, the free version opens each operand at a point derived from `r` rather than at a fresh sumcheck point. That is sound for the same reason the MoveAxis/Broadcast ties are: both sides are multilinear and agree on the hypercube.
- [EST] **When a sumcheck is genuinely needed:** when the index map from output cells to operand cells is not "fix some bits, permute the rest", i.e. (a) Slice with `start` not a multiple of the extent, (b) Concat with a misaligned operand, (c) pad-incoherent Reshape, were it ever admitted. The minimal protocol there is one degree-2 sumcheck over the operand domain, `out(r) = Σ_x in(x)·S(r, x)`, where `S(r, x) = eq(r, x − c)` is an addition-by-constant predicate. `S`'s multilinear extension is evaluable in O(n) by a carry-bit recursion (same family as GKR wiring predicates), so the verifier need not build a table. An alternative for (a)/(b) without any sumcheck is to cut the window into at most ~2·a aligned dyadic blocks, giving that many claims on the operand; that trades a sumcheck for claim fan-out and is only attractive for small a.
- [EST] Unequal operand sizes do not by themselves force a sumcheck: in the aligned formula each operand is evaluated at its own-length sub-point.

### Gaps
- I did not find how often misaligned slices/concats occur in the lowered zoo graphs (needs the lowered dump). In transformer graphs the usual sources are head splits and KV-cache concats, which are typically aligned, but that is unverified here.
- The exact variable ordering (which coordinates of `r` are the axis's high bits) must be read off `coord_to_linear` and `OpeningPoint<BIG_ENDIAN>` when implementing; I did not derive it line by line.

---

## Q2. Claim fan-out: what does reducing k claims on one tensor to one cost today, and is there a cheaper standard approach?

### Takeaway
Fan-out is merged per node with the textbook GKR line-restriction trick generalised to N points (no sumcheck, one univariate of degree n·(N−1)), which is asymptotically fine at O(N²·2^n) prover work but is implemented sequentially over heap-allocated per-coefficient polynomials and cost 1.88 s in the old Qwen trace, more than all structural sumchecks combined. The cheaper standard alternative is to fold the random linear combination of the k claims into the producer's own sumcheck (the Libra/zkCNN style), which removes the extra pass entirely for sumcheck-proven producers.

### Cited Findings
- [CODE] Mechanism: for claims `P(r_1)=v_1 … P(r_N)=v_N`, the prover sends `h(t) = P(l(t))` where `l(i) = r_i`; the verifier checks `h(i) = v_i`, samples `x′`, and continues with the single claim `P(l(x′)) = h(x′)` (`atlas-sumcheck/src/evaluation_reduction/mod.rs:1-13,149-209`). The verifier enforces `deg h ≤ n·(N−1)` (`mod.rs:181-188`).
- [CODE] One claim is a short path: no work, no transcript interaction (`mod.rs:109-122,165-175`; prover skips even the field conversion, `core/ops/eval_reduction.rs:30-41`). The comment there says single-consumer nodes are "~2k" of the nodes.
- [CODE] Multi-claim prover cost: the output tensor is converted to a full-field MLE (`core/ops/eval_reduction.rs:43-48`), then `compute_h` turns each of the 2^n coefficients into its own `UniPoly` (a heap `Vec`) and folds variable by variable with polynomial multiplications in a plain sequential `for` loop (`mod.rs:222-248`). No rayon, no small-scalar path.
- [CODE] It runs for every node, including Constant and Input nodes with several consumers (the dispatcher calls `NodeEvalReduction::prove` for every op under the default flow, `core/ops/mod.rs:231-238`), although the verifier could evaluate public data at each point directly.
- [CODE] Claims are collected per `(NodeOutput(node), consumer)` key (`atlas-sumcheck/src/opening_proof/mod.rs:229-244`); identical points from two consumers are not deduplicated before the reduction.
- [TRACE] `NodeEvalReduction::prove`: 1.881 s over 2172 calls; only 191 calls exceed 1 ms and they account for 1.869 s; the largest are 112 ms, 103 ms, 37 ms. That is 3.7% of the 50.3 s prove and 11% of the 17.06 s IOP in that run.
- [LIT] The line trick is the standard GKR "reducing to verification of a single point" step: the prover sends a univariate `q` of degree at most s_{i+1} claimed to be the restriction of W̃ to the line through the two points; V checks `q(0)`, `q(1)`, picks random r*, continues at `ℓ(r*)` — [Thaler, GKR lecture notes](https://people.cs.georgetown.edu/jthaler/gkrnotes.pdf).
- [LIT] GKR-style zkML descriptions use the other standard option: the two claims on the next layer "are combined into a single claim via random linear combination" — [zkComposer, arXiv 2607.08095](https://arxiv.org/pdf/2607.08095) (as summarised by search; I did not read the full paper).
- [LIT] DeepProve's overview says claims are accumulated so as "to only produce/verify ONE opening proof over a large polynomial" — [deep-prove docs overview](https://raw.githubusercontent.com/Lagrange-Labs/deep-prove/master/docs/src/overview.md). The mechanism is not described on that page.

### Inferences
- [EST] Operation count of the current fold: at variable i there are 2^(n−i−1) pairs, each multiplying a degree-(N−1) polynomial by one of degree i·(N−1); summing gives roughly N²·2^n full-field multiplications (about 4·2^n for N = 2, about 9·2^n for N = 3), plus 2^n small heap allocations. Proof size is n·(N−1)+1 field elements (for N = 2 that is smaller than a degree-2 sumcheck's ~2n).
- [EST] Options, cheapest first:
  1. **Fold the RLC into the producer's sumcheck.** If the producer is proven by an eq-weighted sumcheck whose input claim is `out(r)` (Einsum, Mul, Iff, lookups, the clamp buckets), replace `eq(r, x)` by `Σ_i γ_i·eq(r_i, x)` and the input claim by `Σ_i γ_i·v_i`. No extra rounds, no extra proof elements, about (N−1)·2^n extra multiplications to build the combined eq table. This is how RLC-based GKR variants avoid the line step. Cost: the Gruen split-eq optimisation assumes a single eq point, so a multi-point producer needs either a dense eq table or N split-eq instances; this touches the hot loops (`atlas-algebra/src/poly`), so it must be profiled. The packed clamp buckets already use a cycle weight `Σ_n γ_n·eq(r_n, t − off_n)` (`verifier/global_clamp.rs:9-12`), so they take extra points per node naturally.
  2. **Pass multiplicity through ties.** A tie producer (Identity, Neg, MoveAxis, Broadcast, free Reshape/Slice/Concat) can map each of its k output claims to k operand claims with no reduction, deferring the merge to the next sumcheck producer. This must be bounded: through a residual chain `x_L = x_{L−1} + f(x_{L−1})` multiplicity grows by one per layer, so an unconditional pass-through gives O(depth) claims per tensor. A cap (merge when k exceeds a small threshold) keeps it sound and bounded.
  3. **Skip the reduction for public tensors.** For Constant/Input with k consumers the verifier can evaluate the public MLE at k points: k−1 extra O(2^n) verifier evaluations in exchange for removing an O(N²·2^n) prover fold over what are often the largest tensors (weights). Whether that is a win depends on which side is the bottleneck; with `Constant::verify` already 31% of verify time in the old trace, a better variant is to batch: evaluate the constant once against `Σ γ_i·eq(r_i, ·)`, which is a single pass.
  4. **Keep the line trick but fix the implementation**: evaluate h at n·(N−1)+1 points with a parallel, allocation-free fold using small-scalar (i32) arithmetic for the first variable; deduplicate identical points first. Same protocol, same proof, no soundness change.
  5. **Model-wide batching** (one sumcheck over all nodes of equal size): the reverse-topological order means a node's reduced point is needed before its own op is proven, so only nodes whose consumers are all finished can share a batch. That needs a staged schedule and gives little beyond option 1.
- [EST] Structural rewrites reduce fan-out at the source: every eliminated IsNan node removes one claim on its operand; folding bias Adds into the einsum removes the einsum output as a separately-claimed tensor.

### Gaps
- No current (Akita-era) measurement of eval-reduction time exists in the repo; the 1.88 s figure is from the Oct 20 Dory-era trace.
- The distribution of N (how many nodes have 2, 3, more claims) and of n for multi-claim nodes was not extractable from the trace (spans carry no arguments).
- I did not verify in the zkCNN or Libra PDFs how exactly they fold multiple claims (fetches returned 403 or unreadable PDF streams); the RLC-into-next-sumcheck description is from the zkComposer summary and general background.

---

## Q3. Which of these ops commit witness polynomials? Are intermediate tensors committed? What is committed overall and what fraction is avoidable?

### Takeaway
No intermediate tensor is committed (they are GKR-style virtual polynomials), and of the ops in scope only Gather and the saturating-clamp side of Add/Sub/Sum cause commitments. Since the commitment phase is about 62% of GPT-2 prove time, the avoidable commitment attributable to this op family is the clamp range proof on Add/Sub/Sum outputs.

### Cited Findings
- [CODE] `OpLayout::get_committed_polynomials` is empty by default: Broadcast, Constant, Input, and the tie/standard-sumcheck ops declare `impl OpLayout for X {}` (`verifier/ops/broadcast.rs:30`, `constant.rs:30`, `input.rs:16`).
- [CODE] GatherSmall commits one `GatherRa(node)` polynomial; GatherLarge commits `GatherRaD(node, d)` for d < `instruction_d` (`verifier/ops/gather/small.rs:12-16`, `gather/large.rs:11-19`).
- [CODE] Add/Sub/Sum return `clamp_committed_polys(node)`, which is empty: "A clamped node commits no per-node polynomials: its clamp one-hots live in a packed bucket" (`verifier/clamp_lookups/mod.rs:266-272`). Bucket polys are added at model level (`core/prover.rs:319-322`).
- [CODE] Bucket contents per cycle: 8 dense 4-bit one-hot chunks of `OUT = out + 2^31`, a virtual saturation indicator, and w/4 sparse slack chunks, with w ∈ {40, 48, 56, 64} (`verifier/clamp_split.rs:11-18,96-118`). Nodes are packed first-fit-decreasing into buckets of capacity equal to the largest clamped node, grouped by width (`verifier/global_clamp.rs:73-84,107-115`).
- [CODE] Every non-scalar Add, Sub and Sum is clamped: their `clamp_bound` is unconditionally `Some(...)` and ignores operand information (`atlas-trace/src/op/linear.rs:55-57,85-87,186-195`). `clamp_buckets` includes every node with a bound (`global_clamp.rs:76-84`).
- [CODE] One exemption exists: an output node whose public output is unsaturated checks `acc == out` in the clear and skips the bucket (`verifier/clamp_lookups/mod.rs:242-259`); scalar nodes are also exempt (`clamp_lookups/mod.rs:44-54`).
- [CODE] Add is semantically saturating per step: `sat_chain` clamps each pairwise sum back to i32 (`atlas-trace/src/op/linear.rs:21-32`); the contract "Quantized values are i32" is in CLAUDE.md.
- [CODE/doc] Commit phase 25.2 s of 40.6 s prove on GPT-2/Akita (`README.md:71-74`).
- [LIT] zkLLM commits the model parameters and the auxiliary tensors of its lookup arguments (S, m, A, B per tlookup), proving arithmetic tensor operations by sumcheck down to the prompt and parameters — [zkLLM, arXiv 2404.16109](https://arxiv.org/pdf/2404.16109) §3.4, §4, §6.2.
- [LIT] DeepProve's setup "commits to all the parameters, e.g. the weights, of the model"; the overview does not say intermediate outputs are committed — [deep-prove docs overview](https://raw.githubusercontent.com/Lagrange-Labs/deep-prove/master/docs/src/overview.md).

### Inferences
- [EST] jolt-atlas commits strictly less than those systems on the model side (weights are public and verifier-evaluated, never committed) and the same kind of thing on the non-linear side (lookup/range-check auxiliaries). All commitment in the scoped op family is range-proof material.
- [EST] Fraction avoidable: per clamped element the dense commitment is 8 one-hot chunk entries regardless of which op produced it, so Add/Sub/Sum's share of dense clamp commitment equals their share of clamped elements. In a transformer block the clamped tensors of comparable size are roughly: residual adds (2), bias adds (3–6, architecture dependent), norm-related Sub/Add, versus Einsum/Mul outputs. A plausible range is that Add/Sub/Sum own 30–50% of clamped elements, hence a similar share of the clamp-bucket commit work; this is a guess to be replaced by a count from the lowered graph. The bias-add part of it is removable with no loss of soundness (P3 below); the residual-add part only with a semantics change (P4).

### Gaps
- No measurement splits the 25.2 s commit phase between clamp buckets, remainder buckets and activation lookups, nor between producing ops. The 30–50% figure above is unverified.
- I did not read `core/witness.rs` end to end, so there may be additional per-node committed advice (e.g. ScalarConstDiv's remainder is mentioned at `docs/upstream-inventory.md:130`) outside this op family.

---

## Q4. What do GKR-based zkML systems do for wiring / reshape / concat, and what applies here?

### Takeaway
I could verify only the generic picture: GKR systems express data movement as wiring predicates that the verifier evaluates itself (cheaply when the wiring is structured), intermediate layers are never committed, and multiple claims are merged by a line restriction or a random linear combination. I found no primary-source statement on how zkCNN, zkLLM, zkGPT, DeepProve, Remainder or Expander treat reshape/concat specifically.

### Cited Findings
- [LIT] In GKR the verifier must evaluate the wiring predicates' extensions `add~_i`, `mult~_i` at one random point per layer; these "only depend on the wiring of the circuit, and not on the input", can be done offline, and for data-parallel circuits take time proportional to the sub-computation size and logarithmic in the number of copies — [Thaler, GKR lecture notes](https://people.cs.georgetown.edu/jthaler/gkrnotes.pdf) §4.2 region.
- [LIT] Wiring predicates are Boolean selectors that are 1 only where a wire connects a gate to its inputs in the layer below — [zkComposer, arXiv 2607.08095](https://arxiv.org/pdf/2607.08095) (via search summary).
- [LIT] zkLLM assembles sumcheck proofs "in reverse logical order of the arithmetic circuits, methodically reducing the claimed multilinear extension values of the output y to those associated with the prompt X and the model parameters W", and treats multi-head attention as per-head attention followed by concatenation (its Eq. 4) — [zkLLM, arXiv 2404.16109](https://arxiv.org/pdf/2404.16109) §2, §6.2.
- [LIT] zkCNN's contribution is a sumcheck for FFTs/convolutions with linear prover time; its abstract does not discuss reshape/concat — [zkCNN, ePrint 2021/673](https://eprint.iacr.org/2021/673).
- [LIT] Virgo++ generalises linear-prover GKR "to arbitrary circuits", i.e. layers reading from any earlier layer — [ePrint 2020/1247](https://eprint.iacr.org/2020/1247) (abstract only).
- [LIT] DeepProve (Lagrange) replaced its bespoke linear layers with explicit einsum instantiations, including QKV projections, and uses specialised sumchecks for linear operators with LogUp-GKR lookups for non-linear ones — [Lagrange engineering update Q4 2025](https://lagrange.dev/engineering-updates/lagrange-engineering-update-q4-2025) (via search summary; the page itself returned 403 to direct fetch).
- [LIT, secondary] A blog summary says DeepProve introduced a "ConcatMatMul" layer for multi-head attention, adapting zkLLM techniques — [dev.to summary](https://dev.to/extropy/the-zkml-singularity-a-comprehensive-analysis-of-the-2025-cryptographic-convergence-iln). Secondary source, no protocol detail; treat as unconfirmed.

### Inferences
- [EST] The jolt-atlas tie relations are the "wiring predicate the verifier evaluates in O(log n)" idea specialised to the case where the predicate is a bit-permutation or bit-fixing, so that no sumcheck is needed at all. The existing MoveAxis and Broadcast ties already do this; Reshape/Slice/Concat are the three places where the code still uses the generic "sumcheck against a materialised predicate table" form, and with an O(2^n) rather than O(n) verifier evaluation of the predicate.
- [EST] What transfers: (i) evaluate structured selectors in closed form (eq on fixed bits, shifted-eq by carry recursion) instead of from a table; (ii) fuse data movement into the neighbouring arithmetic layer (the "ConcatMatMul" idea, and DeepProve's everything-is-an-einsum direction, correspond to pushing MoveAxis/Reshape/Concat into einsum index strings here); (iii) RLC claim merging inside the next sumcheck.

### Gaps
- No primary source was readable for zkGPT, Modulus Remainder or Expander on this topic; I make no claims about them.
- zkCNN and zkLLM PDFs could not be searched for reshape/concat handling beyond the passages quoted (zkLLM text was extracted locally; it has no dedicated treatment of reshape/transpose that I could find by keyword). "No reliable source found" for per-layer cost of reshape/concat in any of these systems.
- DeepProve's `layers.md` lists layers by name without describing reshape/concat/add proofs; `commitments.md` (accumulation) was not fetched.

---

## Q5 (objective c). Ranked proposals

### Takeaway
The highest-value items are the ones that remove committed range-proof material or whole nodes (bias folded into the einsum, dead IsNan/Where chains, constant-mask Iff), followed by turning Reshape/Slice/Concat into ties (verifier succinctness and proof size more than prover time) and fixing the eval-reduction implementation. All savings figures are estimates anchored to the stale Oct 20 trace or to the README stage table.

### Cited Findings (code facts each proposal rests on)
- [CODE] Gemm bias is lowered as a separate `Add` after the einsum (`atlas-onnx-import/src/passes.rs:941-946`), and `lower.rs` inserts explicit `Broadcast` nodes so elementwise ops get equal shapes (`atlas-onnx-import/src/lower.rs:6,180`). The einsum is a fused-rescale op: `acc = quotient·D + remainder`, output `SatClamp(quotient)` (CLAUDE.md behavioural contract; `verifier/global_clamp.rs:86-94`).
- [CODE] `IsNaN` lowers directly to `Op::IsNan` and `Where` to `Op::Iff` (`atlas-onnx-import/src/lower.rs:248-250`); constant folding handles them only for constant inputs (`fold_fns.rs:247,635-650`). The prover's relation states "Fixed-point data is never NaN, so `IsNan` outputs zeros" (`verifier/ops/is_nan.rs:34`).
- [CODE] Existing importer passes: `elide`, `reshape_family`, `split_to_slices`, `matmul_gemm_to_einsum` (absorbs only a last-two-axes Transpose on the RHS), `slice_through_projection`, `transpose_to_moveaxis` ("a minimal chain of atlas.MoveAxis nodes"), `expand_to_broadcast`, `repeat_to_gather` (`atlas-onnx-import/src/passes.rs:89,138,165,769,964,1161,1226,1278`). There is no pass that fuses Reshape/MoveAxis chains or pushes general permutations into einsum equations; non-canonical layouts "get explicit Reshape/MoveAxis nodes instead of exotic equations" (`passes.rs:766-768`).
- [CODE] Known open gap: Broadcast into a non-power-of-two axis fills padded lanes with live values (`atlas-trace/src/model.rs:148-156`, ignored test `broadcast_fill_inflates_padded_reduction`).

### Inferences (the ranked list; all [EST])

**P1. Reshape → tie (`out(r) = in(r)`).** 
- Change: `ReshapeRelation` becomes a zero-round tie whenever the model is pad-compiled; delete `build_reshape_selectors` from the hot path.
- Savings: removes an n-round degree-2 sumcheck per Reshape (147 nodes in the old Qwen trace: 0.23 s prover, 0.12 s verifier) and the O(2^n) verifier table. Proof shrinks by ~2n field elements per node.
- Soundness: exact, by the invariant `check_pad_coherent_reshape` already enforces. The verifier must not trust the invariant from the prover; it should either re-check the predicate on the model's dims or the tie must be selected by a model-level flag the verifier derives itself. Keep the selector relation (or reject) for unpadded test models.
- Follow-on: with Reshape a tie, the importer can elide Reshape nodes whose padded flat layout is unchanged, or the claim router can treat them as aliases (no node, no accumulator entry).

**P2. Aligned Slice / Concat → ties; closed-form selector otherwise.**
- Change: Slice with `start % extent == 0` and Concat with dyadic-aligned operands become ties (formulas in Q1). For the misaligned remainder, keep one sumcheck but evaluate the selector by an O(n) shifted-eq formula in the verifier.
- Savings: old trace 0.24 s prover + 0.12 s verifier over 145 nodes; removes all O(2^n) verifier tables for these ops. For Concat it also removes the k lifted witness copies.
- Soundness: exact for aligned cases (multilinear identity). The alignment predicate is public model data. For Concat the tie opens operands at points derived from r, which changes which claims reach the operands (and therefore transcript order): a protocol-level change needing the zoo tests and a real-model run per CLAUDE.md.
- Option: tighten the compiler to reject misaligned concat/slice outright (as it already rejects interior padding) so that no sumcheck variant has to be maintained; check first whether any zoo model needs them.

**P3. Fold bias into the einsum relation (EinsumBias).**
- Change: importer fuses `Einsum → Add(const bias)` into one op with `out = SatClamp(floor((Σ a·b + bias·2^s) / 2^s)) = SatClamp(quotient + bias)`; the remainder is unchanged because `bias·2^s` is a multiple of the divisor. In the relation, the bias enters the accumulation identity as a public term the verifier evaluates (`bias~(r)` broadcast over the non-bias axes).
- Savings per biased linear layer: one Add node, one Broadcast node, one Constant claim path, one clamp-bucket segment of the output size (8 dense one-hot chunks per element of commitment plus its share of the split and booleanity sumchecks), and one level of claim routing. GPT-2 has 48 Gemm nodes in the raw ONNX graph, so up to 48 output-sized clamp segments disappear. This is the only proposal here that directly reduces the commitment phase without touching the i32 contract.
- Soundness/semantics: the fused op differs from the two-op chain only when the einsum's own clamp would have saturated before the bias is added (`SatClamp(SatClamp(q)+b)` vs `SatClamp(q+b)`). The team owns the tracer, so this is a semantics definition, but it changes golden hashes and must be reflected in `atlas-trace` semantics, the clamp bound (`n·2^62 + |bias|`), and the op-surface table. The clamp width may step up one class for extreme biases.
- Same idea applies to `slice_through_projection`'s rebuilt `bias_add` nodes (`passes.rs:1135-1142`).

**P4. Remove clamps on Add/Sub where a static bound proves non-saturation; consider lazy clamping.**
- Change (safe part): make `clamp_bound` for Add/Sub use operand bounds (it already receives `BoundCtx` with constants, `verifier/node_view.rs` `clamp_value_bound`) and extend the analysis to carry per-node magnitude bounds (lookup outputs bounded by their tables, constants exactly, softmax outputs ≤ scale). When `|left| + |right| < 2^31` is provable, the Add is an exact field identity and needs no clamp: it becomes a pure tie with zero commitment.
- Change (aggressive part): let the tracer carry values wider than i32 between clamps (Add exact in i64 with a tracked static width; clamp only when the width budget is exhausted or before a lookup). 
- Savings: each Add freed removes an output-sized clamp segment (same unit as P3). Residual-stream adds are the big ones but are **not** statically boundable (both operands are full-range i32 einsum outputs), so the safe part mostly catches eps-adds, mask-adds and adds after bounded activations.
- Soundness: the safe part is sound iff the bound analysis is sound; an unsound bound is a soundness bug, not a completeness bug, because a wrapped value would be accepted. The bound must be derived by the verifier from the model alone. The aggressive part breaks the "quantized values are i32" contract, changes every downstream bound, and lets residual width grow by one bit per add (48 adds in a 24-layer model), so it needs periodic re-clamping; treat as research, not a quick win.

**P5. Kill dead NaN plumbing in the importer.**
- Change: rewrite `IsNaN(x)` to a zero constant (the relation already asserts this), then fold `Where(const-false, a, b)` to `b` and elide.
- Savings: per attention layer removes an IsNan node (and its unused O(2^n) operand evaluation plus the extra claim it places on the softmax output, which forces an eval reduction there) and an Iff sumcheck (n rounds, degree 3, three operand openings) over an attention-score-sized tensor. Raw counts: IsNaN 12 in GPT-2, 2 in the tiny models; the old Qwen trace shows IsNan 24 and Iff 48.
- Soundness: none at risk; the statement proven is unchanged because the trace semantics already define IsNan as zeros. Only do the Where-fold when the mask is provably constant after the rewrite.

**P6. Constant-mask Iff → tie.**
- When the mask operand is a Constant (causal masks), `out = m·a + (1−m)·b` is linear in a and b with public coefficients, but it is still a product of MLEs, so it is not a zero-round tie in general. It is a tie when additionally one branch is a constant (the usual `Where(mask, scores, −inf)` pattern): then `out(r) = (m·a)~(r) + ((1−m)·b)~(r)` still needs `(m·a)~(r)`, a product. So: keep one sumcheck but of degree 2 with a public table instead of degree 3 with three witness polys, or better, fuse the mask into the softmax/einsum consumer. Modest saving; listed for completeness.

**P7. Eval reduction: fix the implementation, then fold into producers.**
- Step 1 (no protocol change): parallel, allocation-free evaluation of `h` using small-scalar arithmetic; deduplicate equal points; skip for public tensors by letting the verifier evaluate against the combined eq. Expected to recover most of the 1.88 s seen in the old trace (the work is ~N²·2^n multiplications, which should be milliseconds at these sizes when vectorised).
- Step 2 (protocol change): RLC of the k claims inside the producer's sumcheck for eq-weighted producers; bounded pass-through for tie producers (Q2). Removes the per-node `EvalReductionProof` for those nodes.
- Soundness: step 1 none. Step 2 is standard (Schwartz–Zippel over γ) but changes transcript order and the claim lifecycle; needs the soundness sweeps, a `mutate::Site` review, and a real-model run. Interaction with the Gruen split-eq kernels must be benchmarked: losing the single-point eq structure on the hottest sumchecks could cost more than the reduction saves.

**P8. Fuse data-movement chains in the importer / claim router.**
- Compose consecutive MoveAxis nodes into one permutation and cancel inverse pairs; with P1/P2, Reshape·MoveAxis·Reshape chains (head split/merge) become a single point transformation.
- Push MoveAxis into einsum index strings beyond the existing last-two-swap case. The einsum registry has a fixed set of specialised equations (`verifier/ops/einsum/{k_nk_n,mk_kn_mn,abmk_abnk_abmn,dot}.rs`), and the memory notes record that non-canonical/dropped-letter equations broke completeness in fuzzing, so this should go through the generic dot engine and its equivalence tests rather than adding ad hoc variants.
- Eliminate Broadcast nodes feeding elementwise ops by letting the consumer's tie/sumcheck open the un-broadcast operand at the sub-point directly (what `BroadcastRelation::split` computes). 484 Broadcast nodes in the old Qwen trace; each is already nearly free to prove (23 ms total) but each costs an operand `evaluate_mle`, an accumulator entry, a transcript append and possible fan-out. Soundness caveat: the open padded-lane gap for Broadcast into non-power-of-two axes is unaffected by this and still needs its own fix.
- Gather with constant indices (position embeddings, `repeat_to_gather` head gathers) is a fixed row selection: provable as a tie/selector against the dictionary with no committed `ra`. Saves the `GatherRa` commitment and three sumcheck stages for those nodes. Sound because the indices are public model data; must check the index tensor really is a Constant and not an Input.
- Savings: mostly node count, proof size and verifier work; small prover time. Risk: low for MoveAxis composition, medium for einsum index changes.

**P9. Small cleanups.**
- Broadcast: `ones~(r_bc)` is identically 1 for power-of-two dims; drop the tensor build and MLE evaluation in both prover and verifier.
- IsNan (if kept): do not open the operand at all; `produced()` can be empty as for `PublicRelation`.
- Dense `Table` derivation in the verifier (`atlas-sumcheck/src/relation/mod.rs:310-311`): after P1/P2 its remaining users should supply closed-form evaluations so the verifier never materialises a 2^n table.

Rough ranking by expected end-to-end effect: P3 (commitment) > P5 (removes whole sumchecks and fan-out, trivial risk) > P7 step 1 (pure implementation) > P1 > P2 > P4 safe part > P8 > P7 step 2 > P9 > P6 > P4 aggressive part.

### Gaps
- None of the savings were measured on the current tree. Before committing to P3/P4 the commit-phase split per bucket member should be measured (one traced GPT-2 run with per-node clamp element counts would settle it).
- The number of biased linear layers and bias-add output sizes in the *lowered* GPT-2/BGE graphs was not counted (needs the dump example).
- I did not check whether `slice_through_projection` plus `split_to_slices` leave any misaligned Slice in the zoo, which decides whether P2's sumcheck fallback is needed at all.
- P6's benefit depends on how masks are lowered in the current importer (constant vs computed from `And`/`LessOrEqual` on inputs); not traced.
