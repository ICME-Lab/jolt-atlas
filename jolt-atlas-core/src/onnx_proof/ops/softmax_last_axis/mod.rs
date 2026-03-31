/// full pipeline specification (WIP)
///
/// # witness quantities
///
/// Input: `[F, N]` logit matrix X, downstream claim on `softmax_q(r)`.
///
/// | Symbol | Description | Shape |
/// |--------|-------------|-------|
/// | `X[k,j]` | input logits | F × N |
/// | `max_k` | max of feature vector k | F |
/// | `e_k[j]` | one-hot indicator at argmax\_k | F × N |
/// | `z[k,j]` | `max_k − X[k,j]` (≥ 0, fed into LUT) | F × N |
/// | `exp_q[k,j]` | `LUT[z[k,j]]` | F × N |
/// | `exp_sum_q[k]` | `Σ_j exp_q[k,j]` | F |
/// | `inv_sum[k]` | `⌊S² / exp_sum_q[k]⌋` (not committed) | F |
/// | `r_inv[k]` | `S² − inv_sum[k]·exp_sum_q[k]` ∈ `[0, exp_sum_q[k])` | F |
/// | `softmax_q[k,j]` | `⌊exp_q[k,j] · inv_sum[k] / S⌋` | F × N |
/// | `R[k,j]` | `exp_q[k,j]·inv_sum[k] − softmax_q[k,j]·S` ∈ `[0, S)` | F × N |
///
/// # stage 1: combined recip-mult + div-consistency sumcheck
///
/// batches two relations into ONE sumcheck over F×N using random α:
///
///   **relation A (recip-mult):**
///     `exp_q[k,j] · inv_sum[k] = softmax_q[k,j] · S + R[k,j]`
///
///   **relation B (div consistency):**
///     `inv_sum[k] · exp_sum_q[k] + r_inv[k] = S²`
///
/// Combined sumcheck proves:
///   `Σ_{k,j} eq(r, (k,j)) · [ exp_q(k,j) · inv_sum(k)
///                             + α · (inv_sum(k) · exp_sum_q(k) + r_inv(k) − S²) ]
///     = softmax_q(r) · S + R(r)`
///
/// `inv_sum`, `exp_sum_q`, `r_inv` are F-dim polynomials that only depend
/// on the leading `log₂(F)` variables — they are naturally multilinear in
/// the full `log₂(F·N)` variables (constant in trailing `log₂(N)` bits).
///
/// the B-part sums to 0 on the Boolean hypercube, so the input claim
/// is just the A-part: `softmax_q(r)·S + R(r)`.  The random α prevents
/// balancing errors between the two relations.
///
/// degree-3 `log_2(F·N)` rounds.
///
/// after the sumcheck, all claims land at ONE point `r_sc` (= r1).
/// Let `r_lead = first log₂(F) components of r_sc`.  The outputs are:
///   - `exp_q(r_sc)` — virtual, verified by Shout (Stage 4)
///   - `exp_sum_q(r_lead)` — virtual, verified by sum-axis sumcheck (Stage 3)
///   - `r_inv(r_lead)` — algebraically determined: `S² − a·b`
///   - `inv_sum(r_lead) = a` — algebraically determined (see below)
///
/// ## why `inv_sum(r_lead)` is sound without a separate Schwartz-Zippel check
///
/// In other ops (MatMul, Mul, etc.), after a sumcheck the prover sends
/// claims on each witness poly and the verifier checks them via PCS
/// opening (Schwartz-Zippel on the commitment).  `inv_sum` has no
/// commitment, so why is the claim `a` trustworthy?
///
/// The verifier's final-round check is:
///   `eq(r, r_sc) · [ exp_q(r_sc) · a
///                   + α · (a · exp_sum_q(r_lead) + r_inv(r_lead) − S²) ]
///     = final_claim`
///
/// This is LINEAR in `a`.  The values `eq(r, r_sc)`, `exp_q(r_sc)`,
/// `exp_sum_q(r_lead)`, `r_inv(r_lead)`, and `final_claim` are all either
/// transcript-derived or independently verified (`exp_q` via Shout,
/// `exp_sum_q` via sum-axis sumcheck, `r_inv` via PCS opening).
/// So there is exactly ONE value of `a` that satisfies the equation —
/// the verifier computes it.  The prover doesn't get to "choose" `a`;
/// it falls out of the algebra.
///
/// Moreover, the combined sumcheck (via Schwartz-Zippel + random α)
/// guarantees that BOTH relations hold pointwise on the Boolean
/// hypercube.  In particular, Relation B gives:
///   `inv_sum[k] · exp_sum_q[k] + r_inv[k] = S²   for all k`
/// Combined with stage 2's range check (`r_inv[k] ∈ [0, exp_sum_q[k])`),
/// this forces `inv_sum[k] = ⌊S² / exp_sum_q[k]⌋` for every k.
/// So the full polynomial is pinned down everywhere, and the claim
/// `a = inv_sum(r_lead)` is consistent with that unique polynomial.
///
/// # stage 1b: Remainder range check
///
/// Proves: `R[k,j] ∈ [0, S)` for all k,j.
/// Shout identity RC over F·N elements, constant 12-bit bound.
/// (This is the payoff: constant bound, no diff polynomial needed.)
///
/// # stage 1c: Max indicator sumcheck (batched over F vectors)
///
/// Proves: `Σ_j X[k,j] · e_k[j] = max_k` for each k.
/// Batched degree-2, ~log₂(N) rounds.
/// Combined with Stage 4 → `max_k` is the TRUE maximum.
/// Outputs claim on: `X(r1)`.
///
/// # stage 2: Inv-sum remainder range check
///
/// need: `r_inv[k] ∈ [0, exp_sum_q[k])` — a variable upper bound.
/// define `diff[k] = exp_sum_q[k] − 1 − r_inv[k]`.
/// then `diff[k] ≥ 0` iff `r_inv[k] < exp_sum_q[k]`.
/// since `exp_sum_q[k] ≤ N·S`, we have `diff[k] ∈ [0, N·S)`.
/// range-check diff into `[0, N·S)` (constant bound, ~`log_2(N·S)` bits).
/// Verifier also checks `diff(r1_leading) = exp_sum_q(r1_leading) − 1 − r_inv(r1_leading)`.
/// (Note: uses `r1_leading`, not a separate r2 — single evaluation point.)
///
/// # stage 3: Sum-axis-1 sumcheck
///
/// proves: `exp_sum_q[k] = Σ_j exp_q[k,j]`.
/// degree-1, ~log₂(N) rounds.
/// outputs claim on: `exp_q(r2)`.
///
/// # stage 4: check exponentiation via decomposed lookups
///
/// xploits the multiplicative structure of exponentiation: `e^{-(a+b)} = e^{-a} · e^{-b}`.
/// split the centerd logit `z[k,j] = z_hi · B + z_lo` into two digits
/// (B = 2^8 for S = 4096), then look up two *small* sub-tables:
///
///   - `exp_hi[k,j] = LUT_hi[z_hi]`  (~146 entries, indexed by `z >> 8`)
///   - `exp_lo[k,j] = LUT_lo[z_lo]`  (256 entries, indexed by `z & 0xFF`)
///
/// total: **~402 entries** vs 65 536 for the flat LUT (163× smaller).
///
/// The product recovers `exp_q` up to a rounding remainder:
///   `exp_hi[k,j] · exp_lo[k,j] = exp_q[k,j] · S + r_exp[k,j]`
/// with `r_exp[k,j] ∈ [0, S)`.
///
/// ## stag 4a: Algebraic operand link
///
/// Verifier checks: `X(r1) = -(z(r1) - max_k(r_leading))`.
/// (No sumcheck — purely algebraic on claims already in the accumulator.)
///
/// ## stage 4b: Multiplication relation sumcheck
///
/// Proves: `exp_q(r1) · S + r_exp(r1) = Σ_(k,j) eq(k,j, r1) · exp_hi[k, j] · exp_lo[k, j] `
/// for all k,j, with range check `r_exp ∈ [0, S)`.
///
/// #### sub-table Shout lookups:
///
/// 2 Shout instasces over F·N elements each, proving output = LUT[address]
/// via committed one-hot read-address (ra) polynomials:
///   1. `exp_hi[k,j] = LUT_hi[z_hi[k,j]]` — Shout over ~146-entry sub-table
///   2. `exp_lo[k,j] = LUT_lo[z_lo[k,j]]` — Shout over 256-entry sub-table
///
/// implicitly proves `z[k,j] ≥ 0` (i.e. `max_k ≥ X[k,j]`) since the
/// sub-table indices are non-negative by construction.
///
/// ## stage 4d: RAF (Read-Address-Field-element) + digit reconstruction
///
/// 2 RAF sumchecks reconstruct the field-element addresses `z_hi`, `z_lo`
/// from the comitted one-hot ra polynomials:
///   `z_hi(r) = Σ_k (Σ_i 2^i · k_i) · ra_hi(k, r)`
///   `z_lo(r) = Σ_k (Σ_i 2^i · k_i) · ra_lo(k, r)`
/// where `k ∈ {0,1}^log₂(K)` and `k_i` is the i-th bit of k.
///
/// the verifier algebraically checks:
///   `z(r1) = z_hi(r1) · B + z_lo(r1)`
/// via Schwartz-Zippel (if this holds at random r1, the polynomials agree
/// everywhere).  The range constraint `z_lo ∈ [0, B)` is implicit in the
/// RAF — the one-hot ra_lo is committed over a table of size B, so z_lo
/// cannot exceed B1.
///
/// ----
/// # design note: reciprocal-multiply vs direct division
///
/// Naive softmax division:
///   `exp_q[k,j] · S = softmax_q[k,j] · exp_sum_q[k] + R_direct[k,j]`
///
/// with `R_direct[k,j] ∈ [0, exp_sum_q[k])` (variable bound).
/// Requires variable-bound RC over F·N elements:
///   `diff[k,j] = exp_sum_q[k] − 1 − R_direct[k,j] ∈ [0, N·S)`
///
/// Reciprocal-multiply:
///   `inv_sum[k] = ⌊S²/exp_sum_q[k]⌋`
///   `exp_q[k,j] · inv_sum[k] = softmax_q[k,j] · S + R[k,j]`
///
/// now `R[k,j] ∈ [0, S)` — constant bound.
/// So the F·N RC drops to just 12 bits.
///
/// Tradeoff: we introduce `r_inv[k] = S² − inv_sum[k]·exp_sum_q[k]`,
/// which needs its own variable-bound RC, but only over F elements
/// (so much cheaper overall).
use crate::onnx_proof::{
    ops::{
        softmax_last_axis::{
            rc::{ExpRemainderRCProvider, InvSumDiffRCProvider, RemainderRCProvider},
            recip_mult::{RecipMultParams, RecipMultProver, RecipMultVerifier},
        },
        sum::axis::{sum_axis_prover, sum_axis_verifier, SumAxisProvider},
        OperatorProofTrait,
    },
    ProofId, ProofType, Prover, Verifier,
};
use atlas_onnx_tracer::{
    node::ComputationNode,
    ops::{
        softmax::{softmax_last_axis_decomposed, SoftmaxLastAxisTrace},
        SoftmaxLastAxis,
    },
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{OpeningAccumulator, SumcheckId},
    },
    subprotocols::{
        identity_range_check::{identity_rangecheck_prover, identity_rangecheck_verifier},
        sumcheck::{Sumcheck, SumcheckInstanceProof},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, lookup_bits::LookupBits},
};

use crate::utils::dims::SumAxis;

// TODO: Add soundness test
// TODO: Batch sum + max and send sum_k and max_k
// TODO: Add all one-hot checks necessary.

pub mod exponentiation;
/// Proves: R[k,j] ∈ [0, S) for all k,j for some constant S.
///         diff[k] ∈ [0, N·S) for all k for some constant S.
pub mod rc;
/// Proves: Σ_{k,j} eq(r,(k,j)) · [exp_q(k,j)·inv_sum(k) + α·(inv_sum(k)·exp_sum_q(k) + r_inv(k) − S²)]
///       = softmax_q(r)·S + R(r)
pub mod recip_mult;

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for SoftmaxLastAxis {
    #[tracing::instrument(skip_all, name = "SoftmaxLastAxis::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let softmax_input = prover.trace.operand_tensors(node)[0];
        let (_, trace) = softmax_last_axis_decomposed(softmax_input, self.scale);
        SoftmaxLastAxisProver::new(node, trace, self.scale).prove(node, prover)
    }

    #[tracing::instrument(skip_all, name = "SoftmaxLastAxis::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        SoftmaxLastAxisVerifier::new(node, self.scale).verify(node, verifier)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute log₂(N·S), rounded up to the next multiple of 4 (Shout chunk constraint).
fn log_ns(n: usize, scale: usize) -> usize {
    ((n as u32).trailing_zeros() as usize + scale + 3) & !3
}

// ---------------------------------------------------------------------------
// SoftmaxLastAxisProver
// ---------------------------------------------------------------------------

/// Prover for softmax-last-axis operations.
///
/// Proving pipeline (each step has a corresponding method):
///
///  1. [`cache_r_inv`](Self::cache_r_inv) — commit remainder polynomial R
///  2. [`stage1_recip_mult`](Self::stage1_recip_mult) — recip-mult + div-consistency sumcheck
///  3. [`stage1b_remainder_rc`](Self::stage1b_remainder_rc) — R[k,j] ∈ [0, S) range check
///  4. [`stage2_inv_sum_diff_rc`](Self::stage2_inv_sum_diff_rc) — r_inv[k] ∈ [0, exp_sum_q[k]) range check
///  5. [`stage3_sum_axis`](Self::stage3_sum_axis) — exp_sum_q[k] = Σ_j exp_q[k,j]
///
/// See [`pipeline_spec`](Self::pipeline_spec) for the full protocol specification.
pub struct SoftmaxLastAxisProver {
    node_idx: usize,
    scale: i32,
    f_n: [usize; 2],
    trace: SoftmaxLastAxisTrace,
}

impl SoftmaxLastAxisProver {
    fn new(node: &ComputationNode, trace: SoftmaxLastAxisTrace, scale: i32) -> Self {
        let (&n, leading_dims) = node.output_dims.split_last().unwrap();
        let f = leading_dims.iter().product::<usize>();
        Self {
            node_idx: node.idx,
            scale,
            f_n: [f, n],
            trace,
        }
    }

    /// Run the full proving pipeline, returning all stage proofs.
    fn prove<F: JoltField, T: Transcript>(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        self.cache_r_inv(prover);
        let recip_mult_proof = self.stage1_recip_mult(prover);
        let remainder_rc_proof = self.stage1b_remainder_rc(prover);
        let inv_sum_diff_rc_proof = self.stage2_inv_sum_diff_rc(prover);
        let sum_proof = self.stage3_sum_axis(prover);
        self.cache_r_exp(prover);
        let s4_mult_proof = self.stage4_mult(prover);
        let exp_rem_rc_proof = self.stage4b_exp_remainder_rc(prover);
        self.dummy_operand_claim(node, prover);

        // TODO: Batch proofs
        vec![
            (
                ProofId(self.node_idx, ProofType::SoftmaxRecipMult),
                recip_mult_proof,
            ),
            (
                ProofId(self.node_idx, ProofType::SoftmaxRem),
                remainder_rc_proof,
            ),
            (
                ProofId(self.node_idx, ProofType::SoftmaxInvSumDiffRC),
                inv_sum_diff_rc_proof,
            ),
            (ProofId(self.node_idx, ProofType::SoftmaxSum), sum_proof),
            (
                ProofId(self.node_idx, ProofType::SoftmaxExpMult),
                s4_mult_proof,
            ),
            (
                ProofId(self.node_idx, ProofType::SoftmaxExpRemRC),
                exp_rem_rc_proof,
            ),
        ]
    }

    // ── Stage helpers ───────────────────────────────────────────────────────

    /// Cache the r_inv polynomial to the accumulator.
    fn cache_r_inv<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        // cache r_inv(r0)
        let r0 = prover
            .accumulator
            .get_node_output_opening(self.node_idx)
            .0
            .r;
        let r_poly: MultilinearPolynomial<F> = MultilinearPolynomial::from(self.trace.R.clone());
        let r_eval = r_poly.evaluate(&r0);
        prover.accumulator.append_dense(
            &mut prover.transcript,
            CommittedPolynomial::SoftmaxRecipMultRemainder(self.node_idx),
            SumcheckId::Execution,
            r0,
            r_eval,
        );
    }

    /// Stage 1: Combined recip-mult + div-consistency sumcheck.
    ///
    /// Proves: `exp_q[k,j] · inv_sum[k] = softmax_q[k,j] · S + R[k,j]`
    /// batched with div-consistency via random α.
    fn stage1_recip_mult<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
    ) -> SumcheckInstanceProof<F, T> {
        let params = RecipMultParams::new(
            self.node_idx,
            self.scale,
            self.f_n,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut sc_prover = RecipMultProver::initialize(
            self.trace.exp_q.clone(),
            self.trace.inv_sum.clone(),
            self.trace.exp_sum_q.clone(),
            self.trace.r_inv.clone(),
            params,
        );
        Sumcheck::prove(
            &mut sc_prover,
            &mut prover.accumulator,
            &mut prover.transcript,
        )
        .0
    }

    /// Stage 1b: Remainder range check.
    ///
    /// Proves: `R[k,j] ∈ [0, S)` for all k,j.
    /// Shout identity RC over F·N elements, constant 12-bit bound.
    fn stage1b_remainder_rc<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
    ) -> SumcheckInstanceProof<F, T> {
        let r_lookup_bits: Vec<LookupBits> = self
            .trace
            .R
            .iter()
            .map(|&rem| LookupBits::new(rem as u64, prover.preprocessing.scale() as usize))
            .collect();
        let provider = RemainderRCProvider {
            node_idx: self.node_idx,
            scale: prover.preprocessing.scale(),
        };
        let mut sc_prover =
            identity_rangecheck_prover(&provider, r_lookup_bits, &mut prover.accumulator);
        Sumcheck::prove(
            &mut sc_prover,
            &mut prover.accumulator,
            &mut prover.transcript,
        )
        .0
    }

    /// Stage 2: Inv-sum remainder range check.
    ///
    /// Proves: `r_inv[k] ∈ [0, exp_sum_q[k])` for all k via the diff trick:
    ///   `diff[k] = exp_sum_q[k] − 1 − r_inv[k] ∈ [0, N·S)`.
    fn stage2_inv_sum_diff_rc<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
    ) -> SumcheckInstanceProof<F, T> {
        let [_, n] = self.f_n;
        let diff: Vec<i32> = self
            .trace
            .exp_sum_q
            .iter()
            .zip(self.trace.r_inv.iter())
            .map(|(&es, &ri)| es - 1 - ri)
            .collect();
        #[cfg(test)]
        {
            for (k, &d) in diff.iter().enumerate() {
                assert!(
                    d >= 0 && d < (n as i32 * self.scale),
                    "diff[{k}] = {d} not in [0, N·S = {})",
                    n as i32 * self.scale
                );
            }
        }
        let diff_poly: MultilinearPolynomial<F> = MultilinearPolynomial::from(diff.clone());
        let r_leading = prover
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpSum(self.node_idx),
                SumcheckId::Execution,
            )
            .0;
        let diff_eval = diff_poly.evaluate(&r_leading.r);
        prover.accumulator.append_dense(
            &mut prover.transcript,
            CommittedPolynomial::SoftmaxInvSumDiff(self.node_idx),
            SumcheckId::Execution,
            r_leading.r.clone(),
            diff_eval,
        );
        let log_ns = log_ns(n, prover.preprocessing.scale() as usize);
        let diff_lookup_bits: Vec<LookupBits> = diff
            .iter()
            .map(|&d| LookupBits::new(d as u64, log_ns))
            .collect();
        let provider = InvSumDiffRCProvider {
            node_idx: self.node_idx,
            log_k: log_ns,
        };
        let mut sc_prover =
            identity_rangecheck_prover(&provider, diff_lookup_bits, &mut prover.accumulator);
        Sumcheck::prove(
            &mut sc_prover,
            &mut prover.accumulator,
            &mut prover.transcript,
        )
        .0
    }

    /// Stage 3: Sum-axis-1 sumcheck.
    ///
    /// Proves: `exp_sum_q[k] = Σ_j exp_q[k,j]`.
    fn stage3_sum_axis<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
    ) -> SumcheckInstanceProof<F, T> {
        let provider = SoftmaxSumProvider {
            node_idx: self.node_idx,
            f_n: self.f_n,
        };
        let mut sc_prover = sum_axis_prover(&provider, &self.trace.exp_q, &prover.accumulator);
        Sumcheck::prove(
            &mut sc_prover,
            &mut prover.accumulator,
            &mut prover.transcript,
        )
        .0
    }

    /// Cache the r_exp polynomial to the accumulator.
    fn cache_r_exp<F: JoltField, T: Transcript>(&self, prover: &mut Prover<F, T>) {
        // cache r_exp(r1)
        let r1 = prover
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpQ(self.node_idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        let r_poly: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(self.trace.decomposed_exp.r_exp.clone());
        let r_eval = r_poly.evaluate(&r1);
        prover.accumulator.append_dense(
            &mut prover.transcript,
            CommittedPolynomial::SoftmaxExpRemainder(self.node_idx),
            SumcheckId::Execution,
            r1,
            r_eval,
        );
    }

    fn stage4_mult<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
    ) -> SumcheckInstanceProof<F, T> {
        let params =
            exponentiation::MultParams::new(self.node_idx, self.scale, &prover.accumulator);
        let mut sc_prover = exponentiation::MultProver::initialize(
            self.trace.decomposed_exp.exp_hi.clone(),
            self.trace.decomposed_exp.exp_lo.clone(),
            params,
        );
        Sumcheck::prove(
            &mut sc_prover,
            &mut prover.accumulator,
            &mut prover.transcript,
        )
        .0
    }

    /// Stage 4b: Exp-remainder range check.
    ///
    /// Proves: `r_exp[k,j] ∈ [0, S)` for all k,j.
    /// Identical to Stage 1b but reads from `SoftmaxExpRemainder`.
    fn stage4b_exp_remainder_rc<F: JoltField, T: Transcript>(
        &self,
        prover: &mut Prover<F, T>,
    ) -> SumcheckInstanceProof<F, T> {
        let r_lookup_bits: Vec<LookupBits> = self
            .trace
            .decomposed_exp
            .r_exp
            .iter()
            .map(|&rem| LookupBits::new(rem as u64, prover.preprocessing.scale() as usize))
            .collect();
        let provider = ExpRemainderRCProvider {
            node_idx: self.node_idx,
            scale: prover.preprocessing.scale(),
        };
        let mut sc_prover =
            identity_rangecheck_prover(&provider, r_lookup_bits, &mut prover.accumulator);
        Sumcheck::prove(
            &mut sc_prover,
            &mut prover.accumulator,
            &mut prover.transcript,
        )
        .0
    }

    /// Temporary: pass down an operand claim for the rest of the proof system.
    // TODO: rm once full pipeline is implemented
    fn dummy_operand_claim<F: JoltField, T: Transcript>(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) {
        let (opening_point, _claim) = prover.accumulator.get_node_output_opening(self.node_idx);
        let operand = prover.trace.operand_tensors(node)[0];
        let operand_claim = MultilinearPolynomial::from(operand.clone()).evaluate(&opening_point.r);
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(node.inputs[0]),
            SumcheckId::NodeExecution(self.node_idx),
            opening_point,
            operand_claim,
        );
    }
}

// ---------------------------------------------------------------------------
// SoftmaxLastAxisVerifier
// ---------------------------------------------------------------------------

/// Verifier for softmax-last-axis operations.
///
/// Mirror of [`SoftmaxLastAxisProver`] — each stage method corresponds to a
/// prover stage and verifies the associated sumcheck proof.
pub struct SoftmaxLastAxisVerifier {
    node_idx: usize,
    scale: i32,
    f_n: [usize; 2],
}

impl SoftmaxLastAxisVerifier {
    fn new(node: &ComputationNode, scale: i32) -> Self {
        let (&n, leading_dims) = node.output_dims.split_last().unwrap();
        let f = leading_dims.iter().product::<usize>();
        Self {
            node_idx: node.idx,
            scale,
            f_n: [f, n],
        }
    }

    /// Run the full verification pipeline.
    fn verify<F: JoltField, T: Transcript>(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let recip_mult_proof = verifier
            .proofs
            .get(&ProofId(self.node_idx, ProofType::SoftmaxRecipMult))
            .ok_or(ProofVerifyError::MissingProof(self.node_idx))?;
        let remainder_rc_proof = verifier
            .proofs
            .get(&ProofId(self.node_idx, ProofType::SoftmaxRem))
            .ok_or(ProofVerifyError::MissingProof(self.node_idx))?;
        let inv_sum_diff_rc_proof = verifier
            .proofs
            .get(&ProofId(self.node_idx, ProofType::SoftmaxInvSumDiffRC))
            .ok_or(ProofVerifyError::MissingProof(self.node_idx))?;
        let sum_proof = verifier
            .proofs
            .get(&ProofId(self.node_idx, ProofType::SoftmaxSum))
            .ok_or(ProofVerifyError::MissingProof(self.node_idx))?;

        self.cache_r_inv(verifier);
        self.stage1_recip_mult(recip_mult_proof, verifier);
        self.stage1b_remainder_rc(remainder_rc_proof, verifier);
        self.stage2_inv_sum_diff_rc(inv_sum_diff_rc_proof, verifier)?;
        self.stage3_sum_axis(sum_proof, verifier);
        self.cache_r_exp(verifier);
        self.stage4_mult(
            verifier
                .proofs
                .get(&ProofId(self.node_idx, ProofType::SoftmaxExpMult))
                .ok_or(ProofVerifyError::MissingProof(self.node_idx))?,
            verifier,
        );
        self.stage4b_exp_remainder_rc(
            verifier
                .proofs
                .get(&ProofId(self.node_idx, ProofType::SoftmaxExpRemRC))
                .ok_or(ProofVerifyError::MissingProof(self.node_idx))?,
            verifier,
        );
        self.dummy_operand_claim(node, verifier);

        Ok(())
    }

    // ── Stage helpers ───────────────────────────────────────────────────────

    /// Append the remainder polynomial commitment to the accumulator.
    fn cache_r_inv<F: JoltField, T: Transcript>(&self, verifier: &mut Verifier<'_, F, T>) {
        let r = verifier
            .accumulator
            .get_node_output_opening(self.node_idx)
            .0
            .r;
        verifier.accumulator.append_dense(
            &mut verifier.transcript,
            CommittedPolynomial::SoftmaxRecipMultRemainder(self.node_idx),
            SumcheckId::Execution,
            r,
        );
    }

    /// Stage 1: Verify recip-mult + div-consistency sumcheck.
    fn stage1_recip_mult<F: JoltField, T: Transcript>(
        &self,
        proof: &SumcheckInstanceProof<F, T>,
        verifier: &mut Verifier<'_, F, T>,
    ) {
        let rv = RecipMultVerifier::new(
            self.node_idx,
            self.scale,
            self.f_n,
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            proof,
            &rv,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )
        .unwrap(); // TODO: rm
    }

    /// Stage 1b: Verify remainder range check.
    fn stage1b_remainder_rc<F: JoltField, T: Transcript>(
        &self,
        proof: &SumcheckInstanceProof<F, T>,
        verifier: &mut Verifier<'_, F, T>,
    ) {
        let provider = RemainderRCProvider {
            node_idx: self.node_idx,
            scale: verifier.preprocessing.scale(),
        };
        let rv = identity_rangecheck_verifier(&provider, &mut verifier.accumulator);
        Sumcheck::verify(
            proof,
            &rv,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )
        .unwrap(); // TODO: rm
    }

    /// Stage 2: Verify inv-sum diff range check + algebraic consistency.
    ///
    /// Checks: `diff[k] ∈ [0, N·S)` and
    ///   `diff(r_leading) = exp_sum_q(r_leading) − 1 − r_inv(r_leading)`.
    fn stage2_inv_sum_diff_rc<F: JoltField, T: Transcript>(
        &self,
        proof: &SumcheckInstanceProof<F, T>,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let [_, n] = self.f_n;

        // Commit diff polynomial
        let r_leading = verifier
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpSum(self.node_idx),
                SumcheckId::Execution,
            )
            .0;
        verifier.accumulator.append_dense(
            &mut verifier.transcript,
            CommittedPolynomial::SoftmaxInvSumDiff(self.node_idx),
            SumcheckId::Execution,
            r_leading.r,
        );

        // Range check
        let log_ns = log_ns(n, verifier.preprocessing.scale() as usize);
        let provider = InvSumDiffRCProvider {
            node_idx: self.node_idx,
            log_k: log_ns,
        };
        let rv = identity_rangecheck_verifier(&provider, &mut verifier.accumulator);
        Sumcheck::verify(
            proof,
            &rv,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )
        .unwrap(); // TODO: rm

        // Algebraic check: diff(r_leading) = exp_sum_q(r_leading) − 1 − r_inv(r_leading)
        let diff_claim = verifier
            .accumulator
            .get_committed_polynomial_opening(
                CommittedPolynomial::SoftmaxInvSumDiff(self.node_idx),
                SumcheckId::Execution,
            )
            .1;
        let exp_sum_claim = verifier
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpSum(self.node_idx),
                SumcheckId::Execution,
            )
            .1;
        let r_inv_claim = verifier
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxRInv(self.node_idx),
                SumcheckId::Execution,
            )
            .1;
        if diff_claim != exp_sum_claim - F::one() - r_inv_claim {
            return Err(ProofVerifyError::InternalError);
        }

        Ok(())
    }

    /// Stage 3: Verify sum-axis-1 sumcheck.
    fn stage3_sum_axis<F: JoltField, T: Transcript>(
        &self,
        proof: &SumcheckInstanceProof<F, T>,
        verifier: &mut Verifier<'_, F, T>,
    ) {
        let provider = SoftmaxSumProvider {
            node_idx: self.node_idx,
            f_n: self.f_n,
        };
        let sv = sum_axis_verifier(&provider, &verifier.accumulator);
        Sumcheck::verify(
            proof,
            &sv,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )
        .unwrap(); // TODO: rm
    }

    /// Temporary: pass down an operand claim for the rest of the proof system.
    // TODO: rm once full pipeline is implemented
    fn dummy_operand_claim<F: JoltField, T: Transcript>(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) {
        let (opening_point, _claim) = verifier.accumulator.get_node_output_opening(self.node_idx);
        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::NodeOutput(node.inputs[0]),
            SumcheckId::NodeExecution(self.node_idx),
            opening_point,
        );
    }

    fn cache_r_exp<F: JoltField, T: Transcript>(&self, verifier: &mut Verifier<'_, F, T>) {
        let r = verifier
            .accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::SoftmaxExpQ(self.node_idx),
                SumcheckId::Execution,
            )
            .0
            .r;
        verifier.accumulator.append_dense(
            &mut verifier.transcript,
            CommittedPolynomial::SoftmaxExpRemainder(self.node_idx),
            SumcheckId::Execution,
            r,
        );
    }

    fn stage4_mult<F: JoltField, T: Transcript>(
        &self,
        proof: &SumcheckInstanceProof<F, T>,
        verifier: &mut Verifier<'_, F, T>,
    ) {
        let mv =
            exponentiation::MultVerifier::new(self.node_idx, self.scale, &verifier.accumulator);
        Sumcheck::verify(
            proof,
            &mv,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )
        .unwrap(); // TODO: rm
    }

    /// Stage 4b: Verify exp-remainder range check.
    fn stage4b_exp_remainder_rc<F: JoltField, T: Transcript>(
        &self,
        proof: &SumcheckInstanceProof<F, T>,
        verifier: &mut Verifier<'_, F, T>,
    ) {
        let provider = ExpRemainderRCProvider {
            node_idx: self.node_idx,
            scale: verifier.preprocessing.scale(),
        };
        let rv = identity_rangecheck_verifier(&provider, &mut verifier.accumulator);
        Sumcheck::verify(
            proof,
            &rv,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )
        .unwrap(); // TODO: rm
    }
}

// ---------------------------------------------------------------------------
// SoftmaxSumProvider — adapts softmax stage 3 for the generic sum-axis
// sumcheck via SumAxisProvider.
// ---------------------------------------------------------------------------

/// Provider for softmax stage 3: proves `exp_sum_q[k] = Σ_j exp_q[k,j]`.
struct SoftmaxSumProvider {
    node_idx: usize,
    /// `[F, N]` — leading-dim product and last-axis size.
    f_n: [usize; 2],
}

impl SumAxisProvider for SoftmaxSumProvider {
    fn axis(&self) -> SumAxis {
        SumAxis::Axis1
    }

    fn operand_dims(&self) -> [usize; 2] {
        self.f_n
    }

    fn sum_output_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::SoftmaxExpSum(self.node_idx),
            SumcheckId::Execution,
        )
    }

    fn operand_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::SoftmaxExpQ(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }
}

#[cfg(test)]
mod tests {
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };

    use crate::onnx_proof::ops::test::unit_test_op;

    fn softmax_last_axis_model(input_shape: &[usize]) -> Model {
        let mut b = ModelBuilder::with_scale(12);
        let i = b.input(input_shape.to_vec());
        let res = b.softmax_last_axis(i);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_softmax_last_axis() {
        // Realistic pre-softmax attention scores shaped [4, 8, 8] = 256 (power of 2).
        // Non-masked scores sourced from GPT-2 layer 0 (scale=12, multiplier=4096).
        //
        // Layout: 4 attention heads, 8×8 causal attention matrix each.
        // Upper-triangular entries (future tokens) use the GPT-2 causal mask ≈ -2^30.
        // Non-masked scores range roughly [-25000, 50000] in fixed-point.
        const M: i32 = -1_073_741_824; // causal attention mask (≈ -2^30)
        #[rustfmt::skip]
        const GPT2_ATTN_SCORES: &[i32] = &[
            // Head 0 — moderate range (GPT-2 heads 0 + 2)
            //   Non-masked range: [-12738, 5440]
              -492,     M,     M,     M,     M,     M,     M,     M,
              5440, -6182,     M,     M,     M,     M,     M,     M,
              4040, -4700, -3995,     M,     M,     M,     M,     M,
              3503, -4425,   753,  -701,     M,     M,     M,     M,
             -2353, -6284, -4118, -4413,-10752,     M,     M,     M,
             -2247, -7704, -5686, -6371,-12738,-12175,     M,     M,
              -652,   749,   826, -1139, -2820, -5039, -2898,     M,
             -2622, -2193, -5682, -6362, -8036, -4648, -9872,-10283,
            // Head 1 — high variance (GPT-2 heads 1 + 10)
            //   Non-masked range: [-7209, 49207]
             13386,     M,     M,     M,     M,     M,     M,     M,
             11265, 36900,     M,     M,     M,     M,     M,     M,
              9668, 19702, 39017,     M,     M,     M,     M,     M,
             10015, 13052, 21247, 49207,     M,     M,     M,     M,
              7392,  6466,  1150, -4119, 30542,     M,     M,     M,
              4939,   493,   848, -7209,  9139, 24853,     M,     M,
             16554, 12057, 15905,  8038, 13262, 12499,  5472,     M,
             11299,  4069,  6294,  3081,  6412,  8400, 10154,  4726,
            // Head 2 — large positive range (GPT-2 heads 5 + 6)
            //   Non-masked range: [-5332, 47013]
             41677,     M,     M,     M,     M,     M,     M,     M,
             30755, 46148,     M,     M,     M,     M,     M,     M,
             31081, 16155, 43315,     M,     M,     M,     M,     M,
             29381, 15600, 12915, 47013,     M,     M,     M,     M,
             19723, 10507,  5700,  5104, 30550,     M,     M,     M,
             16650,  9028, 11201,  6500, 11179, 20697,     M,     M,
             11184,  4791,   371,  1565, -2793, -5332,  2261,     M,
             -2074, -6790, -7238, -7885, -5024, -5917, -2315, -7353,
            // Head 3 — negative bias (GPT-2 heads 7 + 8)
            //   Non-masked range: [-23295, -1045]
             -7215,     M,     M,     M,     M,     M,     M,     M,
             -7363,-18856,     M,     M,     M,     M,     M,     M,
            -12684,-10596,-20225,     M,     M,     M,     M,     M,
            -11926,-11397,-10691,-16112,     M,     M,     M,     M,
            -16957,-19591,-19715,-17550,-16389,     M,     M,     M,
            -20129,-23135,-23295,-21295,-17981,-17724,     M,     M,
             -2898, -9452, -7226, -7040, -6460,-13308, -6362,     M,
             -8241,-10310,-12414, -5334, -1045, -8036,-10493,-12577,
        ];
        let input_shape = vec![4, 8, 8];
        let input = Tensor::new(Some(GPT2_ATTN_SCORES), &input_shape).unwrap();
        let model = softmax_last_axis_model(&input_shape);
        unit_test_op(model, &[input]);
    }
}
