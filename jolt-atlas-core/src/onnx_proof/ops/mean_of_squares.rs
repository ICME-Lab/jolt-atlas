//! Monolithic fused mean-of-squares (`MeanOfSquares`, issue #194).
//!
//! Computes, per retained output element `h` (the reduced axis has size `N`):
//!
//! ```text
//! acc[h]      = Σ_j x[h,j]²                       (i64 squared sum — no i32 overflow)
//! rescaled[h] = acc[h] / D,        D = N·2^S      (mean rebase, D a per-node constant)
//! out[h]      = SatClamp(rescaled[h])             (i32)
//! ```
//!
//! replacing the old `Square → Sum → Div(N)` decomposition (which squared in i32
//! and could saturate, the reason `pre_rebase_nonlinear` existed). The proof chain
//! (mirrors `Sum`, plus the rescale of [`fused_rebase`]):
//!
//! 1. **Reduction sumcheck** ([`MeanOfSquaresReductionProver`], `RescaleArith`):
//!    `acc(r) = Σ_{h,j} eq(r,h)·x[h,j]²`, reducing to one input opening. Its
//!    initial claim is `acc(r) = rescaled·D + R` (the rescale seam). This is the
//!    einsum **High** eq-schedule (`eq·op·op`, eq over the retained/high bits,
//!    free sum over the reduced/low bits) specialized to one operand.
//! 2. **Clamp** (`Execution` + `RaOneHotChecks`, via [`fused_rebase::prove_pre`]):
//!    `out = SatClamp(rescaled)`, appending `rescaled` (`ClampAcc`) and `R`
//!    (`RescaleRemainder`).
//! 3. **Remainder range check** `R < D` (`RangeCheck` + `RescaleRemainderRaChecks`,
//!    via the [`range_checking`](crate::onnx_proof::range_checking) `UnsignedLT`
//!    lookup; `D` is a constant, so the Teleport-style
//!    [`MeanOfSquaresRangeCheckOperands`] is used).
//!
//! The reduction sums over the *padded* reduced cube (`op.padded_count`, a power
//! of two so the flat-padded operand splits cleanly into `[retained, reduced]`),
//! while the divisor `D = N·2^S` uses the *semantic* count `N` (= `op.count`,
//! any constant — e.g. GPT-2's 768). Zero-padded lanes contribute nothing to
//! `Σx²`, so the two are consistent.
use crate::onnx_proof::ops::{OpLayout, ProveOp, VerifyOp};

use crate::onnx_proof::trace_view::TraceView;
use std::array;

use atlas_trace::{op::MeanOfSquares, trace::LayerData, Op, Trace};

use crate::{
    atlas_relation_types,
    node_view::ComputationNode,
    onnx_proof::{
        clamp_lookups::{clamp_committed_polys, is_scalar, recover_small_int, verify_scalar_clamp},
        deferred_lookups::{DeferredBatch, DeferredOneHot},
        fused_rebase,
        ops::{Prover, Verifier},
        range_checking::{
            range_check_operands::{MeanOfSquaresRangeCheckOperands, RangeCheckOperands},
            RangeCheckProvider,
        },
        relation::{
            append_produced, note_dispatch, reduced_output_setup, relation_input_claim,
            AtlasRelation, RelationDerived, RelationExpr, RelationId, RelationOpening,
            RelationParams, RelationSetup, RelationVerifier,
        },
        ProofId, ProofType,
    },
    utils::opening_access::{AccOpeningAccessor, OpeningIdBuilder, Target},
};
use common::{consts::XLEN, parallel::par_enabled, CommittedPoly, VirtualPoly};
use jolt_claims::{constant, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use joltworks::{
    field::JoltField,
    lookup_tables::unsigned_less_than::UnsignedLessThanTable,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{OpeningAccumulator, OpeningId, ProverOpeningAccumulator},
    },
    subprotocols::{
        shout::RaOneHotEncoding,
        sumcheck::{Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use rayon::prelude::*;

const DEGREE: usize = 3;

/// Fused divisor `D = count·2^scale` (the semantic count, not the padded one).
pub fn mos_divisor(op: &MeanOfSquares, scale: u8) -> i64 {
    (1i64 << scale) * op.count as i64
}

impl OpLayout for MeanOfSquares {
    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        // Saturating clamp one-hot (64-bit) ...
        let mut polys = clamp_committed_polys(node);
        // ... plus the `R < D` range-check one-hot (skipped for scalar outputs).
        if !is_scalar(node) {
            let rc_operands = RangeCheckOperands::<MeanOfSquaresRangeCheckOperands>::new(node);
            let d = rc_operands
                .get_encoding(node)
                .one_hot_params()
                .instruction_d;
            polys.extend((0..d).map(|i| CommittedPoly::MeanOfSquaresRangeCheckRaD(node.idx, i)));
        }
        polys
    }
}

impl<F: JoltField, T: Transcript> ProveOp<F, T> for MeanOfSquares {
    #[tracing::instrument(skip_all, name = "MeanOfSquares::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        let scalar = is_scalar(node);
        let mut proofs = Vec::new();

        // (1+2) Append `R` (RescaleRemainder) + `rescaled` (ClampAcc) and prove
        // the clamp `out = SatClamp(rescaled)`. The returned remainder is
        // unused: MoS range-checks `R < D` via its own operand-based lookup.
        let (pre_proofs, _remainder) = fused_rebase::prove_pre(node, prover);
        proofs.extend(pre_proofs);

        // (3) Reduction sumcheck `acc(r) = Σ eq·x²` (claim `rescaled·D + R`).
        let params = RelationParams::new(
            MeanOfSquaresRelation::new(node.clone()),
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut reduction = MeanOfSquaresReductionProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut reduction,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        proofs.push((ProofId(node.idx, ProofType::RescaleArith), proof));

        // (4) Remainder range check `R < D`. Scalar outputs prove it in the clear
        // (the one-hot reduction degenerates), checked by the verifier.
        if !scalar {
            // Built now, proven after the node loop (`RangeCheck` batch); its
            // one-hot checks follow that batch.
            let rc_provider = RangeCheckProvider::<MeanOfSquaresRangeCheckOperands>::new(node);
            let (rc_sumcheck, lookup_indices) = rc_provider
                .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
                    &prover.trace,
                    &mut prover.accumulator,
                    &mut prover.transcript,
                );
            prover.defer(DeferredBatch::RangeCheck, Box::new(rc_sumcheck));
            let rc_operands = RangeCheckOperands::<MeanOfSquaresRangeCheckOperands>::new(node);
            prover.deferred_onehots.push((
                DeferredOneHot::RangeCheck(rc_operands.get_encoding(node)),
                lookup_indices,
            ));
        }

        proofs
    }
}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for MeanOfSquares {
    #[tracing::instrument(skip_all, name = "MeanOfSquares::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let scalar = is_scalar(node);

        // (1+2) Clamp + remainder advice.
        fused_rebase::verify_pre(node, verifier)?;

        // (3) Reduction sumcheck.
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RescaleArith))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let reduction = RelationVerifier::<F, _>::new(
            MeanOfSquaresRelation::new(node.clone()),
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            proof,
            &reduction,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // (4) Remainder range check `R < D`.
        if scalar {
            // Scalar fallback: `rescaled`, `R`, and `out` open in the clear.
            let accessor = AccOpeningAccessor::new(&verifier.accumulator, node);
            let rescaled_claim = accessor.get_advice(VirtualPoly::ClampAcc).1;
            let output_claim = accessor.get_reduced_opening().1;
            verify_scalar_clamp(rescaled_claim, output_claim, "MeanOfSquares")?;

            let r_claim = accessor.get_advice(VirtualPoly::RescaleRemainder).1;
            verify_scalar_remainder(r_claim, mos_divisor(self, node.scale))?;
        } else {
            let rc_provider = RangeCheckProvider::<MeanOfSquaresRangeCheckOperands>::new(node);
            let rc_verifier = rc_provider.read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
                &mut verifier.accumulator,
                &mut verifier.transcript,
            );
            verifier.defer(DeferredBatch::RangeCheck, Box::new(rc_verifier));
            let rc_operands = RangeCheckOperands::<MeanOfSquaresRangeCheckOperands>::new(node);
            verifier
                .deferred_onehots
                .push(DeferredOneHot::RangeCheck(rc_operands.get_encoding(node)));
        }
        Ok(())
    }
}

/// Verify a scalar mean-of-squares remainder directly: `0 ≤ R < D`.
fn verify_scalar_remainder<F: JoltField>(r_claim: F, divisor: i64) -> Result<(), ProofVerifyError> {
    let value = recover_small_int(r_claim).ok_or_else(|| {
        ProofVerifyError::InvalidOpeningProof(
            "mean-of-squares (scalar): remainder claim is not a small integer".to_string(),
        )
    })?;
    if value < 0 || value >= divisor {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "mean-of-squares (scalar): remainder must lie in [0, D)".to_string(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Reduction sumcheck: acc(r) = Σ_{h,j} eq(r,h)·x[h,j]²
// ---------------------------------------------------------------------------

crate::bind_relation!(MeanOfSquares, MeanOfSquaresRelation);

/// The mean-of-squares reduction as a relation, binding high-to-low over the operand laid out
/// `[retained (high), reduced (low)]`:
///
/// ```text
/// input claim   rescaled(r₀)·D + rem(r₀)      (D = N·2^S, the rescale divisor)
/// output claim  eq(r₀, r′_retained) · x(r′)²
/// ```
///
/// [`MeanOfSquaresReductionProver`] is its prover; the verifier is the relation's.
///
/// **S ⊆ R.** The op accumulates `acc = Σ_reduced x²` in `i64` per retained index and splits it
/// with `atlas_trace::semantics::fixed::rebase` by `D` into `rescaled·D + rem`, so
/// `Σ eq(r₀, h)·x(h, j)²` over the hypercube is `acc(r₀)`; zero-padded lanes add nothing to the
/// sum. The bound `rem < D` is the range check, outside the relation.
pub struct MeanOfSquaresRelation {
    node: ComputationNode,
}

impl MeanOfSquaresRelation {
    fn op(&self) -> &MeanOfSquares {
        let Op::MeanOfSquares(op) = &self.node.operator else {
            panic!("expected MeanOfSquares operator")
        };
        op
    }

    /// The retained (output) variables: the eq cube, bound in the first rounds.
    fn log_retained(&self) -> usize {
        self.node.pow2_padded_num_output_elements().log_2()
    }

    /// The reduced (summed) variables, bound last.
    fn log_reduced(&self) -> usize {
        // The sumcheck cube splits the flat-padded operand into `[retained, reduced]`, so the
        // padded reduced count sizes the low rounds; the divisor stays semantic.
        let padded_count = self.op().padded_count;
        assert!(
            padded_count.is_power_of_two(),
            "mean-of-squares reduction requires a power-of-two padded reduced count, got {padded_count}"
        );
        padded_count.log_2()
    }

    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }
}

impl SymbolicSumcheck for MeanOfSquaresRelation {
    atlas_relation_types!();
    type Shape = ComputationNode;

    fn new(node: ComputationNode) -> Self {
        Self { node }
    }

    fn id() -> RelationId {
        RelationId::MeanOfSquares
    }

    fn rounds(&self) -> usize {
        self.log_retained() + self.log_reduced()
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        let divisor = mos_divisor(self.op(), self.node.scale);
        let rescaled = self.ids().advice(VirtualPoly::ClampAcc);
        let remainder = self.ids().advice(VirtualPoly::RescaleRemainder);
        constant(F::from_u64(divisor as u64)) * opening(RelationOpening::Id(rescaled))
            + opening(RelationOpening::Id(remainder))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        let x = opening(RelationOpening::Id(self.produced()[0]));
        derived(RelationDerived::RetainedEq) * x.clone() * x
    }
}

impl AtlasRelation for MeanOfSquaresRelation {
    type Spec = MeanOfSquares;

    fn node(&self) -> &ComputationNode {
        &self.node
    }

    fn setup<F: JoltField, T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> RelationSetup<F> {
        reduced_output_setup(accumulator, &self.node)
    }

    fn produced(&self) -> Vec<OpeningId> {
        vec![self.ids().nodeio(Target::Input(0))]
    }

    fn opening_points<F: JoltField>(
        &self,
        _setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Vec<Vec<F>> {
        vec![challenges.to_vec()]
    }

    fn derive<F: JoltField>(
        &self,
        id: &RelationDerived,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> F {
        match id {
            RelationDerived::RetainedEq => {
                EqPolynomial::mle(&setup.incoming_point, &challenges[..self.log_retained()])
            }
            other => panic!("a mean of squares derives no {other:?}"),
        }
    }
}

/// The specialized prover of a [`MeanOfSquaresRelation`]: the operand squared against the
/// retained-dim eq table, read by index rather than expanded. Completeness-only:
/// `relation_prover_equiv_mean_of_squares` checks its messages against the relation's reference
/// messages.
pub struct MeanOfSquaresReductionProver<F: JoltField> {
    relation: MeanOfSquaresRelation,
    setup: RelationSetup<F>,
    log_retained: usize,
    log_reduced: usize,
    /// The (flat, padded) input operand, laid out `[retained (high), reduced (low)]`.
    operand: MultilinearPolynomial<F>,
    /// Eq-poly over the retained (output) variables.
    eq: MultilinearPolynomial<F>,
    /// `eq` final claim, cached once its cube is fully bound (after `log_retained`).
    eq_bound_claim: Option<F>,
}

impl<F: JoltField> MeanOfSquaresReductionProver<F> {
    /// Initialize by loading the operand and the retained-dim eq-poly.
    #[tracing::instrument(skip_all, name = "MeanOfSquaresReductionProver::initialize")]
    pub fn initialize(trace: &Trace, params: RelationParams<F, MeanOfSquaresRelation>) -> Self {
        let (relation, setup) = params.into_parts();
        note_dispatch(&relation);
        let LayerData { operands, .. } = TraceView::layer_data(trace, &relation.node);
        let [operand] = operands[..] else {
            panic!("Expected one operand for MeanOfSquares operation")
        };
        let (log_retained, log_reduced) = (relation.log_retained(), relation.log_reduced());
        let operand = MultilinearPolynomial::from(operand.padded_pow2());
        assert_eq!(
            operand.get_num_vars(),
            log_retained + log_reduced,
            "mean-of-squares operand size must equal retained·reduced (reduced count must be a power of two)"
        );
        let eq = MultilinearPolynomial::from(EqPolynomial::evals(&setup.incoming_point));
        // Scalar output (`log_retained == 0`): there are no retained rounds to
        // bind, so the eq cube (over zero variables) is the constant 1 from the
        // start.
        let eq_bound_claim = (log_retained == 0).then(F::one);
        Self {
            relation,
            setup,
            log_retained,
            log_reduced,
            operand,
            eq,
            eq_bound_claim,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MeanOfSquaresReductionProver<F> {
    fn degree(&self) -> usize {
        self.relation.degree()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        relation_input_claim(&self.relation, &self.setup, accumulator)
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        append_produced(
            &self.relation,
            &self.setup,
            accumulator,
            transcript,
            sumcheck_challenges,
            &[self.operand.final_claim()],
        );
    }
}
impl<F: JoltField> MeanOfSquaresReductionProver<F> {
    fn round_message(&mut self, round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let half = self.operand.len() / 2;
        let log_retained = self.log_retained;
        let log_reduced = self.log_reduced;
        let evals: [F; DEGREE] = (0..half)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let l = self
                    .operand
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let e = if round < log_retained {
                    self.eq
                        .sumcheck_evals_array::<DEGREE>(i >> log_reduced, BindingOrder::HighToLow)
                } else {
                    [self.eq_bound_claim.expect("eq claim should be cached"); DEGREE]
                };
                array::from_fn(|k| l[k] * l[k] * e[k])
            })
            .reduce(
                || [F::zero(); DEGREE],
                |a, b| array::from_fn(|k| a[k] + b[k]),
            );
        UnivariatePoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn bind(&mut self, r: F, round: usize) {
        let r_j = F::challenge_from_field(r);
        self.operand.bind_parallel(r_j, BindingOrder::HighToLow);
        if round < self.log_retained {
            self.eq.bind_parallel(r_j, BindingOrder::HighToLow);
            if round == self.log_retained - 1 {
                self.eq_bound_claim = Some(self.eq.final_claim());
            }
        }
    }
}

impl<F: JoltField> ProveRounds<F> for MeanOfSquaresReductionProver<F> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r, round - 1);
        }
        Ok(self.round_message(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, self.relation.rounds() - 1);
        Ok(())
    }
}
