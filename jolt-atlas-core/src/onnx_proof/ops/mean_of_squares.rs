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
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::range_checking::RangeCheckProve;
pub use jolt_atlas_verifier::onnx_proof::ops::mean_of_squares::*;

use crate::onnx_proof::trace_view::TraceView;
use std::array;

use atlas_trace::{op::MeanOfSquares, trace::LayerData, Trace};

use crate::{
    node_view::ComputationNode,
    onnx_proof::{
        clamp_lookups::is_scalar,
        deferred_lookups::{DeferredBatch, DeferredOneHot},
        fused_rebase,
        ops::Prover,
        range_checking::{
            range_check_operands::{MeanOfSquaresRangeCheckOperands, RangeCheckOperands},
            RangeCheckProvider,
        },
        relation::{
            append_produced, note_dispatch, relation_input_claim, RelationParams, RelationSetup,
        },
        ProofId, ProofType,
    },
};
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    },
};
use atlas_lookup_tables::unsigned_less_than::UnsignedLessThanTable;
use atlas_sumcheck::{
    opening_proof::prover::ProverOpeningAccumulator,
    sumcheck::{prover::SumcheckInstanceProver, Sumcheck, SumcheckInstanceProof},
};
use atlas_transcript::Transcript;
use common::{consts::XLEN, parallel::par_enabled};
use jolt_claims::SymbolicSumcheck;
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use rayon::prelude::*;

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
