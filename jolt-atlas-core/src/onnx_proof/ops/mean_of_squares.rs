//! Monolithic fused mean-of-squares (`MeanOfSquares`, issue #194).
//!
//! Computes, per retained output element `h` (the reduced axis has size `N`):
//!
//! ```text
//! acc[h] = Σ_j x[h,j]²                        (i64 squared sum)
//! out[h] = acc[h] / D,        D = N·2^S       (Euclidean quotient; an error outside i32)
//! ```
//!
//! The proof chain (mirrors `Sum`, plus the rescale of
//! [`fused_rebase`](crate::onnx_proof::fused_rebase)):
//!
//! 1. **Remainder advice**: `R = acc mod D` is appended (`RescaleRemainder`).
//! 2. **Reduction sumcheck** ([`MeanOfSquaresReductionProver`], `RescaleArith`):
//!    `acc(r) = Σ_{h,j} eq(r,h)·x[h,j]²`, reducing to one input opening. Its
//!    initial claim is `acc(r) = out(r)·D + R(r)` (the rescale seam). This is the
//!    einsum **High** eq-schedule (`eq·op·op`, eq over the retained/high bits,
//!    free sum over the reduced/low bits) specialized to one operand.
//! 3. **Range claims** in the packed range buckets
//!    ([`range_bucket::node_range_claims`](crate::onnx_proof::range_bucket::node_range_claims)),
//!    after the nodes' steps: the remainder bound `R < D` and, unless the node is a model
//!    output, the output range `out + 2^31 ∈ [0, 2^32)`.
//!
//! The reduction sums over the *padded* reduced cube (`op.padded_count`, a power
//! of two so the flat-padded operand splits cleanly into `[retained, reduced]`),
//! while the divisor `D = N·2^S` uses the *semantic* count `N` (= `op.count`,
//! any constant — e.g. GPT-2's 768). Zero-padded lanes contribute nothing to
//! `Σx²`, so the two are consistent.
pub use jolt_atlas_verifier::onnx_proof::ops::mean_of_squares::*;

use crate::onnx_proof::trace_view::TraceView;
use std::array;

use atlas_trace::{trace::LayerData, Trace};

use crate::onnx_proof::relation::{
    note_dispatch, OpRel, RelationParams, RelationSetup, RelationWitness,
};
use atlas_algebra::poly::univariate::UnivariatePoly;
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq::EqPolynomial,
        multilinear::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    },
};
use atlas_claims::SymbolicSumcheck;
use atlas_sumcheck::engine::{prover::ProveRounds, SumcheckError};
use common::parallel::par_enabled;
use rayon::prelude::*;

/// The specialized prover of a [`squared_sum_reduction`]: the operand squared against the
/// retained-dim eq table, read by index rather than expanded. Completeness-only:
/// `relation_prover_equiv_mean_of_squares` checks its messages against the relation's reference
/// messages.
pub struct MeanOfSquaresReductionProver<F: JoltField> {
    relation: OpRel<F>,
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
    pub fn initialize(trace: &Trace, params: RelationParams<F, OpRel<F>>) -> Self {
        let (relation, setup) = params.into_parts();
        note_dispatch(&relation);
        let LayerData { operands, .. } = TraceView::layer_data(trace, &relation.node);
        let [operand] = operands[..] else {
            panic!("Expected one operand for MeanOfSquares operation")
        };
        let log_retained = log_retained(&relation.node);
        let (_, log_reduced) =
            reduction(&relation.node).expect("the relation was built for a mean of squares");
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

impl<F: JoltField> RelationWitness<F> for MeanOfSquaresReductionProver<F> {
    type Relation = OpRel<F>;

    fn relation(&self) -> &OpRel<F> {
        &self.relation
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    fn produced_claims(&self) -> Vec<F> {
        vec![self.operand.final_claim()]
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
