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
use crate::onnx_proof::ops::{OpLayout, VerifyOp};
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use atlas_sumcheck::relation::verify_relation;

use atlas_trace::{op::MeanOfSquares, Op};

use crate::{
    atlas_relation_types,
    node_view::ComputationNode,
    onnx_proof::{
        clamp_lookups::{clamp_committed_polys, is_scalar, recover_small_int, verify_scalar_clamp},
        deferred_lookups::{DeferredBatch, DeferredOneHot},
        fused_rebase,
        ops::Verifier,
        range_checking::{
            range_check_operands::{MeanOfSquaresRangeCheckOperands, RangeCheckOperands},
            RangeCheckProvider,
        },
        relation::{
            reduced_output_setup, AtlasRelation, RelationDerived, RelationExpr, RelationId,
            RelationOpening, RelationSetup, RelationVerifier,
        },
        ProofId, ProofType,
    },
    utils::opening_access::{AccOpeningAccessor, OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::JoltField,
    poly::eq_poly::EqPolynomial,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_lookup_tables::unsigned_less_than::UnsignedLessThanTable;
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, OpeningId},
    shout::RaOneHotEncoding,
};
use atlas_transcript::Transcript;
use common::{consts::XLEN, CommittedPoly, VirtualPoly};
use jolt_claims::{constant, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

/// Degree of the execution sumcheck.
pub const DEGREE: usize = 3;

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
        verify_relation(
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
    /// The node being proven.
    pub node: ComputationNode,
}

impl MeanOfSquaresRelation {
    fn op(&self) -> &MeanOfSquares {
        let Op::MeanOfSquares(op) = &self.node.operator else {
            panic!("expected MeanOfSquares operator")
        };
        op
    }

    /// The retained (output) variables: the eq cube, bound in the first rounds.
    pub fn log_retained(&self) -> usize {
        self.node.pow2_padded_num_output_elements().log_2()
    }

    /// The reduced (summed) variables, bound last.
    pub fn log_reduced(&self) -> usize {
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

impl Produces for MeanOfSquaresRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![self.ids().nodeio(Target::Input(0))]
    }
}

impl<F: JoltField> Relation<F> for MeanOfSquaresRelation {
    fn opening_points(&self, _setup: &RelationSetup<F>, challenges: &[F]) -> Vec<Vec<F>> {
        vec![challenges.to_vec()]
    }

    fn derive(&self, id: &RelationDerived, setup: &RelationSetup<F>, challenges: &[F]) -> F {
        match id {
            RelationDerived::RetainedEq => {
                EqPolynomial::mle(&setup.incoming_point, &challenges[..self.log_retained()])
            }
            other => panic!("a mean of squares derives no {other:?}"),
        }
    }
}

impl<F: JoltField> SetupRelation<F> for MeanOfSquaresRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> RelationSetup<F> {
        reduced_output_setup(accumulator, &self.node)
    }
}

impl AtlasRelation for MeanOfSquaresRelation {
    type Spec = MeanOfSquares;

    fn node(&self) -> &ComputationNode {
        &self.node
    }
}
