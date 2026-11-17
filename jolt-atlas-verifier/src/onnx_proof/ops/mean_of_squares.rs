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
//! 2. **Reduction sumcheck** (`MeanOfSquaresReductionProver`, `RescaleArith`):
//!    `acc(r) = Σ_{h,j} eq(r,h)·x[h,j]²`, reducing to one input opening. Its
//!    initial claim is `acc(r) = out(r)·D + R(r)` (the rescale seam). This is the
//!    einsum **High** eq-schedule (`eq·op·op`, eq over the retained/high bits,
//!    free sum over the reduced/low bits) specialized to one operand.
//! 3. **Range claims** in the packed range buckets
//!    ([`global_clamp::node_range_claims`](crate::onnx_proof::global_clamp::node_range_claims)),
//!    after the node loop: the remainder bound `R < D` and, unless the node is a model
//!    output, the output range `out + 2^31 ∈ [0, 2^32)`.
//!
//! The reduction sums over the *padded* reduced cube (`op.padded_count`, a power
//! of two so the flat-padded operand splits cleanly into `[retained, reduced]`),
//! while the divisor `D = N·2^S` uses the *semantic* count `N` (= `op.count`,
//! any constant — e.g. GPT-2's 768). Zero-padded lanes contribute nothing to
//! `Σx²`, so the two are consistent.
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::relation::not_derived;
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};

use atlas_trace::{op::MeanOfSquares, trace::ModelExecutionIO, Model, Op};

use crate::{
    atlas_relation_types,
    node_view::ComputationNode,
    onnx_proof::{
        plan::{OpOpen, OpPlan, OpRelation, Plan, Step},
        relation::{
            reduced_output_setup, AtlasRelation, RelationDerived, RelationExpr, RelationId,
            RelationOpening, RelationSetup,
        },
        ProofType,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::field::Ring;
use atlas_algebra::{
    field::JoltField,
    poly::eq::EqPolynomial,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_claims::{constant, derived, opening, SymbolicSumcheck};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_transcript::Transcript;
use common::VirtualPoly;

/// Degree of the execution sumcheck.
pub const DEGREE: usize = 3;

/// Fused divisor `D = count·2^scale` (the semantic count, not the padded one).
pub fn mos_divisor(op: &MeanOfSquares, scale: u8) -> i64 {
    (1i64 << scale) * op.count as i64
}

impl OpLayout for MeanOfSquares {}

/// The plan: the evaluation reduction; the remainder advice (1); the reduction sumcheck (2).
/// The remainder bound `R < D` and the output's range are proven in the packed range buckets
/// after the node loop (`global_clamp::node_range_claims`).
impl OpPlan for MeanOfSquares {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Reduce,
            Step::Open(OpOpen::RescaleRemainder(node.clone())),
            Step::Relation {
                proof: ProofType::RescaleArith,
                relation: OpRelation::MeanOfSquares(MeanOfSquaresRelation::new(node.clone())),
            },
        ]))
    }
}

// ---------------------------------------------------------------------------
// Reduction sumcheck: acc(r) = Σ_{h,j} eq(r,h)·x[h,j]²
// ---------------------------------------------------------------------------

crate::bind_relation!(MeanOfSquares, MeanOfSquaresRelation);

/// The mean-of-squares reduction as a relation, binding high-to-low over the operand laid out
/// `[retained (high), reduced (low)]`:
///
/// ```text
/// input claim   out(r₀)·D + rem(r₀)           (D = N·2^S, the rescale divisor)
/// output claim  eq(r₀, r′_retained) · x(r′)²
/// ```
///
/// `MeanOfSquaresReductionProver` is its prover; the verifier is the relation's.
///
/// **S ⊆ R.** The op accumulates `acc = Σ_reduced x²` in `i64` per retained index and splits it
/// with `atlas_trace::semantics::fixed::rebase` by `D` into `out·D + rem`, failing when the
/// quotient leaves `i32`, so `Σ eq(r₀, h)·x(h, j)²` over the hypercube is `acc(r₀)` on every
/// execution that exists; zero-padded lanes add nothing to the sum. The bounds `rem < D` and
/// `out ∈ i32` are the range checks, outside the relation.
///
/// **No wrap.** The operand's cells are `i32`, so a sum of `n` squares lies in `[0, n·2^62]`.
/// The range claims give `out ∈ [−2^31, 2^31)` (a model output is a public `i32` tensor) and
/// `rem ∈ [0, D)` with `D < 2^31` (the range check requires the divisor to fit `i32`), so
/// `|out·D + rem| < 2^63`. For any tensor that fits in memory (`n < 2^64`) both sides are
/// integers of magnitude at most `2^126`, less than half the 128-bit field modulus, so their
/// equality in the field is their equality over the integers.
pub struct MeanOfSquaresRelation {
    /// The node being proven.
    pub node: ComputationNode,
    /// The rescale divisor `D` and log₂ of the padded reduced count; `None` unless the node is
    /// a mean of squares with a power-of-two padded reduced count, which the setup requires.
    reduction: Option<(i64, usize)>,
}

impl MeanOfSquaresRelation {
    fn reduction(&self) -> Result<(i64, usize), ProofVerifyError> {
        self.reduction.ok_or_else(|| {
            ProofVerifyError::Malformed(format!(
                "model: node {} is not a mean of squares with a power-of-two padded reduced count",
                self.node.idx
            ))
        })
    }

    /// The retained (output) variables: the eq cube, bound in the first rounds.
    pub fn log_retained(&self) -> usize {
        self.node.pow2_padded_num_output_elements().log_2()
    }

    /// The reduced (summed) variables, bound last.
    pub fn log_reduced(&self) -> usize {
        // The sumcheck cube splits the flat-padded operand into `[retained, reduced]`, so the
        // padded reduced count sizes the low rounds; the divisor stays semantic. The setup
        // rejects a node without a reduction, so the zero is never proven against.
        self.reduction.map_or(0, |(_, log_reduced)| log_reduced)
    }

    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }
}

impl SymbolicSumcheck for MeanOfSquaresRelation {
    atlas_relation_types!();

    fn id(&self) -> RelationId {
        RelationId::MeanOfSquares
    }

    fn rounds(&self) -> usize {
        self.log_retained() + self.log_reduced()
    }

    fn degree(&self) -> usize {
        DEGREE
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        // The setup rejects a node without a reduction, so the zero is never proven against.
        let divisor = self.reduction.map_or(0, |(divisor, _)| divisor);
        let remainder = self.ids().advice(VirtualPoly::RescaleRemainder);
        constant(F::from_u64(divisor as u64))
            * opening(RelationOpening::ReducedNodeOutput(self.node.idx))
            + opening(RelationOpening::Id(remainder))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        let x = opening(RelationOpening::Id(self.ids().nodeio(Target::Input(0))));
        derived(RelationDerived::RetainedEq) * x.clone() * x
    }
}

impl MeanOfSquaresRelation {
    /// The relation for the given shape.
    pub fn new(node: ComputationNode) -> Self {
        let reduction = match &node.operator {
            Op::MeanOfSquares(op) if op.padded_count.is_power_of_two() => {
                Some((mos_divisor(op, node.scale), op.padded_count.log_2()))
            }
            _ => None,
        };
        Self { node, reduction }
    }
}

impl Produces for MeanOfSquaresRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![self.ids().nodeio(Target::Input(0))]
    }
}

impl<F: JoltField> Relation<F> for MeanOfSquaresRelation {
    fn opening_points(
        &self,
        _setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<Vec<Vec<F>>, ProofVerifyError> {
        Ok(vec![challenges.to_vec()])
    }

    fn derive(
        &self,
        id: &RelationDerived,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<F, ProofVerifyError> {
        Ok(match id {
            RelationDerived::RetainedEq => {
                let retained = challenges.get(..self.log_retained()).ok_or_else(|| {
                    ProofVerifyError::Malformed(format!(
                        "mean of squares: {} challenges for {} retained variables",
                        challenges.len(),
                        self.log_retained()
                    ))
                })?;
                EqPolynomial::mle(&setup.incoming_point, retained)
            }
            other => return Err(not_derived(other)),
        })
    }
}

impl<F: JoltField> SetupRelation<F> for MeanOfSquaresRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        self.reduction()?;
        reduced_output_setup(accumulator, &self.node)
    }
}

impl AtlasRelation for MeanOfSquaresRelation {
    type Spec = MeanOfSquares;

    fn node(&self) -> &ComputationNode {
        &self.node
    }
}
