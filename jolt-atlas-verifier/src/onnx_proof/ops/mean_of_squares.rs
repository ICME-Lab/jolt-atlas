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
use crate::onnx_proof::ops::OpLayout;
use crate::onnx_proof::relation::not_derived;
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};

use atlas_trace::{op::MeanOfSquares, trace::ModelExecutionIO, Model, Op};

use crate::{
    atlas_relation_types,
    node_view::ComputationNode,
    onnx_proof::{
        clamp_lookups::is_scalar,
        deferred_lookups::DeferredBatch,
        plan::{clamp_steps, OpOpen, OpPlan, OpRangeCheck, OpRelation, OpTie, Plan, Step},
        range_checking::range_check_operands::{
            MeanOfSquaresRangeCheckOperands, RangeCheckOperands,
        },
        relation::{
            reduced_output_setup, AtlasRelation, RelationDerived, RelationExpr, RelationId,
            RelationOpening, RelationSetup,
        },
        ProofType,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::JoltField,
    poly::eq_poly::EqPolynomial,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, OpeningId},
    shout::RaOneHotEncoding,
};
use atlas_transcript::Transcript;
use common::{CommittedPoly, VirtualPoly};
use jolt_claims::{constant, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

/// Degree of the execution sumcheck.
pub const DEGREE: usize = 3;

/// Fused divisor `D = count·2^scale` (the semantic count, not the padded one).
pub fn mos_divisor(op: &MeanOfSquares, scale: u8) -> i64 {
    (1i64 << scale) * op.count as i64
}

impl OpLayout for MeanOfSquares {
    fn reduction_flow(&self) -> crate::onnx_proof::ops::ReductionFlow {
        crate::onnx_proof::ops::ReductionFlow::Custom
    }

    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        // The saturating clamp's one-hots live in the packed clamp buckets; here only the
        // `R < D` range-check one-hot (skipped for scalar outputs).
        let mut polys = Vec::new();
        if !is_scalar(node) {
            let rc_operands = RangeCheckOperands::<MeanOfSquaresRangeCheckOperands>::new(node)?;
            let d = rc_operands
                .get_encoding(node)
                .one_hot_params()
                .instruction_d;
            polys.extend((0..d).map(|i| CommittedPoly::MeanOfSquaresRangeCheckRaD(node.idx, i)));
        }
        Ok(polys)
    }
}

/// The plan: the evaluation reduction; the remainder advice and the clamp of the pre-clamp
/// quotient (1+2); the reduction sumcheck (3); the remainder range check `R < D` (4), in the
/// clear for a scalar node and in the post-loop range-check batches otherwise.
impl OpPlan for MeanOfSquares {
    fn plan(
        &self,
        node: &ComputationNode,
        model: &Model,
        io: &ModelExecutionIO,
    ) -> Result<Plan, ProofVerifyError> {
        let mut steps = vec![
            Step::Reduce,
            Step::Open(OpOpen::RescaleRemainder(node.clone())),
        ];
        steps.extend(clamp_steps(node, model, io));
        steps.push(Step::Relation {
            proof: ProofType::RescaleArith,
            relation: OpRelation::MeanOfSquares(MeanOfSquaresRelation::new(node.clone())),
        });
        if is_scalar(node) {
            steps.push(Step::Tie {
                relation: OpTie::ScalarClamp(node.clone()),
            });
            steps.push(Step::Tie {
                relation: OpTie::ScalarRemainder {
                    node: node.clone(),
                    bound: mos_divisor(self, node.scale),
                },
            });
        } else {
            steps.push(Step::Defer {
                batch: DeferredBatch::RangeCheck,
                relation: OpRelation::RangeCheck(OpRangeCheck::MeanOfSquares, node.clone()),
            });
            steps.push(Step::Defer {
                batch: DeferredBatch::RangeCheckRaChecks,
                relation: OpRelation::RangeCheckOneHot(OpRangeCheck::MeanOfSquares, node.clone()),
            });
        }
        Ok(Plan(steps))
    }
}

// ---------------------------------------------------------------------------
// Reduction sumcheck: acc(r) = Σ_{h,j} eq(r,h)·x[h,j]²
// ---------------------------------------------------------------------------

crate::impl_plan_op!(@verify MeanOfSquares);
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
    type Shape = ComputationNode;

    fn new(node: ComputationNode) -> Self {
        let reduction = match &node.operator {
            Op::MeanOfSquares(op) if op.padded_count.is_power_of_two() => {
                Some((mos_divisor(op, node.scale), op.padded_count.log_2()))
            }
            _ => None,
        };
        Self { node, reduction }
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
        // The setup rejects a node without a reduction, so the zero is never proven against.
        let divisor = self.reduction.map_or(0, |(divisor, _)| divisor);
        let rescaled = self.ids().advice(VirtualPoly::ClampAcc);
        let remainder = self.ids().advice(VirtualPoly::RescaleRemainder);
        constant(F::from_u64(divisor as u64)) * opening(RelationOpening::Id(rescaled))
            + opening(RelationOpening::Id(remainder))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        let x = opening(RelationOpening::Id(self.ids().nodeio(Target::Input(0))));
        derived(RelationDerived::RetainedEq) * x.clone() * x
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
