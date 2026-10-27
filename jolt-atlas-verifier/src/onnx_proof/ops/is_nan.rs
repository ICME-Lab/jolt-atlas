use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    atlas_relation_types, impl_tie_proof_api,
    onnx_proof::{
        relation::{
            reduced_output_setup, AtlasRelation, RelationExpr, RelationId, RelationOpening,
            RelationSetup, TieRelation,
        },
        Verifier,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::JoltField, poly::multilinear_polynomial::EvaluateMle, utils::errors::ProofVerifyError,
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{op::IsNan, Trace};
use atlas_transcript::Transcript;
use jolt_claims::{opening, Expr, SymbolicSumcheck};
use jolt_field::Ring;

impl_tie_proof_api!(IsNan, ZeroOutputRelation);

/// `IsNan` as a tie at the output point `r`: its output is zero everywhere.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  0
/// ```
///
/// The operand is still opened at `r` and handed to its producer's reduction, though the
/// output expression does not read it.
///
/// **S ⊆ R.** Fixed-point data is never NaN, so `atlas_trace::op::IsNan` outputs zeros.
pub struct ZeroOutputRelation {
    node: ComputationNode,
}

impl SymbolicSumcheck for ZeroOutputRelation {
    atlas_relation_types!();
    type Shape = ComputationNode;

    fn new(node: ComputationNode) -> Self {
        Self { node }
    }

    fn id() -> RelationId {
        RelationId::ZeroOutput
    }

    fn rounds(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::ReducedNodeOutput(self.node.idx))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        Expr::zero()
    }
}

impl Produces for ZeroOutputRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![OpeningIdBuilder::new(&self.node).nodeio(Target::Input(0))]
    }
}

impl<F: JoltField> Relation<F> for ZeroOutputRelation {}

impl<F: JoltField> SetupRelation<F> for ZeroOutputRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> RelationSetup<F> {
        reduced_output_setup(accumulator, &self.node)
    }
}

impl AtlasRelation for ZeroOutputRelation {
    type Spec = IsNan;

    fn node(&self) -> &ComputationNode {
        &self.node
    }
}

impl<F: JoltField> TieRelation<F> for ZeroOutputRelation {
    fn claims(
        &self,
        trace: &Trace,
        setup: &RelationSetup<F>,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        let operand = TraceView::operand_tensors(trace, &self.node)[0];
        vec![operand.evaluate_mle(&setup.incoming_point)]
    }
}
