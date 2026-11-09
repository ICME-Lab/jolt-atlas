use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
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
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::op::IsNan;
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
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        reduced_output_setup(accumulator, &self.node)
    }
}

impl AtlasRelation for ZeroOutputRelation {
    type Spec = IsNan;

    fn node(&self) -> &ComputationNode {
        &self.node
    }
}

impl<F: JoltField> TieRelation<F> for ZeroOutputRelation {}
