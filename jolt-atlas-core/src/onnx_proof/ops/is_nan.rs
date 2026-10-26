use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    atlas_relation_types, impl_tie_proof_api,
    onnx_proof::{
        ops::OperatorProofTrait,
        relation::{
            reduced_output_setup, AtlasRelation, EngineRelation, RelationExpr, RelationId,
            RelationOpening, RelationSetup,
        },
        ProofId, Prover, Verifier,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_trace::{op::IsNan, Trace};
use jolt_claims::{opening, Expr, SymbolicSumcheck};
use jolt_field::Ring;
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningAccumulator, OpeningId},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

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

impl AtlasRelation for ZeroOutputRelation {
    type Spec = IsNan;

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
        vec![OpeningIdBuilder::new(&self.node).nodeio(Target::Input(0))]
    }
}

impl EngineRelation for ZeroOutputRelation {
    fn witnesses<F: JoltField>(&self, trace: &Trace) -> Vec<MultilinearPolynomial<F>> {
        let operand = TraceView::operand_tensors(trace, &self.node)[0];
        vec![MultilinearPolynomial::from(operand.padded_pow2())]
    }
}
