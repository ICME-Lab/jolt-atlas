use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    impl_standard_sumcheck_proof_api,
    onnx_proof::{
        relation::{
            AtlasRelation, EngineRelation, RelationChallenge, RelationDerived, RelationExpr,
            RelationId, RelationOpening, RelationSetup,
        },
        ProofId, ProofType, Verifier,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{op::Iff, trace::LayerData, Trace};
use atlas_transcript::Transcript;
use jolt_claims::{
    constant, derived, opening, NoChallenges, NoInputs, NoOutputs, SymbolicSumcheck,
};
use jolt_field::Ring;

impl_standard_sumcheck_proof_api!(Iff, relation IffRelation);

/// Conditional selection `out = mask ? a : b` as a relation:
///
/// ```text
/// input claim   out(r₀)                (the reduced output opening)
/// output claim  eq(r₀, r′) · (mask·a + (1 − mask)·b)
/// ```
///
/// **S ⊆ R.** `Iff`'s semantics require a Boolean mask and compute `m·a + (1 − m)·b` exactly
/// (`atlas_trace::op::Iff`), so the output equals the summand at every hypercube point and its
/// MLE at `r₀` is the sum.
pub struct IffRelation {
    node: ComputationNode,
}

impl IffRelation {
    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }
}

impl SymbolicSumcheck for IffRelation {
    type RelationId = RelationId;
    type OpeningId = RelationOpening;
    type DerivedId = RelationDerived;
    type ChallengeId = RelationChallenge;
    type Shape = ComputationNode;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = NoInputs<C>;
    type Outputs<C> = NoOutputs<C>;

    fn new(node: ComputationNode) -> Self {
        Self { node }
    }

    fn id() -> RelationId {
        RelationId::Select
    }

    fn rounds(&self) -> usize {
        self.node.pow2_padded_num_output_elements().log_2()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::ReducedNodeOutput(self.node.idx))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        let [mask, a, b] = [0, 1, 2].map(|position| {
            opening(RelationOpening::Id(
                self.ids().nodeio(Target::Input(position)),
            ))
        });
        derived(RelationDerived::Eq) * (mask.clone() * a + (constant(F::one()) - mask) * b)
    }
}

impl Produces for IffRelation {
    fn produced(&self) -> Vec<OpeningId> {
        (0..3)
            .map(|position| self.ids().nodeio(Target::Input(position)))
            .collect()
    }
}

impl<F: JoltField> Relation<F> for IffRelation {}

impl<F: JoltField> SetupRelation<F> for IffRelation {
    fn setup<T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> RelationSetup<F> {
        RelationSetup {
            incoming_point: accumulator.get_node_output_opening(self.node.idx).0.r,
            challenges: Vec::new(),
        }
    }
}

impl AtlasRelation for IffRelation {
    type Spec = Iff;

    fn node(&self) -> &ComputationNode {
        &self.node
    }
}

impl<F: JoltField> EngineRelation<F> for IffRelation {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, &self.node);
        let [mask, a, b] = operands[..] else {
            panic!("Expected three operands for Iff operation")
        };
        [mask, a, b]
            .map(|operand| MultilinearPolynomial::from(operand.padded_pow2()))
            .into()
    }
}
