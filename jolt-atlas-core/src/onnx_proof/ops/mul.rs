use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    impl_fused_rescale_proof_api,
    onnx_proof::{
        fused_rebase,
        ops::OperatorProofTrait,
        relation::{
            AtlasRelation, RelationChallenge, RelationDerived, RelationExpr, RelationId,
            RelationOpening, RelationParams, RelationProver, RelationVerifier,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_trace::{op::Mul, trace::LayerData, Trace};
use common::VirtualPoly;
use jolt_claims::{
    constant, derived, opening, NoChallenges, NoInputs, NoOutputs, SymbolicSumcheck,
};
use jolt_field::Ring;
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningAccumulator, OpeningId},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

impl_fused_rescale_proof_api!(Mul, MulParams, MulProver, MulVerifier);

/// Element-wise multiplication as a relation:
///
/// ```text
/// input claim   q(r₀)·2^S + rem(r₀)    (fused: pre-clamp quotient and remainder)
///               out(r₀)                (raw product: the reduced output opening)
/// output claim  eq(r₀, r′) · L(r′) · R(r′)
/// ```
///
/// The clamp lookup and the remainder range check of a fused node are separate protocols
/// ([`fused_rebase`]) around this one.
#[derive(Clone)]
pub struct MulRelation {
    node: ComputationNode,
}

impl MulRelation {
    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }
}

impl SymbolicSumcheck for MulRelation {
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
        RelationId::Mul
    }

    fn rounds(&self) -> usize {
        self.node.pow2_padded_num_output_elements().log_2()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        match fused_rebase::rebase_bits(&self.node) {
            Some(bits) if bits > 0 => {
                let quotient = self.ids().advice(VirtualPoly::ClampAcc);
                let remainder = self.ids().advice(VirtualPoly::RescaleRemainder);
                constant(F::pow2(bits as usize)) * opening(RelationOpening::Id(quotient))
                    + opening(RelationOpening::Id(remainder))
            }
            _ => opening(RelationOpening::ReducedNodeOutput(self.node.idx)),
        }
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        let [left, right] = [Target::Input(0), Target::Input(1)]
            .map(|operand| opening(RelationOpening::Id(self.ids().nodeio(operand))));
        derived(RelationDerived::Eq) * left * right
    }
}

impl AtlasRelation for MulRelation {
    fn incoming_point<F: JoltField>(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F> {
        accumulator.get_node_output_opening(self.node.idx).0.r
    }

    fn produced(&self) -> Vec<OpeningId> {
        vec![
            self.ids().nodeio(Target::Input(0)),
            self.ids().nodeio(Target::Input(1)),
        ]
    }

    fn witnesses<F: JoltField>(&self, trace: &Trace) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, &self.node);
        let [left, right] = operands[..] else {
            panic!("Expected two operands for Mul operation")
        };
        vec![
            MultilinearPolynomial::from(left.padded_pow2()),
            MultilinearPolynomial::from(right.padded_pow2()),
        ]
    }
}

/// The Mul relation with its incoming point.
pub type MulParams<F> = RelationParams<F, MulRelation>;
/// The Mul relation's prover.
pub type MulProver<F> = RelationProver<F, MulRelation>;
/// The Mul relation's verifier.
pub type MulVerifier<F> = RelationVerifier<F, MulRelation>;
