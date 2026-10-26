use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    impl_fused_rescale_proof_api,
    onnx_proof::{
        fused_rebase,
        ops::OperatorProofTrait,
        relation::{
            AtlasRelation, RelationChallenge, RelationDerived, RelationExpr, RelationId,
            RelationOpening, RelationSetup,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_trace::{
    op::{Mul, OpSpec},
    trace::LayerData,
    Trace,
};
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
use std::marker::PhantomData;

impl_fused_rescale_proof_api!(Mul, relation MulRelation);

/// Element-wise multiplication as a relation.
pub type MulRelation = ProductRelation<Mul>;

/// An element-wise product `out = L · R` (`Mul`, and `And` on booleans) as a relation:
///
/// ```text
/// input claim   q(r₀)·2^S + rem(r₀)    (fused rescale: pre-clamp quotient and remainder)
///               out(r₀)                (raw product: the reduced output opening)
/// output claim  eq(r₀, r′) · L(r′) · R(r′)
/// ```
///
/// The incoming point is the node output's reduced opening point. The clamp lookup and the
/// remainder range check of a fused node are separate protocols ([`fused_rebase`]) around this
/// one.
///
/// **S ⊆ R.** A raw product's output is `L·R` exactly, so `Σ eq(r₀,·)·L·R` is the output's MLE
/// at `r₀`. A fused product splits the `i64` accumulation `L·R` with
/// `atlas_trace::semantics::fixed::rebase` into `q·2^S + rem` (Euclidean), so the same sum
/// equals `q(r₀)·2^S + rem(r₀)` for the pre-clamp quotient `q`; the clamped output is never
/// used here, so saturated executions satisfy it too.
pub struct ProductRelation<S> {
    node: ComputationNode,
    _spec: PhantomData<fn() -> S>,
}

impl<S> ProductRelation<S> {
    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }
}

impl<S: OpSpec + 'static> SymbolicSumcheck for ProductRelation<S> {
    type RelationId = RelationId;
    type OpeningId = RelationOpening;
    type DerivedId = RelationDerived;
    type ChallengeId = RelationChallenge;
    type Shape = ComputationNode;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = NoInputs<C>;
    type Outputs<C> = NoOutputs<C>;

    fn new(node: ComputationNode) -> Self {
        Self {
            node,
            _spec: PhantomData,
        }
    }

    fn id() -> RelationId {
        RelationId::Product
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

impl<S: OpSpec + 'static> AtlasRelation for ProductRelation<S> {
    type Spec = S;

    fn node(&self) -> &ComputationNode {
        &self.node
    }

    fn setup<F: JoltField, T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> RelationSetup<F> {
        RelationSetup {
            incoming_point: accumulator.get_node_output_opening(self.node.idx).0.r,
            challenges: Vec::new(),
        }
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
            panic!("Expected two operands for a product")
        };
        vec![
            MultilinearPolynomial::from(left.padded_pow2()),
            MultilinearPolynomial::from(right.padded_pow2()),
        ]
    }
}
