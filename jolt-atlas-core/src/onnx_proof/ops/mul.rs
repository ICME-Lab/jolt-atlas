use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    impl_fused_rescale_proof_api,
    onnx_proof::{
        fused_rebase,
        relation::{
            AtlasRelation, EngineRelation, RelationChallenge, RelationDerived, RelationExpr,
            RelationId, RelationOpening, RelationSetup,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_trace::{
    op::{And, Cube, Mul, OpSpec, Square},
    trace::LayerData,
    Trace,
};
use jolt_claims::{derived, opening, NoChallenges, NoInputs, NoOutputs, SymbolicSumcheck};
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

/// An op whose output is a product of its operands, by operand position.
pub trait ProductShape {
    /// The operand position of each factor: `[0, 1]` for `L·R`, `[0, 0]` for `X²`.
    const FACTORS: &'static [usize];
}

impl ProductShape for Mul {
    const FACTORS: &'static [usize] = &[0, 1];
}

impl ProductShape for And {
    const FACTORS: &'static [usize] = &[0, 1];
}

impl ProductShape for Square {
    const FACTORS: &'static [usize] = &[0, 0];
}

impl ProductShape for Cube {
    const FACTORS: &'static [usize] = &[0, 0, 0];
}

/// An element-wise product of operands (`Mul`, `Square`, `Cube`, and `And` on booleans) as a
/// relation, `Π` running over the op's [`ProductShape::FACTORS`]:
///
/// ```text
/// input claim   q(r₀)·2^S + rem(r₀)    (fused rescale: pre-clamp quotient and remainder)
///               out(r₀)                (raw product: the reduced output opening)
/// output claim  eq(r₀, r′) · Π operand(r′)
/// ```
///
/// The incoming point is the node output's reduced opening point. The clamp lookup and the
/// remainder range check of a fused node are separate protocols ([`fused_rebase`]) around this
/// one.
///
/// **S ⊆ R.** A raw product's output is `Π operand` exactly, so `Σ eq(r₀,·)·Π operand` is the
/// output's MLE at `r₀`. A fused product splits the `i64` accumulation `Π operand` with
/// `atlas_trace::semantics::fixed::rebase` into `q·2^S + rem` (Euclidean, `S` the op's rebase
/// bits: twice the scale for `Cube`), so the same sum equals `q(r₀)·2^S + rem(r₀)` for the
/// pre-clamp quotient `q`; the clamped output is never used here, so saturated executions
/// satisfy it too.
pub struct ProductRelation<S> {
    node: ComputationNode,
    _spec: PhantomData<fn() -> S>,
}

impl<S: ProductShape> ProductRelation<S> {
    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }

    /// The operand positions the product reads, each once, in first-factor order.
    fn operands() -> Vec<usize> {
        let mut operands = Vec::new();
        for &position in S::FACTORS {
            if !operands.contains(&position) {
                operands.push(position);
            }
        }
        operands
    }
}

impl<S: OpSpec + ProductShape + 'static> SymbolicSumcheck for ProductRelation<S> {
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
        S::FACTORS.len() + 1
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        fused_rebase::input_expression(&self.node)
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        S::FACTORS
            .iter()
            .fold(derived(RelationDerived::Eq), |product, &position| {
                product
                    * opening(RelationOpening::Id(
                        self.ids().nodeio(Target::Input(position)),
                    ))
            })
    }
}

impl<S: OpSpec + ProductShape + 'static> AtlasRelation for ProductRelation<S> {
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
        Self::operands()
            .into_iter()
            .map(|position| self.ids().nodeio(Target::Input(position)))
            .collect()
    }
}

impl<S: OpSpec + ProductShape + 'static> EngineRelation for ProductRelation<S> {
    fn witnesses<F: JoltField>(
        &self,
        trace: &Trace,
        _setup: &RelationSetup<F>,
    ) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, &self.node);
        Self::operands()
            .into_iter()
            .map(|position| MultilinearPolynomial::from(operands[position].padded_pow2()))
            .collect()
    }
}
