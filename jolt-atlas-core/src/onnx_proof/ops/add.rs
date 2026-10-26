use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    onnx_proof::{
        clamp_lookups::{
            clamp_committed_polys, is_scalar, prove_clamp_lookup, verify_clamp_lookup,
            verify_scalar_clamp,
        },
        ops::OperatorProofTrait,
        relation::{
            prove_tie, verify_tie, AtlasRelation, EngineRelation, RelationChallenge,
            RelationDerived, RelationExpr, RelationId, RelationOpening, RelationSetup,
        },
        ProofId, Prover, Verifier,
    },
    utils::opening_access::{AccOpeningAccessor, OpeningIdBuilder, Target},
};
use atlas_trace::{
    op::{Add, OpSpec, Sub},
    trace::LayerData,
    Trace,
};
use common::{CommittedPoly, VirtualPoly};
use jolt_claims::{constant, opening, NoChallenges, NoInputs, NoOutputs, SymbolicSumcheck};
use jolt_field::Ring;
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{EvaluateMle, MultilinearPolynomial},
        opening_proof::{OpeningAccumulator, OpeningId},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use std::marker::PhantomData;

crate::bind_relation!(Add, AccumulationRelation<Add>);

/// Proof for saturating element-wise addition: `output = SatClamp(left + right)`.
///
/// The accumulation is widened to `i64` (`acc = left + right`) and then clamped
/// to `i32`, so `Add` is no longer a pure linear identity. The proof has three
/// parts (mirroring [`super::relu`], but over a 64-bit clamp address):
///
/// 1. **Clamp lookup** ([`ClampLookupProvider`], `ProofType::Execution`): a
///    prefix-suffix read-raf sumcheck proving `output(r) = SatClamp(acc(r))` and
///    tying the accumulation MLE ([`VirtualPoly::ClampAcc`], the `raf`) to the
///    one-hot read-address poly ([`VirtualPoly::ClampRa`]).
/// 2. **One-hot checks** ([`ClampEncoding`], `ProofType::RaOneHotChecks`):
///    booleanity + hamming-weight + ra-virtualization of `ClampRa` against its
///    committed decomposition ([`CommittedPoly::ClampRaD`]).
/// 3. **Operand tie** ([`AccumulationRelation`], no sumcheck): the prover opens
///    `left`/`right` at the same output point `r`, and the verifier checks
///    `left(r) + right(r) == acc(r)`. `acc(r)` is the lookup's `raf` claim,
///    already tied to `ClampRa` in (1).
///
/// A scalar node skips (1) and (2): its operands open in the clear and the verifier
/// recomputes the clamp.
///
/// The accumulation is re-executed from the operands at proving time
/// ([`clamp_intermediate`](crate::onnx_proof::clamp_lookups)); it is not stored
/// in the trace.
impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Add {
    #[tracing::instrument(skip_all, name = "Add::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        prove_accumulation::<Add, F, T>(node, prover)
    }

    #[tracing::instrument(skip_all, name = "Add::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_accumulation::<Add, F, T>(node, verifier)
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        clamp_committed_polys(node)
    }
}

/// A saturating `left ± right`.
pub trait AccumulationShape: OpSpec + 'static {
    /// `+1` for `Add`, `−1` for `Sub`.
    const SIGN: i64;
}

impl AccumulationShape for Add {
    const SIGN: i64 = 1;
}

impl AccumulationShape for Sub {
    const SIGN: i64 = -1;
}

/// The pre-clamp accumulation of a non-scalar saturating `left ± right`, a tie at the output
/// point `r`:
///
/// ```text
/// input claim   acc(r)                 (the clamp lookup's raf opening, `ClampAcc`)
/// output claim  left(r) ± right(r)
/// ```
///
/// **S ⊆ R.** The op accumulates `left ± right` in `i64` before the clamp, and the clamp
/// lookup's `acc` is that accumulation, so the identity holds at every element.
pub struct AccumulationRelation<S> {
    node: ComputationNode,
    _spec: PhantomData<fn() -> S>,
}

impl<S> AccumulationRelation<S> {
    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }
}

impl<S: AccumulationShape> SymbolicSumcheck for AccumulationRelation<S> {
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
        RelationId::Accumulation
    }

    fn rounds(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::Id(
            self.ids().advice(VirtualPoly::ClampAcc),
        ))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        let [left, right] = [0, 1].map(|position| {
            opening(RelationOpening::Id(
                self.ids().nodeio(Target::Input(position)),
            ))
        });
        left + constant(F::from_i64(S::SIGN)) * right
    }
}

impl<S: AccumulationShape> AtlasRelation for AccumulationRelation<S> {
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
}

impl<S: AccumulationShape> EngineRelation for AccumulationRelation<S> {
    fn witnesses<F: JoltField>(&self, trace: &Trace) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, &self.node);
        let [left, right] = operands[..] else {
            panic!("Expected two operands for {}", self.node.operator)
        };
        vec![
            MultilinearPolynomial::from(left.padded_pow2()),
            MultilinearPolynomial::from(right.padded_pow2()),
        ]
    }
}

/// Prove a saturating `left ± right`: the clamp lookup and the accumulation tie, or for a
/// scalar node the operands at the output point.
pub(crate) fn prove_accumulation<S: AccumulationShape, F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
    if is_scalar(node) {
        let (opening_point, _claim) =
            AccOpeningAccessor::new(&prover.accumulator, node).get_reduced_opening();
        let LayerData { operands, .. } = TraceView::layer_data(&prover.trace, node);
        let [left, right] = operands[..] else {
            panic!("Expected two operands for {}", node.operator)
        };
        let left_claim = left.evaluate_mle(&opening_point.r);
        let right_claim = right.evaluate_mle(&opening_point.r);
        let mut provider = AccOpeningAccessor::new(&mut prover.accumulator, node)
            .into_provider(&mut prover.transcript, opening_point);
        provider.append_nodeio(Target::Input(0), left_claim);
        provider.append_nodeio(Target::Input(1), right_claim);
        return Vec::new();
    }
    let results = prove_clamp_lookup(node, prover, None);
    prove_tie(
        &AccumulationRelation::<S>::new(node.clone()),
        &prover.trace,
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results
}

/// Verify a saturating `left ± right`: the clamp lookup and the accumulation tie, or for a
/// scalar node the clamp recomputed from the operands in the clear.
pub(crate) fn verify_accumulation<S: AccumulationShape, F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    if is_scalar(node) {
        let (opening_point, output_claim) =
            AccOpeningAccessor::new(&verifier.accumulator, node).get_reduced_opening();
        let mut provider = AccOpeningAccessor::new(&mut verifier.accumulator, node)
            .into_provider(&mut verifier.transcript, opening_point);
        provider.append_nodeio(Target::Input(0));
        provider.append_nodeio(Target::Input(1));
        let left_claim = provider.get_nodeio(Target::Input(0)).1;
        let right_claim = provider.get_nodeio(Target::Input(1)).1;
        drop(provider);
        let acc = left_claim + F::from_i64(S::SIGN) * right_claim;
        return verify_scalar_clamp(acc, output_claim, &node.operator.to_string());
    }
    verify_clamp_lookup(node, verifier)?;
    verify_tie(
        &AccumulationRelation::<S>::new(node.clone()),
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )
}
