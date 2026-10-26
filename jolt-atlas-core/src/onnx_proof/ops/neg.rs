use crate::node_view::ComputationNode;
use crate::{
    atlas_relation_types, impl_tie_proof_api,
    onnx_proof::{
        ops::OperatorProofTrait,
        relation::{
            reduced_output_setup, AtlasRelation, RelationExpr, RelationId, RelationOpening,
            RelationSetup, TieRelation,
        },
        ProofId, Prover, Verifier,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_trace::{
    op::{Identity, Neg, OpSpec},
    Trace,
};
use jolt_claims::{constant, opening, SymbolicSumcheck};
use jolt_field::Ring;
use joltworks::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, OpeningId},
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use std::marker::PhantomData;

impl_tie_proof_api!(Neg, SignedCopyRelation<Neg>);

/// An op whose output is `±` its single operand.
pub trait CopySign: OpSpec + 'static {
    /// `+1` for `Identity`, `−1` for `Neg`.
    const SIGN: i64;
}

impl CopySign for Identity {
    const SIGN: i64 = 1;
}

impl CopySign for Neg {
    const SIGN: i64 = -1;
}

/// `out = ±operand` as a tie at the output point `r`: no sumcheck, the identity holds at `r`
/// because it holds on the hypercube and both sides are multilinear.
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  ±operand(r)
/// ```
///
/// **S ⊆ R.** `Identity` passes its operand through. `Neg` negates element-wise with
/// `saturating_neg`, which agrees with field negation everywhere except `i32::MIN`, sent to
/// `i32::MAX`: an execution negating `i32::MIN` fails the tie (a completeness gap; the
/// verifier accepts no wrong output).
pub struct SignedCopyRelation<S> {
    node: ComputationNode,
    _spec: PhantomData<fn() -> S>,
}

impl<S: CopySign> SymbolicSumcheck for SignedCopyRelation<S> {
    atlas_relation_types!();
    type Shape = ComputationNode;

    fn new(node: ComputationNode) -> Self {
        Self {
            node,
            _spec: PhantomData,
        }
    }

    fn id() -> RelationId {
        RelationId::SignedCopy
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
        constant(F::from_i64(S::SIGN)) * opening(RelationOpening::Id(self.produced()[0]))
    }
}

impl<S: CopySign> AtlasRelation for SignedCopyRelation<S> {
    type Spec = S;

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

impl<S: CopySign> TieRelation for SignedCopyRelation<S> {
    fn claims<F: JoltField>(
        &self,
        _trace: &Trace,
        _setup: &RelationSetup<F>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        // operand(r) = ±out(r): the reduced output claim, signed.
        let output = accumulator.get_node_output_opening(self.node.idx).1;
        vec![F::from_i64(S::SIGN) * output]
    }
}
