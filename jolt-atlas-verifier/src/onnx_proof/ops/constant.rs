use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{OpLayout, VerifyOp};
use crate::onnx_proof::relation::{Relation, SetupRelation};
use crate::{
    atlas_relation_types,
    onnx_proof::{
        ops::Verifier,
        relation::{
            reduced_output_setup, verify_tie, AtlasRelation, RelationDerived, RelationExpr,
            RelationId, RelationOpening, RelationSetup, TieRelation,
        },
    },
};
use atlas_algebra::{
    field::JoltField, poly::multilinear_polynomial::EvaluateMle, utils::errors::ProofVerifyError,
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_trace::{
    op::{Constant, OpSpec},
    Tensor, Trace,
};
use atlas_transcript::Transcript;
use jolt_claims::{derived, opening, SymbolicSumcheck};
use jolt_field::Ring;
use std::{marker::PhantomData, sync::Arc};

crate::bind_relation!(Constant, PublicRelation<Constant>);

impl OpLayout for Constant {}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for Constant {
    #[tracing::instrument(skip_all, name = "Constant::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_tie(
            &PublicRelation::<Constant>::new(PublicShape::new(node.clone(), self.0.clone())),
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )
    }
}

/// A node and the public data its output must equal.
pub struct PublicShape {
    node: ComputationNode,
    value: Arc<Tensor<i32>>,
}

impl PublicShape {
    /// `node` with its public `value`.
    pub fn new(node: ComputationNode, value: Arc<Tensor<i32>>) -> Self {
        Self { node, value }
    }
}

/// An output equal to public data the verifier holds (a model constant, a public input), a tie at
/// the output point `r`:
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  value(r)               (the public data's MLE, derived)
/// ```
///
/// **S ⊆ R.** `Constant` outputs its tensor and `Input` the supplied input, which the verifier
/// holds as the public IO.
pub struct PublicRelation<S> {
    shape: PublicShape,
    _spec: PhantomData<fn() -> S>,
}

impl<S: OpSpec + 'static> SymbolicSumcheck for PublicRelation<S> {
    atlas_relation_types!();
    type Shape = PublicShape;

    fn new(shape: PublicShape) -> Self {
        Self {
            shape,
            _spec: PhantomData,
        }
    }

    fn id() -> RelationId {
        RelationId::PublicValue
    }

    fn rounds(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        opening(RelationOpening::ReducedNodeOutput(self.shape.node.idx))
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        derived(RelationDerived::Public)
    }
}

impl<S: OpSpec + 'static> Relation for PublicRelation<S> {
    fn produced(&self) -> Vec<OpeningId> {
        Vec::new()
    }

    fn derive<F: JoltField>(
        &self,
        id: &RelationDerived,
        setup: &RelationSetup<F>,
        _challenges: &[F],
    ) -> F {
        match id {
            RelationDerived::Public => self.shape.value.evaluate_mle(&setup.incoming_point),
            other => panic!("public data derives no {other:?}"),
        }
    }
}

impl<S: OpSpec + 'static> SetupRelation for PublicRelation<S> {
    fn setup<F: JoltField, T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut T,
    ) -> RelationSetup<F> {
        reduced_output_setup(accumulator, &self.shape.node)
    }
}

impl<S: OpSpec + 'static> AtlasRelation for PublicRelation<S> {
    type Spec = S;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }
}

impl<S: OpSpec + 'static> TieRelation for PublicRelation<S> {
    fn claims<F: JoltField>(
        &self,
        _trace: &Trace,
        _setup: &RelationSetup<F>,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }
}
