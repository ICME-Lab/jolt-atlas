use crate::node_view::ComputationNode;
use crate::onnx_proof::schedule::{OpPlan, ScheduleBuilder};
use crate::onnx_proof::{
    ops::OpLayout,
    relation::{
        not_derived, Custom, OpRel, OpRelKind, RelationDerived, RelationOpening, RelationSetup,
    },
};
use atlas_algebra::{
    field::JoltField, poly::multilinear::EvaluateMle, utils::errors::ProofVerifyError,
};
use atlas_claims::{derived, opening};
use atlas_trace::{op::Constant, Tensor};
use std::sync::Arc;

impl OpLayout for Constant {}

impl OpPlan for Constant {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
        _io: &atlas_trace::trace::ModelExecutionIO,
        schedule: &mut ScheduleBuilder<F>,
    ) -> Result<(), ProofVerifyError> {
        schedule.reduce();
        schedule.tie(public_value(node, OpRelKind::Constant, self.0.clone()));
        Ok(())
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
pub fn public_value<F: JoltField>(
    node: &ComputationNode,
    kind: OpRelKind,
    value: Arc<Tensor<i32>>,
) -> OpRel<F> {
    OpRel {
        degree: 1,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: derived(RelationDerived::Public),
        public: Some(value),
        custom: Custom {
            derive: Some(public_value_at_incoming_point),
            ..Custom::default()
        },
        ..OpRel::base(kind, node)
    }
}

/// [`RelationDerived::Public`] of a [`public_value`]: the public tensor's MLE at the incoming
/// point.
fn public_value_at_incoming_point<F: JoltField>(
    relation: &OpRel<F>,
    id: &RelationDerived,
    setup: &RelationSetup<F>,
    _challenges: &[F],
) -> Result<F, ProofVerifyError> {
    match (id, &relation.public) {
        (RelationDerived::Public, Some(value)) => Ok(value.evaluate_mle(&setup.incoming_point)),
        (RelationDerived::Public, None) => Err(ProofVerifyError::Malformed(format!(
            "node {}'s public-value tie holds no public tensor",
            relation.node.idx
        ))),
        (other, _) => Err(not_derived(other)),
    }
}
