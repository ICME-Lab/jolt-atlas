use crate::node_view::ComputationNode;
use crate::node_view::ModelView;
use crate::onnx_proof::relation::{
    not_derived, Custom, OpRel, OpRelKind, RelationDerived, RelationOpening, RelationSetup,
};
use crate::onnx_proof::schedule::{OpPlan, ScheduleBuilder};
use atlas_algebra::{
    field::JoltField, poly::multilinear::EvaluateMle, utils::errors::ProofVerifyError,
};
use atlas_claims::{derived, opening};
use atlas_trace::{op::Constant, trace::ModelExecutionIO, Model, Tensor};
use std::sync::Arc;

impl OpPlan for Constant {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &atlas_trace::Model,
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

/// The public-value tie of the input node `node`: [`public_value`] with the tensor left out
/// and the input's position among the model's inputs recorded. The public IO supplies the
/// tensor when the tie is checked ([`with_public_input`]). `Err` if the node is not one of the
/// model's inputs.
pub fn public_input<F: JoltField>(
    model: &Model,
    node: &ComputationNode,
) -> Result<OpRel<F>, ProofVerifyError> {
    let position = model
        .input_indices()
        .iter()
        .position(|&idx| idx == node.idx)
        .ok_or_else(|| {
            ProofVerifyError::Malformed(format!(
                "model: input node {} is not one of the model's inputs",
                node.idx
            ))
        })?;
    Ok(OpRel {
        degree: 1,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: derived(RelationDerived::Public),
        public_input: Some(position),
        custom: Custom {
            derive: Some(public_value_at_incoming_point),
            ..Custom::default()
        },
        ..OpRel::base(OpRelKind::Input, node)
    })
}

/// An input's tie ([`public_input`]) with its public tensor filled in from `io`. `Err` if the
/// relation names no input or `io` carries no tensor at its position.
pub fn with_public_input<F: JoltField>(
    relation: &OpRel<F>,
    io: &ModelExecutionIO,
) -> Result<OpRel<F>, ProofVerifyError> {
    let input = relation
        .public_input
        .and_then(|position| io.inputs.get(position))
        .ok_or_else(|| {
            ProofVerifyError::Malformed(format!(
                "public IO: no tensor for input node {}",
                relation.node.idx
            ))
        })?;
    Ok(OpRel {
        public: Some(Arc::new(input.clone())),
        ..relation.clone()
    })
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
