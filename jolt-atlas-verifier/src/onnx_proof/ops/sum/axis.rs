use atlas_algebra::utils::errors::ProofVerifyError;

use atlas_trace::Model;

use crate::node_view::{ComputationNode, ModelView};
use crate::{
    onnx_proof::relation::{Custom, OpRel, OpRelKind, RelationOpening, RelationSetup},
    utils::{
        self,
        dims::{SumAxis, SumConfig},
        opening_access::{OpeningIdBuilder, Target},
    },
};
use atlas_algebra::{field::JoltField, poly::multilinear::BindingOrder, utils::math::Math};
use atlas_claims::opening;

/// The axis reduction of an exact sum as a dense relation, binding the summed axis
/// high-to-low over the operand folded at the output point `r`:
///
/// ```text
/// input claim   out(r)                 (the reduced output opening)
/// output claim  operand(ρ)             (ρ = (r′, r) along axis 0, (r, r′) along axis 1)
/// ```
///
/// The relation's [`dims`](OpRel::dims) are the operand's. `Err` if the node is not a
/// single-axis sum the reduction supports.
///
/// **S ⊆ R.** The op computes `out = Σ_axis operand` exactly and its execution fails when a
/// sum leaves `i32`, so `out(r) = Σ_x operand(x, r)` (or `(r, x)`) on every execution that
/// exists.
///
/// **No wrap.** The operand's cells are `i32`, so a sum over an axis of `n` cells has
/// magnitude at most `n·2^31`, and `out` lies in `[−2^31, 2^31)` by its range claim (a model
/// output by being a public `i32` tensor). For any tensor that fits in memory (`n < 2^64`)
/// both sides are integers of magnitude below `2^95`, far below half the 128-bit field
/// modulus, so their equality in the field is their equality over the integers.
pub fn axis_sum<F: JoltField>(
    model: &Model,
    node: &ComputationNode,
) -> Result<OpRel<F>, ProofVerifyError> {
    let config = utils::dims::sum_config(node, model)?;
    let axis_len = config
        .operand_dims()
        .get(config.axis().axis_index())
        .copied()
        .ok_or_else(|| {
            ProofVerifyError::Malformed(format!(
                "model: sum node {}: the summed axis is not an operand axis",
                node.idx
            ))
        })?;
    let input_dims = node
        .inputs
        .first()
        .map(|&input| model.view(input).output_dims)
        .ok_or_else(|| {
            ProofVerifyError::Malformed(format!("model: sum node {}: no operand", node.idx))
        })?;
    let operand = OpeningIdBuilder::new(node).nodeio(Target::Input(0));
    Ok(OpRel {
        rounds: axis_len.log_2(),
        degree: 1,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx)),
        output: opening(RelationOpening::Id(operand)),
        produced: vec![operand],
        binding: BindingOrder::HighToLow,
        dims: vec![input_dims],
        custom: Custom {
            points: Some(operand_point),
            ..Custom::default()
        },
        ..OpRel::base(OpRelKind::SumAxis, node)
    })
}

/// The normalized 2D shape and axis of an [`axis_sum`].
pub fn config<F: JoltField>(relation: &OpRel<F>) -> Result<SumConfig, ProofVerifyError> {
    utils::dims::sum_config_over(&relation.node, relation.dims_at(0)?)
}

/// The operand's point of an [`axis_sum`]: the bound axis coordinates beside the incoming
/// point, in the operand's axis order.
fn operand_point<F: JoltField>(
    relation: &OpRel<F>,
    setup: &RelationSetup<F>,
    challenges: &[F],
) -> Result<Vec<Vec<F>>, ProofVerifyError> {
    let r = &setup.incoming_point;
    Ok(vec![match config(relation)?.axis() {
        SumAxis::Axis0 => [challenges, r.as_slice()].concat(),
        SumAxis::Axis1 => [r.as_slice(), challenges].concat(),
    }])
}
