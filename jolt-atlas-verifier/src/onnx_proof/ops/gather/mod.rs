use crate::node_view::ComputationNode;
use crate::node_view::ModelView;
use crate::onnx_proof::relation::{
    low_to_high_point, not_derived, Custom, OpRel, OpRelKind, RelationChallenge, RelationDerived,
    RelationOpening, RelationSetup, SetupOpen,
};
use crate::utils::opening_access::{OpeningIdBuilder, Target};
use atlas_algebra::{
    field::JoltField,
    poly::{identity::IdentityPolynomial, multilinear::PolynomialEvaluation},
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_claims::{challenge, derived, opening};
use atlas_trace::{Model, Op};
use common::VirtualPoly;

/// Gather over a large dictionary (read-raf lookup).
pub mod large;
/// Gather over a small dictionary (one-hot reads).
pub mod small;

/// Degree of the execution sumcheck.
pub const DEGREE_BOUND: usize = 2;

fn unsupported(node: &ComputationNode, what: impl std::fmt::Display) -> ProofVerifyError {
    ProofVerifyError::Malformed(format!("model: gather node {}: {what}", node.idx))
}

/// The view of a gather node's operand `position` (0 the dictionary, 1 the indices); `Err` if
/// it is not a node of the model.
fn operand(
    model: &Model,
    node: &ComputationNode,
    position: usize,
) -> Result<ComputationNode, ProofVerifyError> {
    node.inputs
        .get(position)
        .copied()
        .filter(|&input| input < model.num_nodes())
        .map(|input| model.view(input))
        .ok_or_else(|| unsupported(node, format!("operand {position} is not a node")))
}

/// The read of a gather's indices from its dictionary:
///
/// ```text
/// input claim   out(r_out) + γ·index(r_index)           (r_out = r_index ‖ r_word)
/// output claim  ra(ρ ‖ r_index) · (dict(ρ ‖ r_word) + γ·id(ρ))
/// ```
///
/// The relation's [`dims`](OpRel::dims) are the index operand's. `Err` if the node is not a
/// gather or its index operand is not a node of the model.
///
/// **S ⊆ R.** Each output row is the dictionary row its index names, so an honest one-hot `ra`
/// reads the output rows and, through the identity, the indices. The setup takes the reduced
/// output point as its incoming point, draws `γ`, and appends the index opening at `r_index`,
/// the leading coordinates of that point.
pub fn gather<F: JoltField>(
    model: &Model,
    node: &ComputationNode,
) -> Result<OpRel<F>, ProofVerifyError> {
    let num_words = num_words(node)?;
    let indices = operand(model, node, 1)?;
    let ids = OpeningIdBuilder::new(node);
    let index = ids.nodeio(Target::Input(1));
    let ra = ids.advice(VirtualPoly::NodeOutputRa);
    let dict = ids.nodeio(Target::Input(0));
    Ok(OpRel {
        rounds: num_words.log_2(),
        degree: DEGREE_BOUND,
        input: opening(RelationOpening::ReducedNodeOutput(node.idx))
            + challenge(RelationChallenge::Gamma) * opening(RelationOpening::Id(index)),
        output: opening(RelationOpening::Id(ra))
            * (opening(RelationOpening::Id(dict))
                + challenge(RelationChallenge::Gamma) * derived(RelationDerived::Identity)),
        produced: vec![ra, dict],
        draws: vec![RelationChallenge::Gamma],
        setup_opens: vec![SetupOpen {
            id: index,
            range: 0..indices.pow2_padded_num_output_elements().log_2(),
        }],
        dims: vec![indices.output_dims],
        custom: Custom {
            points: Some(read_points),
            derive: Some(identity_at_bound_point),
            ..Custom::default()
        },
        ..OpRel::base(OpRelKind::Gather, node)
    })
}

/// Dictionary rows of a gather `node`, padded to a power of two. `Err` if the node is not a
/// gather.
pub fn num_words(node: &ComputationNode) -> Result<usize, ProofVerifyError> {
    let dict_len = match &node.operator {
        Op::GatherSmall(gather_op) => gather_op.dict_len,
        Op::GatherLarge(gather_op) => gather_op.dict_len,
        _ => return Err(unsupported(node, "expected a Gather operator")),
    };
    Ok(dict_len.next_power_of_two())
}

/// Variables of a [`gather`]'s index tensor: the leading coordinates of the output point.
pub fn lookup_vars<F: JoltField>(relation: &OpRel<F>) -> Result<usize, ProofVerifyError> {
    Ok(relation
        .dims_at(0)?
        .iter()
        .map(|dim| dim.next_power_of_two())
        .product::<usize>()
        .log_2())
}

/// The opening points of a [`gather`]: the bound address `ρ` before the index coordinates
/// (the one-hot `ra`) and before the word coordinates (the dictionary).
fn read_points<F: JoltField>(
    relation: &OpRel<F>,
    setup: &RelationSetup<F>,
    challenges: &[F],
) -> Result<Vec<Vec<F>>, ProofVerifyError> {
    let rho = low_to_high_point(challenges);
    let (r_index, r_word) = setup
        .incoming_point
        .split_at_checked(lookup_vars(relation)?)
        .ok_or_else(|| {
            unsupported(
                &relation.node,
                "the output point is shorter than the index variables",
            )
        })?;
    Ok(vec![[&rho, r_index].concat(), [&rho, r_word].concat()])
}

/// [`RelationDerived::Identity`] of a [`gather`]: the address index at the bound point.
fn identity_at_bound_point<F: JoltField>(
    relation: &OpRel<F>,
    id: &RelationDerived,
    _setup: &RelationSetup<F>,
    challenges: &[F],
) -> Result<F, ProofVerifyError> {
    Ok(match id {
        RelationDerived::Identity => {
            IdentityPolynomial::new(relation.rounds).evaluate(&low_to_high_point(challenges))
        }
        other => return Err(not_derived(other)),
    })
}
