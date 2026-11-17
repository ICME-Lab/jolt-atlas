use crate::onnx_proof::plan::{fused_rescale_plan, OpPlan, OpRelation, Plan};
use atlas_trace::{op::Einsum, trace::ModelExecutionIO, Model, Op};

use crate::node_view::ComputationNode;
use crate::{
    onnx_proof::{
        fused_rebase,
        ops::{
            einsum::{
                abmk_abnk_abmn::AbmkAbnkAbmnLayout,
                dot::{EinsumGeometry, EinsumLayout},
                k_nk_n::KNkNLayout,
                mk_kn_mn::MkKnMnLayout,
            },
            OpLayout,
        },
        relation::{
            not_derived, Custom, OpRel, OpRelKind, RelationDerived, RelationOpening, RelationSetup,
        },
        ProofType,
    },
    utils::{
        dims::{lookup_einsum_config, EinsumDims},
        opening_access::{OpeningIdBuilder, Target},
    },
};
use atlas_algebra::{field::JoltField, poly::point::OpeningPoint, utils::errors::ProofVerifyError};
use atlas_claims::{derived, opening};

pub mod abmk_abnk_abmn;
/// Shared sumcheck engine for einsum contraction patterns.
pub mod dot;
/// `k,nk->n` layout (vector-matrix multiply).
pub mod k_nk_n;
/// `mk,kn->mn` layout (matrix-matrix multiply).
pub mod mk_kn_mn;

// atlas-onnx-import lowers every einsum to one of four canonical full-letter
// equations (`CANONICAL_EINSUMS` in its lower.rs); EINSUM_REGISTRY maps each
// onto one of the three layouts above:
// - mk,kn->mn       → mk_kn_mn
// - mk,nk->mn       → k_nk_n   (vector-matrix; m == 1 enforced at dispatch)
// - amk,ank->amn    → abmk_abnk_abmn (batch pack 1×a)
// - abmk,abnk->abmn → abmk_abnk_abmn

impl OpLayout for Einsum {}

impl OpPlan for Einsum {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        Ok(fused_rescale_plan(
            node,
            ProofType::EinsumMatmul,
            OpRelation::Op(contraction(model, node)?),
        ))
    }
}

fn einsum_equation(computation_node: &ComputationNode) -> Result<String, ProofVerifyError> {
    match &computation_node.operator {
        Op::Einsum(Einsum { equation, .. }) => Ok(equation.to_string()),
        other => Err(ProofVerifyError::Malformed(format!(
            "model: node {} is {other}, not an einsum",
            computation_node.idx
        ))),
    }
}

/// An einsum contraction as a relation:
///
/// ```text
/// input claim   out(r₀)·2^S + rem(r₀)  (fused rescale: the output is the quotient)
///               out(r₀)                (no rescale: the reduced output opening)
/// output claim  L(ρ_L) · R(ρ_R) · EQ
/// ```
///
/// where the operand points `ρ_L`, `ρ_R` and the batch factor `EQ` (one for a plain
/// contraction) come from the pattern's [`EinsumLayout`] given `r₀` and the sumcheck point. The
/// [`dot`] engine is its prover.
///
/// The relation's [`dims`](OpRel::dims) are the flattened dims the pattern's layout folds
/// over: the left operand's, the right operand's and the output's. `Err` if the node is not an
/// einsum of a registered equation over operands of that equation's shape.
///
/// **S ⊆ R.** The einsum accumulates `acc(m, n) = Σ_k L(m, k)·R(k, n)` (batched patterns per
/// batch index) in `i64` and splits it with `atlas_trace::semantics::fixed::rebase` into
/// `out·2^S + rem`, failing when the quotient leaves `i32`; folding the operands at `r₀` turns
/// `acc(r₀)` into the sumcheck's `Σ_h L(h)·R(h)·EQ(h)`, so the input claim equals the sum on
/// every execution that exists.
///
/// **No wrap.** The operands are `i32` cells, so a contraction over `n` cells has
/// `|Σ_k L·R| ≤ n·2^62`. The range claims give `out ∈ [−2^31, 2^31)` (a model output is a
/// public `i32` tensor) and `rem ∈ [0, 2^S)`, so `|out·2^S + rem| < 2^(32+S)`, at most
/// `2^47` since the scale `S` is at most 15 (`atlas_trace::Scale`). For any tensor that fits
/// in memory (`n < 2^64`) both sides are integers of magnitude at most `2^126`, less than
/// half the 128-bit field modulus, so their equality in the field is their equality over the
/// integers.
pub fn contraction<F: JoltField>(
    model: &Model,
    node: &ComputationNode,
) -> Result<OpRel<F>, ProofVerifyError> {
    let config = lookup_einsum_config(&einsum_equation(node)?)?;
    let dims = (config.dims_extractor)(node, model)?;
    let geometry = einsum_geometry(config.equation, &dims)?;
    let ids = OpeningIdBuilder::new(node);
    let produced = [Target::Input(0), Target::Input(1)].map(|operand| ids.nodeio(operand));
    let [left, right] = produced.map(|id| opening(RelationOpening::Id(id)));
    Ok(OpRel {
        rounds: geometry.num_rounds(),
        degree: geometry.schedule().degree(),
        input: fused_rebase::input_expression(node),
        output: derived(RelationDerived::ContractionEq) * left * right,
        produced: produced.to_vec(),
        dims: vec![
            dims.left_operand().to_vec(),
            dims.right_operand().to_vec(),
            dims.output().to_vec(),
        ],
        custom: Custom {
            points: Some(operand_points),
            derive: Some(batch_eq),
            ..Custom::default()
        },
        ..OpRel::base(OpRelKind::Contraction, node)
    })
}

/// The canonical equation a [`contraction`] contracts.
pub fn equation<F: JoltField>(relation: &OpRel<F>) -> Result<&'static str, ProofVerifyError> {
    Ok(lookup_einsum_config(&einsum_equation(&relation.node)?)?.equation)
}

/// The flattened dims a [`contraction`]'s layout folds over.
pub fn dims<F: JoltField>(relation: &OpRel<F>) -> Result<EinsumDims, ProofVerifyError> {
    Ok(EinsumDims::new(
        relation.dims_at(0)?.to_vec(),
        relation.dims_at(1)?.to_vec(),
        relation.dims_at(2)?.to_vec(),
    ))
}

/// The layout of a [`contraction`]'s pattern. `Err` if the dims are not the equation's.
pub fn layout<F: JoltField>(
    relation: &OpRel<F>,
) -> Result<Box<dyn EinsumLayout<F>>, ProofVerifyError> {
    einsum_layout(equation(relation)?, &dims(relation)?)
}

/// The operand points of a [`contraction`]: the layout's scatter of the sumcheck point.
fn operand_points<F: JoltField>(
    relation: &OpRel<F>,
    setup: &RelationSetup<F>,
    challenges: &[F],
) -> Result<Vec<Vec<F>>, ProofVerifyError> {
    let (left, right) = layout(relation)?
        .operand_points(&OpeningPoint::new(setup.incoming_point.clone()), challenges);
    Ok(vec![left, right])
}

/// [`RelationDerived::ContractionEq`] of a [`contraction`]: the layout's batch eq factor.
fn batch_eq<F: JoltField>(
    relation: &OpRel<F>,
    id: &RelationDerived,
    setup: &RelationSetup<F>,
    challenges: &[F],
) -> Result<F, ProofVerifyError> {
    Ok(match id {
        RelationDerived::ContractionEq => layout(relation)?
            .output_eq(&OpeningPoint::new(setup.incoming_point.clone()), challenges),
        other => return Err(not_derived(other)),
    })
}

fn unexpected_equation(equation: &str) -> ProofVerifyError {
    ProofVerifyError::Malformed(format!(
        "model: unexpected canonical einsum equation: {equation}"
    ))
}

/// The round geometry of a canonical equation's layout. `Err` if the equation is not
/// canonical or the dims are not its.
fn einsum_geometry(
    equation: &str,
    einsum_dims: &EinsumDims,
) -> Result<Box<dyn EinsumGeometry>, ProofVerifyError> {
    Ok(match equation {
        "mk,kn->mn" => Box::new(MkKnMnLayout::new(einsum_dims)?),
        "k,nk->n" => Box::new(KNkNLayout::new(einsum_dims)?),
        "abmk,abnk->abmn" => Box::new(AbmkAbnkAbmnLayout::new(einsum_dims)?),
        other => return Err(unexpected_equation(other)),
    })
}

/// Build the [`EinsumLayout`] for a canonical equation; every registered
/// equation is handled by the shared [`dot`] engine. `Err` if the equation is not canonical or
/// the dims are not its.
fn einsum_layout<F: JoltField>(
    equation: &str,
    einsum_dims: &EinsumDims,
) -> Result<Box<dyn EinsumLayout<F>>, ProofVerifyError> {
    Ok(match equation {
        "mk,kn->mn" => Box::new(MkKnMnLayout::new(einsum_dims)?),
        "k,nk->n" => Box::new(KNkNLayout::new(einsum_dims)?),
        "abmk,abnk->abmn" => Box::new(AbmkAbnkAbmnLayout::new(einsum_dims)?),
        other => return Err(unexpected_equation(other)),
    })
}
