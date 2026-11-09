use crate::onnx_proof::ops::{OpLayout, VerifyOp};
use crate::onnx_proof::relation::not_derived;
use crate::onnx_proof::relation::Produces;
use crate::onnx_proof::relation::{Relation, SetupRelation};
use atlas_sumcheck::relation::verify_relation;
use atlas_trace::{op::Einsum, Model, Op};

use crate::node_view::ComputationNode;
use crate::{
    onnx_proof::{
        fused_rebase,
        ops::{
            einsum::{
                abmk_abnk_abmn::AbmkAbnkAbmnLayout,
                dot::{EinsumGeometry, EinsumLayout, EqSchedule},
                k_nk_n::KNkNLayout,
                mk_kn_mn::MkKnMnLayout,
            },
            Verifier,
        },
        relation::{
            AtlasRelation, RelationChallenge, RelationDerived, RelationExpr, RelationId,
            RelationOpening, RelationSetup, RelationVerifier,
        },
        ProofId, ProofType,
    },
    utils::{
        dims::{lookup_einsum_config, EinsumDims},
        opening_access::{OpeningIdBuilder, Target},
    },
};
use atlas_algebra::{
    field::JoltField, poly::opening_point::OpeningPoint, utils::errors::ProofVerifyError,
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_transcript::Transcript;
use common::CommittedPoly;
use jolt_claims::{derived, opening, NoChallenges, NoInputs, NoOutputs, SymbolicSumcheck};
use jolt_field::Ring;

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

crate::bind_relation!(Einsum, ContractionRelation);

impl OpLayout for Einsum {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(fused_rebase::committed_polys(node))
    }
}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for Einsum {
    #[tracing::instrument(skip_all, name = "Einsum::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let fused = fused_rebase::fuses_rebase(node);

        if fused {
            fused_rebase::verify_pre(node, verifier)?;
        }

        let matmul_proof = verifier
            .proofs
            .take(ProofId(node.idx, ProofType::EinsumMatmul))?;
        let relation = ContractionRelation::for_node(&verifier.preprocessing.model, node.clone())?;
        let matmul =
            RelationVerifier::new(relation, &verifier.accumulator, &mut verifier.transcript)?;
        verify_relation(
            matmul_proof,
            &matmul,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        if fused {
            fused_rebase::verify_post(node, verifier)?;
        }

        Ok(())
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

/// The canonical pattern an einsum node contracts, with its dims and the round geometry of
/// the pattern's layout over them.
#[derive(Clone)]
pub struct ContractionShape {
    node: ComputationNode,
    equation: &'static str,
    dims: EinsumDims,
    num_rounds: usize,
    schedule: EqSchedule,
}

/// An einsum contraction as a relation:
///
/// ```text
/// input claim   q(r₀)·2^S + rem(r₀)    (fused rescale: pre-clamp quotient and remainder)
///               out(r₀)                (no rescale: the reduced output opening)
/// output claim  L(ρ_L) · R(ρ_R) · EQ
/// ```
///
/// where the operand points `ρ_L`, `ρ_R` and the batch factor `EQ` (one for a plain
/// contraction) come from the pattern's [`EinsumLayout`] given `r₀` and the sumcheck point. The
/// [`dot`] engine is its prover.
///
/// **S ⊆ R.** The einsum accumulates `acc(m, n) = Σ_k L(m, k)·R(k, n)` (batched patterns per
/// batch index) in `i64` and splits it with `atlas_trace::semantics::fixed::rebase` into
/// `q·2^S + rem`; folding the operands at `r₀` turns `acc(r₀)` into the sumcheck's
/// `Σ_h L(h)·R(h)·EQ(h)`, so the input claim equals the sum for every execution, saturated ones
/// included (the clamp acts on `q`, outside the relation).
pub struct ContractionRelation {
    shape: ContractionShape,
}

impl ContractionRelation {
    /// The relation of an einsum `node` of `model`. `Err` if the node is not an einsum of a
    /// registered equation over operands of that equation's shape.
    pub fn for_node(model: &Model, node: ComputationNode) -> Result<Self, ProofVerifyError> {
        let config = lookup_einsum_config(&einsum_equation(&node)?)?;
        let dims = (config.dims_extractor)(&node, model)?;
        let geometry = einsum_geometry(config.equation, &dims)?;
        Ok(Self::new(ContractionShape {
            node,
            equation: config.equation,
            dims,
            num_rounds: geometry.num_rounds(),
            schedule: geometry.schedule(),
        }))
    }

    /// The canonical equation the node contracts.
    pub fn equation(&self) -> &'static str {
        self.shape.equation
    }

    /// The flattened dims the pattern's layout folds over.
    pub fn dims(&self) -> &EinsumDims {
        &self.shape.dims
    }

    /// The pattern's layout at field `F`. `Err` if the dims are not the equation's.
    pub fn layout<F: JoltField>(&self) -> Result<Box<dyn EinsumLayout<F>>, ProofVerifyError> {
        einsum_layout(self.shape.equation, &self.shape.dims)
    }

    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.shape.node)
    }
}

impl SymbolicSumcheck for ContractionRelation {
    type RelationId = RelationId;
    type OpeningId = RelationOpening;
    type DerivedId = RelationDerived;
    type ChallengeId = RelationChallenge;
    type Shape = ContractionShape;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = NoInputs<C>;
    type Outputs<C> = NoOutputs<C>;

    fn new(shape: ContractionShape) -> Self {
        Self { shape }
    }

    fn id() -> RelationId {
        RelationId::Contraction
    }

    fn rounds(&self) -> usize {
        self.shape.num_rounds
    }

    fn degree(&self) -> usize {
        self.shape.schedule.degree()
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        fused_rebase::input_expression(&self.shape.node)
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        let [left, right] = [Target::Input(0), Target::Input(1)]
            .map(|operand| opening(RelationOpening::Id(self.ids().nodeio(operand))));
        derived(RelationDerived::ContractionEq) * left * right
    }
}

impl Produces for ContractionRelation {
    fn produced(&self) -> Vec<OpeningId> {
        vec![
            self.ids().nodeio(Target::Input(0)),
            self.ids().nodeio(Target::Input(1)),
        ]
    }
}

impl<F: JoltField> Relation<F> for ContractionRelation {
    fn opening_points(
        &self,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<Vec<Vec<F>>, ProofVerifyError> {
        let (left, right) = self
            .layout::<F>()?
            .operand_points(&OpeningPoint::new(setup.incoming_point.clone()), challenges);
        Ok(vec![left, right])
    }

    fn derive(
        &self,
        id: &RelationDerived,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<F, ProofVerifyError> {
        Ok(match id {
            RelationDerived::ContractionEq => self
                .layout::<F>()?
                .output_eq(&OpeningPoint::new(setup.incoming_point.clone()), challenges),
            other => return Err(not_derived(other)),
        })
    }
}

impl<F: JoltField> SetupRelation<F> for ContractionRelation {
    fn setup<Tr: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut Tr,
    ) -> Result<RelationSetup<F>, ProofVerifyError> {
        Ok(RelationSetup {
            incoming_point: accumulator
                .get_node_output_opening(self.shape.node.idx)?
                .0
                .r,
            challenges: Vec::new(),
        })
    }
}

impl AtlasRelation for ContractionRelation {
    type Spec = Einsum;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }
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
