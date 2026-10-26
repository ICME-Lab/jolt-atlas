use crate::onnx_proof::ops::{OpLayout, VerifyOp};
use atlas_trace::{op::Einsum, Model, Op};

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
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, OpeningId},
    sumcheck::Sumcheck,
};
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
    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        fused_rebase::committed_polys(node)
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
            .get(&ProofId(node.idx, ProofType::EinsumMatmul))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let relation = ContractionRelation::for_node(&verifier.preprocessing.model, node.clone());
        let matmul =
            RelationVerifier::new(relation, &verifier.accumulator, &mut verifier.transcript);
        Sumcheck::verify(
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

fn einsum_equation(computation_node: &ComputationNode) -> String {
    match &computation_node.operator {
        Op::Einsum(Einsum { equation, .. }) => equation.to_string(),
        _ => panic!("Unexpected operator"),
    }
}

/// The canonical pattern an einsum node contracts, with its dims.
#[derive(Clone)]
pub struct ContractionShape {
    node: ComputationNode,
    equation: &'static str,
    dims: EinsumDims,
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
    geometry: Box<dyn EinsumGeometry>,
}

impl ContractionRelation {
    /// The relation of an einsum `node` of `model`.
    pub fn for_node(model: &Model, node: ComputationNode) -> Self {
        let config = lookup_einsum_config(&einsum_equation(&node));
        let dims = (config.dims_extractor)(&node, model);
        Self::new(ContractionShape {
            node,
            equation: config.equation,
            dims,
        })
    }

    /// The pattern's layout at field `F`.
    pub fn layout<F: JoltField>(&self) -> Box<dyn EinsumLayout<F>> {
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
        let geometry = einsum_geometry(shape.equation, &shape.dims);
        Self { shape, geometry }
    }

    fn id() -> RelationId {
        RelationId::Contraction
    }

    fn rounds(&self) -> usize {
        self.geometry.num_rounds()
    }

    fn degree(&self) -> usize {
        self.geometry.schedule().degree()
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

impl AtlasRelation for ContractionRelation {
    type Spec = Einsum;

    fn node(&self) -> &ComputationNode {
        &self.shape.node
    }

    fn setup<F: JoltField, Tr: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut Tr,
    ) -> RelationSetup<F> {
        RelationSetup {
            incoming_point: accumulator.get_node_output_opening(self.shape.node.idx).0.r,
            challenges: Vec::new(),
        }
    }

    fn produced(&self) -> Vec<OpeningId> {
        vec![
            self.ids().nodeio(Target::Input(0)),
            self.ids().nodeio(Target::Input(1)),
        ]
    }

    fn opening_points<F: JoltField>(
        &self,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Vec<Vec<F>> {
        let (left, right) = self
            .layout::<F>()
            .operand_points(&OpeningPoint::new(setup.incoming_point.clone()), challenges);
        vec![left, right]
    }

    fn derive<F: JoltField>(
        &self,
        id: &RelationDerived,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> F {
        match id {
            RelationDerived::ContractionEq => self
                .layout::<F>()
                .output_eq(&OpeningPoint::new(setup.incoming_point.clone()), challenges),
            other => panic!("a contraction derives no {other:?}"),
        }
    }
}

/// The round geometry of a canonical equation's layout.
fn einsum_geometry(equation: &str, einsum_dims: &EinsumDims) -> Box<dyn EinsumGeometry> {
    match equation {
        "mk,kn->mn" => Box::new(MkKnMnLayout::new(einsum_dims)),
        "k,nk->n" => Box::new(KNkNLayout::new(einsum_dims)),
        "abmk,abnk->abmn" => Box::new(AbmkAbnkAbmnLayout::new(einsum_dims)),
        other => panic!("unexpected canonical einsum equation: {other}"),
    }
}

/// Build the [`EinsumLayout`] for a canonical equation; every registered
/// equation is handled by the shared [`dot`] engine.
fn einsum_layout<F: JoltField>(
    equation: &str,
    einsum_dims: &EinsumDims,
) -> Box<dyn EinsumLayout<F>> {
    match equation {
        "mk,kn->mn" => Box::new(MkKnMnLayout::new(einsum_dims)),
        "k,nk->n" => Box::new(KNkNLayout::new(einsum_dims)),
        "abmk,abnk->abmn" => Box::new(AbmkAbnkAbmnLayout::new(einsum_dims)),
        other => panic!("unexpected canonical einsum equation: {other}"),
    }
}
