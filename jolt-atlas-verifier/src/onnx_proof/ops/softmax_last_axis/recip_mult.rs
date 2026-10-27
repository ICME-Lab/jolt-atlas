use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::{
    low_to_high_point, Produces, Relation, RelationDerived, RelationExpr, RelationId,
    RelationOpening, RelationSetup, RelationVerifier,
};
use crate::utils::opening_access::{AccOpeningAccessor, OpeningIdBuilder};
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    utils::math::Math,
};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId};
use atlas_transcript::Transcript;
use common::VirtualPoly;
use jolt_claims::{constant, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

/// Degree of the reciprocal-multiplication sumcheck.
pub const DEGREE_BOUND: usize = 3;

/// Shared prover/verifier parameters for softmax reciprocal multiplication proof.
#[derive(Clone)]
pub struct RecipMultParams<F: JoltField> {
    /// Computation node reference.
    pub node: ComputationNode,
    /// Random evaluation point.
    pub r: Vec<F>,
    /// Quantisation scale value S.
    pub S: i32,
    /// `[F, N]` — leading-dim product and last-axis size.
    pub F_N: [usize; 2],
}

impl<F: JoltField> RecipMultParams<F> {
    /// Create new parameters for reciprocal multiplication operation.
    pub fn new(
        node: ComputationNode,
        S: i32,
        F_N: [usize; 2],
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut impl Transcript,
    ) -> Self {
        let r = AccOpeningAccessor::new(accumulator, &node)
            .get_reduced_opening()
            .0
            .r;
        Self { r, S, node, F_N }
    }

    /// Returns `[F, N]`.
    pub fn F_N(&self) -> [usize; 2] {
        self.F_N
    }

    /// Returns the leading-dimension product `F`.
    pub fn F(&self) -> usize {
        self.F_N()[0]
    }

    /// Returns the last-axis size `N`.
    pub fn N(&self) -> usize {
        self.F_N()[1]
    }

    /// Returns log2 of the leading-dimension product `F`.
    pub fn log_F(&self) -> usize {
        self.F().log_2()
    }

    /// Returns log2 of the last-axis size `N`.
    pub fn log_N(&self) -> usize {
        self.N().log_2()
    }
}

/// The softmax output as the exponents times the rows' inverse sums:
///
/// ```text
/// input claim   S·out(r) + R(r)
/// output claim  eq(r, ρ) · exp_q(ρ) · inv_sum(ρ_k)    (ρ_k the row coordinates of ρ)
/// ```
///
/// **S ⊆ R.** `S·out[k, j] + R[k, j] = exp_q[k, j]·inv_sum[k]` on the hypercube, with remainder
/// `R` range-checked elsewhere; `inv_sum` is verifier-computed from the claimed row sums.
pub struct RecipMultRelation<F: JoltField> {
    params: RecipMultParams<F>,
    inv_sum: Vec<F>,
}

impl<F: JoltField> RecipMultRelation<F> {
    /// The relation over `params`, with the rows' inverse sums.
    pub fn with_inv_sum(params: RecipMultParams<F>, inv_sum: Vec<F>) -> Self {
        Self { params, inv_sum }
    }

    /// The parameters.
    pub fn params(&self) -> &RecipMultParams<F> {
        &self.params
    }

    /// The relation's setup: the reduced output point.
    pub fn relation_setup(&self) -> RelationSetup<F> {
        RelationSetup {
            incoming_point: self.params.r.clone(),
            challenges: vec![],
        }
    }

    /// The verifier of the product.
    pub fn into_verifier(self) -> RelationVerifier<F, Self> {
        let setup = self.relation_setup();
        RelationVerifier::at(self, setup)
    }

    fn advice(&self, poly: fn(usize) -> VirtualPoly) -> RelationOpening {
        RelationOpening::Id(OpeningIdBuilder::new(&self.params.node).advice(poly))
    }
}

impl<F: JoltField> SymbolicSumcheck for RecipMultRelation<F> {
    crate::atlas_relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::SoftmaxRecipProduct
    }

    fn rounds(&self) -> usize {
        self.params.r.len()
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        opening::<G, _, _, _>(RelationOpening::ReducedNodeOutput(self.params.node.idx))
            * constant(G::from_i64(self.params.S as i64))
            + opening(self.advice(VirtualPoly::SoftmaxRecipMultRemainder))
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        derived::<G, _, _, _>(RelationDerived::Eq)
            * opening(self.advice(VirtualPoly::SoftmaxExpQ))
            * derived(RelationDerived::RowPublic)
    }
}

impl<F: JoltField> Produces for RecipMultRelation<F> {
    fn produced(&self) -> Vec<OpeningId> {
        vec![OpeningIdBuilder::new(&self.params.node).advice(VirtualPoly::SoftmaxExpQ)]
    }
}

impl<F: JoltField> Relation<F> for RecipMultRelation<F> {
    fn derive(&self, id: &RelationDerived, setup: &RelationSetup<F>, challenges: &[F]) -> F {
        let point = low_to_high_point(challenges);
        match id {
            RelationDerived::Eq => EqPolynomial::mle(&setup.incoming_point, &point),
            RelationDerived::RowPublic => MultilinearPolynomial::from(self.inv_sum.clone())
                .evaluate(&point[..self.params.log_F()]),
            other => panic!("{other:?} is not derived by this relation"),
        }
    }
}
