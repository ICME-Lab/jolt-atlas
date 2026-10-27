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
        opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
    },
    utils::math::Math,
};
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, OpeningId},
    sumcheck_verifier::SumcheckInstanceParams,
};
use atlas_transcript::Transcript;
use common::VirtualPoly;
use jolt_claims::{derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

/// Degree of the exp-sum sumcheck.
pub const DEGREE_BOUND: usize = 2;

/// Shared prover/verifier parameters for softmax exponential sum proof.
#[derive(Clone)]
pub struct ExpSumParams<F: JoltField> {
    /// Computation node reference.
    pub node: ComputationNode,
    /// Random evaluation point.
    pub r0_k: Vec<F>,
    /// `[F, N]` — leading-dim product and last-axis size.
    pub F_N: [usize; 2],
}

impl<F: JoltField> ExpSumParams<F> {
    /// Create new parameters for exponential sum operation.
    pub fn new(
        node: ComputationNode,
        F_N: [usize; 2],
        accumulator: &dyn OpeningAccumulator<F>,
        _transcript: &mut impl Transcript,
    ) -> Self {
        let log_f = F_N[0].log_2();
        let accessor = AccOpeningAccessor::new(accumulator, &node);
        let r = accessor.get_reduced_opening().0;
        let r0_k = r.split_at(log_f).0.r;
        Self { r0_k, node, F_N }
    }

    /// Returns log2 of the leading-dimension product `F`.
    fn log_F(&self) -> usize {
        self.F_N[0].log_2()
    }

    /// Returns log2 of the last-axis size `N`.
    pub fn log_N(&self) -> usize {
        self.F_N[1].log_2()
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ExpSumParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.node);
        accessor.get_advice(VirtualPoly::SoftmaxExpSum).1
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.log_F() + self.log_N()
    }

    // output = eq_eval * exp_q = Challenge(0) * Opening(exp_q)
}

impl<F: JoltField> ExpSumParams<F> {
    /// The verifier of the row sums, set up at the reduced output's row coordinates.
    pub fn into_verifier(self) -> RelationVerifier<F, Self> {
        let setup = RelationSetup {
            incoming_point: self.r0_k.clone(),
            challenges: vec![],
        };
        RelationVerifier::at(self, setup)
    }
}

/// The row sums of the quantized exponents:
///
/// ```text
/// input claim   exp_sum(r_k)
/// output claim  eq(r_k, ρ_k) · exp_q(ρ)              (ρ_k the row coordinates of ρ)
/// ```
///
/// **S ⊆ R.** `exp_sum[k] = Σ_j exp_q[k, j]`, so summing `eq(r_k, k)·exp_q[k, j]` over the
/// hypercube gives `exp_sum(r_k)`.
impl<F: JoltField> SymbolicSumcheck for ExpSumParams<F> {
    crate::atlas_relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::SoftmaxExpSum
    }

    fn rounds(&self) -> usize {
        self.log_F() + self.log_N()
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        opening(RelationOpening::Id(
            OpeningIdBuilder::new(&self.node).advice(VirtualPoly::SoftmaxExpSum),
        ))
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        derived::<G, _, _, _>(RelationDerived::RetainedEq)
            * opening(RelationOpening::Id(self.produced()[0]))
    }
}

impl<F: JoltField> Produces for ExpSumParams<F> {
    fn produced(&self) -> Vec<OpeningId> {
        vec![OpeningIdBuilder::new(&self.node).advice(VirtualPoly::SoftmaxExpQ)]
    }
}

impl<F: JoltField> Relation<F> for ExpSumParams<F> {
    fn derive(&self, id: &RelationDerived, setup: &RelationSetup<F>, challenges: &[F]) -> F {
        match id {
            RelationDerived::RetainedEq => EqPolynomial::mle(
                &setup.incoming_point,
                &low_to_high_point(challenges)[..self.log_F()],
            ),
            other => panic!("{other:?} is not derived by this relation"),
        }
    }
}
