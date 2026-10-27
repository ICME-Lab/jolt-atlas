use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::{
    Produces, Relation, RelationDerived, RelationExpr, RelationId, RelationOpening, RelationSetup,
    RelationVerifier,
};
use crate::utils::opening_access::{AccOpeningAccessor, OpeningIdBuilder};
use atlas_algebra::{
    field::JoltField,
    poly::opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
};
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, OpeningId},
    sumcheck_verifier::SumcheckInstanceParams,
};
use common::VirtualPoly;
use jolt_claims::{constant, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

/// Degree of the exponentiation product sumcheck.
pub const DEGREE_BOUND: usize = 3;

// TODO: refactor code to use mul sum-check

/// Shared prover/verifier parameters for softmax exp multiplication proof.
#[derive(Clone)]
pub struct MultParams<F: JoltField> {
    /// Random evaluation point.
    pub r: Vec<F>,
    /// Quantisation scale exponent.
    pub S: i32,
    /// Computation node reference.
    pub node: ComputationNode,
}

impl<F: JoltField> MultParams<F> {
    /// Create new parameters for softmax exp multiplication operation.
    pub fn new(node: ComputationNode, S: i32, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let r = AccOpeningAccessor::new(accumulator, &node)
            .get_advice(VirtualPoly::SoftmaxExpQ)
            .0
            .r;
        Self { r, S, node }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for MultParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.node);
        let exp_q_claim = accessor.get_advice(VirtualPoly::SoftmaxExpQ).1;
        let R_claim = accessor.get_advice(VirtualPoly::SoftmaxExpRemainder).1;
        exp_q_claim * F::from_i32(self.S) + R_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.r.len()
    }

    // output = eq_eval * exp_hi * exp_lo = Challenge(0) * Opening(exp_hi) * Opening(exp_lo)
}

impl<F: JoltField> MultParams<F> {
    /// The verifier of the product, set up at the exponent's opening point.
    pub fn into_verifier(self) -> RelationVerifier<F, Self> {
        let setup = RelationSetup {
            incoming_point: self.r.clone(),
            challenges: vec![],
        };
        RelationVerifier::at(self, setup)
    }

    fn advice(&self, poly: fn(usize) -> VirtualPoly) -> RelationOpening {
        RelationOpening::Id(OpeningIdBuilder::new(&self.node).advice(poly))
    }
}

/// The exponent's split into its two table digits:
///
/// ```text
/// input claim   S·exp_q(r) + R(r)
/// output claim  eq(r, ρ) · exp_hi(ρ) · exp_lo(ρ)
/// ```
///
/// **S ⊆ R.** The quantized exponent is the digits' product rescaled by `S`, with remainder `R`
/// (range-checked elsewhere): `S·exp_q + R = exp_hi·exp_lo` on the hypercube.
impl<F: JoltField> SymbolicSumcheck for MultParams<F> {
    crate::atlas_relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::SoftmaxExpProduct
    }

    fn rounds(&self) -> usize {
        self.r.len()
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        constant::<G, _, _, _>(G::from_i64(self.S as i64))
            * opening(self.advice(VirtualPoly::SoftmaxExpQ))
            + opening(self.advice(VirtualPoly::SoftmaxExpRemainder))
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        derived::<G, _, _, _>(RelationDerived::Eq)
            * opening(self.advice(VirtualPoly::SoftmaxExpHi))
            * opening(self.advice(VirtualPoly::SoftmaxExpLo))
    }
}

impl<F: JoltField> Produces for MultParams<F> {
    fn produced(&self) -> Vec<OpeningId> {
        let builder = OpeningIdBuilder::new(&self.node);
        vec![
            builder.advice(VirtualPoly::SoftmaxExpHi),
            builder.advice(VirtualPoly::SoftmaxExpLo),
        ]
    }
}

impl<F: JoltField> Relation<F> for MultParams<F> {}
