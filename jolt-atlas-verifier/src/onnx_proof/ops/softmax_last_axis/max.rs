use crate::node_view::ComputationNode;
use crate::onnx_proof::relation::{
    low_to_high_point, Produces, Relation, RelationChallenge, RelationDerived, RelationExpr,
    RelationId, RelationOpening, RelationSetup, RelationVerifier,
};
use crate::utils::opening_access::{AccOpeningAccessor, OpeningIdBuilder, Target};
use atlas_algebra::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        opening_point::{OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN},
    },
    utils::{index_to_field_bitvector, math::Math},
};
use atlas_sumcheck::{
    opening_proof::{OpeningAccumulator, OpeningId},
    sumcheck_verifier::SumcheckInstanceParams,
};
use common::VirtualPoly;
use jolt_claims::{challenge, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

/// Degree of the max-indicator sumcheck.
pub const DEGREE_BOUND: usize = 3;

/// Shared parameters for the max indicator sumcheck instance.
#[derive(Clone)]
pub struct MaxIndicatorParams<F: JoltField> {
    /// Computation node reference.
    pub node: ComputationNode,
    /// Random evaluation point.
    pub r1_k: Vec<F>,
    /// max_k(r1_k)
    pub input_claim: F,
    /// Softmax dims
    pub F_N: [usize; 2],
    /// Max indicator
    pub e: Vec<u32>,
    /// Argmax index per feature (length F)
    pub argmax_k: Vec<usize>,
}

impl<F: JoltField> MaxIndicatorParams<F> {
    /// Create a new instance of max indicator sumcheck parameters.
    pub fn new(
        node: ComputationNode,
        F_N: [usize; 2],
        argmax_k: Vec<usize>,
        input_claim: F,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let [F, N] = F_N;
        let log_f = F.log_2();

        // Get r1 (the point from Stage 1 reciprocal multiplication output)
        let accessor = AccOpeningAccessor::new(accumulator, &node);
        let r1 = accessor.get_advice(VirtualPoly::SoftmaxExpQ).0;
        let r1_k = r1.split_at(log_f).0.r;

        let mut e: Vec<u32> = vec![0; F * N];
        // Only set the argmax positions — everything else is already 0
        for k in 0..F {
            e[k * N + argmax_k[k]] = 1;
        }
        Self {
            r1_k: r1_k.to_vec(),
            input_claim,
            F_N,
            e,
            argmax_k,
            node,
        }
    }

    /// Returns log2 of the leading-dimension product `F`.
    pub fn log_F(&self) -> usize {
        self.F_N[0].log_2()
    }

    /// Returns log2 of the last-axis size `N`.
    pub fn log_N(&self) -> usize {
        self.F_N[1].log_2()
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for MaxIndicatorParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        self.input_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.F_N.iter().product::<usize>().log_2()
    }

    // output = eq(r1_k, r_k) * e_claim * X_claim = Challenge(0) * Opening(X)
    // e_claim is verifier-computable from argmax_k, folded into Challenge(0)
}

impl<F: JoltField> MaxIndicatorParams<F> {
    /// The verifier of the row maxima, set up at the exponent point's row coordinates.
    pub fn into_verifier(self) -> RelationVerifier<F, Self> {
        let setup = RelationSetup {
            incoming_point: self.r1_k.clone(),
            challenges: vec![(RelationChallenge::Claim, self.input_claim)],
        };
        RelationVerifier::at(self, setup)
    }

    /// `e(r)`, the argmax indicator's MLE, in `O(F·log N)`: `e` has one nonzero per row, at
    /// `k·N + argmax_k[k]`, so `e(r_k, r_j) = Σ_k eq(r_k, k)·eq(r_j, argmax_k[k])`.
    fn indicator_at(&self, r: &[F]) -> F {
        let (r_k, r_j) = r.split_at(self.log_F());
        EqPolynomial::evals(r_k)
            .iter()
            .zip(self.argmax_k.iter())
            .map(|(&eq_k, &argmax_j)| {
                let y = index_to_field_bitvector::<F>(argmax_j as u64, r_j.len());
                eq_k * EqPolynomial::mle(r_j, &y)
            })
            .sum()
    }
}

/// The row maxima as reads of the input at the public argmax:
///
/// ```text
/// input claim   max(r_k)                              (verifier-computed from the claimed maxima)
/// output claim  eq(r_k, ρ_k) · e(ρ) · x(ρ)            (e the argmax indicator)
/// ```
///
/// **S ⊆ R.** `e` is one at each row's argmax and zero elsewhere, so `Σ_j e[k, j]·x[k, j]` is
/// the input at the argmax, the claimed `max_k`.
impl<F: JoltField> SymbolicSumcheck for MaxIndicatorParams<F> {
    crate::atlas_relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::SoftmaxMax
    }

    fn rounds(&self) -> usize {
        self.F_N.iter().product::<usize>().log_2()
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        challenge(RelationChallenge::Claim)
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        derived::<G, _, _, _>(RelationDerived::RetainedEq)
            * derived(RelationDerived::Indicator)
            * opening(RelationOpening::Id(self.produced()[0]))
    }
}

impl<F: JoltField> Produces for MaxIndicatorParams<F> {
    fn produced(&self) -> Vec<OpeningId> {
        vec![OpeningIdBuilder::new(&self.node).nodeio(Target::Input(0))]
    }
}

impl<F: JoltField> Relation<F> for MaxIndicatorParams<F> {
    fn derive(&self, id: &RelationDerived, setup: &RelationSetup<F>, challenges: &[F]) -> F {
        let point = low_to_high_point(challenges);
        match id {
            RelationDerived::RetainedEq => {
                EqPolynomial::mle(&setup.incoming_point, &point[..self.log_F()])
            }
            RelationDerived::Indicator => self.indicator_at(&point),
            other => panic!("{other:?} is not derived by this relation"),
        }
    }
}
