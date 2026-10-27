use crate::relation::Produces;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use common::CommittedPoly;
use rayon::prelude::*;

use crate::{
    opening_proof::{OpeningId, SumcheckId},
    relation::{
        low_to_high_point, Relation, RelationChallenge, RelationExpr, RelationId, RelationOpening,
        RelationSetup, RelationVerifier, SparseGroup,
    },
};
use atlas_algebra::field::JoltField;
use jolt_claims::{challenge, opening, SymbolicSumcheck};
use jolt_field::Ring;

/// Degree bound of the sumcheck round polynomials in [`HammingWeightRelation`].
const DEGREE_BOUND: usize = 1;

pub struct HammingWeightSumcheckParams<F: JoltField> {
    pub d: usize,
    pub num_rounds: usize,
    pub gamma_powers: Vec<F>,
    pub polynomial_types: Vec<CommittedPoly>,
    pub sumcheck_id: SumcheckId,
    pub r_cycle: Vec<F>,
}

/// `Σ_i γ^i · Σ_k ra_i(k, r_cycle) = Σ_i γ^i`: each one-hot chunk has Hamming weight one.
///
/// ```text
/// input claim   Σ_i γ^i
/// output claim  Σ_i γ^i · ra_i(r′, r_cycle)        (r′ the low-to-high bound address point)
/// ```
///
/// **S ⊆ R.** An honest chunk is one-hot in every cycle, so its sum over the address is one.
/// The setup's incoming point is `r_cycle`; `γ^i` are [`RelationChallenge::Batching`].
pub struct HammingWeightRelation {
    d: usize,
    num_rounds: usize,
    polynomial_types: Vec<CommittedPoly>,
    sumcheck_id: SumcheckId,
}

impl HammingWeightRelation {
    /// The sumcheck the chunks' openings come from.
    pub fn sumcheck_id(&self) -> SumcheckId {
        self.sumcheck_id
    }
}

impl SymbolicSumcheck for HammingWeightRelation {
    crate::relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::HammingWeight
    }

    fn rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        (0..self.d).fold(RelationExpr::zero(), |acc, i| {
            acc + challenge(RelationChallenge::Batching(i))
        })
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        (0..self.d).fold(RelationExpr::zero(), |acc, i| {
            acc + challenge::<G, _, _, _>(RelationChallenge::Batching(i))
                * opening(RelationOpening::Id(OpeningId::new(
                    self.polynomial_types[i],
                    self.sumcheck_id,
                )))
        })
    }
}

impl Produces for HammingWeightRelation {
    fn produced(&self) -> Vec<OpeningId> {
        self.polynomial_types
            .iter()
            .map(|&poly| OpeningId::new(poly, self.sumcheck_id))
            .collect()
    }
}

impl<F: JoltField> Relation<F> for HammingWeightRelation {
    fn opening_points(&self, setup: &RelationSetup<F>, challenges: &[F]) -> Vec<Vec<F>> {
        let point = [low_to_high_point(challenges), setup.incoming_point.clone()].concat();
        vec![point; self.d]
    }

    fn sparse_groups(&self) -> Option<Vec<SparseGroup>> {
        Some(vec![SparseGroup {
            polynomials: self.polynomial_types.clone(),
            sumcheck: self.sumcheck_id,
            address_vars: self.num_rounds,
        }])
    }
}

/// The verifier of a Hamming-weight check: its relation.
pub type HammingWeightSumcheckVerifier<F> = RelationVerifier<F, HammingWeightRelation>;

impl<F: JoltField> HammingWeightSumcheckParams<F> {
    /// The relation these parameters instantiate, with its setup.
    pub fn into_parts(self) -> (HammingWeightRelation, RelationSetup<F>) {
        let setup = RelationSetup {
            incoming_point: self.r_cycle,
            challenges: self
                .gamma_powers
                .into_iter()
                .enumerate()
                .map(|(i, gamma)| (RelationChallenge::Batching(i), gamma))
                .collect(),
        };
        let relation = HammingWeightRelation {
            d: self.d,
            num_rounds: self.num_rounds,
            polynomial_types: self.polynomial_types,
            sumcheck_id: self.sumcheck_id,
        };
        (relation, setup)
    }

    /// The verifier of the check.
    pub fn into_verifier(self) -> HammingWeightSumcheckVerifier<F> {
        let (relation, setup) = self.into_parts();
        RelationVerifier::at(relation, setup)
    }
}

pub mod prover;
