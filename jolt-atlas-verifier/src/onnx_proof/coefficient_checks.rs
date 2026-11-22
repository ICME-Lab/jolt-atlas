//! Atlas sumchecks for Boolean and base-four coefficient domains.
//!
//! The lookup coefficient and packed-digit checks reuse the vanishing polynomials
//! of Atlas's original Booleanity and digit-range sumchecks. The existing model
//! relations still check the Hamming weight and recompose the packed values.

use atlas_algebra::{
    field::{JoltField, Ring},
    poly::eq::EqPolynomial,
    utils::errors::ProofVerifyError,
};
use atlas_claims::{challenge, constant, derived, opening, SymbolicSumcheck};
use atlas_sumcheck::{
    opening_proof::{OpeningId, SumcheckId},
    relation::{
        low_to_high_point, not_derived, DenseGroup, Produces, Relation, RelationChallenge,
        RelationDerived, RelationExpr, RelationId, RelationOpening, RelationSetup,
        RelationVerifier, SparseGroup,
    },
};
use atlas_transcript::Transcript;
use common::CommittedPoly;
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
enum Domain {
    OneHot { address_vars: usize },
    BaseFour,
}

/// A coefficient check for a set of committed polynomials of the same original shape.
#[derive(Clone)]
pub struct CoefficientCheck {
    domain: Domain,
    polynomials: Vec<CommittedPoly>,
    num_vars: usize,
}

impl CoefficientCheck {
    /// Check that each coefficient in the original one-hot domain is 0 or 1.
    pub fn one_hot(
        polynomials: Vec<CommittedPoly>,
        address_vars: usize,
        row_vars: usize,
    ) -> Result<Self, ProofVerifyError> {
        let num_vars = address_vars
            .checked_add(row_vars)
            .filter(|&vars| vars < usize::BITS as usize && address_vars <= u16::BITS as usize)
            .ok_or_else(|| invalid("invalid one-hot domain"))?;
        Self::new(Domain::OneHot { address_vars }, polynomials, num_vars)
    }

    /// Check that the given packed-range digit columns have coefficients in 0..4.
    pub fn base_four(
        polynomials: Vec<CommittedPoly>,
        num_vars: usize,
    ) -> Result<Self, ProofVerifyError> {
        Self::new(Domain::BaseFour, polynomials, num_vars)
    }

    fn new(
        domain: Domain,
        polynomials: Vec<CommittedPoly>,
        num_vars: usize,
    ) -> Result<Self, ProofVerifyError> {
        if polynomials.is_empty()
            || num_vars >= usize::BITS as usize
            || polynomials.iter().collect::<BTreeSet<_>>().len() != polynomials.len()
        {
            return Err(invalid("invalid coefficient check shape"));
        }
        Ok(Self {
            domain,
            polynomials,
            num_vars,
        })
    }

    /// Draw the equality point and column batching coefficients.
    ///
    /// Call after the commitment to these polynomials has been bound to the transcript.
    pub fn draw<F: JoltField>(&self, transcript: &mut impl Transcript) -> RelationSetup<F> {
        RelationSetup {
            incoming_point: transcript.challenge_vector(self.num_vars),
            challenges: transcript
                .challenge_scalar_powers(self.polynomials.len())
                .into_iter()
                .enumerate()
                .map(|(i, weight)| (RelationChallenge::Batching(i), weight))
                .collect(),
        }
    }

    /// Verify the coefficient relation and register the resulting opening claims.
    pub fn into_verifier<F: JoltField>(self, setup: RelationSetup<F>) -> RelationVerifier<F, Self> {
        RelationVerifier::at(self, setup)
    }

    /// The checked polynomials in the required witness order.
    pub fn polynomials(&self) -> &[CommittedPoly] {
        &self.polynomials
    }

    /// Address variables for a sparse coefficient check.
    pub fn one_hot_address_vars(&self) -> Option<usize> {
        match self.domain {
            Domain::OneHot { address_vars } => Some(address_vars),
            Domain::BaseFour => None,
        }
    }

    fn sumcheck_id(&self) -> SumcheckId {
        match self.domain {
            Domain::OneHot { .. } => SumcheckId::AkitaBooleanity,
            Domain::BaseFour => SumcheckId::AkitaDigitRange,
        }
    }

    fn bound_point<F: Copy>(&self, challenges: &[F]) -> Vec<F> {
        match self.domain {
            Domain::OneHot { address_vars } => {
                let (address, row) = challenges.split_at(address_vars);
                [low_to_high_point(address), low_to_high_point(row)].concat()
            }
            Domain::BaseFour => low_to_high_point(challenges),
        }
    }
}

impl SymbolicSumcheck for CoefficientCheck {
    atlas_sumcheck::relation_types!();

    fn id(&self) -> RelationId {
        match self.domain {
            Domain::OneHot { .. } => RelationId::AkitaBooleanity,
            Domain::BaseFour => RelationId::AkitaDigitRange,
        }
    }

    fn rounds(&self) -> usize {
        self.num_vars
    }

    fn degree(&self) -> usize {
        match self.domain {
            Domain::OneHot { .. } => 3,
            Domain::BaseFour => 5,
        }
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        RelationExpr::zero()
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        let base = self.degree() - 1;
        let vanishing =
            self.produced()
                .into_iter()
                .enumerate()
                .fold(RelationExpr::zero(), |sum, (i, id)| {
                    let value = || opening(RelationOpening::Id(id));
                    let residual = (1..base).fold(value(), |product, root| {
                        product * (value() - constant(G::from_u64(root as u64)))
                    });
                    sum + challenge::<G, _, _, _>(RelationChallenge::Batching(i)) * residual
                });
        derived::<G, _, _, _>(RelationDerived::Eq) * vanishing
    }
}

impl Produces for CoefficientCheck {
    fn produced(&self) -> Vec<OpeningId> {
        self.polynomials
            .iter()
            .map(|&poly| OpeningId::new(poly, self.sumcheck_id()))
            .collect()
    }
}

impl<F: JoltField> Relation<F> for CoefficientCheck {
    fn opening_points(
        &self,
        _setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<Vec<Vec<F>>, ProofVerifyError> {
        Ok(vec![self.bound_point(challenges); self.polynomials.len()])
    }

    fn sparse_groups(&self) -> Option<Vec<SparseGroup>> {
        let Domain::OneHot { address_vars } = self.domain else {
            return None;
        };
        Some(vec![SparseGroup {
            polynomials: self.polynomials.clone(),
            sumcheck: self.sumcheck_id(),
            address_vars,
        }])
    }

    fn dense_groups(&self) -> Option<Vec<DenseGroup>> {
        matches!(self.domain, Domain::BaseFour).then(|| {
            vec![DenseGroup {
                polynomials: self.polynomials.clone(),
                sumcheck: self.sumcheck_id(),
            }]
        })
    }

    fn derive(
        &self,
        id: &RelationDerived,
        setup: &RelationSetup<F>,
        challenges: &[F],
    ) -> Result<F, ProofVerifyError> {
        match id {
            RelationDerived::Eq => Ok(EqPolynomial::mle(
                &setup.incoming_point,
                &self.bound_point(challenges),
            )),
            other => Err(not_derived(other)),
        }
    }
}

fn invalid(message: &str) -> ProofVerifyError {
    ProofVerifyError::Malformed(message.to_owned())
}
