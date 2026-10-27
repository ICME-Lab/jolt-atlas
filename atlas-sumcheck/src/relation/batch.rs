//! Relations driven through upstream `jolt-sumcheck`: a batch of set-up relations is one
//! front-loaded batched sumcheck ([`verify_relation_batch`]), a single relation one plain
//! sumcheck ([`verify_relation`]). The verifier checks the batch's final claim against the
//! relations' output expressions and appends their produced openings.

use super::{relation_input_claim, resolve, Relation, RelationVerifier};
use crate::{
    opening_proof::VerifierOpeningAccumulator,
    sumcheck::{absorb_input_claims, SumcheckInstanceProof},
};
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    utils::errors::ProofVerifyError,
};
use atlas_transcript::Transcript;
use jolt_sumcheck::ClearSumcheckRecorder;

/// A set-up relation as a member of a batch, its relation type erased.
pub trait DynRelationVerifier<F: JoltField, T: Transcript> {
    /// The relation's rounds.
    fn rounds(&self) -> usize;

    /// The degree of the relation's round polynomials.
    fn degree(&self) -> usize;

    /// The input claim, from the accumulator's openings and the setup.
    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F;

    /// The expected final claim at the relation's bound point (its suffix of the batch's
    /// challenges).
    fn output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F;

    /// Append the produced openings' requests at the bound point.
    fn append_produced(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    );
}

impl<F: JoltField, T: Transcript, R: Relation<F>> DynRelationVerifier<F, T>
    for RelationVerifier<F, R>
{
    fn rounds(&self) -> usize {
        self.relation.rounds()
    }

    fn degree(&self) -> usize {
        self.relation.degree()
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        relation_input_claim(&self.relation, &self.setup, accumulator)
    }

    fn output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let challenges: Vec<F> = sumcheck_challenges.to_vec().into_opening();
        self.relation.output_expression::<F>().evaluate(
            |opening| resolve(accumulator, opening),
            |challenge| self.setup.challenge(challenge),
            |derived| self.relation.derive(derived, &self.setup, &challenges),
        )
    }

    fn append_produced(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let challenges: Vec<F> = sumcheck_challenges.to_vec().into_opening();
        self.relation
            .append_produced_requests(&self.setup, &challenges, accumulator, transcript);
    }
}

/// Verify a batch of relations proven as one front-loaded batched sumcheck (Jim Posen,
/// ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA)). A relation with fewer
/// rounds than the batch is bound in its last rounds, and its input claim is scaled by two per
/// round it sits out. Returns the batch's challenges.
#[tracing::instrument(skip_all, name = "verify_relation_batch")]
pub fn verify_relation_batch<F: JoltField, T: Transcript>(
    proof: &SumcheckInstanceProof<F>,
    relations: Vec<&dyn DynRelationVerifier<F, T>>,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<Vec<F::Challenge>, ProofVerifyError> {
    let max_degree = relations.iter().map(|r| r.degree()).max().unwrap();
    let max_rounds = relations.iter().map(|r| r.rounds()).max().unwrap();

    let input_claims: Vec<F> = relations
        .iter()
        .map(|r| r.input_claim(accumulator))
        .collect();
    absorb_input_claims(
        &mut ClearSumcheckRecorder::<F>::new(),
        &input_claims,
        transcript,
    );
    let coefficients: Vec<F> = transcript.challenge_vector(relations.len());

    let claim: F = relations
        .iter()
        .zip(&input_claims)
        .zip(&coefficients)
        .map(|((r, input_claim), coefficient)| {
            input_claim.mul_pow_2(max_rounds - r.rounds()) * coefficient
        })
        .sum();
    let (final_claim, r_sumcheck) = proof.verify(claim, max_rounds, max_degree, transcript)?;

    let expected: F = relations
        .iter()
        .zip(&coefficients)
        .map(|(r, coefficient)| {
            let r_slice = &r_sumcheck[max_rounds - r.rounds()..];
            r.append_produced(accumulator, transcript, r_slice);
            r.output_claim(accumulator, r_slice) * coefficient
        })
        .sum();
    if final_claim != expected {
        return Err(ProofVerifyError::SumcheckVerificationError);
    }
    Ok(r_sumcheck)
}

/// Verify one relation proven as its own sumcheck. Returns its challenges.
#[tracing::instrument(skip_all, name = "verify_relation")]
pub fn verify_relation<F: JoltField, T: Transcript>(
    proof: &SumcheckInstanceProof<F>,
    relation: &dyn DynRelationVerifier<F, T>,
    accumulator: &mut VerifierOpeningAccumulator<F>,
    transcript: &mut T,
) -> Result<Vec<F::Challenge>, ProofVerifyError> {
    let input_claim = relation.input_claim(accumulator);
    absorb_input_claims(
        &mut ClearSumcheckRecorder::<F>::new(),
        &[input_claim],
        transcript,
    );
    let (final_claim, r_sumcheck) = proof.verify(
        input_claim,
        relation.rounds(),
        relation.degree(),
        transcript,
    )?;
    relation.append_produced(accumulator, transcript, &r_sumcheck);
    if final_claim != relation.output_claim(accumulator, &r_sumcheck) {
        return Err(ProofVerifyError::SumcheckVerificationError);
    }
    Ok(r_sumcheck)
}
