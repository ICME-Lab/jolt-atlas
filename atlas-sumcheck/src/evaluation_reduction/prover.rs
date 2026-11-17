//! The prover side of [`super`].

use super::*;
use crate::opening_proof::Opening;
use atlas_algebra::{
    field::JoltField, poly::multilinear::MultilinearPolynomial, utils::errors::ProvingError,
};
use atlas_transcript::ProvingTranscript;

impl<F: JoltField> EvalReductionInstance<F> {
    /// Reduce the instance's openings to one, sending `h` through the transcript. A single
    /// opening is the reduced instance and nothing is sent.
    pub fn prove<T: ProvingTranscript>(
        &self,
        witness: &EvalReductionWitness<F>,
        transcript: &mut T,
    ) -> Result<ReducedInstance<F>, ProvingError> {
        if self.openings.is_empty() {
            return Err(ProvingError::EmptyInput);
        }

        let opening_points = self.openings.iter().map(|(r, _)| r).collect::<Vec<_>>();
        let num_vars = witness.mle.get_num_vars();

        if let Some(opening) = opening_points.iter().find(|r| r.len() != num_vars) {
            return Err(ProvingError::InvalidInputLength(
                witness.mle.get_num_vars(),
                opening.len(),
            ));
        }

        if opening_points.len() == 1 {
            // Short path:
            // If there's only one opening, we can skip the reduction and directly return the claim as the reduced instance.
            let (r, claim) = &self.openings[0];
            return Ok(ReducedInstance {
                r: r.clone(),
                claim: *claim,
            });
        }

        // i'th vector of this Vec<Vec<F>> is the vector of i'th variables across all evaluation points.
        let ri_vec = group_by_variable(&opening_points);
        // h = mle ∘ l, where l is the unique n-1 degree polynomial defined by the n opening points.
        let h = compute_h(&witness.mle, &ri_vec);

        #[cfg(test)]
        {
            for (i, (_, claim)) in self.openings.iter().enumerate() {
                let eval_at_i = h.evaluate(F::from_u32(i as u32));
                assert_eq!(eval_at_i, *claim, "h does not match opening claim at t={i}");
            }
        }

        // One coefficient per power up to the degree bound: the count the verifier reads.
        let mut coefficients = h.coefficients().to_vec();
        let count = h_coefficients(num_vars, opening_points.len());
        assert!(
            coefficients.len() <= count,
            "h has {} coefficients, above its degree bound's {count}",
            coefficients.len()
        );
        coefficients.resize(count, F::zero());
        transcript.append_message(H_BEGIN_TRANSCRIPT_LABEL);
        transcript.send_scalars(&coefficients);
        transcript.append_message(H_END_TRANSCRIPT_LABEL);
        let x_prime: F::Challenge = transcript.challenge_scalar_optimized::<F>();

        Ok(ReducedInstance {
            r: eval_on_l(&ri_vec, x_prime.into()),
            claim: h.evaluate(x_prime.into()),
        })
    }
}

impl EvalReductionProtocol {
    pub fn prove<F: JoltField, T: ProvingTranscript>(
        openings: &[&Opening<F>],
        output_mle: MultilinearPolynomial<F>,
        transcript: &mut T,
    ) -> Result<ReducedInstance<F>, ProvingError> {
        let witness = EvalReductionWitness::new(output_mle);
        let instance = EvalReductionInstance::new(openings);

        instance.prove(&witness, transcript)
    }
}
