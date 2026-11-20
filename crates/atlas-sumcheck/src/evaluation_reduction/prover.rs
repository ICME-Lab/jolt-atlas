//! The prover side of [`super`].

use super::*;
use crate::opening_proof::Opening;
use atlas_algebra::{
    field::JoltField, poly::multilinear::MultilinearPolynomial, utils::errors::ProvingError,
};
use atlas_transcript::ProvingTranscript;

/// The part of a reduction that reads the witness: `h`, the witness restricted to the line
/// through the opening points, ready to send ([`EvalReductionInstance::send`]). `None` for a
/// single opening, which is its own reduced instance.
pub struct PreparedReduction<F: JoltField> {
    h: Option<(UnivariatePoly<F>, Vec<Vec<F>>)>,
}

impl<F: JoltField> EvalReductionInstance<F> {
    /// Reduce the instance's openings to one, sending `h` through the transcript. A single
    /// opening is the reduced instance and nothing is sent.
    pub fn prove<T: ProvingTranscript>(
        &self,
        witness: &EvalReductionWitness<F>,
        transcript: &mut T,
    ) -> Result<ReducedInstance<F>, ProvingError> {
        let prepared = self.prepare_with(Some(witness), || witness.clone())?;
        Ok(self.send(prepared, transcript))
    }

    /// The part of [`Self::prove`] that reads the witness and not the transcript. The witness
    /// is built only when there is more than one opening to reduce.
    pub fn prepare(
        &self,
        witness: impl FnOnce() -> EvalReductionWitness<F>,
    ) -> Result<PreparedReduction<F>, ProvingError> {
        self.prepare_with(None, witness)
    }

    /// [`Self::prepare`] on `given` when there is one, else on the witness `build` makes.
    fn prepare_with(
        &self,
        given: Option<&EvalReductionWitness<F>>,
        build: impl FnOnce() -> EvalReductionWitness<F>,
    ) -> Result<PreparedReduction<F>, ProvingError> {
        if self.openings.is_empty() {
            return Err(ProvingError::EmptyInput);
        }
        if self.openings.len() == 1 {
            // The one opening is the reduced instance.
            return Ok(PreparedReduction { h: None });
        }
        let built;
        let witness = match given {
            Some(witness) => witness,
            None => {
                built = build();
                &built
            }
        };
        let opening_points = self.openings.iter().map(|(r, _)| r).collect::<Vec<_>>();
        let num_vars = witness.mle.get_num_vars();

        if let Some(opening) = opening_points.iter().find(|r| r.len() != num_vars) {
            return Err(ProvingError::InvalidInputLength(
                witness.mle.get_num_vars(),
                opening.len(),
            ));
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
        Ok(PreparedReduction {
            h: Some((h, ri_vec)),
        })
    }

    /// The part of [`Self::prove`] that writes the transcript: `h` sent, the point on its
    /// line drawn, and the reduced instance there.
    pub fn send<T: ProvingTranscript>(
        &self,
        prepared: PreparedReduction<F>,
        transcript: &mut T,
    ) -> ReducedInstance<F> {
        let Some((h, ri_vec)) = prepared.h else {
            let (r, claim) = &self.openings[0];
            return ReducedInstance {
                r: r.clone(),
                claim: *claim,
            };
        };
        let num_vars = ri_vec.len();
        // One coefficient per power up to the degree bound: the count the verifier reads.
        let mut coefficients = h.coefficients().to_vec();
        let count = h_coefficients(num_vars, self.openings.len());
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

        ReducedInstance {
            r: eval_on_l(&ri_vec, x_prime.into()),
            claim: h.evaluate(x_prime.into()),
        }
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
