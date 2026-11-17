//! The prover side of [`super`].

use super::*;
use crate::opening_proof::Opening;
use atlas_algebra::{
    field::JoltField, poly::multilinear::MultilinearPolynomial, utils::errors::ProvingError,
};
use atlas_transcript::ProvingTranscript;

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
