//! The prover side of [`super`].

use super::*;
use crate::opening_proof::Opening;
use atlas_algebra::{
    field::JoltField, poly::multilinear_polynomial::MultilinearPolynomial,
    utils::errors::ProvingError,
};
use atlas_transcript::Transcript;

impl EvalReductionProtocol {
    pub fn prove<F: JoltField, T: Transcript>(
        openings: &[&Opening<F>],
        output_mle: MultilinearPolynomial<F>,
        transcript: &mut T,
    ) -> Result<(EvalReductionProof<F>, ReducedInstance<F>), ProvingError> {
        let witness = EvalReductionWitness::new(output_mle);
        let instance = EvalReductionInstance::new(openings);

        instance.prove(&witness, transcript)
    }
}
