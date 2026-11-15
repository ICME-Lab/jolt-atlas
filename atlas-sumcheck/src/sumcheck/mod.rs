use crate::engine::{verify_compressed, CompressedSumcheckProof, SumcheckClaim, SumcheckError};
use atlas_algebra::{
    field::JoltField,
    poly::univariate::CompressedPoly,
    utils::{errors::ProofVerifyError, serialization::deserialize_vec},
};
use atlas_transcript::Transcript;

use ark_serialize::*;

#[derive(CanonicalSerialize, Debug, Clone)]
pub struct SumcheckInstanceProof<F: JoltField> {
    pub compressed_polys: Vec<CompressedPoly<F>>,
}

impl<F: JoltField> Valid for SumcheckInstanceProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.compressed_polys.check()
    }
}

impl<F: JoltField> CanonicalDeserialize for SumcheckInstanceProof<F> {
    /// The round count is prover-supplied, so the rounds decode through [`deserialize_vec`].
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            compressed_polys: deserialize_vec(reader, compress, validate)?,
        })
    }
}

impl<F: JoltField> SumcheckInstanceProof<F> {
    pub fn new(compressed_polys: Vec<CompressedPoly<F>>) -> SumcheckInstanceProof<F> {
        SumcheckInstanceProof { compressed_polys }
    }

    /// Verify this sumcheck proof.
    /// Note: Verification does not execute the final check of sumcheck protocol: g_v(r_v) = oracle_g(r),
    /// as the oracle is not passed in. Expected that the caller will implement.
    ///
    /// Params
    /// - `claim`: Claimed evaluation
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    pub fn verify<ProofTranscript: Transcript>(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F::Challenge>), ProofVerifyError> {
        if self.compressed_polys.len() != num_rounds {
            return Err(ProofVerifyError::InvalidInputLength(
                num_rounds,
                self.compressed_polys.len(),
            ));
        }
        let claim = SumcheckClaim {
            num_vars: num_rounds,
            degree: degree_bound,
            claimed_sum: claim,
        };
        let proof = CompressedSumcheckProof {
            round_polynomials: self.compressed_polys.clone(),
        };
        let (value, point) =
            verify_compressed(&claim, &proof, transcript).map_err(|e| match e {
                SumcheckError::ProofOfWork { .. } => ProofVerifyError::ProofOfWork,
                _ => ProofVerifyError::SumcheckVerificationError,
            })?;
        let r = point.into_iter().map(F::challenge_from_field).collect();
        Ok((value, r))
    }
}
