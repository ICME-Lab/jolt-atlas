use crate::engine::{verify_compressed, CompressedSumcheckProof, SumcheckClaim, SumcheckError};
use atlas_algebra::{
    field::JoltField, poly::univariate::CompressedPoly, utils::errors::ProofVerifyError,
};
use atlas_transcript::Transcript;

use atlas_algebra::serialize::{Canonical, Error};
use std::io::{Read, Write};

#[derive(Debug, Clone)]
pub struct SumcheckInstanceProof<F: JoltField> {
    pub compressed_polys: Vec<CompressedPoly<F>>,
}

/// The rounds as a `Vec` (its `u64` length, then each round's compressed polynomial).
impl<F: JoltField> Canonical for SumcheckInstanceProof<F> {
    fn encode<W: Write>(&self, writer: W) -> Result<(), Error> {
        self.compressed_polys.encode(writer)
    }

    fn encoded_len(&self) -> usize {
        self.compressed_polys.encoded_len()
    }

    fn decode<R: Read>(reader: R) -> Result<Self, Error> {
        Ok(Self {
            compressed_polys: Vec::decode(reader)?,
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
