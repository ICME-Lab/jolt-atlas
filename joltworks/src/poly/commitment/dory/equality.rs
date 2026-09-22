//! Equality of two hiding commitments under one registered Dory basis.
//!
//! Each adjacent computation must constrain its own commitment to the actual
//! tensor it consumes or produces. This proof then shows that those tensors
//! agree, even when their commitment blinds differ. It does not authenticate
//! host labels or establish either computation by itself.

use super::{DoryCommitment, DoryHint, DoryVerifierSetup};
use crate::{
    field::JoltField,
    transcripts::{Blake2bTranscript, Transcript},
    utils::errors::ProofVerifyError,
};
use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::{
    backends::arkworks::{ArkFr, ArkGT},
    primitives::arithmetic::Group,
};

/// The registered edge fixes context, encoding, and tensor shape. Both stage
/// verifiers must check these same parameters and use the same Dory basis.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HiddenValueEdge {
    pub context: Vec<u8>,
    pub encoding: Vec<u8>,
    pub shape: Vec<usize>,
    pub producer: DoryCommitment,
    pub consumer: DoryCommitment,
}

/// A Schnorr argument for `producer - consumer = ht * (r_p - r_c)`.
/// Dory's binding property then binds both commitments to the same polynomial.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HiddenValueEqualityProof {
    pub nonce_commitment: ArkGT,
    pub response: Fr,
}

fn challenge(edge: &HiddenValueEdge, setup: &DoryVerifierSetup, nonce: &ArkGT) -> Fr {
    let mut transcript = Blake2bTranscript::new(b"Atlas/hidden-value-equality/v1");
    transcript.append_serializable(setup);
    transcript.append_serializable(edge);
    transcript.append_serializable(nonce);
    transcript.challenge_scalar()
}

fn check_shape(edge: &HiddenValueEdge, setup: &DoryVerifierSetup) -> Result<(), ProofVerifyError> {
    let size = edge.shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d));
    if edge.context.is_empty()
        || edge.encoding.is_empty()
        || edge.shape.is_empty()
        || size.is_none_or(|n| {
            n == 0
                || n.checked_next_power_of_two()
                    .is_none_or(|m| m.ilog2() as usize > setup.0.max_log_n)
        })
        || setup.0.ht == ArkGT::identity()
    {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "Invalid registered hidden edge".into(),
        ));
    }
    Ok(())
}

impl HiddenValueEqualityProof {
    pub fn prove(
        edge: &HiddenValueEdge,
        producer_hint: &DoryHint,
        consumer_hint: &DoryHint,
        setup: &DoryVerifierSetup,
    ) -> Result<Self, ProofVerifyError> {
        check_shape(edge, setup)?;
        let delta = producer_hint.commit_blind.0 - consumer_hint.commit_blind.0;
        if edge.producer.0 - edge.consumer.0 != setup.0.ht.scale(&ArkFr(delta)) {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Hidden edge values differ".into(),
            ));
        }
        let nonce = Fr::random(&mut rand::thread_rng());
        let nonce_commitment = setup.0.ht.scale(&ArkFr(nonce));
        let c = challenge(edge, setup, &nonce_commitment);
        Ok(Self {
            nonce_commitment,
            response: nonce + c * delta,
        })
    }

    pub fn verify(
        &self,
        edge: &HiddenValueEdge,
        setup: &DoryVerifierSetup,
    ) -> Result<(), ProofVerifyError> {
        check_shape(edge, setup)?;
        let c = challenge(edge, setup, &self.nonce_commitment);
        let lhs = setup.0.ht.scale(&ArkFr(self.response));
        let rhs = self.nonce_commitment + (edge.producer.0 - edge.consumer.0).scale(&ArkFr(c));
        if lhs != rhs {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Hidden edge equality rejected".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{
        commitment::{commitment_scheme::CommitmentScheme, dory::DoryScheme},
        multilinear_polynomial::MultilinearPolynomial,
    };

    #[test]
    fn hidden_edge_proof_binds_values_and_registered_context() {
        let pp = DoryScheme::setup_prover(8);
        let vp = DoryScheme::setup_verifier(&pp);
        let tensor = MultilinearPolynomial::from(vec![7i64, -8, 9, 11]);
        let (producer, ph) = DoryScheme::commit_zk(&tensor, &pp);
        let (consumer, ch) = DoryScheme::commit_zk(&tensor, &pp);
        assert_ne!(producer, consumer);
        let edge = HiddenValueEdge {
            context: b"session/action-to-formalization".to_vec(),
            encoding: b"i32-field-v1".to_vec(),
            shape: vec![2, 2],
            producer,
            consumer,
        };
        let proof = HiddenValueEqualityProof::prove(&edge, &ph, &ch, &vp).unwrap();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).unwrap();
        let decoded = HiddenValueEqualityProof::deserialize_compressed(bytes.as_slice()).unwrap();
        decoded.verify(&edge, &vp).unwrap();
        let different = MultilinearPolynomial::from(vec![7i64, -8, 9, 12]);
        let (wrong, wh) = DoryScheme::commit_zk(&different, &pp);
        let mut modified = edge.clone();
        modified.consumer = wrong;
        assert!(decoded.verify(&modified, &vp).is_err());
        assert!(HiddenValueEqualityProof::prove(&modified, &ph, &wh, &vp).is_err());
        let mut modified = edge.clone();
        modified.context.push(0);
        assert!(decoded.verify(&modified, &vp).is_err());
        let mut modified = edge.clone();
        modified.shape = vec![4];
        assert!(decoded.verify(&modified, &vp).is_err());
        let mut modified = edge.clone();
        modified.encoding = b"bytes-v1".to_vec();
        assert!(decoded.verify(&modified, &vp).is_err());
        let mut modified = edge.clone();
        std::mem::swap(&mut modified.producer, &mut modified.consumer);
        assert!(decoded.verify(&modified, &vp).is_err());
        let mut forged = decoded;
        forged.response += Fr::from(1u64);
        assert!(forged.verify(&edge, &vp).is_err());
    }
}
