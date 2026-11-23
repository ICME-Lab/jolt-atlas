//! Validate Akita payload boundaries before decoding the owned Jolt types.

use super::opening_error;
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_pcs::akita::{AkitaBatchProof, AkitaCommitment};

// These tuples mirror the pinned Jolt serde layouts. Borrowing the last field
// checks its length against the input without allocating a byte buffer.
type CommitmentWire<'a> = (u32, [u8; 32], usize, usize, usize, usize, &'a [u8]);
type OpeningWire<'a> = ([u8; 32], &'a [u8]);

pub(super) fn decode_commitment(
    bytes: &[u8],
) -> Result<(AkitaCommitment, usize), ProofVerifyError> {
    let config = bincode::config::standard();
    let (_, used) =
        bincode::serde::borrow_decode_from_slice::<CommitmentWire<'_>, _>(bytes, config)
            .map_err(opening_error)?;
    let frame = bytes
        .get(..used)
        .ok_or_else(|| opening_error("invalid Akita commitment length"))?;
    bincode::serde::decode_from_slice(frame, config).map_err(opening_error)
}

pub(super) fn decode_opening_proof(bytes: &[u8]) -> Result<AkitaBatchProof, ProofVerifyError> {
    let config = bincode::config::standard();
    let (_, used) = bincode::serde::borrow_decode_from_slice::<OpeningWire<'_>, _>(bytes, config)
        .map_err(opening_error)?;
    if used != bytes.len() {
        return Err(opening_error("trailing Akita opening proof bytes"));
    }
    bincode::serde::decode_from_slice(bytes, config)
        .map(|(proof, _)| proof)
        .map_err(opening_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_commitment_frame_boundaries() -> Result<(), Box<dyn std::error::Error>> {
        let config = bincode::config::standard();
        let commitment = AkitaCommitment::default();
        let mut frame = bincode::serde::encode_to_vec(&commitment, config)?;
        let mut sequence = frame.clone();
        sequence.extend_from_slice(&frame);
        assert_eq!(decode_commitment(&sequence)?, (commitment, frame.len()));
        frame.pop();
        assert!(decode_commitment(&frame).is_err());
        Ok(())
    }

    #[test]
    fn checks_opening_frame_boundaries() -> Result<(), Box<dyn std::error::Error>> {
        let config = bincode::config::standard();
        let payload: OpeningWire<'_> = ([7; 32], &[1, 2, 3]);
        let mut frame = bincode::serde::encode_to_vec(payload, config)?;
        assert!(decode_opening_proof(&frame).is_ok());
        frame.push(0);
        assert!(decode_opening_proof(&frame).is_err());
        frame.truncate(frame.len() - 2);
        assert!(decode_opening_proof(&frame).is_err());
        Ok(())
    }
}
