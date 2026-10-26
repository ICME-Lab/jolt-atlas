//! Proof decoding whose memory is bounded by the bytes actually read.
//!
//! ark's `Vec<T>` decoder pre-allocates the length it reads, which in a proof is prover-supplied:
//! a few forged bytes request an arbitrarily large allocation (or a capacity-overflow panic)
//! before any element is read. [`deserialize_vec`] reads the same encoding but grows the vector
//! as elements actually decode, so a forged length fails at the end of the input instead.

use ark_serialize::{CanonicalDeserialize, Compress, Read, SerializationError, Validate};

/// The most elements a decoded length pre-allocates.
const PREALLOCATE_LIMIT: usize = 1 << 10;

/// Decode a `Vec<T>` in ark's encoding (a `u64` length, then the elements), allocating no more
/// than the decoded elements need.
pub fn deserialize_vec<T: CanonicalDeserialize, R: Read>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<Vec<T>, SerializationError> {
    let len = u64::deserialize_with_mode(&mut reader, compress, validate)?;
    let len = usize::try_from(len).map_err(|_| SerializationError::InvalidData)?;
    let mut values = Vec::with_capacity(len.min(PREALLOCATE_LIMIT));
    for _ in 0..len {
        values.push(T::deserialize_with_mode(&mut reader, compress, validate)?);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::bn254::Fr;
    use ark_serialize::CanonicalSerialize;
    use jolt_field::Field;

    fn decode<T: CanonicalDeserialize>(bytes: &[u8]) -> Result<Vec<T>, SerializationError> {
        deserialize_vec(bytes, Compress::Yes, Validate::Yes)
    }

    #[test]
    fn reads_ark_vec_encoding() {
        let mut rng = ark_std::test_rng();
        let values: Vec<Fr> = (0..5).map(|_| Fr::random(&mut rng)).collect();
        let mut bytes = Vec::new();
        values.serialize_compressed(&mut bytes).unwrap();
        assert_eq!(decode::<Fr>(&bytes).unwrap(), values);
    }

    #[test]
    fn forged_length_fails_without_allocating_it() {
        // A length of 2⁶⁴ − 1 with no elements behind it: ark's decoder would request that
        // capacity up front.
        let bytes = u64::MAX.to_le_bytes();
        assert!(decode::<Fr>(&bytes).is_err());
        assert!(decode::<u8>(&bytes).is_err());
    }
}
