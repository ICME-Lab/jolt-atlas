//! Canonical torus encoding of a BN254 target-group commitment.
//! For x = a + b*w of norm one, with w^2 = v, encode t = b/(a+1).
//! Recover a = (1+v*t^2)/(1-v*t^2), b = 2*t/(1-v*t^2).
//! Identity maps to zero. The excluded point -1 has order two and is not in GT.
//! This changes transport only; use the original commitment in transcripts.
use super::DoryCommitment;
use ark_bn254::{Fq12, Fq12Config, Fq6};
use ark_ec::pairing::PairingOutput;
use ark_ff::{AdditiveGroup, Field, Fp12Config};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use dory::backends::arkworks::ArkGT;

/// A 192-byte transport wrapper around the original 384-byte GT commitment.
/// Checked decoding enforces the original prime-order subgroup membership.
/// Do not substitute this encoding for the original transcript representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactDoryCommitment(pub DoryCommitment);

fn times_nonresidue(mut x: Fq6) -> Fq6 {
    Fq12Config::mul_fp6_by_nonresidue_in_place(&mut x);
    x
}

impl CanonicalSerialize for CompactDoryCommitment {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let x = self.0 .0 .0 .0;
        // In particular, reject the zero field element instead of mapping it
        // to the identity. Full subgroup validation remains on checked decode.
        if x.c0.square() - times_nonresidue(x.c1.square()) != Fq6::ONE {
            return Err(SerializationError::InvalidData);
        }
        let inverse = (x.c0 + Fq6::ONE)
            .inverse()
            .ok_or(SerializationError::InvalidData)?;
        (x.c1 * inverse).serialize_with_mode(writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        Fq6::ZERO.serialized_size(compress)
    }
}
// BN254 p - r = 6u^2, in little-endian limbs, where u = 4965661367192848881.
const P_MINUS_R: [u64; 2] = [0xf83e9682e87cfd46, 0x6f4d8248eeb859fb];

impl Valid for CompactDoryCommitment {
    fn check(&self) -> Result<(), SerializationError> {
        let x = self.0 .0 .0 .0;
        // For nonzero x, x^r = 1 iff x^p = x^(p-r). Frobenius computes x^p
        // exactly, while p-r has 127 bits. Zero must be rejected explicitly.
        // Use ordinary field powering: arbitrary torus inputs need not be
        // cyclotomic, so cyclotomic squaring would not be valid here.
        if x != Fq12::ZERO && x.frobenius_map(1) == x.pow(P_MINUS_R) {
            Ok(())
        } else {
            Err(SerializationError::InvalidData)
        }
    }
}
impl CanonicalDeserialize for CompactDoryCommitment {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let t = Fq6::deserialize_with_mode(reader, compress, validate)?;
        let vt2 = times_nonresidue(t.square());
        let inverse = (Fq6::ONE - vt2)
            .inverse()
            .ok_or(SerializationError::InvalidData)?;
        let x = Fq12::new((Fq6::ONE + vt2) * inverse, t.double() * inverse);
        let result = Self(DoryCommitment(ArkGT(PairingOutput(x))));
        // Norm one is not sufficient: the torus contains elements outside GT.
        if validate == Validate::Yes {
            result.check()?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Bn254, Fr, G1Affine, G2Affine};
    use ark_ec::{pairing::Pairing, AffineRepr};
    use ark_ff::PrimeField;

    fn bytes<T: CanonicalSerialize>(value: &T) -> Vec<u8> {
        let mut data = Vec::new();
        value.serialize_compressed(&mut data).unwrap();
        data
    }
    #[test]
    fn compact_gt_roundtrips_identity_powers_and_inverses_exactly() {
        let base = Bn254::pairing(G1Affine::generator(), G2Affine::generator()).0;
        for i in 0..32u64 {
            for scalar in [Fr::from(i), -Fr::from(i)] {
                let original = DoryCommitment(ArkGT(PairingOutput(base.pow(scalar.into_bigint()))));
                let encoded = bytes(&CompactDoryCommitment(original));
                assert_eq!(encoded.len(), 192);
                assert_eq!(bytes(&original).len(), 384);
                let restored =
                    CompactDoryCommitment::deserialize_compressed(encoded.as_slice()).unwrap();
                assert_eq!(restored.0, original);
                assert_eq!(bytes(&restored.0), bytes(&original));
                assert_eq!(bytes(&restored), encoded);
            }
        }
    }
    #[test]
    fn compact_gt_rejects_noncanonical_fields_and_non_subgroup_torus_points() {
        assert!(CompactDoryCommitment::deserialize_compressed([255u8; 192].as_slice()).is_err());
        // This is a well-formed norm-one element but is not in the prime-order GT.
        let bad = bytes(&Fq6::ONE);
        assert!(CompactDoryCommitment::deserialize_compressed(bad.as_slice()).is_err());
        assert!(CompactDoryCommitment::deserialize_compressed([0u8; 191].as_slice()).is_err());
    }
    #[test]
    fn compact_gt_rejects_zero_and_the_excluded_order_two_point() {
        for x in [Fq12::ZERO, -Fq12::ONE] {
            let invalid = CompactDoryCommitment(DoryCommitment(ArkGT(PairingOutput(x))));
            assert!(invalid.serialize_compressed(Vec::new()).is_err());
        }
    }
}

#[cfg(test)]
#[path = "compact_subgroup_tests.rs"]
mod subgroup_tests;
