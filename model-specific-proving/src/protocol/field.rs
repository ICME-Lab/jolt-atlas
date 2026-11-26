//! Arithmetic and challenge sampling for the prime field used by BitZ.

use crypto_primitives::LiftElement;
use jolt_field::{
    CanonicalBytes, CanonicalEncoding, Field, NaiveAccumulator, Ring, WithAccumulator,
};
use jolt_transcript::{
    ChallengeSetSize, Encoding, FieldTranscript, Keccak, ProverState, VerificationError,
    VerifierState, MAX_CHALLENGE_BYTES,
};
use num_traits::{ConstOne, ConstZero, Zero};

const PRIME: u128 = (1 << 121) - 73;
pub(crate) type Scalar = world_field::Fq<PRIME>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct F(pub(crate) Scalar);

jolt_field::impl_ring_ops!(impl[] F {
    add(a, b): F(a.0 + b.0),
    sub(a, b): F(a.0 - b.0),
    mul(a, b): F(a.0 * b.0),
    neg(a): F(-a.0),
    zero: F(Scalar::ZERO),
    one: F(Scalar::ONE),
});
jolt_field::impl_serde_bytes!(impl[] F, 16);

impl std::fmt::Display for F {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Ring for F {
    fn from_u64(value: u64) -> Self {
        Self(Scalar::from(value))
    }

    fn from_i64(value: i64) -> Self {
        Self::from_i128(value.into())
    }

    fn from_u128(value: u128) -> Self {
        Self(Scalar::from(value))
    }

    fn from_i128(value: i128) -> Self {
        let magnitude = Scalar::from(value.unsigned_abs());
        Self(if value < 0 { -magnitude } else { magnitude })
    }
}

impl WithAccumulator for F {
    type SumAccumulator = NaiveAccumulator<Self>;
    type Accumulator = NaiveAccumulator<Self>;
    type SmallScalarAccumulator = NaiveAccumulator<Self>;
    type SignedProductAccumulator = NaiveAccumulator<Self>;
}

impl Field for F {
    fn inverse(&self) -> Option<Self> {
        (!self.is_zero()).then(|| Self::from_u128(Scalar::META.pow(self.0.lift(), PRIME - 2)))
    }

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        loop {
            let mut bytes = [0; 16];
            rng.fill_bytes(&mut bytes);
            if let Some(value) = sample(u128::from_le_bytes(bytes)) {
                return value;
            }
        }
    }
}

impl CanonicalBytes for F {
    const NUM_BYTES: usize = 16;

    fn to_bytes_le(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.0.lift().to_le_bytes());
    }
}

impl CanonicalEncoding for F {
    const MODULUS_BITS: u32 = 121;
    const CHALLENGE_SET_LIMBS: [u64; 4] = [PRIME as u64, (PRIME >> 64) as u64, 0, 0];

    fn num_bits(&self) -> u32 {
        128 - self.0.lift().leading_zeros()
    }

    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        bytes.iter().rev().fold(Self::from_u64(0), |value, &byte| {
            value.mul_u64(256) + Self::from_u64(byte.into())
        })
    }

    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        Self::from_u128_checked(u128::from_le_bytes(bytes.try_into().ok()?))
    }

    fn to_u128_checked(&self) -> Option<u128> {
        Some(self.0.lift())
    }

    fn from_u128_checked(value: u128) -> Option<Self> {
        (value < PRIME).then(|| Self::from_u128(value))
    }

    fn from_u128_reduced(value: u128) -> Self {
        Self::from_u128(value)
    }
}

impl Encoding for F {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0.lift().to_le_bytes()
    }
}

fn sample(word: u128) -> Option<F> {
    F::from_u128_checked(word & ((1 << F::MODULUS_BITS) - 1))
}

macro_rules! transcript {
    ($type:ty) => {
        impl FieldTranscript<F> for $type {
            const CHALLENGE_SET: ChallengeSetSize =
                ChallengeSetSize::from_limbs(F::CHALLENGE_SET_LIMBS);

            fn challenge_vector(&mut self, count: usize) -> Result<Vec<F>, VerificationError> {
                if count == 0 {
                    return Ok(Vec::new());
                }
                let bytes = count
                    .checked_add(2)
                    .and_then(|n| n.checked_mul(16))
                    .filter(|&n| n <= MAX_CHALLENGE_BYTES)
                    .ok_or(VerificationError)?;
                self.verifier_message_as(bytes, |bytes| {
                    let values: Vec<_> = bytes
                        .as_chunks::<16>()
                        .0
                        .iter()
                        .copied()
                        .map(u128::from_le_bytes)
                        .filter_map(sample)
                        .take(count)
                        .collect();
                    (values.len() == count)
                        .then_some(values)
                        .ok_or(VerificationError)
                })
            }
        }
    };
}

transcript!(ProverState<Keccak>);
transcript!(VerifierState<'_, Keccak>);
