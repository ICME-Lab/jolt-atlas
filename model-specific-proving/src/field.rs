//! Jolt transcript and polynomial traits for World BitZ's prime field.

use crypto_primitives::LiftElement;
use jolt_field::{CanonicalBytes, CanonicalEncoding, Field, Ring};
use jolt_transcript::{
    ChallengeSetSize, Encoding, FieldTranscript, Keccak, ProverState, VerificationError,
    VerifierState, MAX_CHALLENGE_BYTES,
};
use num_traits::{ConstOne, ConstZero, Zero};

const MODULUS: u128 = (1 << 119) - 69;
pub type Scalar = world_field::Fq<MODULUS>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct F(pub Scalar);

jolt_field::impl_ring_ops!(impl[] F {
    add(a, b): Self(a.0 + b.0),
    sub(a, b): Self(a.0 - b.0),
    mul(a, b): Self(a.0 * b.0),
    neg(a): Self(-a.0),
    zero: Self(Scalar::ZERO),
    one: Self(Scalar::ONE),
});
jolt_field::impl_serde_bytes!(impl[] F, 16);

impl std::fmt::Display for F {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Ring for F {
    fn from_u64(value: u64) -> Self {
        Self::from_u128(u128::from(value))
    }

    fn from_i64(value: i64) -> Self {
        Self::from_i128(i128::from(value))
    }

    fn from_u128(value: u128) -> Self {
        Self(Scalar::from(value))
    }

    fn from_i128(value: i128) -> Self {
        let magnitude = Self::from_u128(value.unsigned_abs());
        if value < 0 {
            -magnitude
        } else {
            magnitude
        }
    }
}

impl Field for F {
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let mut power = *self;
        let mut inverse = Self(Scalar::ONE);
        let mut exponent = MODULUS - 2;
        while exponent != 0 {
            if exponent & 1 != 0 {
                inverse *= power;
            }
            power *= power;
            exponent >>= 1;
        }
        Some(inverse)
    }

    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        loop {
            let mut bytes = [0; 16];
            rng.fill_bytes(&mut bytes);
            if let Some(value) = candidate(bytes) {
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
    const MODULUS_BITS: u32 = 119;
    const CHALLENGE_SET_LIMBS: [u64; 4] = [MODULUS as u64, (MODULUS >> 64) as u64, 0, 0];

    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        Self::from_u128_checked(u128::from_le_bytes(bytes.try_into().ok()?))
    }

    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        let base = Self::from_u64(256);
        bytes
            .iter()
            .rev()
            .fold(Self::zero(), |a, &b| a * base + Self::from_u8(b))
    }

    fn num_bits(&self) -> u32 {
        128 - self.0.lift().leading_zeros()
    }

    fn to_u128_checked(&self) -> Option<u128> {
        Some(self.0.lift())
    }

    fn from_u128_checked(value: u128) -> Option<Self> {
        (value < MODULUS).then(|| Self::from_u128(value))
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

fn candidate(bytes: [u8; 16]) -> Option<F> {
    F::from_u128_checked(u128::from_le_bytes(bytes) & ((1 << 119) - 1))
}

fn sample(bytes: &[u8], count: usize) -> Result<Vec<F>, VerificationError> {
    let values: Vec<_> = bytes
        .as_chunks::<16>()
        .0
        .iter()
        .copied()
        .filter_map(candidate)
        .take(count)
        .collect();
    if values.len() == count {
        Ok(values)
    } else {
        Err(VerificationError)
    }
}

fn sample_len(count: usize) -> Result<usize, VerificationError> {
    count
        .checked_add(1)
        .and_then(|n| n.checked_mul(16))
        .filter(|&n| n <= MAX_CHALLENGE_BYTES)
        .ok_or(VerificationError)
}

impl FieldTranscript<F> for ProverState<Keccak> {
    const CHALLENGE_SET: ChallengeSetSize = ChallengeSetSize::from_limbs(F::CHALLENGE_SET_LIMBS);

    fn challenge_vector(&mut self, count: usize) -> Result<Vec<F>, VerificationError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let len = sample_len(count)?;
        self.verifier_message_as(len, |bytes| sample(bytes, count))
    }
}

impl FieldTranscript<F> for VerifierState<'_, Keccak> {
    const CHALLENGE_SET: ChallengeSetSize = ChallengeSetSize::from_limbs(F::CHALLENGE_SET_LIMBS);

    fn challenge_vector(&mut self, count: usize) -> Result<Vec<F>, VerificationError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let len = sample_len(count)?;
        self.verifier_message_as(len, |bytes| sample(bytes, count))
    }
}
