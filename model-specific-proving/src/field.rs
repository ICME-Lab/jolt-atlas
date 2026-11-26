//! Jolt transcript and polynomial traits for World BitZ's prime field.

use jolt_field::{CanonicalBytes, CanonicalEncoding, Field, Ring};
use jolt_transcript::{
    ChallengeSetSize, Encoding, FieldTranscript, Keccak, ProverState, ProverTranscript,
    VerificationError, VerifierState, VerifierTranscript, MAX_CHALLENGE_BYTES,
};
use num_traits::Zero;

const MASK: u128 = (1 << 119) - 1;
const MODULUS: u128 = MASK - 68;
pub type Scalar = world_field::Fq<MODULUS>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct F(u128);

jolt_field::impl_ring_ops!(impl[] F {
    add(a, b): Self(canonical(a.0 + b.0)),
    sub(a, b): Self(canonical((a.0 + MODULUS) - b.0)),
    mul(a, b): a.multiply(b),
    neg(a): Self(canonical(MODULUS - a.0)),
    zero: Self(0),
    one: Self(1),
});
jolt_field::impl_serde_bytes!(impl[] F, 16);

impl std::fmt::Display for F {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Ring for F {
    fn from_u64(value: u64) -> Self {
        Self(u128::from(value))
    }

    fn from_i64(value: i64) -> Self {
        let x = Self::from_u64(value.unsigned_abs());
        if value < 0 {
            -x
        } else {
            x
        }
    }

    fn from_u128(value: u128) -> Self {
        Self(canonical(fold(value)))
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

#[inline]
fn canonical(value: u128) -> u128 {
    if value < MODULUS {
        value
    } else {
        value - MODULUS
    }
}

#[inline]
fn fold(value: u128) -> u128 {
    (value & MASK) + 69 * (value >> 119)
}

impl F {
    #[inline]
    fn multiply(self, other: Self) -> Self {
        let [a, b] = [u128::from(self.0 as u64), self.0 >> 64];
        let [c, d] = [u128::from(other.0 as u64), other.0 >> 64];
        let low = a * c;
        let middle = a * d + b * c + (low >> 64);
        let upper = b * d + (middle >> 64);
        let tail = u128::from(low as u64) | ((middle & ((1 << 55) - 1)) << 64);
        let high = (u128::from(middle as u64) >> 55) | (upper << 9);
        Self(canonical(fold(tail + 69 * high)))
    }

    pub(crate) fn scalar(self) -> Scalar {
        Scalar::from(self.0)
    }

    pub(crate) fn mul_integer(self, integer: i128) -> Self {
        let magnitude = integer.unsigned_abs();
        let Ok(word) = u64::try_from(magnitude) else {
            return self * Self::from_i128(integer);
        };
        let x = self.0;
        let lower = u128::from(x as u64) * u128::from(word);
        let upper = (x >> 64) * u128::from(word) + (lower >> 64);
        let low_bits = (upper as u64 & ((1 << 55) - 1)) as u128;
        let y = Self::from_u128(((low_bits << 64) | u128::from(lower as u64)) + (upper >> 55) * 69);
        if integer < 0 {
            -y
        } else {
            y
        }
    }
}

impl Field for F {
    fn inverse(&self) -> Option<Self> {
        (!self.is_zero()).then(|| Self::from_u128(Scalar::META.pow(self.0, MODULUS - 2)))
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
        out.copy_from_slice(&self.0.to_le_bytes());
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
        128 - self.0.leading_zeros()
    }

    fn to_u128_checked(&self) -> Option<u128> {
        Some(self.0)
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
        self.0.to_le_bytes()
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

pub(crate) trait Mixing<E: Field>: FieldTranscript<E> + Sized {
    fn grind_mix(&mut self, bits: u32) -> Result<(), VerificationError>;

    fn mix(&mut self, count: usize, loss: usize) -> Result<Vec<E>, &'static str> {
        let bits =
            jolt_sumcheck::round_pow_bits::<E, Self>(loss).map_err(|_| "invalid challenge size")?;
        self.grind_mix(bits).map_err(|_| "invalid proof of work")?;
        self.challenge_vector(count)
            .map_err(|_| "invalid transcript challenge")
    }
}

impl Mixing<F> for ProverState<Keccak> {
    fn grind_mix(&mut self, bits: u32) -> Result<(), VerificationError> {
        self.grind(bits)
    }
}

impl Mixing<F> for VerifierState<'_, Keccak> {
    fn grind_mix(&mut self, bits: u32) -> Result<(), VerificationError> {
        self.check_grind(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto_primitives::LiftElement;

    #[test]
    fn field_operations_match_world_on_carries_and_boundaries() {
        let boundary = [
            0,
            1,
            68,
            69,
            1 << 55,
            1 << 64,
            MODULUS - 2,
            MODULUS - 1,
            MODULUS,
            u128::MAX,
        ];
        let check = |a, b| {
            let (aa, bb) = (F::from_u128(a), F::from_u128(b));
            let (sa, sb) = (Scalar::from(a), Scalar::from(b));
            assert_eq!((aa * bb).0, (sa * sb).lift());
            assert_eq!((aa + bb).0, (sa + sb).lift());
            assert_eq!((aa - bb).0, (sa - sb).lift());
            assert_eq!(
                aa.mul_integer(i128::from(b as i64)),
                aa * F::from_i64(b as i64)
            );
        };
        for a in boundary {
            for b in boundary {
                check(a, b);
            }
        }
        let mut seed = 12345u128;
        let mut random = || {
            seed = seed
                .wrapping_mul(47026247687942121848144207491837523525)
                .wrapping_add(1);
            seed ^ (seed >> 79)
        };
        for _ in 0..10000 {
            check(random(), random());
        }
    }
}
