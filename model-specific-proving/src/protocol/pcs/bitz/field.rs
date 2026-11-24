use crypto_primitives::LiftElement;
use jolt_field::{
    CanonicalBytes, CanonicalEncoding, Field, NaiveAccumulator, Ring, WithAccumulator,
};
use jolt_transcript::{
    ChallengeSetSize, Encoding, FieldTranscript, Keccak, ProverState, VerificationError,
    VerifierState, MAX_CHALLENGE_BYTES,
};
use num_traits::{ConstOne, ConstZero, Zero};
use std::fmt::{self, Display};

const Q: u128 = (1 << 121) - 73;
pub(super) type Fq = world_field::Fq<Q>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BitzField(pub(super) Fq);

jolt_field::impl_ring_ops!(impl[] BitzField {
    add(a, b): Self(a.0 + b.0),
    sub(a, b): Self(a.0 - b.0),
    mul(a, b): Self(a.0 * b.0),
    neg(a): Self(-a.0),
    zero: Self(Fq::ZERO),
    one: Self(Fq::ONE),
});
jolt_field::impl_serde_bytes!(impl[] BitzField, 16);

impl Display for BitzField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Ring for BitzField {
    fn from_u64(x: u64) -> Self {
        Self(Fq::from(x))
    }
    fn from_i64(x: i64) -> Self {
        Self::from_i128(x.into())
    }
    fn from_u128(x: u128) -> Self {
        Self(Fq::from(x))
    }
    fn from_i128(x: i128) -> Self {
        let magnitude = Self::from_u128(x.unsigned_abs());
        if x < 0 {
            -magnitude
        } else {
            magnitude
        }
    }
}

impl WithAccumulator for BitzField {
    type SumAccumulator = NaiveAccumulator<Self>;
    type Accumulator = NaiveAccumulator<Self>;
    type SmallScalarAccumulator = NaiveAccumulator<Self>;
    type SignedProductAccumulator = NaiveAccumulator<Self>;
}

impl Field for BitzField {
    fn inverse(&self) -> Option<Self> {
        (!self.is_zero()).then(|| Self::from_u128(Fq::META.pow(self.0.lift(), Q - 2)))
    }
    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        loop {
            let mut word = [0; 16];
            rng.fill_bytes(&mut word);
            if let Some(field) = candidate(word) {
                return field;
            }
        }
    }
}

impl CanonicalBytes for BitzField {
    const NUM_BYTES: usize = 16;
    fn to_bytes_le(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.0.lift().to_le_bytes());
    }
}

impl CanonicalEncoding for BitzField {
    const MODULUS_BITS: u32 = Fq::META.bits;
    const CHALLENGE_SET_LIMBS: [u64; 4] = [Q as u64, (Q >> 64) as u64, 0, 0];
    fn num_bits(&self) -> u32 {
        128 - self.0.lift().leading_zeros()
    }
    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        bytes
            .iter()
            .rev()
            .fold(Self::zero(), |x, &b| x.mul_u64(256) + Self::from_u8(b))
    }
    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        Self::from_u128_checked(u128::from_le_bytes(bytes.try_into().ok()?))
    }
    fn to_u128_checked(&self) -> Option<u128> {
        Some(self.0.lift())
    }
    fn from_u128_checked(x: u128) -> Option<Self> {
        (x < Q).then(|| Self::from_u128(x))
    }
    fn from_u128_reduced(x: u128) -> Self {
        Self::from_u128(x)
    }
}

impl Encoding for BitzField {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0.lift().to_le_bytes()
    }
}

fn candidate(word: [u8; 16]) -> Option<BitzField> {
    BitzField::from_u128_checked(u128::from_le_bytes(word) & ((1 << BitzField::MODULUS_BITS) - 1))
}

macro_rules! transcript {
    ($state:ty) => {
        impl FieldTranscript<BitzField> for $state {
            const CHALLENGE_SET: ChallengeSetSize =
                ChallengeSetSize::from_limbs(BitzField::CHALLENGE_SET_LIMBS);

            fn challenge_vector(&mut self, n: usize) -> Result<Vec<BitzField>, VerificationError> {
                if n == 0 {
                    return Ok(Vec::new());
                }
                // For q = 2^121 - 73, exhausting two spare candidates has probability below 2^-280.
                let budget = n
                    .checked_add(2)
                    .and_then(|n| n.checked_mul(16))
                    .filter(|&bytes| bytes <= MAX_CHALLENGE_BYTES)
                    .ok_or(VerificationError)?;
                self.verifier_message_as(budget, |bytes| {
                    let values: Vec<_> = bytes
                        .as_chunks::<16>()
                        .0
                        .iter()
                        .copied()
                        .filter_map(candidate)
                        .take(n)
                        .collect();
                    (values.len() == n)
                        .then_some(values)
                        .ok_or(VerificationError)
                })
            }
        }
    };
}
transcript!(ProverState<Keccak>);
transcript!(VerifierState<'_, Keccak>);
