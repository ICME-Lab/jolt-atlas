//! How many proof-of-work bits a Fiat-Shamir challenge needs.
//!
//! A cheating prover often wins a protocol step only if a verifier challenge
//! lands in a small bad set. The step's loss factor `L` is a bound on that
//! set: a challenge drawn uniformly from the challenge set `C` is bad with
//! probability at most `L / |C|`. For example, a degree-`d` univariate
//! sumcheck round has `L = d`, because two distinct polynomials of degree at
//! most `d` agree on at most `d` points.
//!
//! Under Fiat-Shamir the prover can retry, at one hash call per fresh
//! challenge. A proof of work of `b` bits right before the challenge (`grind`
//! and `check_grind`) passes on each attempt with probability `2^-b`, so in
//! the random-oracle model each hash call yields a bad challenge with
//! probability at most `L / |C| · 2^-b`. The least `b` with
//! `|C| · 2^b ≥ L · 2^λ` makes this at most `2^-λ`: λ bits of security per
//! protected challenge. [`grind_bits_for_loss`] returns that `b`, comparing
//! exact integers.
//!
//! `|C|` is the size of an ideal uniform draw. A codec's distance from
//! uniform ([`ChallengeCodec`]) is charged separately.

use std::error::Error;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// The largest supported difficulty. The nonce space, all of `u32` at this
/// difficulty, still holds `2^7` times the expected number of trials.
pub const MAX_POW_BITS: u32 = 25;

/// The largest security level [`grind_bits_for_loss`] accepts. With a `u64`
/// loss factor it keeps `L · 2^λ` below `2^256 − 1`, the largest size a
/// [`ChallengeSetSize`] stores exactly.
const MAX_SECURITY_BITS: u32 = 192;

/// The exact number of values a challenge can take.
///
/// Sizes of at least `2^256 − 1` are all stored as `2^256 − 1`. Every target
/// [`grind_bits_for_loss`] compares against is smaller, so its answer is exact
/// for any size; two stored sizes at that cap compare equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChallengeSetSize {
    // `hi` first, so the derived order is numeric.
    hi: u128,
    lo: u128,
}

impl ChallengeSetSize {
    const SATURATED: Self = Self {
        hi: u128::MAX,
        lo: u128::MAX,
    };

    /// A set of exactly `2^bits` challenges, such as challenges masked to
    /// `bits` bits. From `bits = 256` on, the stored size is `2^256 − 1`.
    pub const fn power_of_two(bits: u32) -> Self {
        match bits {
            0..=127 => Self {
                hi: 0,
                lo: 1 << bits,
            },
            128..=255 => Self {
                hi: 1 << (bits - 128),
                lo: 0,
            },
            _ => Self::SATURATED,
        }
    }

    /// The `modulus^degree` elements of a field: a prime field for
    /// `degree = 1`, or its degree-`degree` extension. Primality of `modulus`
    /// is not checked.
    ///
    /// # Errors
    /// [`GrindBitsError::InvalidFieldOrder`] if `modulus < 2` or `degree == 0`.
    pub const fn field_order(modulus: u128, degree: u32) -> Result<Self, GrindBitsError> {
        if modulus < 2 || degree == 0 {
            Err(GrindBitsError::InvalidFieldOrder)
        } else {
            Ok(Self::from_u128(modulus).pow(degree))
        }
    }

    pub(crate) const fn from_u128(value: u128) -> Self {
        Self { hi: 0, lo: value }
    }

    /// `self^exponent`, saturating.
    pub(crate) const fn pow(self, mut exponent: u32) -> Self {
        let mut base = self;
        let mut result = Self::from_u128(1);
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = result.saturating_mul(base);
            }
            base = base.saturating_mul(base);
            exponent >>= 1;
        }
        result
    }

    /// `min(self · other, 2^256 − 1)`. A saturated factor stands for any value
    /// at or above the cap, and the result is still the capped product.
    const fn saturating_mul(self, other: Self) -> Self {
        // Two factors of at least 2^128 multiply to at least 2^256.
        let (wide, narrow) = if self.hi == 0 {
            (other, self.lo)
        } else if other.hi == 0 {
            (self, other.lo)
        } else {
            return Self::SATURATED;
        };
        let (carry, lo) = widening_mul(wide.lo, narrow);
        match wide.hi.checked_mul(narrow) {
            Some(high) => match high.checked_add(carry) {
                Some(hi) => Self { hi, lo },
                None => Self::SATURATED,
            },
            None => Self::SATURATED,
        }
    }
}

/// The 256-bit product `a · b` as its `(high, low)` 128-bit halves.
const fn widening_mul(a: u128, b: u128) -> (u128, u128) {
    const LOW: u128 = u64::MAX as u128;
    let (a1, a0) = (a >> 64, a & LOW);
    let (b1, b0) = (b >> 64, b & LOW);
    let low = a0 * b0;
    let cross_a = a1 * b0;
    let cross_b = a0 * b1;
    // Below 3 · 2^64.
    let middle = (low >> 64) + (cross_a & LOW) + (cross_b & LOW);
    let high = a1 * b1 + (cross_a >> 64) + (cross_b >> 64) + (middle >> 64);
    (high, (middle << 64) | (low & LOW))
}

/// `⌈loss · 2^security_bits / 2^bits⌉`, at most `2^256 − 2^192` for
/// `security_bits ≤ MAX_SECURITY_BITS`. An integer size `s` satisfies
/// `s ≥ ⌈x / 2^bits⌉` exactly when `s · 2^bits ≥ x`.
fn target(loss: u64, security_bits: u32, bits: u32) -> ChallengeSetSize {
    let loss = u128::from(loss);
    match security_bits.checked_sub(bits) {
        Some(shift) => {
            ChallengeSetSize::from_u128(loss).saturating_mul(ChallengeSetSize::power_of_two(shift))
        }
        None => ChallengeSetSize::from_u128(loss.div_ceil(1 << (bits - security_bits))),
    }
}

/// The least proof-of-work difficulty `b ≤ MAX_POW_BITS` with
/// `|C| · 2^b ≥ L · 2^λ`, for loss factor `L = loss_factor`, challenge set
/// `C` and security level `λ = security_bits`.
///
/// With that difficulty, each hash call yields a bad challenge with
/// probability at most `2^-λ` (see the module docs). The result is a valid
/// difficulty for `grind` and `check_grind`; `0` means no proof of work.
///
/// # Errors
/// [`GrindBitsError::ZeroLoss`] if `loss_factor == 0`,
/// [`GrindBitsError::SecurityBitsTooLarge`] if `security_bits > 192`, and
/// [`GrindBitsError::Unattainable`] if `MAX_POW_BITS` bits do not suffice.
pub fn grind_bits_for_loss(
    loss_factor: u64,
    challenge_set: ChallengeSetSize,
    security_bits: u32,
) -> Result<u32, GrindBitsError> {
    if loss_factor == 0 {
        return Err(GrindBitsError::ZeroLoss);
    }
    if security_bits > MAX_SECURITY_BITS {
        return Err(GrindBitsError::SecurityBitsTooLarge);
    }
    (0..=MAX_POW_BITS)
        .find(|&bits| challenge_set >= target(loss_factor, security_bits, bits))
        .ok_or(GrindBitsError::Unattainable)
}

/// Why a proof-of-work difficulty or challenge-set size was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrindBitsError {
    /// The loss factor is zero.
    ZeroLoss,
    /// The security level exceeds 192 bits.
    SecurityBitsTooLarge,
    /// The modulus is below 2 or the extension degree is zero.
    InvalidFieldOrder,
    /// Even [`MAX_POW_BITS`] bits of proof of work fall short.
    Unattainable,
}

impl Display for GrindBitsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.write_str(match self {
            Self::ZeroLoss => "proof-of-work loss factor must be nonzero",
            Self::SecurityBitsTooLarge => "security level exceeds 192 bits",
            Self::InvalidFieldOrder => "field modulus must be at least 2 and degree nonzero",
            Self::Unattainable => {
                "security target needs more than MAX_POW_BITS bits of proof of work"
            }
        })
    }
}

impl Error for GrindBitsError {}

// The challenge codec: how full-width field challenges decode from squeezed bytes. It lives with
// the field (rather than the transcript) because `JoltField::ChallengeCodecField` is bounded on
// it and `atlas-transcript` depends on this crate.

use crate::field::{ExtField, Fp128, Fp32, Fp64, PseudoMersenne};
use sealed::Sealed;

mod sealed {
    pub trait Sealed {}
}

/// Decodes a fixed-length uniform byte tape into one message of field elements.
///
/// Implementations must budget conventional TV ≤ 2^-161 for the entire message,
/// including rejection-pool exhaustion, rather than for each element separately.
/// No bytes are drawn adaptively. Scalar draws use `len == 1` of this same codec.
/// Empty messages consume zero bytes and decode to an empty vector.
///
/// # Pseudo-Mersenne fields
///
/// Candidates have `NUM_BYTES` little-endian bytes, masked to `MODULUS_BITS`;
/// checked decoding accepts exactly the representatives below the prime.
/// Masking makes each candidate uniform on [0, 2^b); acceptance gives a
/// uniform element of [0, p). Accepted values are independent of acceptance
/// indicators, so the first `len` accepted values are jointly uniform even
/// conditioned on pool success.
///
/// With rejection probability ρ = c/2^b ≤ 2^-R, R = b - ceil(log2(c)),
/// failure among B = len+k candidates needs at least k+1 rejections. A union
/// bound gives Pr[fail] ≤ C(B,k+1) ρ^(k+1) ≤ 2^(-(k+1)(R-L)), where
/// L = ceil(log2(B)). Choose the least k with (k+1)(R-L) ≥ 161. Once L ≥ R,
/// increasing k cannot make the bound attainable. All arithmetic is integral.
///
/// The trait is sealed: every implementation lives in this module, next to
/// the argument for its TV bound.
///
/// # Use outside this crate
///
/// `ChallengeCodec` bounds the field of every `ProverTranscript` and
/// `VerifierTranscript` `field_challenges` call, so a protocol that draws
/// field challenges from these transcripts names it as a bound. Callers
/// cannot implement it. The widths and decodings above are part of the
/// transcript's wire format: changing one changes every proof that draws
/// challenges in that field.
pub trait ChallengeCodec: Sized + Sealed {
    /// `|Self|`, the number of values one challenge can take. Proof-of-work
    /// difficulties are priced from it ([`grind_bits_for_loss`]).
    const CHALLENGE_SET_SIZE: ChallengeSetSize;

    /// Bytes squeezed for `len` elements; depends only on the type and `len`.
    /// Returns `None` on arithmetic overflow or an unattainable rejection bound.
    fn challenge_bytes(len: usize) -> Option<usize>;

    /// Decodes exactly `challenge_bytes(len)` bytes in little-endian order.
    /// Returns `None` on a wrong/unsupported length or rejection-pool exhaustion.
    /// Allocates only the returned vector, with capacity `len`.
    fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>>;
}

/// The most bytes one challenge message may squeeze.
pub const MAX_CHALLENGE_BYTES: usize = 1 << 24;

fn ceil_log2(n: u128) -> u32 {
    u128::BITS - n.saturating_sub(1).leading_zeros()
}

#[expect(non_snake_case, reason = "R and L match the pooled-rejection bound")]
fn pool_bytes<F: PseudoMersenne>(len: usize) -> Option<usize> {
    if len == 0 {
        return Some(0);
    }
    let R = F::MODULUS_BITS.checked_sub(ceil_log2(F::OFFSET))?;
    let mut k = 0usize;
    loop {
        let candidates = len.checked_add(k)?;
        let L = ceil_log2(u128::try_from(candidates).ok()?);
        let margin = R.checked_sub(L).filter(|&margin| margin > 0)?;
        if k.checked_add(1)?
            .checked_mul(usize::try_from(margin).ok()?)?
            >= 161
        {
            return candidates.checked_mul(F::NUM_BYTES);
        }
        k = k.checked_add(1)?;
    }
}

/// `|F|^degree`, where a pseudo-Mersenne base has order
/// `2^MODULUS_BITS - OFFSET`.
const fn pseudo_mersenne_power<F: PseudoMersenne>(degree: usize) -> ChallengeSetSize {
    let modulus = (u128::MAX >> (128 - F::MODULUS_BITS)) - (F::OFFSET - 1);
    ChallengeSetSize::from_u128(modulus).pow(degree as u32)
}

fn decode_pool<F, E, const D: usize>(bytes: &[u8], len: usize) -> Option<Vec<E>>
where
    F: PseudoMersenne,
    E: ExtField<F>,
{
    if D != E::DEGREE || bytes.len() != pool_bytes::<F>(len.checked_mul(D)?)? {
        return None;
    }
    let mask = u128::MAX.checked_shr(128u32.checked_sub(F::MODULUS_BITS)?)?;
    let mut accepted = bytes.chunks_exact(F::NUM_BYTES).filter_map(|candidate| {
        let mut word = [0u8; 16];
        word.get_mut(..F::NUM_BYTES)?.copy_from_slice(candidate);
        let masked = (u128::from_le_bytes(word) & mask).to_le_bytes();
        F::from_bytes_le_checked(masked.get(..F::NUM_BYTES)?)
    });
    let mut result = Vec::with_capacity(len);
    for _ in 0..len {
        let mut coordinates = [F::default(); D];
        for coordinate in &mut coordinates {
            *coordinate = accepted.next()?;
        }
        // ExtField::from_base_fn supplies exactly 0..DEGREE; D is checked above.
        result.push(E::from_base_fn(|i| {
            coordinates.get(i).copied().unwrap_or_default()
        }));
    }
    Some(result)
}

macro_rules! prime_codec {
    ($field:ident, $word:ty) => {
        impl<const P: $word> Sealed for $field<P> {}

        impl<const P: $word> ChallengeCodec for $field<P> {
            const CHALLENGE_SET_SIZE: ChallengeSetSize = pseudo_mersenne_power::<Self>(1);

            fn challenge_bytes(len: usize) -> Option<usize> {
                pool_bytes::<Self>(len)
            }

            fn decode_challenges(bytes: &[u8], len: usize) -> Option<Vec<Self>> {
                decode_pool::<Self, Self, 1>(bytes, len)
            }
        }
    };
}

prime_codec!(Fp32, u32);
prime_codec!(Fp64, u64);
prime_codec!(Fp128, u128);

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::{
        grind_bits_for_loss, ChallengeSetSize, GrindBitsError, MAX_POW_BITS, MAX_SECURITY_BITS,
    };
    use num::BigUint;

    fn big(size: ChallengeSetSize) -> BigUint {
        (BigUint::from(size.hi) << 128u32) | BigUint::from(size.lo)
    }

    fn capped(value: BigUint) -> BigUint {
        value.min((BigUint::from(1u8) << 256u32) - 1u8)
    }

    /// The least `b ≤ MAX_POW_BITS` with `|C| · 2^b ≥ L · 2^λ`, in unbounded
    /// integers.
    fn oracle(loss: u64, cardinality: &BigUint, security_bits: u32) -> Option<u32> {
        let target = BigUint::from(loss) << security_bits;
        (0..=MAX_POW_BITS).find(|&bits| (cardinality << bits) >= target)
    }

    const P128: u128 = u128::MAX - 274;
    const P64: u128 = (1 << 64) - 59;

    /// `(modulus, degree)` pairs on both sides of the 256-bit cap.
    const FIELDS: [(u128, u32); 9] = [
        (2, 1),
        (P64, 2),
        (P128, 1),
        (P128, 2),
        (P128, 3),
        (u128::MAX, 2),
        (u128::MAX, 3),
        (3, 161),
        (3, 162),
    ];

    const POWERS: [u32; 12] = [0, 1, 63, 64, 102, 103, 125, 127, 128, 192, 255, 1000];

    fn sets() -> Vec<(ChallengeSetSize, BigUint)> {
        let fields = FIELDS.iter().map(|&(modulus, degree)| {
            (
                ChallengeSetSize::field_order(modulus, degree).unwrap(),
                BigUint::from(modulus).pow(degree),
            )
        });
        let powers = POWERS.iter().map(|&bits| {
            (
                ChallengeSetSize::power_of_two(bits),
                BigUint::from(1u8) << bits,
            )
        });
        fields.chain(powers).collect()
    }

    #[test]
    fn sizes_are_exact_below_the_cap_and_saturate_above() {
        for (size, exact) in sets() {
            assert_eq!(big(size), capped(exact.clone()), "{exact}");
        }
        assert!(BigUint::from(3u8).pow(161) < BigUint::from(1u8) << 256u32);
        assert!(BigUint::from(3u8).pow(162) > BigUint::from(1u8) << 256u32);
        assert_eq!(
            ChallengeSetSize::field_order(3, 162).unwrap(),
            ChallengeSetSize::power_of_two(256)
        );
    }

    #[test]
    fn grind_bits_match_the_unbounded_oracle() {
        let losses = [
            1,
            2,
            3,
            8,
            (1 << 24) - 1,
            1 << 24,
            (1 << 24) + 1,
            (1 << 25) - 1,
            1 << 25,
            (1 << 25) + 1,
            u64::MAX,
        ];
        for (size, exact) in sets() {
            for loss in losses {
                for security_bits in [0, 5, 64, 100, 128, MAX_SECURITY_BITS] {
                    assert_eq!(
                        grind_bits_for_loss(loss, size, security_bits).ok(),
                        oracle(loss, &exact, security_bits),
                        "size={exact}, loss={loss}, security_bits={security_bits}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_last_supported_difficulty_is_exact() {
        let fp128 = ChallengeSetSize::field_order(P128, 1).unwrap();
        assert_eq!(grind_bits_for_loss((1 << 25) - 1, fp128, 128), Ok(25));
        assert_eq!(
            grind_bits_for_loss(1 << 25, fp128, 128),
            Err(GrindBitsError::Unattainable)
        );
        // Equality reaches the target: 2^103 · 2^25 = 1 · 2^128.
        let set = ChallengeSetSize::power_of_two(103);
        assert_eq!(grind_bits_for_loss(1, set, 128), Ok(MAX_POW_BITS));
        assert_eq!(
            grind_bits_for_loss(2, set, 128),
            Err(GrindBitsError::Unattainable)
        );
        assert_eq!(
            grind_bits_for_loss(1, ChallengeSetSize::power_of_two(102), 128),
            Err(GrindBitsError::Unattainable)
        );
        // Difficulty above λ rounds the target up: 2 · 2^6 ≥ 3 · 2^5 > 2 · 2^5.
        assert_eq!(
            grind_bits_for_loss(3, ChallengeSetSize::power_of_two(1), 5),
            Ok(6)
        );
    }

    #[test]
    fn pseudo_mersenne_codecs_count_their_fields() {
        use super::ChallengeCodec;
        use crate::field::{Prime128Offset275, Prime24Offset3, Prime64Offset59};

        let order = |modulus, degree| ChallengeSetSize::field_order(modulus, degree).unwrap();
        assert_eq!(Prime128Offset275::CHALLENGE_SET_SIZE, order(P128, 1));
        assert_eq!(Prime24Offset3::CHALLENGE_SET_SIZE, order((1 << 24) - 3, 1));
        assert_eq!(Prime64Offset59::CHALLENGE_SET_SIZE, order(P64, 1));
    }

    #[test]
    fn rejects_invalid_inputs() {
        let set = ChallengeSetSize::power_of_two(128);
        assert_eq!(
            grind_bits_for_loss(0, set, 128),
            Err(GrindBitsError::ZeroLoss)
        );
        assert_eq!(
            grind_bits_for_loss(1, set, MAX_SECURITY_BITS + 1),
            Err(GrindBitsError::SecurityBitsTooLarge)
        );
        for (modulus, degree) in [(0, 1), (1, 1), (2, 0)] {
            assert_eq!(
                ChallengeSetSize::field_order(modulus, degree),
                Err(GrindBitsError::InvalidFieldOrder)
            );
        }
    }
}
