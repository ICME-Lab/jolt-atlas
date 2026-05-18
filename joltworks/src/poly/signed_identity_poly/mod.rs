use super::multilinear_polynomial::PolynomialEvaluation;
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    poly::{
        identity_poly::{OperandPolynomial, OperandSide, ShiftHalfSuffixPolynomial},
        prefix_suffix::{
            CachedPolynomial, Prefix, PrefixPolynomial, PrefixRegistry, PrefixSuffixPolynomial,
            SuffixPolynomial,
        },
        signed_identity_poly::{
            prefixes::{OperandMsbPrefixPoly, SignedIdentityPrefixPoly},
            suffixes::{OneSuffixPolynomial, SignedIdentitySuffixPoly},
        },
    },
    utils::math::Math,
};
use allocative::Allocative;
use std::sync::{Arc, RwLock};

mod prefixes;
mod suffixes;

// -------- Signed Identity Poly --------

/// Multilinear polynomial that interprets a boolean input vector as a signed integer.
///
/// Used by the RAF sum-check as `sum_{k,j} ra(k, j) * Self(k)`.
///
/// Layout: the input length equals `2 * X_LEN`. The first `X_LEN` coordinates are high bits
/// (skipped in the evaluation), the coordinate at index `X_LEN` is the sign bit, and the remaining
/// `X_LEN - 1` coordinates are the magnitude (value) bits. Let `u` be the lower `X_LEN` bits
/// interpreted as an unsigned integer and `s` the sign bit. Then
/// `u - s * 2^X_LEN = value_bits - s * 2^(X_LEN - 1)`, which is exactly the two's-complement
/// signed interpretation.
/// However this means we are restricted to even number of variables.
///
/// In the prime field, negatives are represented modulo `p`: for `m > 0`, `-m` is encoded as
/// `p - m` (equivalently `-m mod p`).
#[derive(Clone, Debug, Allocative, Default)]
pub struct SignedIdentityPoly<F: JoltField, const X_LEN: usize> {
    _field: core::marker::PhantomData<F>,
}

impl<F: JoltField, const X_LEN: usize> PolynomialEvaluation<F> for SignedIdentityPoly<F, X_LEN> {
    /// Self(x) = (sum_{i=0}^{X_LEN-1} x[X_LEN + i] * 2^i) - x[X_LEN] * 2^X_LEN
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * X_LEN);
        let mut y = F::zero();
        r.iter()
            .skip(X_LEN) // skip high bits
            .rev()
            .enumerate()
            .for_each(|(i, &r_i)| {
                let w = F::from_u64(i.pow2() as u64);
                y += r_i * w;
            });
        let sign_bit = r[X_LEN];
        let w = F::from_u64(X_LEN.pow2() as u64);
        y - sign_bit * w
    }
}

impl<F: JoltField, const X_LEN: usize> PrefixSuffixPolynomial<F, 2>
    for SignedIdentityPoly<F, X_LEN>
{
    fn suffixes(&self) -> [Box<dyn SuffixPolynomial<F> + Sync>; 2] {
        [
            Box::new(OneSuffixPolynomial),
            Box::new(SignedIdentitySuffixPoly::<F, X_LEN>::default()),
        ]
    }

    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        registry: &mut PrefixRegistry<F>,
    ) -> [Option<Arc<RwLock<CachedPolynomial<F>>>>; 2] {
        let identity = registry.get_or_insert(Prefix::SignedIdentity, |cp| {
            SignedIdentityPrefixPoly::<F, X_LEN>::default().prefix_polynomial(cp, chunk_len, phase)
        });
        [Some(identity), None]
    }
}

// -------- Signed Operand Poly --------

/// Multilinear polynomial that interprets one interleaved operand as a signed integer.
///
/// Used when evaluating either the left or right operand lane from an interleaved bit-vector.
///
/// Layout: the input length equals `2 * X_LEN` and bits are interleaved by operand.
/// For the left operand, relevant coordinates are `x[2*i]`; for the right operand,
/// relevant coordinates are `x[2*i + 1]`. In each lane, the first coordinate is the sign bit
/// and the remaining `X_LEN - 1` coordinates are value bits. Let `u` be the lane interpreted as
/// an unsigned integer and `s` the sign bit (MSB). Then
/// `u - s * 2^X_LEN = value_bits - s * 2^(X_LEN - 1)`, which is exactly the two's-complement
/// signed interpretation.
///
/// In the prime field, negatives are represented modulo `p`: for `m > 0`, `-m` is encoded as
/// `p - m` (equivalently `-m mod p`).
#[derive(Clone, Debug)]
pub struct SignedOperandPoly<F: JoltField, const X_LEN: usize> {
    _field: core::marker::PhantomData<F>,
    side: OperandSide,
}

impl<F: JoltField, const X_LEN: usize> SignedOperandPoly<F, X_LEN> {
    pub fn new(side: OperandSide) -> Self {
        Self {
            _field: core::marker::PhantomData,
            side,
        }
    }
}

impl<F: JoltField, const X_LEN: usize> PolynomialEvaluation<F> for SignedOperandPoly<F, X_LEN> {
    /// Left Operand Case: Self(x) = (sum_{i=0}^{X_LEN-1} x[2*i] * 2^i) - x[0] * 2^X_LEN
    /// Right Operand Case: Self(x) = (sum_{i=0}^{X_LEN-1} x[2*i+1] * 2^i) - x[1] * 2^X_LEN
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let len = r.len();
        debug_assert_eq!(len, 2 * X_LEN);
        let shift = match self.side {
            OperandSide::Left => 0,
            OperandSide::Right => 1,
        };
        let sign_bit = r[shift];
        let y: F = (0..X_LEN)
            .map(|i| {
                let r_i: F = r[2 * i + shift].into();
                r_i.mul_u128(1u128 << (X_LEN - 1 - i))
            })
            .sum();
        y - sign_bit * F::from_u64(X_LEN.pow2() as u64)
    }
}

impl<F: JoltField, const X_LEN: usize> PrefixSuffixPolynomial<F, 3>
    for SignedOperandPoly<F, X_LEN>
{
    fn suffixes(&self) -> [Box<dyn SuffixPolynomial<F> + Sync>; 3] {
        [
            Box::new(ShiftHalfSuffixPolynomial),
            Box::new(OperandPolynomial::new(2 * X_LEN, self.side)),
            Box::new(OneSuffixPolynomial),
        ]
    }

    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        prefix_registry: &mut PrefixRegistry<F>,
    ) -> [Option<Arc<RwLock<CachedPolynomial<F>>>>; 3] {
        match self.side {
            OperandSide::Left => {
                let lo = prefix_registry.get_or_insert(Prefix::LeftOperand, |cp| {
                    OperandPolynomial::new(2 * X_LEN, OperandSide::Left)
                        .prefix_polynomial(cp, chunk_len, phase)
                });
                let msb = prefix_registry.get_or_insert(Prefix::LeftOperandMSB, |cp| {
                    OperandMsbPrefixPoly::<F, X_LEN>::new(OperandSide::Left)
                        .prefix_polynomial(cp, chunk_len, phase)
                });
                [Some(lo), None, Some(msb)]
            }
            OperandSide::Right => {
                let ro = prefix_registry.get_or_insert(Prefix::RightOperand, |cp| {
                    OperandPolynomial::new(2 * X_LEN, OperandSide::Right)
                        .prefix_polynomial(cp, chunk_len, phase)
                });
                let msb = prefix_registry.get_or_insert(Prefix::RightOperandMSB, |cp| {
                    OperandMsbPrefixPoly::<F, X_LEN>::new(OperandSide::Right)
                        .prefix_polynomial(cp, chunk_len, phase)
                });
                [Some(ro), None, Some(msb)]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        field::JoltField,
        poly::{
            identity_poly::OperandSide,
            multilinear_polynomial::PolynomialEvaluation,
            prefix_suffix::{tests::prefix_suffix_decomposition_test, Prefix},
            signed_identity_poly::{SignedIdentityPoly, SignedOperandPoly},
        },
        utils::{index_to_field_bitvector, interleave_bits},
    };
    use ark_bn254::Fr;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn test_signed_identity_poly_evaluation() {
        for i in 0..256usize {
            let i = i as i8;
            let bit_vector: Vec<Fr> = index_to_field_bitvector(i as u8 as u64, 16);
            assert_eq!(
                Fr::from_i8(i),
                SignedIdentityPoly::<Fr, 8>::default().evaluate(&bit_vector)
            );
        }
    }

    #[test]
    fn test_signed_operand_poly_evaluation() {
        for i in 0..256usize {
            let lo = i as i8;
            let ro = i as i8;
            let index = interleave_bits(lo as u8 as u32, ro as u8 as u32);
            let bit_vector: Vec<Fr> = index_to_field_bitvector(index, 16);
            assert_eq!(
                Fr::from_i8(lo),
                SignedOperandPoly::<Fr, 8>::new(OperandSide::Left).evaluate(&bit_vector),
                "failed for value = {lo}, bit_vector = {bit_vector:?}"
            );
            assert_eq!(
                Fr::from_i8(ro),
                SignedOperandPoly::<Fr, 8>::new(OperandSide::Right).evaluate(&bit_vector),
                "failed for value = {ro}"
            );
        }
    }

    #[test]
    fn test_signed_identity_poly_random_evaluations() {
        let mut rng = StdRng::seed_from_u64(404);
        for _ in 0..1000 {
            let i = rng.gen();
            let bit_vector: Vec<Fr> = index_to_field_bitvector(i as u32 as u64, 64);
            assert_eq!(
                Fr::from_i32(i),
                SignedIdentityPoly::<Fr, 32>::default().evaluate(&bit_vector)
            );
        }

        // i32::MIN edge case
        let i = i32::MIN;
        let bit_vector: Vec<Fr> = index_to_field_bitvector(i as u32 as u64, 64);
        assert_eq!(
            Fr::from_i32(i),
            SignedIdentityPoly::<Fr, 32>::default().evaluate(&bit_vector)
        );
    }

    #[test]
    fn signed_identity_poly_prefix_suffix_decomposition() {
        prefix_suffix_decomposition_test::<16, 2, 2, true, _>(
            SignedIdentityPoly::<Fr, 8>::default(),
            vec![Prefix::SignedIdentity],
        );
    }

    #[test]
    fn signed_operand_poly_prefix_suffix_decomposition() {
        prefix_suffix_decomposition_test::<16, 2, 3, false, _>(
            SignedOperandPoly::<Fr, 8>::new(OperandSide::Left),
            vec![Prefix::LeftOperand, Prefix::LeftOperandMSB],
        );
        prefix_suffix_decomposition_test::<16, 2, 3, false, _>(
            SignedOperandPoly::<Fr, 8>::new(OperandSide::Right),
            vec![Prefix::RightOperand, Prefix::RightOperandMSB],
        );
    }
}
