use super::multilinear_polynomial::PolynomialEvaluation;
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    poly::{
        multilinear_polynomial::MultilinearPolynomial,
        prefix_suffix::{
            CachedPolynomial, Prefix, PrefixCheckpoints, PrefixPolynomial, PrefixRegistry,
            PrefixSuffixPolynomial, SuffixPolynomial,
        },
    },
    utils::{lookup_bits::LookupBits, math::Math},
};
use allocative::Allocative;
use num::Integer;
use std::sync::{Arc, RwLock};

// -------- Signed Identity Poly --------

#[derive(Clone, Debug, Allocative, Default)]
pub struct SignedIdentityPoly<F: JoltField, const X_LEN: usize> {
    _field: core::marker::PhantomData<F>,
}

impl<F: JoltField, const X_LEN: usize> PolynomialEvaluation<F> for SignedIdentityPoly<F, X_LEN> {
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * X_LEN);

        let mut positive_case = F::zero();

        // if x < 0, abs(x) = -x = (!x) + 1
        let mut negative_case = F::one();

        r.iter()
            .skip(X_LEN  /* skip high bits */ + 1 /* skip sign bit */)
            .rev()
            .enumerate()
            .for_each(|(i, &r_i)| {
                positive_case += r_i * F::from_u64(i.pow2() as u64);
                negative_case += (F::one() - r_i) * F::from_u64(i.pow2() as u64)
            });

        let sign_bit = r[X_LEN];
        positive_case * (F::one() - sign_bit) - negative_case * sign_bit
    }
}

impl<F: JoltField, const X_LEN: usize> PrefixSuffixPolynomial<F, 4>
    for SignedIdentityPoly<F, X_LEN>
{
    fn suffixes(&self) -> [Box<dyn SuffixPolynomial<F> + Sync>; 4] {
        [
            Box::new(OneSuffixPolynomial),
            Box::new(ReluSuffixPoly::<F, X_LEN>::default()),
            Box::new(OneSuffixPolynomial),
            Box::new(NegReluSuffixPoly::<F, X_LEN>::default()),
        ]
    }

    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        registry: &mut PrefixRegistry<F>,
    ) -> [Option<Arc<RwLock<CachedPolynomial<F>>>>; 4] {
        let relu = registry.get_or_insert(Prefix::Relu, |cp| {
            ReluPrefixPoly::<F, X_LEN>::default().prefix_polynomial(cp, chunk_len, phase)
        });
        let not_msb = registry.get_or_insert(Prefix::NotMsb, |cp| {
            NotMsbPrefixPoly::<F, X_LEN>::default().prefix_polynomial(cp, chunk_len, phase)
        });
        let anti_relu = registry.get_or_insert(Prefix::AntiRelu, |cp| {
            AntiReluPrefixPoly::<F, X_LEN>::default().prefix_polynomial(cp, chunk_len, phase)
        });
        let neg_msb = registry.get_or_insert(Prefix::NegMsb, |cp| {
            NegMsbPrefixPoly::<F, X_LEN>::default().prefix_polynomial(cp, chunk_len, phase)
        });
        [Some(relu), Some(not_msb), Some(anti_relu), Some(neg_msb)]
    }
}

// -------- -MSB Poly --------

#[derive(Clone, Debug, Allocative, Default)]
pub struct NegMsbPrefixPoly<F: JoltField, const X_LEN: usize> {
    _field: core::marker::PhantomData<F>,
}

impl<F: JoltField, const X_LEN: usize> PrefixPolynomial<F> for NegMsbPrefixPoly<F, X_LEN> {
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        phase: usize,
    ) -> CachedPolynomial<F> {
        debug_assert!(chunk_len.is_even());
        let bound_value = checkpoints[Prefix::NegMsb].unwrap_or(-F::one());
        let evals = match phase_position(chunk_len, phase, X_LEN) {
            PhasePosition::AtMsb => (0..chunk_len.pow2())
                .map(|i| {
                    let sign_bit = ((i >> (chunk_len - 1)) & 1) as u32;
                    -F::from_u32(sign_bit)
                })
                .collect(),
            _ => vec![bound_value; chunk_len.pow2()],
        };
        make_cached_poly(evals, chunk_len)
    }
}

// -------- One Suffix Poly --------

pub struct OneSuffixPolynomial;
impl<F: JoltField> SuffixPolynomial<F> for OneSuffixPolynomial {
    fn suffix_mle(&self, _b: LookupBits) -> u64 {
        1
    }
}

// -------- Relu Poly --------

#[derive(Clone, Debug, Allocative, Default)]
pub struct ReluPrefixPoly<F: JoltField, const X_LEN: usize> {
    _field: core::marker::PhantomData<F>,
}

#[derive(Clone, Debug, Allocative, Default)]
pub struct ReluSuffixPoly<F: JoltField, const X_LEN: usize> {
    _field: core::marker::PhantomData<F>,
}

impl<F: JoltField, const X_LEN: usize> PrefixPolynomial<F> for ReluPrefixPoly<F, X_LEN> {
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        phase: usize,
    ) -> CachedPolynomial<F> {
        debug_assert!(chunk_len.is_even());
        let bound_value = checkpoints[Prefix::Relu].unwrap_or(F::zero());
        let evals = match phase_position(chunk_len, phase, X_LEN) {
            PhasePosition::BeforeMsb => vec![F::zero(); chunk_len.pow2()],
            // TODO: Add assert as this case only works if num_phases * chunk_len == X_LEN * 2
            PhasePosition::AtMsb => (0..chunk_len.pow2())
                .map(|i| {
                    let sign_bit = ((i >> (chunk_len - 1)) & 1) as u32;
                    let res =
                        bound_value + F::from_u64(((i << chunk_len) % (1 << (X_LEN - 1))) as u64);
                    res * (F::one() - F::from_u32(sign_bit))
                })
                .collect(),
            PhasePosition::AfterMsb => {
                let not_sign_bit = checkpoints[Prefix::NotMsb].unwrap();
                (0..chunk_len.pow2())
                    .map(|i| bound_value + F::from_u64(i as u64) * not_sign_bit)
                    .collect()
            }
        };
        make_cached_poly(evals, chunk_len)
    }
}

impl<F: JoltField, const X_LEN: usize> SuffixPolynomial<F> for ReluSuffixPoly<F, X_LEN> {
    fn suffix_mle(&self, bits: LookupBits) -> u64 {
        let mut b: u64 = bits.into();
        b %= 1 << X_LEN;
        if bits.len() >= X_LEN {
            let sign_bit = b >> (X_LEN - 1);
            b * (1 - sign_bit)
        } else {
            b
        }
    }
}

// -------- !MSB Poly (complement of MsbPoly) --------

#[derive(Clone, Debug, Allocative, Default)]
pub struct NotMsbPrefixPoly<F: JoltField, const X_LEN: usize> {
    _field: core::marker::PhantomData<F>,
}

impl<F: JoltField, const X_LEN: usize> PrefixPolynomial<F> for NotMsbPrefixPoly<F, X_LEN> {
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        phase: usize,
    ) -> CachedPolynomial<F> {
        debug_assert!(chunk_len.is_even());
        let bound_value = checkpoints[Prefix::NotMsb].unwrap_or(F::one());
        let evals = match phase_position(chunk_len, phase, X_LEN) {
            PhasePosition::AtMsb => (0..chunk_len.pow2())
                .map(|i| {
                    let sign_bit = ((i >> (chunk_len - 1)) & 1) as u32;
                    F::one() - F::from_u32(sign_bit)
                })
                .collect(),
            _ => vec![bound_value; chunk_len.pow2()],
        };
        make_cached_poly(evals, chunk_len)
    }
}

// -------- AntiRelu Poly (-Relu(-x) = min(0, x)) --------

#[derive(Clone, Debug, Allocative, Default)]
pub struct AntiReluPrefixPoly<F: JoltField, const X_LEN: usize> {
    _field: core::marker::PhantomData<F>,
}

impl<F: JoltField, const X_LEN: usize> PrefixPolynomial<F> for AntiReluPrefixPoly<F, X_LEN> {
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        phase: usize,
    ) -> CachedPolynomial<F> {
        debug_assert!(chunk_len.is_even());
        let bound_value = checkpoints[Prefix::AntiRelu].unwrap_or(F::zero());
        let evals = match phase_position(chunk_len, phase, X_LEN) {
            PhasePosition::BeforeMsb => vec![F::zero(); chunk_len.pow2()],
            // TODO: Add assert as this case only works if num_phases * chunk_len == X_LEN * 2
            PhasePosition::AtMsb => (0..chunk_len.pow2())
                .map(|i| {
                    let sign_bit = ((i >> (chunk_len - 1)) & 1) as u32;
                    let res = bound_value
                        + F::from_u64(
                            ((!i << chunk_len/* TODO: this should be suffix len */)
                                % (1 << (X_LEN - 1))) as u64,
                        )
                        + F::one();
                    -res * F::from_u32(sign_bit)
                })
                .collect(),
            PhasePosition::AfterMsb => {
                let neg_sign_bit = checkpoints[Prefix::NegMsb].unwrap();
                (0..chunk_len.pow2())
                    .map(|i| bound_value + F::from_u64((!i % (1 << chunk_len) /* TODO: make this more robust as this is kinda hacky as this only works if there is only one phase after AtMSB phase*/) as u64) * neg_sign_bit)
                    .collect()
            }
        };
        make_cached_poly(evals, chunk_len)
    }
}

// -------- NegRelu Poly --------
#[derive(Clone, Debug, Allocative, Default)]
pub struct NegReluSuffixPoly<F: JoltField, const X_LEN: usize> {
    _field: core::marker::PhantomData<F>,
}

impl<F: JoltField, const X_LEN: usize> SuffixPolynomial<F> for NegReluSuffixPoly<F, X_LEN> {
    fn suffix_mle(&self, bits: LookupBits) -> u64 {
        let b: u64 = bits.into();
        let num_bits = bits.len();

        if num_bits >= X_LEN {
            // Two's complement negation, gated by sign bit: neg_relu(x) = is_negative * |x|
            let is_negative = (b >> (X_LEN - 1)) & 1;
            let abs_value = (!b % (1 << (X_LEN - 1))) + 1;
            abs_value * is_negative
        } else {
            // Partial suffix: flip the available bits (!b truncated to num_bits).
            !b % (1 << num_bits)
        }
    }
}

// -------- Helpers --------

/// Classifies the current phase relative to the MSB boundary.
enum PhasePosition {
    BeforeMsb,
    AtMsb,
    AfterMsb,
}

fn phase_position(chunk_len: usize, phase: usize, x_len: usize) -> PhasePosition {
    let j = chunk_len * phase;
    if j < x_len {
        PhasePosition::BeforeMsb
    } else if j == x_len {
        PhasePosition::AtMsb
    } else {
        PhasePosition::AfterMsb
    }
}

/// Wraps evals into a `CachedPolynomial` with standard cache capacity.
fn make_cached_poly<F: JoltField>(evals: Vec<F>, chunk_len: usize) -> CachedPolynomial<F> {
    CachedPolynomial::new(MultilinearPolynomial::from(evals), (chunk_len - 1).pow2())
}

#[cfg(test)]
mod tests {
    use crate::{
        field::JoltField,
        poly::{
            multilinear_polynomial::PolynomialEvaluation,
            prefix_suffix::{tests::prefix_suffix_decomposition_test, Prefix},
            signed_identity_poly::SignedIdentityPoly,
        },
        utils::index_to_field_bitvector,
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
        prefix_suffix_decomposition_test::<8, 2, 4, _>(
            SignedIdentityPoly::<Fr, 4>::default(),
            vec![Prefix::Relu, Prefix::AntiRelu],
        );
    }
}
