use crate::{
    field::JoltField, poly::prefix_suffix::SuffixPolynomial, utils::lookup_bits::LookupBits,
};

pub struct OneSuffixPolynomial;
impl<F: JoltField> SuffixPolynomial<F> for OneSuffixPolynomial {
    fn suffix_mle(&self, _b: LookupBits) -> u64 {
        1
    }
}

#[derive(Clone, Debug, Default)]
pub struct SignedIdentitySuffixPoly<F: JoltField, const X_LEN: usize> {
    _field: core::marker::PhantomData<F>,
}

impl<F: JoltField, const X_LEN: usize> SuffixPolynomial<F> for SignedIdentitySuffixPoly<F, X_LEN> {
    fn suffix_mle(&self, bits: LookupBits) -> u64 {
        debug_assert!(X_LEN <= 32);
        let b = u64::from(bits) % (1 << X_LEN);
        if bits.len() < X_LEN {
            return b;
        }
        let sign_bit = (b >> (X_LEN - 1)) & 1;
        (b as i64 - (sign_bit << X_LEN) as i64) as u64
    }
}
