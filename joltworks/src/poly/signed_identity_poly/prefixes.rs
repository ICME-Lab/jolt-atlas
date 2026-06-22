use num::Integer;

use crate::{
    field::JoltField,
    poly::{
        identity_poly::OperandSide,
        multilinear_polynomial::MultilinearPolynomial,
        prefix_suffix::{CachedPolynomial, Prefix, PrefixCheckpoints, PrefixPolynomial},
        signed_identity_poly::SignedIdentityPoly,
    },
    utils::math::Math,
};

#[derive(Clone, Debug)]
pub struct OperandMsbPrefixPoly<F: JoltField, const X_LEN: usize> {
    _field: core::marker::PhantomData<F>,
    side: OperandSide,
}

impl<F: JoltField, const X_LEN: usize> OperandMsbPrefixPoly<F, X_LEN> {
    pub fn new(side: OperandSide) -> Self {
        Self {
            _field: core::marker::PhantomData,
            side,
        }
    }
}

impl<F: JoltField, const X_LEN: usize> PrefixPolynomial<F> for OperandMsbPrefixPoly<F, X_LEN> {
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        phase: usize,
    ) -> CachedPolynomial<F> {
        debug_assert!(chunk_len.is_even());
        let bound_value = match self.side {
            OperandSide::Left => checkpoints[Prefix::LeftOperandMSB].unwrap_or(F::zero()),
            OperandSide::Right => checkpoints[Prefix::RightOperandMSB].unwrap_or(F::zero()),
        };
        let shift = match self.side {
            OperandSide::Left => 1,
            OperandSide::Right => 2,
        };
        let evals = match phase {
            0 => (0..chunk_len.pow2())
                .map(|i| {
                    let sign_bit = ((i >> (chunk_len - shift)) & 1) as u32;
                    sign_correction::<F>(sign_bit, X_LEN)
                })
                .collect(),
            _ => vec![bound_value; chunk_len.pow2()],
        };
        make_cached_poly(evals, chunk_len)
    }
}

impl<F: JoltField> PrefixPolynomial<F> for SignedIdentityPoly<F> {
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        phase: usize,
    ) -> CachedPolynomial<F> {
        let xlen = self.num_vars;
        debug_assert!(chunk_len.is_even());
        debug_assert!(
            chunk_len * (phase + 1) <= xlen,
            "total_len must equal X_LEN"
        );
        let suffix_len = xlen - chunk_len * (phase + 1);
        let bound_value = checkpoints[Prefix::SignedIdentity].unwrap_or(F::zero());
        let evals = match phase_position(chunk_len, phase) {
            PhasePosition::AtMsb => (0..chunk_len.pow2())
                .map(|i| {
                    let sign_bit = ((i >> (chunk_len - 1)) & 1) as u32;
                    let res = bound_value + F::from_u64(((i << suffix_len) % (1 << xlen)) as u64);
                    res + sign_correction::<F>(sign_bit, xlen)
                })
                .collect(),
            PhasePosition::AfterMsb => (0..chunk_len.pow2())
                .map(|i| bound_value + F::from_u64((i << suffix_len) as u64))
                .collect(),
            _ => panic!(
                "Invalid phase for SignedIdentityPrefixPoly: phase {} with chunk_len {}",
                phase, chunk_len
            ),
        };
        make_cached_poly(evals, chunk_len)
    }
}

/// Wraps evals into a `CachedPolynomial` with standard cache capacity.
fn make_cached_poly<F: JoltField>(evals: Vec<F>, chunk_len: usize) -> CachedPolynomial<F> {
    CachedPolynomial::new(MultilinearPolynomial::from(evals), (chunk_len - 1).pow2())
}

/// Returns `-sign_bit * 2^X_LEN` as a field element.
fn sign_correction<F: JoltField>(sign_bit: u32, xlen: usize) -> F {
    -F::from_u32(sign_bit) * F::from_u64(xlen.pow2() as u64)
}

/// Classifies the current phase relative to the MSB boundary.
#[allow(clippy::enum_variant_names)]
enum PhasePosition {
    BeforeMsb,
    AtMsb,
    AfterMsb,
}

fn phase_position(chunk_len: usize, phase: usize) -> PhasePosition {
    let j = chunk_len * phase;
    if j == 0 {
        PhasePosition::AtMsb
    } else {
        PhasePosition::AfterMsb
    }
}

#[derive(Clone, Debug, Default)]
pub struct SignedIdentityPrefixPoly<F: JoltField, const X_LEN: usize> {
    _field: core::marker::PhantomData<F>,
}

impl<F: JoltField, const X_LEN: usize> PrefixPolynomial<F> for SignedIdentityPrefixPoly<F, X_LEN> {
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        phase: usize,
    ) -> CachedPolynomial<F> {
        debug_assert!(chunk_len.is_even());
        debug_assert!(
            chunk_len * (phase + 1) <= X_LEN * 2,
            "total_len must equal X_LEN * 2"
        );
        let suffix_len = X_LEN * 2 - chunk_len * (phase + 1);
        let bound_value = checkpoints[Prefix::SignedIdentity].unwrap_or(F::zero());
        let evals = match phase_position(chunk_len, phase) {
            PhasePosition::BeforeMsb => vec![F::zero(); chunk_len.pow2()], // suffix handles
            PhasePosition::AtMsb => (0..chunk_len.pow2())
                .map(|i| {
                    let sign_bit = ((i >> (chunk_len - 1)) & 1) as u32;
                    let res = bound_value + F::from_u64(((i << suffix_len) % (1 << X_LEN)) as u64);
                    res + sign_correction::<F>(sign_bit, X_LEN)
                })
                .collect(),
            PhasePosition::AfterMsb => (0..chunk_len.pow2())
                .map(|i| bound_value + F::from_u64((i << suffix_len) as u64))
                .collect(),
        };
        make_cached_poly(evals, chunk_len)
    }
}
