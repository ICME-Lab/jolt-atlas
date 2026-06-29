use super::multilinear_polynomial::{BindingOrder, PolynomialBinding, PolynomialEvaluation};
use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    poly::{
        identity_poly::{
            IdentityPolynomial, OperandPolynomial, OperandSide, ShiftHalfSuffixPolynomial,
        },
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

/// Multilinear polynomial encoding a single signed integer for sumcheck.
///
/// Interprets an `n`-bit boolean input vector as an `n`-bit two's-complement signed integer.
/// The most-significant variable (index 0 in the `r` slice, highest bit) is the sign bit.
///
/// `Self(r) = sum_{i=0}^{n-1} r[n-1-i] * 2^i - r[0] * 2^n`
///
/// Used in the RAF sumcheck as the identity factor:
/// `sum_k ra(k, j) * Self(k)` folds to a claim on the signed integer value at the sumcheck point.
///
/// Unlike [`BinarySignedIdentityPoly`], this struct is not parameterized by an `X_LEN` constant
/// and does not assume an interleaved / two-operand key layout. It is used directly in
/// single-operand activation proofs (tanh, erf, sigmoid) where the lookup index is the
/// quotient of the input divided by the fixed-point scale.
#[derive(Clone, Debug, Allocative)]
pub struct SignedIdentityPoly<F: JoltField> {
    num_vars: usize,
    num_bound_vars: usize,
    bound_value: F,
}

impl<F: JoltField> SignedIdentityPoly<F> {
    pub fn new(num_vars: usize) -> Self {
        SignedIdentityPoly {
            num_vars,
            num_bound_vars: 0,
            bound_value: F::zero(),
        }
    }

    fn sign_bit_pos(&self) -> usize {
        self.num_vars - 1
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for SignedIdentityPoly<F> {
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        let len = r.len();
        debug_assert_eq!(len, self.num_vars);

        let mut y = F::zero();
        r.iter().rev().enumerate().for_each(|(i, &r_i)| {
            let w = F::from_u64(i.pow2() as u64);
            y += w * r_i;
        });
        let sign_bit = r[0].into();
        let w = F::from_u128(1 << self.num_vars);
        y - sign_bit * w
    }

    fn batch_evaluate<C>(_polys: &[&Self], _r: &[C]) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!("Currently unused")
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        let mut evals = vec![F::zero(); degree];
        let m = match order {
            BindingOrder::LowToHigh => {
                // Weight of the variable being bound this round.
                let slope = F::from_u128(1 << self.num_bound_vars);
                // Two's-complement sign penalty: sign bit contributes -2^(n-1),
                // encoded here as (slope_at_sign_bit - 2^n).
                let sign_penalty = F::from_u64(self.num_vars.pow2() as u64);

                if self.sign_bit_pos() == self.num_bound_vars {
                    // Binding the sign bit: p(0) = bound_value (sign=0, no penalty),
                    // p(1) = bound_value + slope - sign_penalty.
                    evals[0] = self.bound_value;
                    slope - sign_penalty
                } else {
                    // Sign bit is still unbound; read it from `index` to apply
                    // the sign penalty to the base value before extrapolating.
                    let sign_bit_offset = self.sign_bit_pos() - self.num_bound_vars - 1;
                    let sign_bit = (index >> sign_bit_offset) & 1;
                    let base = self.bound_value - sign_penalty * F::from_u64(sign_bit as u64);

                    // Upper unbound bits collectively contribute 2^(num_bound_vars+1) * index.
                    evals[0] = base + slope.mul_u64(2 * index as u64);
                    slope
                }
            }
            BindingOrder::HighToLow => {
                unimplemented!("Currently unused")
            }
        };

        let mut eval = evals[0] + m;
        for i in 1..degree {
            eval += m;
            evals[i] = eval;
        }
        evals
    }
}

impl<F: JoltField> PolynomialBinding<F> for SignedIdentityPoly<F> {
    fn is_bound(&self) -> bool {
        self.num_bound_vars != 0
    }

    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        debug_assert!(self.num_bound_vars < self.num_vars);

        match order {
            BindingOrder::LowToHigh => {
                if self.num_bound_vars == self.sign_bit_pos() {
                    self.bound_value -= F::from_u128(1 << self.num_vars) * r;
                }

                self.bound_value += F::from_u128(1 << self.num_bound_vars) * r;
            }
            BindingOrder::HighToLow => {
                unimplemented!("Currently unused")
            }
        }
        self.num_bound_vars += 1;
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        // Binding is constant time, no parallelism necessary
        self.bind(r, order);
    }

    fn final_claim(&self) -> F {
        debug_assert_eq!(self.num_vars, self.num_bound_vars);
        self.bound_value
    }
}

impl<F: JoltField> PrefixSuffixPolynomial<F, 2> for SignedIdentityPoly<F> {
    fn suffixes(&self) -> [Box<dyn SuffixPolynomial<F> + Sync>; 2] {
        [
            Box::new(OneSuffixPolynomial),
            Box::new(
                IdentityPolynomial::<F>::new(0), /* num-vars is not used for suffix impl for identity poly */
            ),
        ]
    }

    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        registry: &mut PrefixRegistry<F>,
    ) -> [Option<Arc<RwLock<CachedPolynomial<F>>>>; 2] {
        let identity = registry.get_or_insert(Prefix::SignedIdentity, |cp| {
            self.clone().prefix_polynomial(cp, chunk_len, phase)
        });
        [Some(identity), None]
    }
}

pub struct OneSuffixPolynomial;
impl<F: JoltField> SuffixPolynomial<F> for OneSuffixPolynomial {
    fn suffix_mle(&self, _b: LookupBits) -> u64 {
        1
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
                    let res =
                        bound_value + F::from_u128(((i << suffix_len) as u128) % (1u128 << xlen));
                    res + sign_correction::<F>(sign_bit, xlen)
                })
                .collect(),
            PhasePosition::AfterMsb => (0..chunk_len.pow2())
                .map(|i| bound_value + F::from_u128((i << suffix_len) as u128))
                .collect(),
        };
        make_cached_poly(evals, chunk_len)
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
    /// Left Operand Case: Self(x) = (sum_{i=0}^{X_LEN-1} x[2*i] * 2^(X_LEN - 1 - i)) - x[0] * 2^X_LEN
    /// Right Operand Case: Self(x) = (sum_{i=0}^{X_LEN-1} x[2*i+1] * 2^(X_LEN - 1 - i)) - x[1] * 2^X_LEN
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

    fn batch_evaluate<C>(_: &[&Self], _: &[C]) -> Vec<F>
    where
        Self: Sized,
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!("Currently unused")
    }

    fn sumcheck_evals(&self, _: usize, _: usize, _: BindingOrder) -> Vec<F> {
        unimplemented!("Currently unused")
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

/// Wraps evals into a `CachedPolynomial` with standard cache capacity.
fn make_cached_poly<F: JoltField>(evals: Vec<F>, chunk_len: usize) -> CachedPolynomial<F> {
    CachedPolynomial::new(MultilinearPolynomial::from(evals), (chunk_len - 1).pow2())
}

/// Returns `-sign_bit * 2^X_LEN` as a field element.
fn sign_correction<F: JoltField>(sign_bit: u32, xlen: usize) -> F {
    -F::from_u32(sign_bit) * F::from_u128(1 << xlen)
}

/// Classifies the current phase relative to the MSB boundary.
#[allow(clippy::enum_variant_names)]
enum PhasePosition {
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

#[cfg(test)]
mod tests {
    use crate::{
        field::JoltField,
        poly::{
            identity_poly::OperandSide,
            multilinear_polynomial::{
                BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
            },
            prefix_suffix::{tests::prefix_suffix_decomposition_test, Prefix},
            signed_identity_poly::{SignedIdentityPoly, SignedOperandPoly},
        },
        utils::{index_to_field_bitvector, interleave_bits},
    };
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    // --- SignedIdentityPoly tests ---

    #[test]
    fn test_signed_identity_poly_evaluation() {
        for i in 0..256usize {
            let i = i as i8;
            let bit_vector: Vec<Fr> = index_to_field_bitvector(i as u8 as u64, 8);
            assert_eq!(
                Fr::from_i8(i),
                SignedIdentityPoly::<Fr>::new(8).evaluate(&bit_vector),
                "bit_vector: {bit_vector:?}, i: {i}"
            );
        }
    }

    #[test]
    fn signed_identity_poly_sumcheck() {
        const NUM_VARS: usize = 8;
        const DEGREE: usize = 3;

        let mut rng = test_rng();
        let mut identity_poly: SignedIdentityPoly<Fr> = SignedIdentityPoly::new(NUM_VARS);
        let mut reference_poly: MultilinearPolynomial<Fr> = MultilinearPolynomial::from(
            (0..(1 << NUM_VARS))
                .map(|i: u32| Fr::from_i8(i as u8 as i8))
                .collect::<Vec<_>>(),
        );

        for j in 0..reference_poly.len() / 2 {
            let identity_poly_evals =
                identity_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
            let reference_poly_evals =
                reference_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
            assert_eq!(identity_poly_evals, reference_poly_evals, "j: {j:04b}");
        }

        for round in 0..NUM_VARS {
            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            identity_poly.bind(r, BindingOrder::LowToHigh);
            reference_poly.bind(r, BindingOrder::LowToHigh);
            for j in 0..reference_poly.len() / 2 {
                let identity_poly_evals =
                    identity_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                let reference_poly_evals =
                    reference_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                assert_eq!(
                    identity_poly_evals,
                    reference_poly_evals,
                    "j: {j:04b}, round: {}",
                    round + 1
                );
            }
        }

        assert_eq!(identity_poly.final_claim(), reference_poly.final_claim());
    }

    #[test]
    fn signed_identity_poly_prefix_suffix_decomposition() {
        prefix_suffix_decomposition_test::<16, 2, 2, true, _>(
            SignedIdentityPoly::<Fr>::new(16),
            vec![Prefix::SignedIdentity],
        );
    }

    // --- SignedOperandPoly tests ---

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
    fn test_signed_operand_poly_random_evaluations() {
        use crate::utils::uninterleave_bits;

        let mut rng = StdRng::seed_from_u64(404);
        for _ in 0..1000 {
            let i = rng.gen();
            let bit_vector: Vec<Fr> = index_to_field_bitvector(i, 64);
            let (left, right) = uninterleave_bits(i);
            assert_eq!(
                Fr::from_i32(left as i32),
                SignedOperandPoly::<Fr, 32>::new(OperandSide::Left).evaluate(&bit_vector)
            );
            assert_eq!(
                Fr::from_i32(right as i32),
                SignedOperandPoly::<Fr, 32>::new(OperandSide::Right).evaluate(&bit_vector)
            );
        }
    }

    #[test]
    fn signed_operand_poly_prefix_suffix_decomposition() {
        prefix_suffix_decomposition_test::<16, 2, 3, true, _>(
            SignedOperandPoly::<Fr, 8>::new(OperandSide::Left),
            vec![Prefix::LeftOperand, Prefix::LeftOperandMSB],
        );
        prefix_suffix_decomposition_test::<16, 2, 3, true, _>(
            SignedOperandPoly::<Fr, 8>::new(OperandSide::Right),
            vec![Prefix::RightOperand, Prefix::RightOperandMSB],
        );
    }
}
