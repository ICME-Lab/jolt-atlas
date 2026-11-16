use super::multilinear::{BindingOrder, PolynomialBinding, PolynomialEvaluation};
use crate::{
    field::JoltField,
    poly::{
        identity::IdentityPolynomial,
        multilinear::MultilinearPolynomial,
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
/// The address is one operand's two's-complement value: the lookup index of the prefix-suffix
/// read-rafs (ReLU, the clamps, the trig downscale).
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

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        let mut evals = vec![F::zero(); degree];
        let m = match order {
            BindingOrder::LowToHigh => {
                // Weight of the variable being bound this round.
                let slope = F::from_u128(1 << self.num_bound_vars);
                // Two's-complement sign penalty: sign bit contributes -2^(n-1),
                // encoded here as (slope_at_sign_bit - 2^n). Use a u128 shift so
                // this stays correct for `num_vars >= 64` (64-bit clamp lookups),
                // matching `evaluate`'s `from_u128(1 << num_vars)`.
                let sign_penalty = F::from_u128(1u128 << self.num_vars);

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
                        bound_value + F::from_u128(((i as u128) << suffix_len) % (1u128 << xlen));
                    res + sign_correction::<F>(sign_bit, xlen)
                })
                .collect(),
            PhasePosition::AfterMsb => (0..chunk_len.pow2())
                .map(|i| bound_value + F::from_u128((i as u128) << suffix_len))
                .collect(),
        };
        make_cached_poly(evals, chunk_len)
    }
}

// -------- Signed Operand Poly --------

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
    use crate::field::fp128::Fp128 as Fr;
    use crate::field::Field as _;
    use crate::field::Ring;
    use crate::utils::test_rng;
    use crate::{
        field::JoltField,
        poly::{
            multilinear::{
                BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
            },
            prefix_suffix::{tests::prefix_suffix_decomposition_test, Prefix},
            signed_identity::SignedIdentityPoly,
        },
        utils::index_to_field_bitvector,
    };

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
}
