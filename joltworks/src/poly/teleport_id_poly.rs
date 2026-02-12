use allocative::Allocative;

use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};

use super::multilinear_polynomial::{BindingOrder, PolynomialBinding, PolynomialEvaluation};
const X_LEN: usize = 32;

/// This polynomial encodes the function f, that maps a n-bits 2's complement integer to a 32-bits signed integer, then casts it to an unsigned.
/// Example:
/// ```text
/// n = 4:
/// input (binary)  | input (decimal) | n-bit 2's comp | output (binary) | output (decimal)
/// 0000            | 0               | 0              | 0000...0000     | 0
/// 0001            | 1               | 1              | 0000...0001     | 1
/// 1000            | 8               | -8             | 1111...1000     | 4294967288 (2^32 - 8)
/// 1111            | 15              | -1             | 1111...1111     | 4294967295 (2^32 - 1)
/// ```
#[derive(Clone, Debug, Allocative)]
pub struct TeleportIdPolynomial<F: JoltField> {
    num_vars: usize,
    num_bound_vars: usize,
    bound_value: F,
}

impl<F: JoltField> TeleportIdPolynomial<F> {
    pub fn new(num_vars: usize) -> Self {
        TeleportIdPolynomial {
            num_vars,
            num_bound_vars: 0,
            bound_value: F::zero(),
        }
    }

    // returns the bit being treated at current round of binding
    pub fn current_bit(&self, order: BindingOrder) -> usize {
        match order {
            BindingOrder::LowToHigh => self.num_bound_vars,
            BindingOrder::HighToLow => self.num_vars - 1 - self.num_bound_vars,
        }
    }

    /// Returns the index of the sign bit, which encodes whether the value is negative in 2's complement form.
    pub fn sign_bit(&self) -> usize {
        self.num_vars - 1
    }
}

impl<F: JoltField> PolynomialBinding<F> for TeleportIdPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_bound_vars != 0
    }

    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        debug_assert!(self.num_bound_vars < self.num_vars);

        let current_bit = self.current_bit(order);

        match order {
            BindingOrder::LowToHigh => {
                if current_bit < self.sign_bit() {
                    self.bound_value += F::from_u128(1 << current_bit) * r
                } else {
                    // sign_bit encodes whether the value is negative, in 2's complement form.
                    // Hence when sign_bit is 1, we first substract the corresponding value (2^sign_bit_idx),
                    // then add 2^32 to get u32 2's complement representation.
                    self.bound_value += F::from_i64((1 << X_LEN) - (1 << self.sign_bit())) * r;
                }
            }
            BindingOrder::HighToLow => {
                unimplemented!("HighToLow binding is not yet needed, hence not implemented");
                // self.bound_value += self.bound_value;
                // self.bound_value = self.bound_value + r;
            }
        }
        self.num_bound_vars += 1;
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        // Binding is constant time, no parallelism necessary
        self.bind(r, order);
    }

    fn final_sumcheck_claim(&self) -> F {
        debug_assert_eq!(self.num_vars, self.num_bound_vars);
        self.bound_value
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for TeleportIdPolynomial<F> {
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        let len = r.len();
        debug_assert_eq!(len, self.num_vars);

        let mut eval: F = (1..=self.sign_bit())
            .map(|i| r[i].into().mul_u128(1 << (len - 1 - i)))
            .sum();
        eval += r[0].into().mul_i128((1 << X_LEN) - (1 << self.sign_bit()));
        eval
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
                let current_bit = self.current_bit(order);
                if current_bit < self.sign_bit() {
                    // Recover position of sign bit in index: which is offset by the number of bound variables + the currently treated bit.
                    let sign_bit_in_idx = self.sign_bit() - self.num_bound_vars - 1;
                    // Recover sign from index value: the sign bit is the bit at position `sign_bit_in_idx` in `index`.
                    let sign = index >> sign_bit_in_idx & 1;
                    let unsigned_index = index & !(1 << sign_bit_in_idx);
                    let m = F::from_u128(1 << self.num_bound_vars);
                    evals[0] = self.bound_value
                        + (m + m).mul_u64(unsigned_index as u64)
                        + F::from_u128(sign as u128 * ((1 << X_LEN) - (1 << self.sign_bit())));
                    m
                } else {
                    let m = F::from_u128((1 << X_LEN) - (1 << self.sign_bit()));
                    evals[0] = self.bound_value;
                    m
                }
            }
            BindingOrder::HighToLow => {
                unimplemented!("HighToLow binding is not yet needed, hence not implemented");
                // let m = F::from_u128(1 << (self.num_vars - 1 - self.num_bound_vars));
                // evals[0] = self.bound_value * (m + m) + F::from_u64(index as u64);
                // m
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

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::test_rng;

    use crate::{
        field::JoltField,
        poly::{
            multilinear_polynomial::{
                BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
            },
            teleport_id_poly::TeleportIdPolynomial,
        },
        utils::index_to_field_bitvector,
    };

    #[test]
    fn identity_poly() {
        const NUM_VARS: usize = 4;
        const DEGREE: usize = 3;

        let mut rng = test_rng();

        let mut reference_vec: Vec<i32> = Vec::new();
        reference_vec.extend(0..1 << (NUM_VARS - 1));
        reference_vec.extend(-1 << (NUM_VARS - 1)..0);
        println!("reference_vec: {reference_vec:?}");
        // Needed for compatibility: we first convert to unsigned form (i.e. 2's complement representation), then convert to field elements.
        let reference_vec: Vec<u32> = reference_vec.into_iter().map(|x| x as u32).collect();
        let mut identity_poly: TeleportIdPolynomial<Fr> = TeleportIdPolynomial::new(NUM_VARS);
        let mut reference_poly: MultilinearPolynomial<Fr> =
            MultilinearPolynomial::from(reference_vec);

        let input_vec: Vec<usize> = (0..(1 << NUM_VARS)).collect();
        let eval_points: Vec<_> = input_vec
            .iter()
            .map(|&i| index_to_field_bitvector::<Fr>(i as u64, NUM_VARS))
            .collect();
        for eval_point in eval_points.iter() {
            let eval = identity_poly.evaluate(eval_point);
            let expected_eval = reference_poly.evaluate(eval_point);
            assert_eq!(
                eval, expected_eval,
                "Evaluation mismatch for input {eval_point:?}"
            );
        }

        for j in 0..reference_poly.len() / 2 {
            let identity_poly_evals =
                identity_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
            let reference_poly_evals =
                reference_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
            assert_eq!(
                identity_poly_evals, reference_poly_evals,
                "mismatch at variable {j:04b}"
            );
        }

        for bound_var in 0..NUM_VARS {
            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            identity_poly.bind(r, BindingOrder::LowToHigh);
            reference_poly.bind(r, BindingOrder::LowToHigh);
            for j in 0..reference_poly.len() / 2 {
                let identity_poly_evals =
                    identity_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                let reference_poly_evals =
                    reference_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                assert_eq!(
                    identity_poly_evals, reference_poly_evals,
                    "mismatch at variable {j:04b} after binding variable {bound_var}"
                );
            }
        }

        assert_eq!(
            identity_poly.final_sumcheck_claim(),
            reference_poly.final_sumcheck_claim()
        );
    }
}
