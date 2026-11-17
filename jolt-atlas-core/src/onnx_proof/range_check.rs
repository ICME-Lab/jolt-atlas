//! The prover side of the packed range checks
//! ([`jolt_atlas_verifier::onnx_proof::range_check`] states the protocol): the value sumcheck
//! over a bucket's packed values ([`ValueProver`]), which opens the bucket's dense
//! base-[`DIGIT_BASE`] digit columns ([`digit_columns`]).
use crate::onnx_proof::relation::{low_to_high_point, RelationSetup, RelationWitness};
use atlas_algebra::poly::univariate::UnivariatePoly;
use atlas_algebra::{
    field::JoltField,
    poly::eq::EqPolynomial,
    poly::multilinear::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
};
use atlas_claims::SymbolicSumcheck;
use atlas_sumcheck::engine::{prover::ProveRounds, SumcheckError};
use common::parallel::par_enabled;
pub use jolt_atlas_verifier::onnx_proof::range_check::*;
use rayon::prelude::*;

/// Prover of the value sumcheck.
pub struct ValueProver<F: JoltField> {
    params: ValueParams<F>,
    setup: RelationSetup<F>,
    w: MultilinearPolynomial<F>,
    value: MultilinearPolynomial<F>,
    /// The digit columns the packed values recompose, most significant first.
    digits: Vec<Vec<u8>>,
    /// The challenges bound so far, first round first.
    challenges: Vec<F>,
}

impl<F: JoltField> ValueProver<F> {
    /// Build from resolved parameters, the packed values and their digit columns.
    pub fn new(params: ValueParams<F>, values: Vec<u64>, digits: Vec<Vec<u8>>) -> Self {
        let w = MultilinearPolynomial::from(params.cycle.evals());
        Self {
            setup: params.relation_setup(),
            params,
            w,
            value: MultilinearPolynomial::from(values),
            digits,
            challenges: Vec::new(),
        }
    }
}

impl<F: JoltField> RelationWitness<F> for ValueProver<F> {
    type Relation = ValueParams<F>;

    fn relation(&self) -> &ValueParams<F> {
        &self.params
    }

    fn setup(&self) -> &RelationSetup<F> {
        &self.setup
    }

    /// Every digit column at the bound point. The rounds run on the packed values; the
    /// digits are read once, here, against the `eq` table of the point.
    fn produced_claims(&self) -> Vec<F> {
        let point = low_to_high_point(&self.challenges);
        let weights = EqPolynomial::<F>::evals(&point);
        self.digits
            .par_iter()
            .map(|column| {
                // A digit is below the base: sum the weights per digit value, then scale.
                let mut by_digit = [F::zero(); DIGIT_BASE as usize];
                for (&digit, &weight) in column.iter().zip(&weights) {
                    by_digit[digit as usize] += weight;
                }
                by_digit
                    .iter()
                    .enumerate()
                    .skip(1)
                    .fold(F::zero(), |acc, (digit, &sum)| {
                        acc + sum.mul_u64(digit as u64)
                    })
            })
            .collect()
    }
}

impl<F: JoltField> ValueProver<F> {
    fn round_message(&mut self, _round: usize, previous_claim: F) -> UnivariatePoly<F> {
        let half = self.w.len() / 2;
        let [e0, e2] = (0..half)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|g| {
                let w0 = self.w.get_bound_coeff(2 * g);
                let w1 = self.w.get_bound_coeff(2 * g + 1);
                let v0 = self.value.get_bound_coeff(2 * g);
                let v1 = self.value.get_bound_coeff(2 * g + 1);
                [w0 * v0, (w1 + w1 - w0) * (v1 + v1 - v0)]
            })
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
        UnivariatePoly::from_evals_and_hint(previous_claim, &[e0, e2])
    }

    fn bind(&mut self, r: F, _round: usize) {
        let r_j = F::challenge_from_field(r);
        self.challenges.push(r);
        self.w.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.value.bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField> ProveRounds<F> for ValueProver<F> {
    fn num_rounds(&self) -> usize {
        self.params.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r, round - 1);
        }
        Ok(self.round_message(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind, self.params.rounds() - 1);
        Ok(())
    }
}

/// `values` in base [`DIGIT_BASE`]: `n` digit columns, most significant first.
pub fn digit_columns<T: Copy + Into<u64> + Sync>(values: &[T], n: usize) -> Vec<Vec<u8>> {
    let digit_bits = DIGIT_BASE.ilog2();
    (0..n)
        .map(|j| {
            let shift = (n - 1 - j) as u32 * digit_bits;
            values
                .par_iter()
                .with_min_len(par_enabled())
                .map(|&v| ((v.into() >> shift) % DIGIT_BASE) as u8)
                .collect()
        })
        .collect()
}

/// Test-only tamper of one bucket's digits, for the witness tamper sweep's `BucketDigit` sites:
/// the trace holds no digits, so a trace tamper cannot reach them. The sweep sets the tamper
/// before proving and clears it after; the bucket's witnesses read it in
/// [`BucketWitnesses`](crate::onnx_proof::range_bucket::BucketWitnesses).
#[cfg(test)]
pub(crate) mod digit_tamper {
    use super::DIGIT_BASE;
    use std::sync::Mutex;

    /// How the digits of one cycle are tampered.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum DigitTamperForm {
        /// `d_j + b` and the next more significant digit `− 1`, at the most significant `j`
        /// whose more significant neighbour is nonzero, on the first cycle from the tamper's
        /// that has one: the value is unchanged and one digit is out of range.
        KeepValue,
        /// The most significant digit `+ b`: the digits recompose the value plus `b^n`, and the
        /// value relation reads that value.
        RaiseTop,
    }

    /// A digit tamper of `bucket` at `cycle`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct DigitTamper {
        pub bucket: usize,
        pub cycle: usize,
        pub form: DigitTamperForm,
    }

    static ACTIVE: Mutex<Option<DigitTamper>> = Mutex::new(None);

    /// Set (or, with `None`, clear) the tamper the next proof applies.
    pub(crate) fn set(tamper: Option<DigitTamper>) {
        *ACTIVE.lock().unwrap() = tamper;
    }

    /// The active tamper's cycle and form, if it names bucket `bucket`.
    pub(crate) fn for_bucket(bucket: usize) -> Option<(usize, DigitTamperForm)> {
        ACTIVE
            .lock()
            .unwrap()
            .filter(|t| t.bucket == bucket)
            .map(|t| (t.cycle, t.form))
    }

    /// Apply `form` to the digit columns at `cycle`.
    pub(crate) fn apply(digits: &mut [Vec<u8>], cycle: usize, form: DigitTamperForm) {
        let base = DIGIT_BASE as u8;
        match form {
            DigitTamperForm::KeepValue => {
                // The first cycle from `cycle` on with a nonzero digit above the least
                // significant.
                let movable = (cycle..digits[0].len()).find_map(|t| {
                    (1..digits.len())
                        .find(|&j| digits[j - 1][t] >= 1)
                        .map(|j| (t, j))
                });
                if let Some((t, j)) = movable {
                    digits[j][t] += base;
                    digits[j - 1][t] -= 1;
                }
            }
            DigitTamperForm::RaiseTop => digits[0][cycle] += base,
        }
    }
}
