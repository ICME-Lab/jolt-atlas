use allocative::Allocative;

use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};

use super::multilinear_polynomial::{BindingOrder, PolynomialBinding, PolynomialEvaluation};

/// The multilinear polynomial selecting a one-hot chunk's live addresses.
///
/// An address is dead when it sets one of the chunk's top `dead_bits` bits. The selector
/// is the product of `1 − k_b` over those bits: a table of `scale` on its first
/// `2^live_vars` entries and 0 on the others. It keeps that shape under binding, so no
/// table is stored.
#[derive(Clone, Debug, Allocative)]
pub struct AddressSelector<F: JoltField> {
    /// Unbound variables the selector doesn't depend on.
    live_vars: usize,
    /// Unbound variables among the dead bits.
    dead_vars: usize,
    /// The product of `1 − r` over the dead bits bound so far.
    scale: F,
    is_bound: bool,
}

impl<F: JoltField> AddressSelector<F> {
    pub fn new(log_k_chunk: usize, dead_bits: usize) -> Self {
        Self {
            live_vars: log_k_chunk - dead_bits,
            dead_vars: dead_bits,
            scale: F::one(),
            is_bound: false,
        }
    }

    /// The table entry at `index`, over the unbound variables.
    fn entry(&self, index: usize) -> F {
        if index >> self.live_vars == 0 {
            self.scale
        } else {
            F::zero()
        }
    }

    fn bind_dead_var(&mut self, r: F::Challenge) {
        self.dead_vars -= 1;
        self.scale *= F::one() - r;
    }
}

impl<F: JoltField> PolynomialBinding<F> for AddressSelector<F> {
    fn is_bound(&self) -> bool {
        self.is_bound
    }

    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        match order {
            BindingOrder::LowToHigh if self.live_vars > 0 => self.live_vars -= 1,
            BindingOrder::LowToHigh => self.bind_dead_var(r),
            BindingOrder::HighToLow if self.dead_vars > 0 => self.bind_dead_var(r),
            BindingOrder::HighToLow => self.live_vars -= 1,
        }
        self.is_bound = true;
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        // Binding is constant time, no parallelism necessary
        self.bind(r, order);
    }

    fn final_claim(&self) -> F {
        debug_assert_eq!(self.live_vars + self.dead_vars, 0);
        self.scale
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for AddressSelector<F> {
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), self.live_vars + self.dead_vars);
        r[..self.dead_vars]
            .iter()
            .fold(self.scale, |acc, &r_i| acc * (F::one() - r_i.into()))
    }

    fn batch_evaluate<C>(_polys: &[&Self], _r: &[C]) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!("Currently unused")
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        let (at_0, at_1) = match order {
            BindingOrder::LowToHigh => (self.entry(2 * index), self.entry(2 * index + 1)),
            BindingOrder::HighToLow => {
                let half = 1 << (self.live_vars + self.dead_vars - 1);
                (self.entry(index), self.entry(index + half))
            }
        };
        let slope = at_1 - at_0;
        let mut evals = vec![at_0; degree];
        let mut eval = at_1;
        for value in evals.iter_mut().skip(1) {
            eval += slope;
            *value = eval;
        }
        evals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::multilinear_polynomial::MultilinearPolynomial,
        transcripts::{Blake2bTranscript, Transcript},
    };
    use ark_bn254::Fr;

    const LOG_K_CHUNK: usize = 4;
    const DEGREE: usize = 3;

    #[test]
    fn matches_its_table_in_every_bound_state() {
        let mut transcript = Blake2bTranscript::default();
        for order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            for dead_bits in 0..=LOG_K_CHUNK {
                let table: Vec<Fr> = (0..1usize << LOG_K_CHUNK)
                    .map(|k| Fr::from(u64::from(k >> (LOG_K_CHUNK - dead_bits) == 0)))
                    .collect();
                let mut expected = MultilinearPolynomial::from(table);
                let mut selector = AddressSelector::<Fr>::new(LOG_K_CHUNK, dead_bits);

                let point = transcript.challenge_vector_optimized::<Fr>(LOG_K_CHUNK);
                for &r in &point {
                    for g in 0..expected.len() / 2 {
                        assert_eq!(
                            selector.sumcheck_evals(g, DEGREE, order),
                            expected.sumcheck_evals(g, DEGREE, order),
                            "{order:?}, dead_bits = {dead_bits}, g = {g}",
                        );
                    }
                    selector.bind(r, order);
                    expected.bind(r, order);
                }
                assert_eq!(selector.final_claim(), expected.final_claim());

                let big_endian: Vec<_> = match order {
                    BindingOrder::LowToHigh => point.iter().rev().copied().collect(),
                    BindingOrder::HighToLow => point.clone(),
                };
                assert_eq!(
                    AddressSelector::<Fr>::new(LOG_K_CHUNK, dead_bits).evaluate(&big_endian),
                    expected.final_claim(),
                );
            }
        }
    }
}
