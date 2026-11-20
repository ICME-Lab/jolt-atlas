//! Reed–Solomon codes over `F_{2^128}` in the novel polynomial basis of Lin, Chung and Han.
//!
//! A message `c ∈ F^{2^k}` is the polynomial `f = Σ_i c_i · X_i`, with `X_i = Π_j Ŵ_j^{i_j}` for
//! the bits `i_j` of `i`. `W_j` is the polynomial vanishing on the span of `β_0, …, β_{j−1}`
//! (`β_j = x^j`) and `Ŵ_j = W_j / W_j(β_j)`; each is `F_2`-linear. The codeword is `f` on the
//! `2^{k+c}` points `ω_a = Σ_j a_j β_j`, where `2^{-c}` is the rate.
//!
//! Row `a` of the generator matrix is the tensor `⊗_j (1, Ŵ_j(ω_a))`, which is what lets a
//! verifier evaluate a codeword symbol of a folded message without the message.

use super::gf128::Gf128;
use rayon::prelude::*;

/// A code with messages of `2^log_message` symbols and codewords of
/// `2^(log_message + log_inv_rate)`.
#[derive(Clone, Debug)]
pub struct Code {
    log_message: usize,
    log_inv_rate: usize,
    /// `basis_values[j][i] = Ŵ_j(β_i)` for `j < log_message`, `i < log_message + log_inv_rate`.
    basis_values: Vec<Vec<Gf128>>,
}

impl Code {
    pub fn new(log_message: usize, log_inv_rate: usize) -> Self {
        let log_block = log_message + log_inv_rate;
        assert!(
            log_block <= 64,
            "the evaluation domain is indexed by a `usize`"
        );
        // `W_0(x) = x`, and `W_{j+1}(x) = W_j(x) · (W_j(x) + W_j(β_j))`.
        let mut w: Vec<Gf128> = (0..log_block).map(|i| Gf128(1u128 << i)).collect();
        let mut basis_values = Vec::with_capacity(log_message);
        for j in 0..log_message {
            let pivot = w[j];
            #[allow(clippy::expect_used)]
            let pivot_inverse = pivot
                .inverse()
                .expect("β_j is outside the span W_j vanishes on");
            basis_values.push(w.iter().map(|&value| value * pivot_inverse).collect());
            for value in &mut w {
                *value = *value * (*value + pivot);
            }
        }
        Self {
            log_message,
            log_inv_rate,
            basis_values,
        }
    }

    pub fn log_message(&self) -> usize {
        self.log_message
    }

    pub fn log_block(&self) -> usize {
        self.log_message + self.log_inv_rate
    }

    /// `Ŵ_j(ω_a)`, by linearity over the set bits of `a`.
    fn twiddle(&self, j: usize, a: usize) -> Gf128 {
        let mut acc = Gf128::ZERO;
        let mut bits = a;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            acc += self.basis_values[j][i];
            bits &= bits - 1;
        }
        acc
    }

    /// The factors `Ŵ_j(ω_a)`, `j < log_message`, of generator row `a`: its entry at message
    /// index `i` is the product of the factors at the set bits of `i`.
    pub fn row_factors(&self, a: usize) -> Vec<Gf128> {
        (0..self.log_message).map(|j| self.twiddle(j, a)).collect()
    }

    /// Encodes every column of a row-major matrix with `2^log_message` rows of `width` symbols,
    /// returning the row-major matrix of `2^log_block` rows.
    pub fn encode_columns(&self, message: &[Gf128], width: usize) -> Vec<Gf128> {
        let rows = 1usize << self.log_message;
        assert_eq!(message.len(), rows * width, "message matrix shape");
        let block = message.len() << self.log_inv_rate;
        // Each coset of the message span starts from the message itself: the basis polynomials
        // above `log_message` have zero coefficients.
        let mut code = Vec::with_capacity(block);
        for _ in 0..1usize << self.log_inv_rate {
            code.extend_from_slice(message);
        }
        for j in (0..self.log_message).rev() {
            let half = (1usize << j) * width;
            code.par_chunks_mut(2 * half)
                .enumerate()
                .for_each(|(b, chunk)| {
                    let twiddle = self.twiddle(j, b << (j + 1));
                    let (low, high) = chunk.split_at_mut(half);
                    for (u, v) in low.iter_mut().zip(high.iter_mut()) {
                        *u += twiddle * *v;
                        *v += *u;
                    }
                });
        }
        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    fn random(rng: &mut StdRng) -> Gf128 {
        Gf128(((rng.next_u64() as u128) << 64) | rng.next_u64() as u128)
    }

    /// `Ŵ_j(point)` from the definition: the product of `point − u` over the span of
    /// `β_0, …, β_{j−1}`, divided by the same product at `β_j`.
    fn basis_polynomial(j: usize, point: Gf128) -> Gf128 {
        let vanishing = |at: Gf128| {
            (0..1u128 << j)
                .map(|u| at + Gf128(u))
                .fold(Gf128::ONE, |acc, factor| acc * factor)
        };
        vanishing(point) * vanishing(Gf128(1 << j)).inverse().unwrap()
    }

    #[test]
    fn row_factors_are_the_normalized_subspace_polynomials() {
        let code = Code::new(5, 2);
        for a in [0usize, 1, 2, 37, 127] {
            let factors = code.row_factors(a);
            for (j, factor) in factors.iter().enumerate() {
                assert_eq!(
                    *factor,
                    basis_polynomial(j, Gf128(a as u128)),
                    "j={j} a={a}"
                );
            }
        }
    }

    #[test]
    fn encoding_evaluates_the_message_polynomial_on_the_domain() {
        let mut rng = StdRng::seed_from_u64(7);
        for (log_message, log_inv_rate, width) in [(0, 1, 1), (1, 1, 2), (4, 1, 3), (5, 2, 1)] {
            let code = Code::new(log_message, log_inv_rate);
            let rows = 1usize << log_message;
            let message: Vec<Gf128> = (0..rows * width).map(|_| random(&mut rng)).collect();
            let encoded = code.encode_columns(&message, width);
            assert_eq!(encoded.len(), width << code.log_block());
            for a in 0..1usize << code.log_block() {
                let factors = code.row_factors(a);
                for column in 0..width {
                    let expected = (0..rows).fold(Gf128::ZERO, |acc, i| {
                        let entry = (0..log_message)
                            .filter(|j| (i >> j) & 1 == 1)
                            .fold(Gf128::ONE, |product, j| product * factors[j]);
                        acc + entry * message[i * width + column]
                    });
                    assert_eq!(
                        encoded[a * width + column],
                        expected,
                        "a={a} column={column}"
                    );
                }
            }
        }
    }

    #[test]
    fn distinct_messages_differ_in_more_than_half_the_block() {
        let mut rng = StdRng::seed_from_u64(9);
        let code = Code::new(6, 1);
        let a: Vec<Gf128> = (0..64).map(|_| random(&mut rng)).collect();
        let mut b = a.clone();
        b[17] += Gf128::ONE;
        let (ca, cb) = (code.encode_columns(&a, 1), code.encode_columns(&b, 1));
        let differing = ca.iter().zip(&cb).filter(|(x, y)| x != y).count();
        // A nonzero polynomial of degree below 64 has at most 63 roots among 128 points.
        assert!(differing >= 128 - 63);
    }
}
