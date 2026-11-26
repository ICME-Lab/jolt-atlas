//! The degree-two sumcheck: from `sum_x W(x) V(x) = h_0` over two tables to
//! `MLE[V](rho) = v`, the verifier checking `MLE[W](rho) * v = h_n` with the
//! `MLE[W](rho)` its caller computes.
//!
//! Each round splits off the lowest remaining variable of both tables and
//! sends the round polynomial
//!
//! ```text
//! p(X) = sum_{x'} MLE[W](X, x') MLE[V](X, x') = a_0 + a_1 X + a_2 X^2.
//! ```
//!
//! In characteristic two `p(0) + p(1) = a_1 + a_2`, so the running claim
//! `h = p(0) + p(1)` fixes `a_1 = h + a_2` and only `(a_0, a_2)` is sent
//! ([`RoundMessage`]). Both sides continue with `h' = p(rho)` and the prover
//! folds its tables at `rho`. There is no per-round check: a wrong message
//! in any round surfaces in the closing check or the commitment opening.
//! Each challenge has error at most `2 / |F128|`, reduced by its grinding factor.
//! The opening authenticates the closing evaluation, including when its weight is zero.

use crate::{GRINDING_LABEL, VerifyError};
use common::SecurityLevel;
use field::{F128, Wide256};
#[cfg(feature = "parallel")]
use poly::parallel::workload_size;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use transcript::{ProverState, VerifierState};

/// `(a_0, a_2)` of `p(X) = a_0 + a_1 X + a_2 X^2`; `a_1` the running claim
/// implies.
pub(crate) type RoundMessage = [F128; 2];

/// Entries below which [`folded`] runs on one thread: the pool's overhead
/// is that of a few hundred thousand multiplications.
#[cfg(feature = "parallel")]
const PARALLEL_FOLD_MIN: usize = 1 << 18;

/// The public weights `W` and the prover's values `V` over the same
/// variables, folded together as the rounds bind them.
#[derive(Debug, Clone)]
pub(crate) struct Pair {
    weights: Vec<F128>,
    values: Vec<F128>,
}

impl Pair {
    /// Two tables of the same power-of-two length.
    pub(crate) fn new(weights: Vec<F128>, values: Vec<F128>) -> Self {
        debug_assert_eq!(weights.len(), values.len());
        debug_assert!(weights.len().is_power_of_two());
        Self { weights, values }
    }

    /// `(MLE[W](rho), MLE[V](rho))` once every variable is bound.
    fn bound(&self) -> (F128, F128) {
        debug_assert_eq!(self.weights.len(), 1);
        (self.weights[0], self.values[0])
    }

    /// `(a_0, a_2)`: `a_0 = sum w_0 v_0` and `a_2 = sum (w_0 + w_1)(v_0 + v_1)`
    /// over adjacent entries, since `MLE[W](X, x') = w_0 + X (w_0 + w_1)`
    /// and likewise for `MLE[V]`.
    fn message(&self) -> RoundMessage {
        message(&self.weights, &self.values)
    }

    fn fold(&mut self, challenge: F128) {
        fold(&mut self.weights, challenge);
        fold(&mut self.values, challenge);
    }
}

pub(super) fn message(weights: &[F128], values: &[F128]) -> RoundMessage {
    fn coeffs(weights: &[F128], values: &[F128]) -> (Wide256, Wide256) {
        // Large tables are split into cache-sized chunks summed on the Rayon pool.
        #[cfg(feature = "parallel")]
        {
            // An even chunk length keeps every pair inside one chunk.
            let chunk = workload_size::<F128>() & !1;
            if weights.len() > chunk {
                return weights
                    .par_chunks(chunk)
                    .zip(values.par_chunks(chunk))
                    .map(|(w, v)| coefficients_serial(w, v))
                    .reduce(
                        || (Wide256::zero(), Wide256::zero()),
                        |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                    );
            }
        }
        coefficients_serial(weights, values)
    }

    let (a0, a2) = coeffs(weights, values);
    [a0.reduce(), a2.reduce()]
}

/// Runs the sumcheck over `pair`, folding it in place, and writes the
/// closing evaluation. Returns the challenges in the order they were drawn
/// and `v = MLE[V](rho)`.
pub(crate) fn prove(
    pair: &mut Pair,
    claim: F128,
    security: SecurityLevel,
    transcript: &mut ProverState,
) -> (Vec<F128>, F128) {
    let rounds = pair.weights.len().trailing_zeros() as usize;
    let (point, running) = prove_rounds(pair, rounds, claim, security, transcript);
    let evaluation = prove_evaluation(pair, running, transcript);
    (point, evaluation)
}

/// Replays the sumcheck from the records and checks the closing evaluation
/// against `weight(rho) = MLE[W](rho)`. Returns the challenges and `v`.
pub(crate) fn verify(
    rounds: usize,
    claim: F128,
    weight: impl FnOnce(&[F128]) -> F128,
    security: SecurityLevel,
    transcript: &mut VerifierState<'_>,
) -> Result<(Vec<F128>, F128), VerifyError> {
    let (point, running) = verify_rounds(rounds, claim, security, transcript)?;
    let evaluation = verify_evaluation(weight(&point), running, transcript)?;
    Ok((point, evaluation))
}

/// Runs `rounds` rounds over `pair`, folding it in place. Returns the
/// challenges in the order they were drawn and the running claim.
pub(super) fn prove_rounds(
    pair: &mut Pair,
    rounds: usize,
    mut claim: F128,
    security: SecurityLevel,
    transcript: &mut ProverState,
) -> (Vec<F128>, F128) {
    let mut point = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let message = pair.message();
        transcript.prover_message(&message);
        transcript.grind(GRINDING_LABEL, security.grinding_bits(2));
        let challenge: F128 = transcript.verifier_message();
        claim = super::advance(claim, message, challenge);
        point.push(challenge);
        pair.fold(challenge);
    }
    (point, claim)
}

/// Replays `rounds` rounds from the records alone. Returns the challenges
/// and the running claim.
fn verify_rounds(
    rounds: usize,
    mut claim: F128,
    security: SecurityLevel,
    transcript: &mut VerifierState<'_>,
) -> Result<(Vec<F128>, F128), VerifyError> {
    let mut point = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let message: RoundMessage = transcript
            .prover_message()
            .map_err(|_| VerifyError::MalformedProof)?;
        transcript
            .grind(GRINDING_LABEL, security.grinding_bits(2))
            .map_err(|_| VerifyError::MalformedProof)?;
        let challenge: F128 = transcript.verifier_message();
        claim = super::advance(claim, message, challenge);
        point.push(challenge);
    }
    Ok((point, claim))
}

/// Writes `v = MLE[V](rho)`, the one entry left in the folded pair.
fn prove_evaluation(pair: &Pair, claim: F128, transcript: &mut ProverState) -> F128 {
    let (weight, evaluation) = pair.bound();
    debug_assert_eq!(weight * evaluation, claim);
    transcript.prover_message(&evaluation);
    evaluation
}

/// Reads `v` and checks `MLE[W](rho) * v = h` for the caller's `MLE[W](rho)`.
fn verify_evaluation(
    bound_weight: F128,
    claim: F128,
    transcript: &mut VerifierState<'_>,
) -> Result<F128, VerifyError> {
    let evaluation: F128 = transcript
        .prover_message()
        .map_err(|_| VerifyError::MalformedProof)?;
    if bound_weight * evaluation != claim {
        return Err(VerifyError::EvaluationMismatch);
    }
    Ok(evaluation)
}

/// Fixes the lowest remaining variable of `table` at `challenge`.
pub(crate) fn fold(table: &mut Vec<F128>, challenge: F128) {
    *table = folded(table, challenge)
}

/// [`fold`] into a fresh table, for a table that is only borrowed.
pub(crate) fn folded(table: &[F128], challenge: F128) -> Vec<F128> {
    let entry = |pair: &[F128]| pair[0] + challenge * (pair[0] + pair[1]);
    #[cfg(feature = "parallel")]
    if table.len() >= PARALLEL_FOLD_MIN {
        return table.par_chunks_exact(2).map(entry).collect();
    }
    table.chunks_exact(2).map(entry).collect()
}

pub(super) fn folded_with_scale(table: &[F128], challenge: F128, scale: F128) -> Vec<F128> {
    let entry = |pair: &[F128]| scale * (pair[0] + challenge * (pair[0] + pair[1]));
    #[cfg(feature = "parallel")]
    if table.len() >= PARALLEL_FOLD_MIN {
        return table.par_chunks_exact(2).map(entry).collect();
    }
    table.chunks_exact(2).map(entry).collect()
}

fn coefficients_serial(weights: &[F128], values: &[F128]) -> (Wide256, Wide256) {
    let mut a0 = Wide256::zero();
    let mut a2 = Wide256::zero();
    for (w, v) in weights.chunks_exact(2).zip(values.chunks_exact(2)) {
        a0 += Wide256::mul(w[0], v[0]);
        a2 += Wide256::mul(w[0] + w[1], v[0] + v[1]);
    }
    (a0, a2)
}

#[cfg(test)]
mod tests {
    use common::SecurityLevel::Bits100;
    use num_traits::{ConstOne, ConstZero};
    use poly::DenseMultilinearExtension;
    use poly::eq_table;
    use poly::f128::{evaluate, inner_product};
    use transcript::{build_prover, build_verifier};

    use super::*;
    use crate::test_util::{random, random_elements, rng};

    fn pair(n: usize, rng: &mut rand_pcg::Pcg64) -> Pair {
        Pair::new(random_elements(rng, 1 << n), random_elements(rng, 1 << n))
    }

    /// `sum_x MLE[W](x, x') MLE[V](x, x')` at `x`, by interpolating each pair.
    fn round_polynomial(pair: &Pair, x: F128) -> F128 {
        let interpolate = |entries: &[F128]| entries[0] + x * (entries[0] + entries[1]);
        pair.weights
            .chunks_exact(2)
            .zip(pair.values.chunks_exact(2))
            .map(|(w, v)| interpolate(w) * interpolate(v))
            .sum()
    }

    #[test]
    fn the_message_is_the_round_polynomial() {
        let mut rng = rng(1);
        let pair = pair(5, &mut rng);
        let claim = inner_product(&pair.weights, &pair.values);
        let message = pair.message();
        assert_eq!(
            round_polynomial(&pair, F128::ZERO) + round_polynomial(&pair, F128::ONE),
            claim
        );
        assert_eq!(message[0], round_polynomial(&pair, F128::ZERO));
        let rho = random(&mut rng);
        assert_eq!(
            crate::advance(claim, message, rho),
            round_polynomial(&pair, rho)
        );
    }

    #[test]
    fn an_honest_run_closes_and_a_false_claim_does_not() {
        let mut rng = rng(3);
        let mut pair = pair(6, &mut rng);
        let (weights, values) = (pair.weights.clone(), pair.values.clone());
        let claim = inner_product(&weights, &values);

        let mut prover = build_prover("post_gkr-tests", "sumcheck");
        let (point, running) = prove_rounds(&mut pair, 6, claim, Bits100, &mut prover);
        let evaluation = prove_evaluation(&pair, running, &mut prover);
        let proof = prover.finish();
        assert_eq!(proof.narg_string.len(), (2 * 6 + 1) * 16);

        let extension = |table: &[F128]| {
            DenseMultilinearExtension {
                evaluations: table.to_vec(),
            }
            .evaluate(&point)
            .unwrap()
        };
        assert_eq!(evaluation, extension(&values));
        assert_eq!(evaluate(&weights, &point), extension(&weights));

        let mut verifier = build_verifier("post_gkr-tests", "sumcheck", &proof);
        let (same_point, same_running) = verify_rounds(6, claim, Bits100, &mut verifier).unwrap();
        assert_eq!(same_point, point);
        assert_eq!(same_running, running);
        assert_eq!(
            verify_evaluation(evaluate(&weights, &point), running, &mut verifier),
            Ok(evaluation)
        );
        assert!(verifier.check_eof().is_ok());

        // The same records against a claim one off: the gap survives every
        // round and the closing check catches it.
        let mut verifier = build_verifier("post_gkr-tests", "sumcheck", &proof);
        let (_, running) = verify_rounds(6, claim + F128::ONE, Bits100, &mut verifier).unwrap();
        assert_eq!(
            verify_evaluation(evaluate(&weights, &point), running, &mut verifier),
            Err(VerifyError::EvaluationMismatch)
        );
    }

    #[test]
    fn the_evaluation_and_the_fold_agree_with_the_extension() {
        let mut rng = rng(9);
        for n in [3, 7, 9, 13] {
            let weights = random_elements(&mut rng, 1 << n);
            let point = random_elements(&mut rng, n);
            assert_eq!(
                evaluate(&weights, &point),
                inner_product(&weights, &eq_table(&point)),
                "{n} variables"
            );
            let mut table = weights.clone();
            fold(&mut table, point[0]);
            assert_eq!(folded(&weights, point[0]), table);
        }
    }

    #[test]
    fn a_truncated_reduction_is_malformed() {
        let mut pair = pair(3, &mut rng(10));
        let claim = inner_product(&pair.weights, &pair.values);
        let mut prover = build_prover("post_gkr-tests", "sumcheck");
        prove_rounds(&mut pair, 3, claim, Bits100, &mut prover);
        let mut proof = prover.finish();
        proof.narg_string.truncate(proof.narg_string.len() - 16);
        let mut verifier = build_verifier("post_gkr-tests", "sumcheck", &proof);
        assert_eq!(
            verify_rounds(3, claim, Bits100, &mut verifier).err(),
            Some(VerifyError::MalformedProof)
        );
    }
}
