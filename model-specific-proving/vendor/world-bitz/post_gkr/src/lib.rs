//! Step 6 of 2.1. "The prime field case": from the GKR's inner product
//! claim on the committed bits to the MLE evaluation claim the opening takes.
//!
//! The GKR protocol ends on `<omega (x) eq(., r_c), f> = C - 1`
//! over the `2^t x 2^s` committed bits `f`, a `LinearClaim<F128>` with row
//! factor `omega` and column factor `eq(., r_c)`. A virtualization `h = M f`
//! (2.4. "Virtual F_2-linear transforms, NP-complete multi-domain linear
//! relations, and hybrid proof systems") moves it onto `f` as `<M^T w, f>`,
//! one weight per bit: the same type with a single column of weight one.
//!
//! The opening scheme (`pcs`) proves evaluation claims `MLE[f](r) = v` and
//! does the ring switch from the bits to the packed vector itself (Binius's,
//! as Step 6 cites it). This crate's sumcheck turns the linear claim into
//! such an evaluation claim: all `m = t + s` variables are bound against the
//! bits, the row ones first as the bits are indexed. The verifier's closing
//! weight `MLE[rows](rho_b) MLE[columns](rho_c)` costs `2^t + 2^s`
//! multiplications, so it is linear in the bits only when a factor is.
//!
//! The first round is taken off the packed bits: a bit is zero or one, so
//! the round's coefficients are sums of weights with no multiplication, and
//! the tables the prover folds to have one entry per two bits. The weights
//! are never written out in full: their folded table is the folded row
//! factor tensored with the column factor. Proof: `2m + 1` elements, `rho`
//! low coordinate first ([`OpeningQuery::Mle`]). Each positive grind adds an eight-byte nonce.
//!
//! # Transcript
//!
//! Each round absorbs its coefficients, checks any grinding, then samples its challenge.
//! The closing evaluation follows the last round. The claim is
//! not re-absorbed here: the opening scheme binds its statement before it
//! calls the sumcheck. The round count derives from the claim and is not
//! absorbed either.

mod bounded;
mod projected;
mod sumcheck;
#[cfg(test)]
mod test_util;

use crate::sumcheck::{Pair, RoundMessage};
use common::shape::PACK_BITS;
use common::{LinearClaim, OpeningQuery, SecurityLevel};
use field::F128;
use num_traits::{ConstOne, ConstZero};
#[cfg(feature = "parallel")]
use poly::parallel::workload_size;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use transcript::{ProverState, VerifierState};

/// A reduction the prover cannot run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProveError {
    /// The packed witness is not one element per 128 bits of the claim.
    WitnessLengthMismatch,
    /// The claim's weights against the witness do not give the target.
    ClaimDoesNotHold,
}

/// A reduction the verifier rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyError {
    /// A record is missing or does not decode.
    MalformedProof,
    /// The weights at the challenges times `v` is not the running claim.
    EvaluationMismatch,
}

/// The bits one packed element carries.
const ELEMENT_BITS: usize = 1 << PACK_BITS;

/// Each quadratic round has error at most `2 / |F128|` before grinding.
const GRINDING_LABEL: &[u8] = b"bitz/pcs/sumcheck/v1";

/// The even bit positions of a packed element, `0b01010101...`
const EVEN: u128 = 0x55555555555555555555555555555555;

/// Packed elements per parallel task, enough to fill the cache.
#[cfg(feature = "parallel")]
const ELEMENTS_PER_TASK: usize = workload_size::<F128>() / ELEMENT_BITS;

/// Runs the sumcheck for `claim` on `packed`, the committed `f` column
/// major, and returns the evaluation claim it leaves: `MLE[f](rho) = v` as
/// an [`OpeningQuery::Mle`], `rho` the challenges low coordinate first and
/// `v` the prover's closing evaluation.
#[tracing::instrument(name = "Prove inner-product sumcheck", skip_all)]
pub fn prove(
    claim: &LinearClaim<F128>,
    packed: &[F128],
    security: SecurityLevel,
    transcript: &mut ProverState,
) -> Result<OpeningQuery, ProveError> {
    prove_factors(
        claim.row_weights(),
        claim.column_weights(),
        claim.target(),
        packed,
        security,
        transcript,
    )
}

/// Replays the sumcheck for `claim`, checks that the closing evaluation
/// fits, and returns the evaluation claim for the opening scheme to verify
/// against the commitment.
#[tracing::instrument(name = "Verify inner-product sumcheck", skip_all)]
pub fn verify(
    claim: &LinearClaim<F128>,
    security: SecurityLevel,
    transcript: &mut VerifierState<'_>,
) -> Result<OpeningQuery, VerifyError> {
    verify_factors(
        claim.row_weights(),
        claim.column_weights(),
        claim.target(),
        security,
        transcript,
    )
}

/// [`prove`] on the claim's parts. `rows` has `2^t` weights with `t >= 7`
/// and `columns` `2^s`, as `LinearClaim` guarantees.
fn prove_factors(
    rows: &[F128],
    columns: &[F128],
    target: F128,
    packed: &[F128],
    security: SecurityLevel,
    transcript: &mut ProverState,
) -> Result<OpeningQuery, ProveError> {
    prove_factors_with_cap(rows, columns, target, packed, security, transcript, 1 << 23)
}

fn prove_factors_with_cap(
    rows: &[F128],
    columns: &[F128],
    target: F128,
    packed: &[F128],
    security: SecurityLevel,
    transcript: &mut ProverState,
    dense_entries: usize,
) -> Result<OpeningQuery, ProveError> {
    let factors = Factors { rows, columns };
    if packed.len() != factors.packed_len() {
        return Err(ProveError::WitnessLengthMismatch);
    }
    if (128..=512).contains(&rows.len()) && packed.len() * (ELEMENT_BITS / 2) > dense_entries {
        return projected::prove(factors, packed, target, security, transcript, dense_entries);
    }
    if factors.weighted_sum(packed) != target {
        return Err(ProveError::ClaimDoesNotHold);
    }

    let message = factors.first_message(packed);
    transcript.prover_message(&message);
    transcript.grind(GRINDING_LABEL, security.grinding_bits(2));
    let challenge: F128 = transcript.verifier_message();
    let running = advance(target, message, challenge);
    if rows.len() * columns.len() / 2 > dense_entries {
        return Ok(bounded::prove(
            factors,
            packed,
            challenge,
            running,
            security,
            transcript,
            dense_entries,
        ));
    }
    let mut pair = Pair::new(factors.fold(challenge), bind_first(packed, challenge));
    let (rest, target) = sumcheck::prove(&mut pair, running, security, transcript);
    let mut point = vec![challenge];
    point.extend(rest);
    Ok(OpeningQuery::Mle { point, target })
}

/// [`verify`] on the claim's parts.
fn verify_factors(
    rows: &[F128],
    columns: &[F128],
    target: F128,
    security: SecurityLevel,
    transcript: &mut VerifierState<'_>,
) -> Result<OpeningQuery, VerifyError> {
    let log_rows = rows.len().trailing_zeros() as usize;
    let rounds = log_rows + columns.len().trailing_zeros() as usize;
    // `MLE[rows (x) columns](rho) = MLE[rows](rho_b) MLE[columns](rho_c)`.
    let weight = |point: &[F128]| {
        poly::f128::evaluate(rows, &point[..log_rows])
            * poly::f128::evaluate(columns, &point[log_rows..])
    };
    let (point, target) = sumcheck::verify(rounds, target, weight, security, transcript)?;
    Ok(OpeningQuery::Mle { point, target })
}

/// The weights `rows (x) columns` over the bits, read without being written
/// out: bit `(c << t) | b` weighs `rows[b] columns[c]`, so packed element
/// `e` covers rows `128 (e mod 2^(t-7)) ..` of column `e / 2^(t-7)`.
struct Factors<'a> {
    rows: &'a [F128],
    columns: &'a [F128],
}

impl Factors<'_> {
    fn packed_len(&self) -> usize {
        (self.rows.len() >> PACK_BITS) * self.columns.len()
    }

    /// The row weights element `index` covers, and its column's weight.
    fn element(&self, index: usize) -> (&[F128], F128) {
        let per_column = self.rows.len() >> PACK_BITS;
        let rows = &self.rows[(index % per_column) << PACK_BITS..][..ELEMENT_BITS];
        (rows, self.columns[index / per_column])
    }

    /// `<rows (x) columns, f>`: the weights at the set bits.
    fn weighted_sum(&self, packed: &[F128]) -> F128 {
        let element = |(index, &element): (usize, &F128)| -> F128 {
            let (rows, column) = self.element(index);
            column
                * iter_over_set_bits(element.to_u128())
                    .map(|v| rows[v])
                    .sum::<F128>()
        };
        #[cfg(feature = "parallel")]
        return packed
            .par_iter()
            .enumerate()
            .with_min_len(ELEMENTS_PER_TASK)
            .map(element)
            .sum();
        #[cfg(not(feature = "parallel"))]
        packed.iter().enumerate().map(element).sum()
    }

    /// `(a_0, a_2)` of the first round off the bits: `a_0` sums the weights
    /// at even positions whose bit is set, `a_2` the pair sums `w_0 + w_1`
    /// where the two bits differ, since `(w_0 + w_1)(f_0 + f_1)` is that
    /// exactly then.
    fn first_message(&self, packed: &[F128]) -> RoundMessage {
        let element = |(index, &element): (usize, &F128)| -> (F128, F128) {
            let (rows, column) = self.element(index);
            let bits = element.to_u128();
            let a0: F128 = iter_over_set_bits(bits & EVEN).map(|v| rows[v]).sum();
            let a2: F128 = iter_over_set_bits((bits ^ (bits >> 1)) & EVEN)
                .map(|v| rows[v] + rows[v + 1])
                .sum();
            (column * a0, column * a2)
        };
        let add = |(a0, a2): (F128, F128), (b0, b2): (F128, F128)| (a0 + b0, a2 + b2);
        #[cfg(feature = "parallel")]
        let (a0, a2) = packed
            .par_iter()
            .enumerate()
            .with_min_len(ELEMENTS_PER_TASK)
            .map(element)
            .reduce(|| (F128::ZERO, F128::ZERO), add);
        #[cfg(not(feature = "parallel"))]
        let (a0, a2) = packed
            .iter()
            .enumerate()
            .map(element)
            .fold((F128::ZERO, F128::ZERO), add);
        [a0, a2]
    }

    /// The weights with their first variable bound: the folded row factor
    /// tensored with the column factor, one entry per two bits.
    fn fold(self, challenge: F128) -> Vec<F128> {
        let rows = sumcheck::folded(self.rows, challenge);
        let mut table = Vec::with_capacity(rows.len() * self.columns.len());
        for &column in self.columns {
            table.extend(rows.iter().map(|&row| column * row));
        }
        table
    }
}

/// The positions of the set bits of `word`, ascending.
fn iter_over_set_bits(mut word: u128) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (word != 0).then(|| {
            let position = word.trailing_zeros() as usize;
            word &= word - 1;
            position
        })
    })
}

/// `MLE[f]` with its first variable bound: over each pair of bits
/// `(f_0, f_1)`, `f_0 + rho (f_0 + f_1)`, one of `0`, `1`, `rho` and
/// `1 + rho`.
fn bind_first(packed: &[F128], challenge: F128) -> Vec<F128> {
    let challenge_plus_one = challenge + F128::ONE;
    let values = [F128::ZERO, challenge_plus_one, challenge, F128::ONE];
    let element = |(slot, &element): (&mut [F128], &F128)| {
        let bits = element.to_u128();
        for (pair, value) in slot.iter_mut().enumerate() {
            *value = values[((bits >> (2 * pair)) & 3) as usize];
        }
    };
    let mut table = vec![F128::ZERO; packed.len() * ELEMENT_BITS / 2];
    #[cfg(feature = "parallel")]
    table
        .par_chunks_mut(ELEMENT_BITS / 2)
        .zip(packed)
        .with_min_len(ELEMENTS_PER_TASK)
        .for_each(element);
    #[cfg(not(feature = "parallel"))]
    table
        .chunks_mut(ELEMENT_BITS / 2)
        .zip(packed)
        .for_each(element);
    table
}

/// `h' = p(rho)` with `a_1 = h + a_2`.
fn advance(claim: F128, [a0, a2]: RoundMessage, challenge: F128) -> F128 {
    a0 + challenge * (claim + a2 + challenge * a2)
}

#[cfg(test)]
mod tests {
    use common::SecurityLevel::{Bits100, Bits128};
    use common::Shape;
    use poly::DenseMultilinearExtension;
    use transcript::{Proof, build_prover, build_verifier};

    use super::*;
    use crate::test_util::{Leaf, random, rng};
    use poly::f128::inner_product;

    #[test]
    fn bounded_prover_matches_dense_transcript() {
        let rows = vec![F128::from(5u128); 256];
        let columns = vec![F128::from(7u128); 64];
        let packed = (0..128)
            .map(|i| F128::from(i as u128 + 1))
            .collect::<Vec<_>>();
        let target = Factors {
            rows: &rows,
            columns: &columns,
        }
        .weighted_sum(&packed);
        let mut dense = build_prover("bounded-post-gkr", &b"");
        let mut compact = build_prover("bounded-post-gkr", &b"");
        let expected = prove_factors_with_cap(
            &rows,
            &columns,
            target,
            &packed,
            Bits100,
            &mut dense,
            usize::MAX,
        )
        .unwrap();
        let actual =
            prove_factors_with_cap(&rows, &columns, target, &packed, Bits100, &mut compact, 8)
                .unwrap();
        assert_eq!(mle(&expected), mle(&actual));
        let proof = compact.finish();
        assert_eq!(dense.finish(), proof);
        let mut verifier = build_verifier("bounded-post-gkr", &b"", &proof);
        let verified = verify_factors(&rows, &columns, target, Bits100, &mut verifier).unwrap();
        assert_eq!(mle(&actual), mle(&verified));
        verifier.check_eof().unwrap();
    }

    /// The evaluation claim's parts.
    fn mle(query: &OpeningQuery) -> (&[F128], F128) {
        let OpeningQuery::Mle { point, target } = query else {
            panic!("the sumcheck leaves an evaluation claim");
        };
        (point, *target)
    }

    fn reduced(leaf: &Leaf, security: SecurityLevel) -> (OpeningQuery, Proof) {
        let mut prover = build_prover("post_gkr-tests", "reduce");
        let sent = prove_factors(
            &leaf.rows,
            &leaf.columns,
            leaf.target,
            &leaf.packed,
            security,
            &mut prover,
        )
        .unwrap();
        (sent, prover.finish())
    }

    fn verified(
        leaf: &Leaf,
        proof: &Proof,
        security: SecurityLevel,
    ) -> Result<OpeningQuery, VerifyError> {
        let mut verifier = build_verifier("post_gkr-tests", "reduce", proof);
        let received = verify_factors(
            &leaf.rows,
            &leaf.columns,
            leaf.target,
            security,
            &mut verifier,
        )?;
        assert!(verifier.check_eof().is_ok());
        Ok(received)
    }

    #[test]
    fn set_bits_are_read_in_ascending_order() {
        let element = F128::new(0b1011, 1 << 63);
        assert_eq!(
            iter_over_set_bits(element.to_u128()).collect::<Vec<_>>(),
            vec![0, 1, 3, 127]
        );
        assert_eq!(iter_over_set_bits(0).count(), 0);
        assert_eq!(iter_over_set_bits(u128::MAX).count(), 128);
    }

    /// Factored over several columns, and a single column of weight one:
    /// the shape the transposition leaves.
    #[test]
    fn the_two_sides_agree_on_a_true_claim() {
        for (log_rows, log_columns, seed) in [(8, 2, 63), (10, 0, 64)] {
            let leaf = Leaf::random(log_rows, log_columns, seed);
            let (sent, proof) = reduced(&leaf, Bits100);
            assert_eq!(proof.narg_string.len(), (2 * 10 + 1) * 16);
            assert!(proof.hints.is_empty());

            let received = verified(&leaf, &proof, Bits100).unwrap();
            assert_eq!(sent, received);
            let (point, target) = mle(&received);
            assert_eq!(point.len(), 10);
            assert_eq!(
                leaf.bits_extension().evaluate(point).unwrap(),
                target,
                "{log_rows} x {log_columns}"
            );
        }
    }

    /// Through `LinearClaim`, at the smallest committed shape.
    #[test]
    fn a_linear_claim_reduces_to_an_evaluation_the_witness_satisfies() {
        let leaf = Leaf::sparse(7, 15, 65);
        let claim = LinearClaim::from_shape(
            &Shape::new(7, 15).unwrap(),
            leaf.rows.clone(),
            leaf.columns.clone(),
            leaf.target,
        )
        .unwrap();
        let mut prover = build_prover("post_gkr-tests", "reduce");
        let sent = prove(&claim, &leaf.packed, Bits100, &mut prover).unwrap();
        let proof = prover.finish();
        let mut verifier = build_verifier("post_gkr-tests", "reduce", &proof);
        assert_eq!(verify(&claim, Bits100, &mut verifier), Ok(sent.clone()));
        let (point, target) = mle(&sent);
        assert_eq!(leaf.evaluate(point), target);
    }

    /// The first round off the bits sends what the generic round over the
    /// weights and bits written out would, and folds to the same tables.
    #[test]
    fn the_first_round_off_the_bits_is_the_generic_round() {
        let leaf = Leaf::random(7, 2, 67);
        let factors = Factors {
            rows: &leaf.rows,
            columns: &leaf.columns,
        };
        let weights = leaf.weights();
        let written_out: Vec<F128> = (0..1 << 9)
            .map(|index| F128::from(leaf.bit(index)))
            .collect();
        assert_eq!(factors.weighted_sum(&leaf.packed), leaf.target);
        assert_eq!(inner_product(&weights, &written_out), leaf.target);

        let mut generic = Pair::new(weights.clone(), written_out.clone());
        let mut prover = build_prover("post_gkr-tests", "reduce");
        let (point, _) = sumcheck::prove(&mut generic, leaf.target, Bits100, &mut prover);
        let proof = prover.finish();
        let message = factors.first_message(&leaf.packed);
        assert_eq!(
            proof.narg_string[..32],
            [message[0].to_bytes(), message[1].to_bytes()].concat()
        );

        let fold = |table: Vec<F128>| {
            let mut folded = DenseMultilinearExtension { evaluations: table };
            folded.fold(&point[..1]).unwrap();
            folded.evaluations
        };
        assert_eq!(bind_first(&leaf.packed, point[0]), fold(written_out));
        assert_eq!(factors.fold(point[0]), fold(weights));
    }

    /// Every record off by one bit fails the closing check; a different
    /// target does too; a proof cut short does not decode.
    #[test]
    fn a_reduction_altered_in_transit_is_caught() {
        let mut leaf = Leaf::random(7, 1, 66);
        let (_, proof) = reduced(&leaf, Bits100);
        let records = 2 * 8 + 1;
        assert_eq!(proof.narg_string.len(), records * 16);
        for record in 0..records {
            let mut altered = proof.clone();
            altered.narg_string[record * 16] ^= 1;
            assert_eq!(
                verified(&leaf, &altered, Bits100),
                Err(VerifyError::EvaluationMismatch),
                "record {record}"
            );
        }
        let mut short = proof.clone();
        short.narg_string.truncate(proof.narg_string.len() - 16);
        assert_eq!(
            verified(&leaf, &short, Bits100),
            Err(VerifyError::MalformedProof)
        );

        leaf.target += F128::ONE;
        assert_eq!(
            verified(&leaf, &proof, Bits100),
            Err(VerifyError::EvaluationMismatch)
        );
    }

    #[test]
    fn security_targets_cover_every_quadratic_round() {
        let leaf = Leaf::random(7, 1, 68);
        let rounds = 8;
        for security in [Bits100, Bits128] {
            let (sent, proof) = reduced(&leaf, security);
            let nonce_bytes = usize::from(security == Bits128) * 8;
            assert_eq!(proof.narg_string.len(), rounds * (32 + nonce_bytes) + 16);
            assert_eq!(verified(&leaf, &proof, security), Ok(sent.clone()));
            let (point, target) = mle(&sent);
            assert_eq!(leaf.evaluate(point), target);

            let other = match security {
                Bits100 => Bits128,
                Bits128 => Bits100,
            };
            assert!(verified(&leaf, &proof, other).is_err());

            if nonce_bytes != 0 {
                for round in 0..rounds {
                    let mut altered = proof.clone();
                    let nonce = round * 40 + 32;
                    altered.narg_string[nonce] ^= 1;
                    assert!(
                        verified(&leaf, &altered, security).is_err(),
                        "round {round}"
                    );
                    let mut missing = proof.clone();
                    missing.narg_string.drain(nonce..nonce + 8);
                    assert!(
                        verified(&leaf, &missing, security).is_err(),
                        "round {round}"
                    );
                }
            }
            for length in 0..proof.narg_string.len() {
                let mut short = proof.clone();
                short.narg_string.truncate(length);
                assert!(
                    verified(&leaf, &short, security).is_err(),
                    "length {length}"
                );
            }
        }
    }

    #[test]
    fn the_reduction_refuses_what_it_cannot_prove() {
        let leaf = Leaf::random(7, 1, 56);
        let mut transcript = build_prover("post_gkr-tests", "reduce");
        assert_eq!(
            prove_factors(
                &leaf.rows,
                &leaf.columns,
                leaf.target + random(&mut rng(57)),
                &leaf.packed,
                Bits100,
                &mut transcript
            ),
            Err(ProveError::ClaimDoesNotHold)
        );
        assert_eq!(
            prove_factors(
                &leaf.rows,
                &leaf.columns,
                leaf.target,
                &leaf.packed[1..],
                Bits100,
                &mut transcript
            ),
            Err(ProveError::WitnessLengthMismatch)
        );
        // Nothing was written before the refusals.
        assert!(transcript.finish().narg_string.is_empty());
    }
}
