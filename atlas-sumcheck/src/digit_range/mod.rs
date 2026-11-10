//! Dense digit range check: committed base-`b` digit columns, each digit in `[0, b)`, recompose
//! a value opened elsewhere.
//!
//! For digit columns `d_0, …, d_{n−1}` (most significant first), the point `r` and the claim
//! `V(r)` of the value they decompose, both supplied by the caller:
//!
//! ```text
//! Σ_t eq(r, t) · [ Σ_j γ^j · ∏_{a<b} (d_j(t) − a)  +  β · Σ_j b^(n−1−j) · d_j(t) ]  =  β · V(r)
//! ```
//!
//! of degree `b + 1`, reducing to one opening of every digit column at the bound point.
//!
//! **Soundness.** The caller fixes `V(r)` and has the digits committed before `γ` and `β` are
//! drawn ([`DigitRangeParams::draw`]). As a polynomial in `β` and `γ`, the identity splits into
//! `Σ_t eq(r, t)·∏_{a<b}(d_j(t) − a) = 0` for every `j` and `Σ_j b^(n−1−j)·d_j(r) = V(r)`, so
//! random `β`, `γ` give both with high probability. The first says the multilinear extension
//! of `t ↦ ∏_{a<b}(d_j(t) − a)` vanishes at `r`; `r` is itself drawn after the digits are
//! committed, so by Schwartz–Zippel that extension is zero, i.e. every digit is a root of
//! `∏_{a<b}(x − a)`, which has exactly the roots `0, …, b − 1` in the field. The second ties the
//! value to its digits at `r`, which the caller turns into a pointwise identity by the same
//! argument. With `n` digits the recomposition is in `[0, b^n)` and, for `b^n` far below the
//! field's characteristic, never wraps. Folding the recomposition into the zero-check opens each
//! digit column once.
//!
//! **S ⊆ R.** Honest digits are in `[0, b)`, so every zero-check summand vanishes, and the
//! recomposition at `r` is `V(r)` for the caller's honest value.
//!
//! The relation knows nothing about what the digits encode: the caller names the digit columns,
//! the value opening and the base.

use crate::relation::Produces;
use crate::{
    opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    relation::{
        Relation, RelationChallenge, RelationDerived, RelationExpr, RelationId, RelationOpening,
        RelationSetup, RelationVerifier,
    },
};
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_transcript::Transcript;
use common::CommittedPoly;
use jolt_claims::{challenge, constant, derived, opening, SymbolicSumcheck};
use jolt_field::Ring;

/// The challenge id of `β`, the recomposition's coefficient.
const BETA: RelationChallenge = RelationChallenge::Weight(0);

/// A digit range check, its setup resolved.
#[derive(Clone)]
pub struct DigitRangeParams<F: JoltField> {
    /// The digit base `b ≥ 2`.
    pub base: u64,
    /// The committed digit columns, most significant first.
    pub digits: Vec<CommittedPoly>,
    /// The opening of the value the digits recompose.
    pub value: OpeningId,
    /// The value's opening point `r` (big-endian).
    pub r: Vec<F>,
    /// The zero-check batching coefficients `γ^j`, one per digit column.
    pub gamma_powers: Vec<F>,
    /// The recomposition's coefficient `β`.
    pub beta: F,
    /// The sumcheck the digit openings are registered under.
    pub sumcheck_id: SumcheckId,
}

impl<F: JoltField> DigitRangeParams<F> {
    /// Read `value`'s opening point from the accumulator and draw `γ` and `β`. The caller has
    /// already fixed `value`'s claim and committed the digits. `Err` if the accumulator holds no
    /// claim for `value`.
    pub fn draw<T: Transcript>(
        base: u64,
        digits: Vec<CommittedPoly>,
        value: OpeningId,
        sumcheck_id: SumcheckId,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Result<Self, ProofVerifyError> {
        let r = accumulator.get_virtual_polynomial_opening(value)?.0.r;
        let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(digits.len());
        let beta: F = transcript.challenge_scalar();
        Ok(Self {
            base,
            digits,
            value,
            r,
            gamma_powers,
            beta,
            sumcheck_id,
        })
    }

    /// The recomposition weight `b^(n−1−j)` of every digit column.
    pub fn place_values(&self) -> Vec<u64> {
        place_values(self.base, self.digits.len())
    }

    /// The relation these parameters instantiate, with its setup.
    pub fn into_parts(self) -> (DigitRangeRelation, RelationSetup<F>) {
        let mut challenges: Vec<(RelationChallenge, F)> = self
            .gamma_powers
            .into_iter()
            .enumerate()
            .map(|(j, gamma)| (RelationChallenge::Batching(j), gamma))
            .collect();
        challenges.push((BETA, self.beta));
        let setup = RelationSetup {
            incoming_point: self.r.clone(),
            challenges,
        };
        let relation = DigitRangeRelation {
            base: self.base,
            num_rounds: self.r.len(),
            digits: self.digits,
            value: self.value,
            sumcheck_id: self.sumcheck_id,
        };
        (relation, setup)
    }

    /// The verifier of the check.
    pub fn into_verifier(self) -> DigitRangeVerifier<F> {
        let (relation, setup) = self.into_parts();
        RelationVerifier::at(relation, setup)
    }
}

/// `b^(n−1−j)` for `j < n`: the place value of each digit, most significant first.
pub fn place_values(base: u64, n: usize) -> Vec<u64> {
    (0..n).map(|j| base.pow((n - 1 - j) as u32)).collect()
}

/// `Σ_t eq(r, t)·[Σ_j γ^j·∏_{a<b}(d_j(t) − a) + β·Σ_j b^(n−1−j)·d_j(t)] = β·V(r)`: every digit is
/// in `[0, b)` and the digits recompose `V` (see the module docs).
///
/// ```text
/// input claim   β · V(r)
/// output claim  eq(r, r′) · [Σ_j γ^j · ∏_{a<b}(d_j(r′) − a) + β · Σ_j b^(n−1−j) · d_j(r′)]
/// ```
///
/// at the low-to-high bound point `r′`. The setup's incoming point is `r`; `γ^j` are
/// [`RelationChallenge::Batching`] and `β` is [`RelationChallenge::Weight`]`(0)`.
pub struct DigitRangeRelation {
    base: u64,
    num_rounds: usize,
    digits: Vec<CommittedPoly>,
    value: OpeningId,
    sumcheck_id: SumcheckId,
}

impl DigitRangeRelation {
    /// The digit base `b`.
    pub fn base(&self) -> u64 {
        self.base
    }

    fn digit<G: Ring>(&self, j: usize) -> RelationExpr<G> {
        opening(RelationOpening::Id(OpeningId::new(
            self.digits[j],
            self.sumcheck_id,
        )))
    }
}

impl SymbolicSumcheck for DigitRangeRelation {
    crate::relation_types!();
    type Shape = Self;

    fn new(shape: Self) -> Self {
        shape
    }

    fn id() -> RelationId {
        RelationId::DigitRange
    }

    fn rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree(&self) -> usize {
        self.base as usize + 1
    }

    fn input_expression<G: Ring>(&self) -> RelationExpr<G> {
        challenge::<G, _, _, _>(BETA) * opening(RelationOpening::Id(self.value))
    }

    fn output_expression<G: Ring>(&self) -> RelationExpr<G> {
        let n = self.digits.len();
        let zero_check = (0..n).fold(RelationExpr::zero(), |acc, j| {
            let vanishing = (0..self.base).fold(constant::<G, _, _, _>(G::one()), |p, a| {
                p * (self.digit::<G>(j) - constant(G::from_u64(a)))
            });
            acc + challenge::<G, _, _, _>(RelationChallenge::Batching(j)) * vanishing
        });
        let recomposition = place_values(self.base, n)
            .into_iter()
            .enumerate()
            .fold(RelationExpr::zero(), |acc, (j, place)| {
                acc + constant::<G, _, _, _>(G::from_u64(place)) * self.digit::<G>(j)
            });
        derived::<G, _, _, _>(RelationDerived::Eq)
            * (zero_check + challenge::<G, _, _, _>(BETA) * recomposition)
    }
}

impl Produces for DigitRangeRelation {
    fn produced(&self) -> Vec<OpeningId> {
        self.digits
            .iter()
            .map(|&poly| OpeningId::new(poly, self.sumcheck_id))
            .collect()
    }
}

impl<F: JoltField> Relation<F> for DigitRangeRelation {}

/// The verifier of a digit range check: its relation.
pub type DigitRangeVerifier<F> = RelationVerifier<F, DigitRangeRelation>;

#[cfg(test)]
mod tests {
    use super::prover::DigitRangeProver;
    use super::*;
    use crate::opening_proof::{prover::ProverOpeningAccumulator, VerifierOpeningAccumulator};
    use crate::relation::{prover::prove_relation, verify_relation};
    use atlas_algebra::field::fp128::Fp128 as Fr;
    use atlas_algebra::poly::multilinear_polynomial::{
        MultilinearPolynomial, PolynomialEvaluation,
    };
    use atlas_algebra::poly::opening_point::{OpeningPoint, BIG_ENDIAN};
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, VerifyingTranscript,
    };
    use common::VirtualPoly;
    use jolt_field::Field;
    use jolt_sumcheck::ProveRounds;
    use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};

    const LOG_T: usize = 5;
    const VALUE_BITS: u32 = 32;

    fn value_id() -> OpeningId {
        OpeningId::new(VirtualPoly::GlobalClampOut(0), SumcheckId::NodeExecution(0))
    }

    fn digit_ids(n: usize) -> Vec<CommittedPoly> {
        (0..n)
            .map(|j| CommittedPoly::GlobalClampOutDigit(0, j))
            .collect()
    }

    fn num_digits(base: u64) -> usize {
        (VALUE_BITS / base.ilog2()) as usize
    }

    /// Random values below `2^32` and their base-`b` digits, most significant first.
    fn honest(rng: &mut StdRng, base: u64) -> (Vec<u64>, Vec<Vec<u8>>) {
        let n = num_digits(base);
        let values: Vec<u64> = (0..1 << LOG_T).map(|_| rng.next_u32() as u64).collect();
        let digits = (0..n)
            .map(|j| {
                let shift = (n - 1 - j) as u32 * base.ilog2();
                values
                    .iter()
                    .map(|&v| ((v >> shift) % base) as u8)
                    .collect()
            })
            .collect();
        (values, digits)
    }

    fn recompose(digits: &[Vec<u8>], base: u64) -> Vec<u64> {
        let places = place_values(base, digits.len());
        (0..digits[0].len())
            .map(|t| {
                digits
                    .iter()
                    .zip(&places)
                    .map(|(d, p)| d[t] as u64 * p)
                    .sum()
            })
            .collect()
    }

    fn mle(values: &[u64]) -> MultilinearPolynomial<Fr> {
        MultilinearPolynomial::from(values.iter().map(|&v| Fr::from_u64(v)).collect::<Vec<_>>())
    }

    /// The hypercube sum of the output expression's summand for these digits: what an honest
    /// prover of these columns can claim.
    fn true_sum(params: &DigitRangeParams<Fr>, digits: &[Vec<u8>]) -> Fr {
        let eq = atlas_algebra::poly::eq_poly::EqPolynomial::<Fr>::evals(&params.r);
        let places = params.place_values();
        (0..digits[0].len())
            .map(|t| {
                let summand: Fr = digits
                    .iter()
                    .enumerate()
                    .map(|(j, d)| {
                        let x = Fr::from_u64(d[t] as u64);
                        let vanishing = (0..params.base)
                            .fold(Fr::from_u64(1), |p, a| p * (x - Fr::from_u64(a)));
                        params.gamma_powers[j] * vanishing
                            + params.beta * Fr::from_u64(places[j]) * x
                    })
                    .sum();
                eq[t] * summand
            })
            .sum()
    }

    /// Prove the check over `digits` and verify it against `verifier_value`, the value claim
    /// the verifier holds. `r`, `γ` and `β` are drawn first, as the caller's protocol draws them
    /// after the value is fixed; the prover then claims its digits' true hypercube sum (over
    /// `β`), which is the only claim its round messages can support, and the verifier must
    /// reject unless that sum is `β·verifier_value`.
    fn run(base: u64, digits: &[Vec<u8>], verifier_value: Fr) -> Result<(), ProofVerifyError> {
        let n = digits.len();
        let mut prover_transcript = Blake2bProver::new(b"digit_range");
        let r: Vec<Fr> = prover_transcript.challenge_vector(LOG_T);
        let gamma_powers: Vec<Fr> = prover_transcript.challenge_scalar_powers(n);
        let beta: Fr = prover_transcript.challenge_scalar();
        let params = DigitRangeParams {
            base,
            digits: digit_ids(n),
            value: value_id(),
            r: r.clone(),
            gamma_powers,
            beta,
            sumcheck_id: SumcheckId::DigitRange,
        };
        let prover_value = true_sum(&params, digits) * beta.inverse().unwrap();

        let mut prover_accumulator = ProverOpeningAccumulator::<Fr>::new();
        prover_accumulator.openings.insert(
            value_id(),
            (OpeningPoint::<BIG_ENDIAN, Fr>::new(r.clone()), prover_value),
        );
        let columns: Vec<MultilinearPolynomial<Fr>> = digits
            .iter()
            .map(|d| MultilinearPolynomial::from(d.clone()))
            .collect();
        let mut prover = DigitRangeProver::new(params.clone(), columns);
        let (proof, _) =
            prove_relation(&mut prover, &mut prover_accumulator, &mut prover_transcript);

        let mut verifier_transcript =
            Blake2bVerifier::new(b"digit_range", prover_transcript.narg().to_vec());
        let _: Vec<Fr> = verifier_transcript.challenge_vector(LOG_T);
        let _: Vec<Fr> = verifier_transcript.challenge_scalar_powers(n);
        let _: Fr = verifier_transcript.challenge_scalar();
        let mut verifier_accumulator = VerifierOpeningAccumulator::<Fr>::new();
        for (key, (_, value)) in &prover_accumulator.openings {
            verifier_accumulator
                .openings
                .insert(*key, (OpeningPoint::default(), *value));
        }
        verifier_accumulator.openings.insert(
            value_id(),
            (OpeningPoint::<BIG_ENDIAN, Fr>::new(r), verifier_value),
        );
        verify_relation(
            &proof,
            &params.into_verifier(),
            &mut verifier_accumulator,
            &mut verifier_transcript,
        )
        .map(|_| ())
    }

    fn value_at(values: &[u64], r: &[Fr]) -> Fr {
        mle(values).evaluate(r)
    }

    /// The point `r` that [`run`] draws.
    fn point() -> Vec<Fr> {
        Blake2bProver::new(b"digit_range").challenge_vector(LOG_T)
    }

    fn honest_digits_verify(base: u64) {
        let mut rng = StdRng::seed_from_u64(0xD161 + base);
        let (values, digits) = honest(&mut rng, base);
        assert_eq!(recompose(&digits, base), values);
        let r = point();
        run(base, &digits, value_at(&values, &r)).expect("honest digits verify");
    }

    /// `d_j + b` and the next more significant digit `− 1`: the recomposition is unchanged and
    /// one digit is out of range.
    fn out_of_range_digit_rejected(base: u64) {
        let mut rng = StdRng::seed_from_u64(0x0B0E + base);
        let (values, mut digits) = honest(&mut rng, base);
        let t = rng.gen_range(0..1usize << LOG_T);
        let j = (1..digits.len())
            .find(|&j| digits[j - 1][t] >= 1)
            .expect("a nonzero digit above the least significant");
        digits[j][t] += base as u8;
        digits[j - 1][t] -= 1;
        assert_eq!(
            recompose(&digits, base),
            values,
            "the recomposition is kept"
        );
        let r = point();
        assert!(run(base, &digits, value_at(&values, &r)).is_err());
    }

    /// Honest digits against a value one off theirs.
    fn false_recomposition_rejected(base: u64) {
        let mut rng = StdRng::seed_from_u64(0xFA15 + base);
        let (values, digits) = honest(&mut rng, base);
        let r = point();
        assert!(run(base, &digits, value_at(&values, &r) + Fr::from_u64(1)).is_err());
    }

    #[test]
    fn digit_range_honest_base_4() {
        honest_digits_verify(4);
    }

    #[test]
    fn digit_range_honest_base_2() {
        honest_digits_verify(2);
    }

    #[test]
    fn digit_range_out_of_range_digit_base_4() {
        out_of_range_digit_rejected(4);
    }

    #[test]
    fn digit_range_out_of_range_digit_base_2() {
        out_of_range_digit_rejected(2);
    }

    #[test]
    fn digit_range_false_recomposition_base_4() {
        false_recomposition_rejected(4);
    }

    #[test]
    fn digit_range_false_recomposition_base_2() {
        false_recomposition_rejected(2);
    }

    /// The prover's round messages against the relation's reference messages: the output
    /// expression summed over the hypercube at each round's partially bound point, with the
    /// digits read as their multilinear extensions.
    fn prover_matches_reference(base: u64) {
        let mut rng = StdRng::seed_from_u64(0x2EF0 + base);
        let (_, digits) = honest(&mut rng, base);
        // Arbitrary digits (not in range) so every summand is exercised.
        let digits: Vec<Vec<u8>> = digits
            .iter()
            .map(|d| {
                d.iter()
                    .map(|_| rng.gen_range(0..2 * base as u8 + 1))
                    .collect()
            })
            .collect();
        let n = digits.len();
        let random = |rng: &mut StdRng| <Fr as jolt_field::Field>::random(rng);
        let params = DigitRangeParams {
            base,
            digits: digit_ids(n),
            value: value_id(),
            r: (0..LOG_T).map(|_| random(&mut rng)).collect(),
            gamma_powers: (0..n).map(|_| random(&mut rng)).collect(),
            beta: random(&mut rng),
            sumcheck_id: SumcheckId::DigitRange,
        };
        let columns: Vec<MultilinearPolynomial<Fr>> = digits
            .iter()
            .map(|d| MultilinearPolynomial::from(d.clone()))
            .collect();
        let (relation, setup) = params.clone().into_parts();
        let produced = relation.produced();
        let output = relation.output_expression::<Fr>();
        let summand = |point: &[Fr]| {
            let opening_point = crate::relation::low_to_high_point(point);
            output.evaluate(
                |opening| match opening {
                    RelationOpening::Id(id) => {
                        let j = produced.iter().position(|p| p == id).unwrap();
                        columns[j].evaluate(&opening_point)
                    }
                    RelationOpening::ReducedNodeOutput(node) => {
                        panic!("node {node}'s reduced output is not produced")
                    }
                },
                |challenge| setup.challenge(challenge),
                |derived| relation.derive(derived, &setup, point).unwrap(),
            )
        };
        let challenges: Vec<Fr> = (0..LOG_T).map(|_| random(&mut rng)).collect();
        let degree = <DigitRangeRelation as SymbolicSumcheck>::degree(&relation);
        let mut prover = DigitRangeProver::new(params, columns.clone());
        let mut bind = None;
        let mut claim = Fr::from_u64(0);
        for round in 0..LOG_T {
            let free = LOG_T - round - 1;
            let expected: Vec<Fr> = (0..=degree as u64)
                .map(|x| {
                    (0..1usize << free)
                        .map(|bits| {
                            let mut p = challenges[..round].to_vec();
                            p.push(Fr::from_u64(x));
                            p.extend(
                                (0..free)
                                    .map(|i| Fr::from_u64(((bits >> (free - 1 - i)) & 1) as u64)),
                            );
                            summand(&p)
                        })
                        .sum()
                })
                .collect();
            if round == 0 {
                claim = expected[0] + expected[1];
            }
            let message = prover
                .prove_round(bind, round, claim)
                .unwrap_or_else(|e| panic!("round {round}: {e:?}"));
            for (x, value) in expected.iter().enumerate() {
                assert_eq!(
                    message.evaluate(Fr::from_u64(x as u64)),
                    *value,
                    "base {base}, round {round}, x = {x}"
                );
            }
            claim = message.evaluate(challenges[round]);
            bind = Some(challenges[round]);
        }
    }

    #[test]
    fn digit_range_prover_matches_reference_base_4() {
        prover_matches_reference(4);
    }

    #[test]
    fn digit_range_prover_matches_reference_base_2() {
        prover_matches_reference(2);
    }
}

pub mod prover;
