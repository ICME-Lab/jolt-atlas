//! Sumcheck runner for GKR-style relations with a common equality factor.
//!
//! Operation relations intentionally omit `eq`. This runner owns the split-eq
//! state and attaches the common equality factor while building each round
//! polynomial.

use ark_bn254::Fr;
use ark_ff::PrimeField;
use joltworks::{
    field::JoltField,
    poly::{multilinear_polynomial::BindingOrder, split_eq_poly::GruenSplitEqPolynomial},
    transcripts::Transcript,
};
use rand_core::CryptoRngCore;

use crate::{
    committed_round::{
        challenge_round_poly_optimized, commit_round_poly, prove_round_consistency,
        verify_round_consistency, CommittedRoundPoly, RoundConsistencyProof,
    },
    pedersen::Commitment,
    pedersen::PedersenParams,
    round::{RoundPoly, RoundRelation},
};

#[derive(Debug, Clone)]
pub struct SumCheck<F, R, const LANES: usize>
where
    F: JoltField + PrimeField,
{
    eq: GruenSplitEqPolynomial<F>,
    relation: R,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedSumCheckRound {
    pub poly: RoundPoly<Fr>,
    pub committed_poly: CommittedRoundPoly,
    pub challenge: <Fr as JoltField>::Challenge,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedSumCheckProof {
    pub round_commitments: Vec<Vec<Commitment>>,
    pub consistency_proofs: Vec<RoundConsistencyProof>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedSumCheckProverOutput {
    pub proof: CommittedSumCheckProof,
    pub challenges: Vec<<Fr as JoltField>::Challenge>,
    pub rounds: Vec<CommittedRoundPoly>,
}

impl<F, R, const LANES: usize> SumCheck<F, R, LANES>
where
    F: JoltField + PrimeField,
    R: RoundRelation<F, LANES> + Sync,
{
    pub fn new(eq_point: &[F::Challenge], relation: R) -> Self {
        assert!(!eq_point.is_empty(), "SumCheck needs at least one variable");
        let eq = GruenSplitEqPolynomial::new(eq_point, BindingOrder::LowToHigh);
        assert_eq!(
            eq.len(),
            relation.len() * 2,
            "split-eq and relation length mismatch"
        );
        Self { eq, relation }
    }

    pub fn num_rounds(&self) -> usize {
        self.eq.get_num_vars()
    }

    pub fn remaining_rounds(&self) -> usize {
        self.eq.len().ilog2() as usize
    }

    pub fn is_complete(&self) -> bool {
        self.remaining_rounds() == 0
    }

    pub fn relation(&self) -> &R {
        &self.relation
    }

    pub fn relation_mut(&mut self) -> &mut R {
        &mut self.relation
    }

    pub fn round_poly(&self) -> RoundPoly<F> {
        assert!(!self.is_complete(), "sumcheck is already complete");
        assert_eq!(LANES, self.relation.degree() + 1);

        let relation_evals = self.relation_evals_with_split_eq();
        let current_w = self.eq.get_current_w();
        let eq_constant = F::one() - current_w;
        let eq_linear = current_w + current_w - F::one();
        let current_scalar = self.eq.get_current_scalar();

        if LANES == 3 {
            return self.quadratic_relation_times_eq(
                &relation_evals,
                eq_constant,
                eq_linear,
                current_scalar,
            );
        }

        let relation_poly = RoundPoly::interpolate(&relation_evals);
        let mut coeffs = vec![F::zero(); relation_poly.coeffs.len() + 1];
        for (i, coeff) in relation_poly.coeffs.iter().enumerate() {
            coeffs[i] += *coeff * eq_constant * current_scalar;
            coeffs[i + 1] += *coeff * eq_linear * current_scalar;
        }
        RoundPoly { coeffs }
    }

    pub fn round(&mut self, r: F::Challenge) -> RoundPoly<F> {
        let poly = self.round_poly();
        self.bind(r);
        poly
    }

    pub fn bind(&mut self, r: F::Challenge) {
        assert!(!self.is_complete(), "sumcheck is already complete");
        self.eq.bind(r);
        self.relation.bind(r);
        if !self.is_complete() {
            assert_eq!(
                self.eq.len(),
                self.relation.len() * 2,
                "split-eq and relation length mismatch after bind"
            );
        }
    }

    fn relation_evals_with_split_eq(&self) -> [F; LANES] {
        self.eq
            .par_fold_out_in_unreduced::<9, LANES>(&|g| self.relation.round_evals(g))
    }

    fn quadratic_relation_times_eq(
        &self,
        relation_evals: &[F],
        eq_constant: F,
        eq_linear: F,
        current_scalar: F,
    ) -> RoundPoly<F> {
        assert_eq!(relation_evals.len(), 3);
        let half = JoltField::inverse(&F::from(2_u64)).expect("2 is nonzero");
        let q0 = relation_evals[0];
        let q2 =
            (relation_evals[2] - relation_evals[1] - relation_evals[1] + relation_evals[0]) * half;
        let q1 = relation_evals[1] - q0 - q2;

        RoundPoly {
            coeffs: vec![
                current_scalar * eq_constant * q0,
                current_scalar * (eq_constant * q1 + eq_linear * q0),
                current_scalar * (eq_constant * q2 + eq_linear * q1),
                current_scalar * eq_linear * q2,
            ],
        }
    }
}

impl<R, const LANES: usize> SumCheck<Fr, R, LANES>
where
    R: RoundRelation<Fr, LANES> + Sync,
{
    pub fn committed_round<T, Rng>(
        &mut self,
        params: &PedersenParams,
        transcript: &mut T,
        rng: &mut Rng,
    ) -> CommittedSumCheckRound
    where
        T: Transcript,
        Rng: CryptoRngCore,
    {
        let poly = self.round_poly();
        let committed_poly = commit_round_poly(params, &poly, rng);
        let challenge =
            challenge_round_poly_optimized(params, &committed_poly.commitments, transcript);
        self.bind(challenge);

        CommittedSumCheckRound {
            poly,
            committed_poly,
            challenge,
        }
    }

    pub fn prove<T, Rng>(
        &mut self,
        params: &PedersenParams,
        previous_round: &CommittedRoundPoly,
        previous_challenge: Fr,
        transcript: &mut T,
        rng: &mut Rng,
    ) -> Option<CommittedSumCheckProverOutput>
    where
        T: Transcript,
        Rng: CryptoRngCore,
    {
        let mut round_commitments = Vec::with_capacity(self.remaining_rounds());
        let mut consistency_proofs = Vec::with_capacity(self.remaining_rounds());
        let mut challenges = Vec::with_capacity(self.remaining_rounds());
        let mut rounds = Vec::with_capacity(self.remaining_rounds());
        let mut previous_round = previous_round.clone();
        let mut previous_challenge = previous_challenge;

        while !self.is_complete() {
            let poly = self.round_poly();
            let committed_poly = commit_round_poly(params, &poly, rng);
            let proof = prove_round_consistency(
                params,
                &previous_round,
                &committed_poly,
                previous_challenge,
                transcript,
                rng,
            )?;
            consistency_proofs.push(proof);

            let challenge =
                challenge_round_poly_optimized(params, &committed_poly.commitments, transcript);
            self.bind(challenge);

            round_commitments.push(committed_poly.commitments.clone());
            rounds.push(committed_poly.clone());
            previous_round = committed_poly;
            previous_challenge = challenge.into();
            challenges.push(challenge);
        }

        Some(CommittedSumCheckProverOutput {
            proof: CommittedSumCheckProof {
                round_commitments,
                consistency_proofs,
            },
            challenges,
            rounds,
        })
    }

    pub fn verify<T>(
        params: &PedersenParams,
        proof: &CommittedSumCheckProof,
        previous_commitments: &[Commitment],
        previous_challenge: Fr,
        num_rounds: usize,
        transcript: &mut T,
    ) -> Option<Vec<<Fr as JoltField>::Challenge>>
    where
        T: Transcript,
    {
        if previous_commitments.is_empty()
            || proof.round_commitments.len() != num_rounds
            || proof.consistency_proofs.len() != num_rounds
        {
            return None;
        }

        let expected_commitments_per_round = LANES + 1;
        let mut challenges: Vec<<Fr as JoltField>::Challenge> = Vec::with_capacity(num_rounds);
        let mut previous_commitments = previous_commitments;
        let mut previous_challenge = previous_challenge;
        for (round, commitments) in proof.round_commitments.iter().enumerate() {
            if commitments.len() != expected_commitments_per_round {
                return None;
            }

            if !verify_round_consistency(
                params,
                previous_commitments,
                commitments,
                previous_challenge,
                &proof.consistency_proofs[round],
                transcript,
            ) {
                return None;
            }

            let challenge = challenge_round_poly_optimized(params, commitments, transcript);
            challenges.push(challenge);
            previous_commitments = commitments;
            previous_challenge = challenge.into();
        }
        Some(challenges)
    }
}

#[cfg(test)]
mod tests {
    use ark_ff::{One, Zero};
    use joltworks::{
        field::JoltField, poly::eq_poly::EqPolynomial, transcripts::Blake2bTranscript,
    };
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    use super::*;
    use crate::{
        committed_round::challenge_round_poly_optimized,
        ops::hadamard::Hadamard,
        pedersen::{commit, Opening, PedersenParams},
        round::{DenseMleTable, MleTable},
    };

    #[test]
    fn hadamard_round_poly_includes_split_eq() {
        let eq_challenges = [
            <Fr as JoltField>::Challenge::from(2_u128),
            <Fr as JoltField>::Challenge::from(5_u128),
        ];
        let eq_point = eq_challenges
            .iter()
            .map(|challenge| (*challenge).into())
            .collect::<Vec<Fr>>();
        let eq = EqPolynomial::<Fr>::evals(&eq_point);
        let lhs = [1, 3, 2, 5].map(Fr::from);
        let rhs = [7, 11, 13, 17].map(Fr::from);

        let relation = Hadamard::new::<Fr>(
            DenseMleTable::new(lhs.to_vec()),
            DenseMleTable::new(rhs.to_vec()),
        );
        let sumcheck = SumCheck::<Fr, _, 3>::new(&eq_challenges, relation);

        let poly = sumcheck.round_poly();
        for t in 0..=3 {
            let t = Fr::from(t as u64);
            let relation = Hadamard::new::<Fr>(
                DenseMleTable::new(lhs.to_vec()),
                DenseMleTable::new(rhs.to_vec()),
            );
            let expected = (0..relation.len())
                .map(|i| {
                    DenseMleTable::new(eq.clone()).at(i, t)
                        * relation.lhs.at(i, t)
                        * relation.rhs.at(i, t)
                })
                .fold(Fr::zero(), |acc, term| acc + term);
            assert_eq!(poly.evaluate(t), expected);
        }
    }

    #[test]
    fn bind_advances_relation_and_split_eq_together() {
        let eq_challenges = [
            <Fr as JoltField>::Challenge::from(2_u128),
            <Fr as JoltField>::Challenge::from(5_u128),
        ];
        let lhs = [1, 3, 2, 5].map(Fr::from);
        let rhs = [7, 11, 13, 17].map(Fr::from);
        let challenge = <Fr as JoltField>::Challenge::from(9_u128);

        let relation = Hadamard::new::<Fr>(
            DenseMleTable::new(lhs.to_vec()),
            DenseMleTable::new(rhs.to_vec()),
        );
        let mut sumcheck = SumCheck::<Fr, _, 3>::new(&eq_challenges, relation);
        sumcheck.bind(challenge);

        assert_eq!(sumcheck.remaining_rounds(), 1);
        assert_eq!(sumcheck.relation().len(), 1);
        let _ = sumcheck.round_poly();
    }

    #[test]
    fn hadamard_rounds_preserve_sumcheck_claim() {
        let mut rng = ChaCha20Rng::from_seed([11; 32]);
        let num_vars = 4;
        let len = 1 << num_vars;
        let eq_challenges = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect::<Vec<_>>();
        let lhs = (0..len).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
        let rhs = (0..len).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();

        let relation = Hadamard::new::<Fr>(DenseMleTable::new(lhs), DenseMleTable::new(rhs));
        let mut sumcheck = SumCheck::<Fr, _, 3>::new(&eq_challenges, relation);

        let mut claim = {
            let poly = sumcheck.round_poly();
            poly.evaluate(Fr::zero()) + poly.evaluate(Fr::one())
        };

        while !sumcheck.is_complete() {
            let poly = sumcheck.round_poly();
            assert_eq!(claim, poly.evaluate(Fr::zero()) + poly.evaluate(Fr::one()));

            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            claim = poly.evaluate(r.into());
            sumcheck.bind(r);
        }
    }

    #[test]
    fn committed_round_absorbs_commitments_and_binds_challenge() {
        let mut rng = ChaCha20Rng::from_seed([15; 32]);
        let params = PedersenParams::random(&mut rng);
        let eq_challenges = [
            <Fr as JoltField>::Challenge::from(2_u128),
            <Fr as JoltField>::Challenge::from(5_u128),
        ];
        let lhs = [1, 3, 2, 5].map(Fr::from);
        let rhs = [7, 11, 13, 17].map(Fr::from);
        let relation = Hadamard::new::<Fr>(
            DenseMleTable::new(lhs.to_vec()),
            DenseMleTable::new(rhs.to_vec()),
        );
        let mut sumcheck = SumCheck::<Fr, _, 3>::new(&eq_challenges, relation);

        let mut prover_transcript = Blake2bTranscript::default();
        let round = sumcheck.committed_round(&params, &mut prover_transcript, &mut rng);

        let mut verifier_transcript = Blake2bTranscript::default();
        let expected_challenge = challenge_round_poly_optimized(
            &params,
            &round.committed_poly.commitments,
            &mut verifier_transcript,
        );

        assert_eq!(round.challenge, expected_challenge);
        assert_eq!(sumcheck.remaining_rounds(), 1);
    }

    #[test]
    fn prover_and_verifier_check_committed_round_consistency() {
        let mut rng = ChaCha20Rng::from_seed([16; 32]);
        let params = PedersenParams::random(&mut rng);
        let eq_challenges = [
            <Fr as JoltField>::Challenge::from(2_u128),
            <Fr as JoltField>::Challenge::from(5_u128),
        ];
        let lhs = [1, 3, 2, 5].map(Fr::from);
        let rhs = [7, 11, 13, 17].map(Fr::from);
        let relation = Hadamard::new::<Fr>(
            DenseMleTable::new(lhs.to_vec()),
            DenseMleTable::new(rhs.to_vec()),
        );
        let mut sumcheck = SumCheck::<Fr, _, 3>::new(&eq_challenges, relation);
        let previous_round = scalar_round_for_initial_claim(&params, &sumcheck, &mut rng);

        let mut prover_transcript = Blake2bTranscript::default();
        let proof = sumcheck
            .prove(
                &params,
                &previous_round,
                Fr::from(0_u64),
                &mut prover_transcript,
                &mut rng,
            )
            .expect("valid previous round consistency")
            .proof;

        assert_eq!(proof.round_commitments.len(), eq_challenges.len());
        assert_eq!(proof.consistency_proofs.len(), eq_challenges.len());
        assert!(proof
            .round_commitments
            .iter()
            .all(|commitments| commitments.len() == 4));

        let mut verifier_transcript = Blake2bTranscript::default();
        let challenges = SumCheck::<Fr, Hadamard<DenseMleTable<Fr>, DenseMleTable<Fr>>, 3>::verify(
            &params,
            &proof,
            &previous_round.commitments,
            Fr::from(0_u64),
            eq_challenges.len(),
            &mut verifier_transcript,
        )
        .expect("valid commitment-only proof shape");

        assert_eq!(challenges.len(), eq_challenges.len());
        assert!(sumcheck.is_complete());
    }

    #[test]
    fn verifier_rejects_bad_committed_round_consistency() {
        let mut rng = ChaCha20Rng::from_seed([17; 32]);
        let params = PedersenParams::random(&mut rng);
        let eq_challenges = [
            <Fr as JoltField>::Challenge::from(2_u128),
            <Fr as JoltField>::Challenge::from(5_u128),
        ];
        let lhs = [1, 3, 2, 5].map(Fr::from);
        let rhs = [7, 11, 13, 17].map(Fr::from);
        let relation = Hadamard::new::<Fr>(
            DenseMleTable::new(lhs.to_vec()),
            DenseMleTable::new(rhs.to_vec()),
        );
        let mut sumcheck = SumCheck::<Fr, _, 3>::new(&eq_challenges, relation);
        let previous_round = scalar_round_for_initial_claim(&params, &sumcheck, &mut rng);

        let mut prover_transcript = Blake2bTranscript::default();
        let mut proof = sumcheck
            .prove(
                &params,
                &previous_round,
                Fr::from(0_u64),
                &mut prover_transcript,
                &mut rng,
            )
            .expect("valid previous round consistency")
            .proof;
        proof.consistency_proofs[0]
            .equality
            .blinding_difference_response += Fr::from(1_u64);

        let mut verifier_transcript = Blake2bTranscript::default();
        assert!(
            SumCheck::<Fr, Hadamard<DenseMleTable<Fr>, DenseMleTable<Fr>>, 3>::verify(
                &params,
                &proof,
                &previous_round.commitments,
                Fr::from(0_u64),
                eq_challenges.len(),
                &mut verifier_transcript,
            )
            .is_none()
        );
    }

    fn scalar_round_for_initial_claim<R, Rel, const LANES: usize>(
        params: &PedersenParams,
        sumcheck: &SumCheck<Fr, Rel, LANES>,
        rng: &mut R,
    ) -> CommittedRoundPoly
    where
        R: rand_core::CryptoRngCore,
        Rel: RoundRelation<Fr, LANES> + Sync,
    {
        let poly = sumcheck.round_poly();
        let opening = Opening {
            value: poly.evaluate(Fr::zero()) + poly.evaluate(Fr::one()),
            blinding: Fr::random(rng),
        };
        CommittedRoundPoly {
            commitments: vec![commit(params, &opening)],
            openings: vec![opening],
        }
    }
}
