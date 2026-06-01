use ark_bn254::Fr;
use ark_std::UniformRand;
use joltworks::{field::JoltField, transcripts::Transcript};
use rand_core::CryptoRngCore;

use crate::{
    committed_round::{evaluate_round_commitments, evaluate_round_opening, CommittedRoundPoly},
    gkr_util::{absorb_layer_label, evaluate_eq, evaluate_mle},
    multiplication::{prove_multiplication, verify_multiplication, MultiplicationProof},
    ops::hadamard::Hadamard,
    pedersen::{commit, Commitment, Opening, PedersenParams},
    round::DenseMleTable,
    sumcheck::{CommittedSumCheckProof, SumCheck},
};

pub(crate) const PRODUCT_ROUND_COEFFS: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductLayerProof {
    pub sumcheck_proof: CommittedSumCheckProof,
    pub lhs_at_sumcheck_point_commitment: Commitment,
    pub rhs_at_sumcheck_point_commitment: Commitment,
    pub multiplication_proof: MultiplicationProof,
}

pub(crate) struct ProductLayerClaims {
    pub sumcheck_point: Vec<<Fr as JoltField>::Challenge>,
    pub lhs_at_sumcheck_point: CommitmentOpening,
    pub rhs_at_sumcheck_point: CommitmentOpening,
}

pub(crate) struct ProductLayerVerifierClaims {
    pub sumcheck_point: Vec<<Fr as JoltField>::Challenge>,
    pub lhs_at_sumcheck_point_commitment: Commitment,
    pub rhs_at_sumcheck_point_commitment: Commitment,
}

pub(crate) struct CommitmentOpening {
    pub commitment: Commitment,
    pub opening: Opening,
}

impl CommitmentOpening {
    fn from_opening(params: &PedersenParams, opening: Opening) -> Self {
        Self {
            commitment: commit(params, &opening),
            opening,
        }
    }

    fn scalar_mul(&self, scalar: Fr) -> Self {
        Self {
            commitment: Commitment(self.commitment.0 * scalar),
            opening: Opening {
                value: self.opening.value * scalar,
                blinding: self.opening.blinding * scalar,
            },
        }
    }
}

pub(crate) fn prove_product_layer<T, R>(
    label: &'static [u8],
    params: &PedersenParams,
    lhs: &[Fr],
    rhs: &[Fr],
    claim_point: &[<Fr as JoltField>::Challenge],
    previous_round: &CommittedRoundPoly,
    transcript: &mut T,
    rng: &mut R,
) -> Option<(ProductLayerProof, ProductLayerClaims)>
where
    T: Transcript,
    R: CryptoRngCore,
{
    absorb_layer_label(label, transcript);

    let relation = Hadamard::new::<Fr>(
        DenseMleTable::new(lhs.to_vec()),
        DenseMleTable::new(rhs.to_vec()),
    );
    let mut sumcheck = SumCheck::<Fr, _, 3>::new(claim_point, relation);
    let sumcheck_output = sumcheck.prove(params, previous_round, None, transcript, rng)?;

    let sumcheck_point = sumcheck_output.challenges;
    let sumcheck_eval_challenge: Fr = (*sumcheck_point.last()?).into();
    let sumcheck_eval_round = sumcheck_output.rounds.last()?;
    let sumcheck_eval_at_point_commitment =
        evaluate_round_commitments(&sumcheck_eval_round.commitments, sumcheck_eval_challenge);
    let sumcheck_eval_at_point_opening =
        evaluate_round_opening(sumcheck_eval_round, sumcheck_eval_challenge);

    let lhs_at_sumcheck_point = CommitmentOpening::from_opening(
        params,
        Opening {
            value: evaluate_mle(lhs, &sumcheck_point),
            blinding: Fr::rand(rng),
        },
    );
    let rhs_at_sumcheck_point = CommitmentOpening::from_opening(
        params,
        Opening {
            value: evaluate_mle(rhs, &sumcheck_point),
            blinding: Fr::rand(rng),
        },
    );
    let multiplication_proof = prove_product_relation_at_sumcheck_point(
        params,
        claim_point,
        &sumcheck_point,
        &lhs_at_sumcheck_point,
        &rhs_at_sumcheck_point,
        &sumcheck_eval_at_point_commitment,
        &sumcheck_eval_at_point_opening,
        transcript,
        rng,
    )?;

    Some((
        ProductLayerProof {
            sumcheck_proof: sumcheck_output.proof,
            lhs_at_sumcheck_point_commitment: lhs_at_sumcheck_point.commitment,
            rhs_at_sumcheck_point_commitment: rhs_at_sumcheck_point.commitment,
            multiplication_proof,
        },
        ProductLayerClaims {
            sumcheck_point,
            lhs_at_sumcheck_point,
            rhs_at_sumcheck_point,
        },
    ))
}

pub(crate) fn verify_product_layer<T>(
    label: &'static [u8],
    params: &PedersenParams,
    previous_commitments: &[Commitment],
    claim_point: &[<Fr as JoltField>::Challenge],
    proof: &ProductLayerProof,
    num_rounds: usize,
    transcript: &mut T,
) -> Option<ProductLayerVerifierClaims>
where
    T: Transcript,
{
    if num_rounds == 0 || previous_commitments.is_empty() {
        return None;
    }

    absorb_layer_label(label, transcript);
    let challenges = SumCheck::<Fr, Hadamard<DenseMleTable<Fr>, DenseMleTable<Fr>>, 3>::verify(
        params,
        &proof.sumcheck_proof,
        previous_commitments,
        None,
        claim_point,
        num_rounds,
        transcript,
    )?;

    let sumcheck_eval_challenge: Fr = (*challenges.last()?).into();
    let sumcheck_eval_at_point_commitment = evaluate_round_commitments(
        proof.sumcheck_proof.round_commitments.last()?,
        sumcheck_eval_challenge,
    );
    if !verify_product_relation_at_sumcheck_point(
        params,
        claim_point,
        &challenges,
        &proof.lhs_at_sumcheck_point_commitment,
        &proof.rhs_at_sumcheck_point_commitment,
        &sumcheck_eval_at_point_commitment,
        &proof.multiplication_proof,
        transcript,
    ) {
        return None;
    }

    Some(ProductLayerVerifierClaims {
        sumcheck_point: challenges,
        lhs_at_sumcheck_point_commitment: proof.lhs_at_sumcheck_point_commitment,
        rhs_at_sumcheck_point_commitment: proof.rhs_at_sumcheck_point_commitment,
    })
}

fn prove_product_relation_at_sumcheck_point<T, R>(
    params: &PedersenParams,
    claim_point: &[<Fr as JoltField>::Challenge],
    sumcheck_point: &[<Fr as JoltField>::Challenge],
    lhs_at_sumcheck_point: &CommitmentOpening,
    rhs_at_sumcheck_point: &CommitmentOpening,
    sumcheck_eval_at_point_commitment: &Commitment,
    sumcheck_eval_at_point_opening: &Opening,
    transcript: &mut T,
    rng: &mut R,
) -> Option<MultiplicationProof>
where
    T: Transcript,
    R: CryptoRngCore,
{
    let eq_at_sumcheck_point = evaluate_eq(claim_point, sumcheck_point);
    let binary_product_lhs = lhs_at_sumcheck_point.scalar_mul(eq_at_sumcheck_point);
    prove_multiplication(
        params,
        &binary_product_lhs.commitment,
        &rhs_at_sumcheck_point.commitment,
        sumcheck_eval_at_point_commitment,
        &binary_product_lhs.opening,
        &rhs_at_sumcheck_point.opening,
        sumcheck_eval_at_point_opening,
        transcript,
        rng,
    )
}

fn verify_product_relation_at_sumcheck_point<T>(
    params: &PedersenParams,
    claim_point: &[<Fr as JoltField>::Challenge],
    sumcheck_point: &[<Fr as JoltField>::Challenge],
    lhs_at_sumcheck_point_commitment: &Commitment,
    rhs_at_sumcheck_point_commitment: &Commitment,
    sumcheck_eval_at_point_commitment: &Commitment,
    proof: &MultiplicationProof,
    transcript: &mut T,
) -> bool
where
    T: Transcript,
{
    let eq_at_sumcheck_point = evaluate_eq(claim_point, sumcheck_point);
    let binary_product_lhs_commitment =
        Commitment(lhs_at_sumcheck_point_commitment.0 * eq_at_sumcheck_point);
    verify_multiplication(
        params,
        &binary_product_lhs_commitment,
        rhs_at_sumcheck_point_commitment,
        sumcheck_eval_at_point_commitment,
        proof,
        transcript,
    )
}
