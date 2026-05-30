//! A small committed GKR composition for three product gates.
//!
//! The circuit shape is fixed:
//!
//! ```text
//! E = A * B
//! F = C * D
//! G = E * F
//! ```
//!
//! For each product layer, the prover runs a committed sumcheck for
//! `sum_x eq_R(x) lhs(x) rhs(x) = out(R)`. The incoming scalar claim is treated
//! as a degree-0 round polynomial, so the first-round link and all adjacent
//! round links use the same round-consistency proof.

use ark_bn254::Fr;
use ark_std::UniformRand;
use joltworks::{field::JoltField, transcripts::Transcript};
use rand_core::CryptoRngCore;

use crate::{
    committed_round::{
        evaluate_round_commitments, evaluate_round_opening, scalar_round_poly, CommittedRoundPoly,
    },
    gkr_util::{
        absorb_gkr_statement, absorb_layer_label, evaluate_eq, evaluate_mle, hadamard_values,
        reversed_challenges, validate_tables,
    },
    multiplication::{prove_multiplication, verify_multiplication, MultiplicationProof},
    ops::hadamard::Hadamard,
    pedersen::{commit, Commitment, Opening, PedersenParams},
    round::DenseMleTable,
    sumcheck::{CommittedSumCheckProof, SumCheck},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductLayerProof {
    pub sumcheck_proof: CommittedSumCheckProof,
    pub left_eval_commitment: Commitment,
    pub right_eval_commitment: Commitment,
    pub multiplication_proof: MultiplicationProof,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreeProductGkrProof {
    pub output_commitment: Commitment,
    pub top: ProductLayerProof,
    pub left: ProductLayerProof,
    pub right: ProductLayerProof,
}

struct ProductLayerProverOutput {
    proof: ProductLayerProof,
    challenges: Vec<<Fr as JoltField>::Challenge>,
    left_eval_opening: Opening,
    right_eval_opening: Opening,
}

pub fn prove_three_product_gkr<T, R>(
    params: &PedersenParams,
    a: &[Fr],
    b: &[Fr],
    c: &[Fr],
    d: &[Fr],
    output_point: &[<Fr as JoltField>::Challenge],
    transcript: &mut T,
    rng: &mut R,
) -> Option<ThreeProductGkrProof>
where
    T: Transcript,
    R: CryptoRngCore,
{
    validate_tables(output_point.len(), &[a, b, c, d])?;

    let e = hadamard_values(a, b);
    let f = hadamard_values(c, d);
    let g = hadamard_values(&e, &f);

    let output_opening = Opening {
        value: evaluate_mle(&g, output_point),
        blinding: Fr::rand(rng),
    };
    let output_commitment = commit(params, &output_opening);
    let output_round = scalar_round_poly(output_commitment, output_opening);
    absorb_gkr_statement(params, output_point, &output_commitment, transcript);

    let top_eq_point = reversed_challenges(output_point);
    let top = prove_product_layer(
        b"top",
        params,
        &e,
        &f,
        &top_eq_point,
        &output_round,
        Fr::from(0_u64),
        transcript,
        rng,
    )?;

    let child_eq_point = reversed_challenges(&top.challenges);
    let top_left_round = scalar_round_poly(top.proof.left_eval_commitment, top.left_eval_opening);
    let left = prove_product_layer(
        b"left",
        params,
        a,
        b,
        &child_eq_point,
        &top_left_round,
        Fr::from(0_u64),
        transcript,
        rng,
    )?;

    let top_right_round =
        scalar_round_poly(top.proof.right_eval_commitment, top.right_eval_opening);
    let right = prove_product_layer(
        b"right",
        params,
        c,
        d,
        &child_eq_point,
        &top_right_round,
        Fr::from(0_u64),
        transcript,
        rng,
    )?;

    Some(ThreeProductGkrProof {
        output_commitment,
        top: top.proof,
        left: left.proof,
        right: right.proof,
    })
}

pub fn verify_three_product_gkr<T>(
    params: &PedersenParams,
    proof: &ThreeProductGkrProof,
    num_vars: usize,
    output_point: &[<Fr as JoltField>::Challenge],
    transcript: &mut T,
) -> bool
where
    T: Transcript,
{
    if output_point.len() != num_vars || num_vars == 0 {
        return false;
    }

    absorb_gkr_statement(params, output_point, &proof.output_commitment, transcript);

    let top_eq_point = reversed_challenges(output_point);
    let Some(top_challenges) = verify_product_layer(
        b"top",
        params,
        &[proof.output_commitment],
        Fr::from(0_u64),
        &top_eq_point,
        &proof.top,
        num_vars,
        transcript,
    ) else {
        return false;
    };

    let child_eq_point = reversed_challenges(&top_challenges);
    if verify_product_layer(
        b"left",
        params,
        &[proof.top.left_eval_commitment],
        Fr::from(0_u64),
        &child_eq_point,
        &proof.left,
        top_challenges.len(),
        transcript,
    )
    .is_none()
    {
        return false;
    }

    verify_product_layer(
        b"right",
        params,
        &[proof.top.right_eval_commitment],
        Fr::from(0_u64),
        &child_eq_point,
        &proof.right,
        top_challenges.len(),
        transcript,
    )
    .is_some()
}

fn prove_product_layer<T, R>(
    label: &'static [u8],
    params: &PedersenParams,
    lhs: &[Fr],
    rhs: &[Fr],
    eq_point: &[<Fr as JoltField>::Challenge],
    previous_round: &CommittedRoundPoly,
    previous_challenge: Fr,
    transcript: &mut T,
    rng: &mut R,
) -> Option<ProductLayerProverOutput>
where
    T: Transcript,
    R: CryptoRngCore,
{
    absorb_layer_label(label, transcript);

    let relation = Hadamard::new::<Fr>(
        DenseMleTable::new(lhs.to_vec()),
        DenseMleTable::new(rhs.to_vec()),
    );
    let mut sumcheck = SumCheck::<Fr, _, 3>::new(eq_point, relation);
    let sumcheck_output =
        sumcheck.prove(params, previous_round, previous_challenge, transcript, rng)?;

    let final_challenge: Fr = (*sumcheck_output.challenges.last()?).into();
    let final_round = sumcheck_output.rounds.last()?;
    let output_eval_opening = evaluate_round_opening(final_round, final_challenge);
    let output_eval_commitment =
        evaluate_round_commitments(&final_round.commitments, final_challenge);

    let left_eval_opening = Opening {
        value: evaluate_mle(lhs, &sumcheck_output.challenges),
        blinding: Fr::rand(rng),
    };
    let right_eval_opening = Opening {
        value: evaluate_mle(rhs, &sumcheck_output.challenges),
        blinding: Fr::rand(rng),
    };
    let left_eval_commitment = commit(params, &left_eval_opening);
    let right_eval_commitment = commit(params, &right_eval_opening);
    let eq_eval = evaluate_eq(eq_point, &sumcheck_output.challenges);
    let scaled_left_eval_opening = Opening {
        value: left_eval_opening.value * eq_eval,
        blinding: left_eval_opening.blinding * eq_eval,
    };
    let scaled_left_eval_commitment = Commitment(left_eval_commitment.0 * eq_eval);
    let multiplication_proof = prove_multiplication(
        params,
        &scaled_left_eval_commitment,
        &right_eval_commitment,
        &output_eval_commitment,
        &scaled_left_eval_opening,
        &right_eval_opening,
        &output_eval_opening,
        transcript,
        rng,
    )?;

    Some(ProductLayerProverOutput {
        proof: ProductLayerProof {
            sumcheck_proof: sumcheck_output.proof,
            left_eval_commitment,
            right_eval_commitment,
            multiplication_proof,
        },
        challenges: sumcheck_output.challenges,
        left_eval_opening,
        right_eval_opening,
    })
}

fn verify_product_layer<T>(
    label: &'static [u8],
    params: &PedersenParams,
    previous_commitments: &[Commitment],
    previous_challenge: Fr,
    eq_point: &[<Fr as JoltField>::Challenge],
    proof: &ProductLayerProof,
    num_rounds: usize,
    transcript: &mut T,
) -> Option<Vec<<Fr as JoltField>::Challenge>>
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
        previous_challenge,
        num_rounds,
        transcript,
    )?;

    let final_challenge: Fr = (*challenges.last()?).into();
    let output_eval_commitment = evaluate_round_commitments(
        proof.sumcheck_proof.round_commitments.last()?,
        final_challenge,
    );
    let eq_eval = evaluate_eq(eq_point, &challenges);
    let scaled_left_eval_commitment = Commitment(proof.left_eval_commitment.0 * eq_eval);
    if !verify_multiplication(
        params,
        &scaled_left_eval_commitment,
        &proof.right_eval_commitment,
        &output_eval_commitment,
        &proof.multiplication_proof,
        transcript,
    ) {
        return None;
    }

    Some(challenges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use joltworks::transcripts::Blake2bTranscript;
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    #[test]
    fn proves_three_product_gkr() {
        let mut rng = ChaCha20Rng::from_seed([30; 32]);
        let params = PedersenParams::random(&mut rng);
        let num_vars = 3;
        let len = 1 << num_vars;
        let a = random_values(len, &mut rng);
        let b = random_values(len, &mut rng);
        let c = random_values(len, &mut rng);
        let d = random_values(len, &mut rng);
        let output_point = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect::<Vec<_>>();

        let mut prover_transcript = Blake2bTranscript::default();
        let proof = prove_three_product_gkr(
            &params,
            &a,
            &b,
            &c,
            &d,
            &output_point,
            &mut prover_transcript,
            &mut rng,
        )
        .expect("valid GKR witness");

        let mut verifier_transcript = Blake2bTranscript::default();
        assert!(verify_three_product_gkr(
            &params,
            &proof,
            num_vars,
            &output_point,
            &mut verifier_transcript,
        ));
    }

    #[test]
    fn rejects_bad_top_multiplication_proof() {
        let mut rng = ChaCha20Rng::from_seed([31; 32]);
        let params = PedersenParams::random(&mut rng);
        let num_vars = 2;
        let len = 1 << num_vars;
        let a = random_values(len, &mut rng);
        let b = random_values(len, &mut rng);
        let c = random_values(len, &mut rng);
        let d = random_values(len, &mut rng);
        let output_point = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect::<Vec<_>>();

        let mut prover_transcript = Blake2bTranscript::default();
        let mut proof = prove_three_product_gkr(
            &params,
            &a,
            &b,
            &c,
            &d,
            &output_point,
            &mut prover_transcript,
            &mut rng,
        )
        .expect("valid GKR witness");
        proof.top.multiplication_proof.z_a += Fr::from(1_u64);

        let mut verifier_transcript = Blake2bTranscript::default();
        assert!(!verify_three_product_gkr(
            &params,
            &proof,
            num_vars,
            &output_point,
            &mut verifier_transcript,
        ));
    }

    fn random_values<R: CryptoRngCore>(len: usize, rng: &mut R) -> Vec<Fr> {
        (0..len).map(|_| Fr::rand(rng)).collect()
    }
}
