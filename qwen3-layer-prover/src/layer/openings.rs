use std::{collections::BTreeMap, marker::PhantomData, time::Instant};

use common::CommittedPoly;
use jolt_atlas_core::onnx_proof::ReducedOpeningProof;
use joltworks::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{
            OpeningId, Openings, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
        },
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

use crate::{ProverError, claim::Claim, error::Result};

use super::{tensors::round_site, types::LayerClaims};

/// PCS proof for the claims left unresolved by the layer IOP.
///
/// The layer IOP returns structured `LayerClaims`. We keep that structure until
/// this module so the field name can determine the corresponding
/// `CommittedPoly` and `SumcheckId`; `Claim` itself stays a pure polynomial
/// evaluation claim.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LayerOpeningReductionProof<F, T, PCS>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    pub(crate) opening_claims: Openings<F>,
    pub(crate) proof: Option<ReducedOpeningProof<F, T, PCS>>,
    pub(crate) _marker: PhantomData<(F, T, PCS)>,
}

#[derive(Debug, Clone)]
struct LayerOpeningGroup<F: JoltField, C> {
    claims: Vec<Claim<F, C>>,
    committed_polys: Vec<CommittedPoly>,
    sumcheck: SumcheckId,
}

pub(crate) fn prove_layer_openings<F, T, PCS, C>(
    hidden_out: &Claim<F, C>,
    claims: &LayerClaims<F, C>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> Result<LayerOpeningReductionProof<F, T, PCS>>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
    C: Clone,
{
    let opening_groups = layer_opening_groups(claims);
    let mut accumulator = ProverOpeningAccumulator::new();
    let mut poly_map = BTreeMap::new();

    let t0 = Instant::now();
    append_dense_claim(
        &mut accumulator,
        &mut poly_map,
        hidden_out,
        CommittedPoly::QwenLayerTensor(0),
        SumcheckId::NodeExecution(0),
        transcript,
    );
    append_dense_claim(
        &mut accumulator,
        &mut poly_map,
        &claims.hidden_in_a,
        CommittedPoly::QwenLayerTensor(1),
        SumcheckId::NodeExecution(1),
        transcript,
    );
    append_dense_claim(
        &mut accumulator,
        &mut poly_map,
        &claims.hidden_in_b,
        CommittedPoly::QwenLayerTensor(1),
        SumcheckId::NodeExecution(2),
        transcript,
    );

    // The op provers already appended these opening values to the real layer
    // transcript when the claims were created.  We rebuild the accumulator here
    // only for core's opening-reduction machinery, so append_sparse must write
    // to a throwaway transcript to avoid binding the same claims twice.
    let mut ignored_transcript = transcript.clone();
    for group in opening_groups {
        append_sparse_group(
            &mut accumulator,
            &mut poly_map,
            group,
            &mut ignored_transcript,
        )?;
    }
    eprintln!(
        "timing: prove_layer.openings.collect_claims {:.3}s",
        t0.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();
    let proof = jolt_atlas_core::opening_reduction::prove_reduced_openings::<F, T, PCS>(
        &mut accumulator,
        &poly_map,
        setup,
        transcript,
    );
    eprintln!(
        "timing: prove_layer.openings.reduce_and_pcs {:.3}s",
        t0.elapsed().as_secs_f64()
    );
    let opening_claims = accumulator.take();

    Ok(LayerOpeningReductionProof {
        opening_claims,
        proof,
        _marker: PhantomData,
    })
}

#[allow(dead_code)]
pub(crate) fn verify_layer_openings<F, T, PCS>(
    hidden_out: &Claim<F, PCS::Commitment>,
    claims: &LayerClaims<F, PCS::Commitment>,
    proof: &LayerOpeningReductionProof<F, T, PCS>,
    setup: &PCS::VerifierSetup,
    transcript: &mut T,
) -> std::result::Result<(), joltworks::utils::errors::ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let reduced_proof = proof
        .proof
        .as_ref()
        .ok_or(ProofVerifyError::MissingReductionProof)?;
    let opening_groups = layer_opening_groups(claims);
    let mut accumulator = VerifierOpeningAccumulator::new();
    accumulator.openings = proof.opening_claims.clone();
    let mut commitments_by_id = BTreeMap::new();

    append_dense_verifier::<F, T, PCS>(
        &mut accumulator,
        &mut commitments_by_id,
        &proof.opening_claims,
        hidden_out,
        CommittedPoly::QwenLayerTensor(0),
        SumcheckId::NodeExecution(0),
        transcript,
    )?;
    append_dense_verifier::<F, T, PCS>(
        &mut accumulator,
        &mut commitments_by_id,
        &proof.opening_claims,
        &claims.hidden_in_a,
        CommittedPoly::QwenLayerTensor(1),
        SumcheckId::NodeExecution(1),
        transcript,
    )?;
    append_dense_verifier::<F, T, PCS>(
        &mut accumulator,
        &mut commitments_by_id,
        &proof.opening_claims,
        &claims.hidden_in_b,
        CommittedPoly::QwenLayerTensor(1),
        SumcheckId::NodeExecution(2),
        transcript,
    )?;

    // The RA opening values were already bound by the op-level SHOUT IOP.
    // Rebuilding the opening accumulator must therefore use a throwaway
    // transcript, matching `prove_layer_openings`.
    let mut ignored_transcript = transcript.clone();
    for group in opening_groups {
        append_sparse_verifier::<F, T, PCS>(
            &mut accumulator,
            &mut commitments_by_id,
            &proof.opening_claims,
            group,
            &mut ignored_transcript,
        )?;
    }

    let commitments = commitments_by_id
        .into_values()
        .collect::<Vec<PCS::Commitment>>();
    jolt_atlas_core::opening_reduction::verify_reduced_openings::<F, T, PCS>(
        &mut accumulator,
        &commitments,
        reduced_proof,
        setup,
        transcript,
    )
}

fn append_dense_claim<F, T, C>(
    accumulator: &mut ProverOpeningAccumulator<F>,
    poly_map: &mut BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
    claim: &Claim<F, C>,
    committed_poly: CommittedPoly,
    sumcheck: SumcheckId,
    transcript: &mut T,
) where
    F: JoltField,
    T: Transcript,
{
    let opening_id = OpeningId::new(committed_poly, sumcheck);
    accumulator.append_dense(transcript, opening_id, claim.point.clone(), claim.value);
    poly_map
        .entry(committed_poly)
        .or_insert_with(|| claim.poly.data.clone());
}

#[allow(dead_code)]
fn append_dense_verifier<F, T, PCS>(
    accumulator: &mut VerifierOpeningAccumulator<F>,
    commitments_by_id: &mut BTreeMap<OpeningId, PCS::Commitment>,
    proof_openings: &Openings<F>,
    claim: &Claim<F, PCS::Commitment>,
    committed_poly: CommittedPoly,
    sumcheck: SumcheckId,
    transcript: &mut T,
) -> std::result::Result<(), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    let opening_id = OpeningId::new(committed_poly, sumcheck);
    require_proof_claim(proof_openings, opening_id, claim.value)?;
    commitments_by_id.insert(opening_id, require_commitment(claim)?);
    accumulator.append_dense(transcript, opening_id, claim.point.clone());
    Ok(())
}

fn append_sparse_group<F, T, C>(
    accumulator: &mut ProverOpeningAccumulator<F>,
    poly_map: &mut BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
    group: LayerOpeningGroup<F, C>,
    transcript: &mut T,
) -> Result<()>
where
    F: JoltField,
    T: Transcript,
{
    if group.claims.is_empty() {
        return Ok(());
    }
    if group.claims.len() != group.committed_polys.len() {
        return Err(ProverError::InvalidInput(format!(
            "opening group size mismatch: claims {} committed_polys {}",
            group.claims.len(),
            group.committed_polys.len()
        )));
    }

    let (r_address, r_cycle) = split_onehot_point(&group.claims[0])?;
    let opening_point = [r_address.as_slice(), r_cycle.as_slice()].concat();
    for claim in &group.claims {
        if claim.point != opening_point {
            return Err(ProverError::InvalidInput(
                "all sparse openings in a group must share one point".to_string(),
            ));
        }
    }

    let claims = group
        .claims
        .iter()
        .map(|claim| claim.value)
        .collect::<Vec<_>>();
    for (committed_poly, claim) in group.committed_polys.iter().zip(group.claims.into_iter()) {
        poly_map
            .entry(*committed_poly)
            .or_insert_with(|| claim.poly.data);
    }

    accumulator.append_sparse(
        transcript,
        group.committed_polys,
        group.sumcheck,
        r_address,
        r_cycle,
        claims,
    );
    Ok(())
}

#[allow(dead_code)]
fn append_sparse_verifier<F, T, PCS>(
    accumulator: &mut VerifierOpeningAccumulator<F>,
    commitments_by_id: &mut BTreeMap<OpeningId, PCS::Commitment>,
    proof_openings: &Openings<F>,
    group: LayerOpeningGroup<F, PCS::Commitment>,
    transcript: &mut T,
) -> std::result::Result<(), ProofVerifyError>
where
    F: JoltField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    if group.claims.is_empty() {
        return Ok(());
    }
    if group.claims.len() != group.committed_polys.len() {
        return Err(ProofVerifyError::InvalidOpeningProof(format!(
            "opening group size mismatch: claims {} committed_polys {}",
            group.claims.len(),
            group.committed_polys.len()
        )));
    }

    let opening_point = group.claims[0].point.clone();
    for claim in &group.claims {
        if claim.point != opening_point {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "all sparse openings in a group must share one point".to_string(),
            ));
        }
    }

    for (committed_poly, claim) in group.committed_polys.iter().zip(group.claims.iter()) {
        let opening_id = OpeningId::new(*committed_poly, group.sumcheck);
        require_proof_claim(proof_openings, opening_id, claim.value)?;
        commitments_by_id.insert(opening_id, require_commitment(claim)?);
    }
    accumulator.append_sparse(
        transcript,
        group.committed_polys,
        group.sumcheck,
        opening_point,
    );
    Ok(())
}

#[allow(dead_code)]
fn require_proof_claim<F: JoltField>(
    proof_openings: &Openings<F>,
    opening_id: OpeningId,
    expected: F,
) -> std::result::Result<(), ProofVerifyError> {
    let (_, actual) = proof_openings.get(&opening_id).ok_or_else(|| {
        ProofVerifyError::InvalidOpeningProof(format!("missing opening claim for {opening_id:?}"))
    })?;
    if *actual != expected {
        return Err(ProofVerifyError::InvalidOpeningProof(format!(
            "opening claim mismatch for {opening_id:?}"
        )));
    }
    Ok(())
}

#[allow(dead_code)]
fn require_commitment<F, C>(claim: &Claim<F, C>) -> std::result::Result<C, ProofVerifyError>
where
    F: JoltField,
    C: Clone,
{
    claim.poly.commitment.clone().ok_or_else(|| {
        ProofVerifyError::InvalidOpeningProof("missing commitment for opened claim".to_string())
    })
}

fn split_onehot_point<F: JoltField, C>(claim: &Claim<F, C>) -> Result<(Vec<F>, Vec<F>)> {
    let log_k = match &claim.poly.data {
        MultilinearPolynomial::OneHot(onehot) => onehot.K.trailing_zeros() as usize,
        _ => {
            return Err(ProverError::InvalidInput(
                "sparse opening reduction expected OneHot polynomial".to_string(),
            ));
        }
    };
    if claim.point.len() < log_k {
        return Err(ProverError::InvalidInput(format!(
            "one-hot opening point too short: len {} log_k {}",
            claim.point.len(),
            log_k
        )));
    }
    let (r_address, r_cycle) = claim.point.split_at(log_k);
    Ok((r_address.to_vec(), r_cycle.to_vec()))
}

fn layer_opening_groups<F: JoltField, C: Clone>(
    claims: &LayerClaims<F, C>,
) -> Vec<LayerOpeningGroup<F, C>> {
    let mut out = Vec::new();

    push_round_claims(&mut out, &claims.down_proj_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::DOWN_PROJ, d)
    });
    push_round_claims(&mut out, &claims.silu_up_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::SILU_UP, d)
    });
    push_round_claims(
        &mut out,
        &claims.silu_gate_round_ra,
        CommittedPoly::QwenSiluRoundRaD,
    );
    push_lookup_claims(&mut out, &claims.silu_ra, |d| {
        let split = claims.silu_ra.len() / 2;
        if d < split {
            CommittedPoly::QwenSiluBaseRaD(d)
        } else {
            CommittedPoly::QwenSiluSlopeRaD(d - split)
        }
    });
    push_round_claims(
        &mut out,
        &claims.silu_round_ra,
        CommittedPoly::QwenSiluOutputRoundRaD,
    );
    push_round_claims(&mut out, &claims.gate_proj_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::GATE_PROJ, d)
    });
    push_round_claims(&mut out, &claims.up_proj_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::UP_PROJ, d)
    });
    push_round_claims(&mut out, &claims.rms_norm_mlp_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::RMS_NORM_MLP, d)
    });
    push_round_claims(&mut out, &claims.rms_norm_mlp_norm_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::RMS_NORM_MLP_INTERNAL, d)
    });
    push_round_claims(&mut out, &claims.o_proj_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::O_PROJ, d)
    });
    push_round_claims(&mut out, &claims.pv_matmul_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::CONTEXT, d)
    });
    push_round_claims(&mut out, &claims.softmax_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::SOFTMAX_OUTPUT, d)
    });
    push_round_claims(&mut out, &claims.softmax_floor_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::SOFTMAX_FLOOR, d)
    });
    push_round_claims(&mut out, &claims.softmax_exp_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::SOFTMAX_EXP, d)
    });
    push_lookup_claims(
        &mut out,
        &claims.softmax_input_frac_ra,
        CommittedPoly::QwenSoftmaxInputFracRaD,
    );
    push_lookup_claims(
        &mut out,
        &claims.softmax_ra,
        CommittedPoly::QwenSoftmaxExpRaD,
    );
    push_round_claims(&mut out, &claims.qk_score_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::QK_SCORE_SCALE, d)
    });
    push_round_claims(&mut out, &claims.qk_score_dot_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::QK_SCORE_DOT, d)
    });
    push_round_claims(&mut out, &claims.q_rope_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::Q_ROPE, d)
    });
    push_round_claims(&mut out, &claims.k_rope_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::K_ROPE, d)
    });
    push_round_claims(&mut out, &claims.q_norm_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::Q_NORM, d)
    });
    push_round_claims(&mut out, &claims.q_norm_norm_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::Q_NORM_INTERNAL, d)
    });
    push_round_claims(&mut out, &claims.k_norm_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::K_NORM, d)
    });
    push_round_claims(&mut out, &claims.k_norm_norm_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::K_NORM_INTERNAL, d)
    });
    push_round_claims(&mut out, &claims.q_proj_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::Q_PROJ, d)
    });
    push_round_claims(&mut out, &claims.k_proj_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::K_PROJ, d)
    });
    push_round_claims(&mut out, &claims.v_proj_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::V_PROJ, d)
    });
    push_round_claims(&mut out, &claims.rms_norm_atten_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::RMS_NORM_ATTEN, d)
    });
    push_round_claims(&mut out, &claims.rms_norm_atten_norm_round_ra, |d| {
        CommittedPoly::QwenRoundRaD(round_site::RMS_NORM_ATTEN_INTERNAL, d)
    });

    out
}

fn push_round_claims<F, C>(
    out: &mut Vec<LayerOpeningGroup<F, C>>,
    claims: &[Claim<F, C>],
    committed_poly: impl Fn(usize) -> CommittedPoly,
) where
    F: JoltField,
    C: Clone,
{
    push_shout_claims(out, claims, committed_poly);
}

fn push_lookup_claims<F, C>(
    out: &mut Vec<LayerOpeningGroup<F, C>>,
    claims: &[Claim<F, C>],
    committed_poly: impl Fn(usize) -> CommittedPoly,
) where
    F: JoltField,
    C: Clone,
{
    push_shout_claims(out, claims, committed_poly);
}

fn push_shout_claims<F, C>(
    out: &mut Vec<LayerOpeningGroup<F, C>>,
    claims: &[Claim<F, C>],
    committed_poly: impl Fn(usize) -> CommittedPoly,
) where
    F: JoltField,
    C: Clone,
{
    if claims.is_empty() {
        return;
    }

    let chunk_count = infer_shout_chunk_count(claims.len());
    let sumchecks = match claims.len() / chunk_count {
        1 => [SumcheckId::HammingWeight].as_slice(),
        3 => [
            SumcheckId::RaVirtualization,
            SumcheckId::HammingWeight,
            SumcheckId::Booleanity,
        ]
        .as_slice(),
        _ => panic!("unexpected SHOUT opening claim count: {}", claims.len()),
    };

    for (group, sumcheck) in sumchecks.iter().enumerate() {
        for d in 0..chunk_count {
            let claim = claims[group * chunk_count + d].clone();
            out.push(LayerOpeningGroup {
                claims: vec![claim],
                committed_polys: vec![committed_poly(d)],
                sumcheck: *sumcheck,
            });
        }
    }
}

fn infer_shout_chunk_count(claim_count: usize) -> usize {
    if claim_count % 3 == 0 {
        claim_count / 3
    } else {
        claim_count
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr};
    use joltworks::{
        poly::{
            commitment::{commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG},
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            one_hot_polynomial::OneHotPolynomial,
        },
        transcripts::Blake2bTranscript,
    };

    use crate::{
        claim::{Claim, Poly},
        layer::types::LayerClaims,
    };

    use super::{prove_layer_openings, verify_layer_openings};

    #[test]
    fn verifies_dense_and_sparse_layer_opening_reduction() {
        type Pcs = HyperKZG<Bn254>;

        let setup = Pcs::setup_prover(5);
        let verifier_setup = Pcs::setup_verifier(&setup);
        let hidden_out = claim_with_commitment::<Pcs>(
            MultilinearPolynomial::from(vec![1_i32, 2, 3, 4]),
            vec![Fr::from(2_u64), Fr::from(3_u64)],
            &setup,
        );
        let hidden_in_a = claim_with_commitment::<Pcs>(
            MultilinearPolynomial::from(vec![4_i32, 3, 2, 1]),
            vec![Fr::from(5_u64), Fr::from(7_u64)],
            &setup,
        );
        let hidden_in_b = claim_with_commitment::<Pcs>(
            MultilinearPolynomial::from(vec![4_i32, 3, 2, 1]),
            vec![Fr::from(11_u64), Fr::from(13_u64)],
            &setup,
        );
        let mut claims = empty_claims(hidden_in_a, hidden_in_b);
        claims.q_proj_round_ra = vec![onehot_claim_with_commitment::<Pcs>(
            vec![Some(0), Some(1), Some(2), Some(3)],
            4,
            vec![
                Fr::from(2_u64),
                Fr::from(3_u64),
                Fr::from(5_u64),
                Fr::from(7_u64),
            ],
            &setup,
        )];

        let mut prover_transcript = Blake2bTranscript::default();
        let proof = prove_layer_openings::<Fr, _, Pcs, _>(
            &hidden_out,
            &claims,
            &setup,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = Blake2bTranscript::default();
        verify_layer_openings::<Fr, _, Pcs>(
            &hidden_out,
            &claims,
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
        )
        .unwrap();
    }

    fn claim_with_commitment<PCS>(
        poly: MultilinearPolynomial<Fr>,
        point: Vec<Fr>,
        setup: &PCS::ProverSetup,
    ) -> Claim<Fr, PCS::Commitment>
    where
        PCS: CommitmentScheme<Field = Fr>,
    {
        let value = poly.evaluate(&point);
        let (commitment, _) = PCS::commit(&poly, setup);
        Claim::new(Poly::new(poly, Some(commitment)), point, value)
    }

    fn onehot_claim_with_commitment<PCS>(
        indices: Vec<Option<u16>>,
        k: usize,
        point: Vec<Fr>,
        setup: &PCS::ProverSetup,
    ) -> Claim<Fr, PCS::Commitment>
    where
        PCS: CommitmentScheme<Field = Fr>,
    {
        let poly = MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(indices, k));
        claim_with_commitment::<PCS>(poly, point, setup)
    }

    fn empty_claims<C: Clone>(
        hidden_in_a: Claim<Fr, C>,
        hidden_in_b: Claim<Fr, C>,
    ) -> LayerClaims<Fr, C> {
        LayerClaims {
            hidden_in_a,
            hidden_in_b,
            direct_eval_claims: Vec::new(),
            down_proj_round_ra: Vec::new(),
            silu_up_round_ra: Vec::new(),
            silu_gate_round_ra: Vec::new(),
            silu_ra: Vec::new(),
            silu_round_ra: Vec::new(),
            gate_proj_round_ra: Vec::new(),
            up_proj_round_ra: Vec::new(),
            rms_norm_mlp_round_ra: Vec::new(),
            rms_norm_mlp_norm_round_ra: Vec::new(),
            o_proj_round_ra: Vec::new(),
            pv_matmul_round_ra: Vec::new(),
            softmax_round_ra: Vec::new(),
            softmax_floor_round_ra: Vec::new(),
            softmax_exp_round_ra: Vec::new(),
            softmax_input_frac_ra: Vec::new(),
            softmax_ra: Vec::new(),
            qk_score_round_ra: Vec::new(),
            qk_score_dot_round_ra: Vec::new(),
            q_rope_round_ra: Vec::new(),
            k_rope_round_ra: Vec::new(),
            q_norm_round_ra: Vec::new(),
            q_norm_norm_round_ra: Vec::new(),
            k_norm_round_ra: Vec::new(),
            k_norm_norm_round_ra: Vec::new(),
            q_proj_round_ra: Vec::new(),
            k_proj_round_ra: Vec::new(),
            v_proj_round_ra: Vec::new(),
            rms_norm_atten_round_ra: Vec::new(),
            rms_norm_atten_norm_round_ra: Vec::new(),
        }
    }
}
