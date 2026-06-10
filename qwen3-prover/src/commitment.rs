use ark_bn254::{Bn254, Fr, G1Projective};
use ark_ec::{AffineRepr, CurveGroup};
use joltworks::{
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            hyperkzg::{HyperKZG, HyperKZGCommitment, HyperKZGProverKey},
        },
        multilinear_polynomial::MultilinearPolynomial,
    },
    transcripts::Transcript,
};
pub use qwen3_common::{
    ChunkedCommitments, FRAC_BITS, LayerBitCommitments, LayerCommitments, LayerHiddenCommitments,
};
use thiserror::Error;

use crate::{
    layer::{BitOpeningWitnesses, LayerOpeningWitnesses, LayerShape},
    layer_input::LayerRawWitness,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitLayerParams {
    pub pcs_domain_size: usize,
}

pub fn append_layer_commitments<T>(
    transcript: &mut T,
    commitments: &LayerCommitments<HyperKZGCommitment<Bn254>>,
) where
    T: Transcript,
{
    append_chunked_commitments(transcript, &commitments.hidden_out);
    append_chunked_commitments(transcript, &commitments.hidden_in_a);
    append_chunked_commitments(transcript, &commitments.hidden_in_b);
    append_chunked_commitments(transcript, &commitments.silu_lookup_ra);
    append_chunked_commitments(transcript, &commitments.softmax_lookup_ra);
    append_layer_bit_commitments(transcript, &commitments.bits);
}

fn append_layer_bit_commitments<T>(
    transcript: &mut T,
    commitments: &LayerBitCommitments<HyperKZGCommitment<Bn254>>,
) where
    T: Transcript,
{
    append_bit_commitments(transcript, &commitments.down_proj_output_frac_bits);
    append_bit_commitments(transcript, &commitments.silu_up_output_frac_bits);
    append_bit_commitments(transcript, &commitments.silu_input_frac_bits);
    append_bit_commitments(transcript, &commitments.silu_output_frac_bits);
    append_bit_commitments(transcript, &commitments.gate_proj_output_frac_bits);
    append_bit_commitments(transcript, &commitments.up_proj_output_frac_bits);
    append_bit_commitments(transcript, &commitments.rms_norm_mlp_norm_frac_bits);
    append_bit_commitments(transcript, &commitments.rms_norm_mlp_output_frac_bits);
    append_bit_commitments(transcript, &commitments.o_proj_output_frac_bits);
    append_bit_commitments(transcript, &commitments.pv_matmul_output_frac_bits);
    append_bit_commitments(transcript, &commitments.softmax_floor_frac_bits);
    append_bit_commitments(transcript, &commitments.softmax_output_frac_bits);
    append_bit_commitments(transcript, &commitments.softmax_exp_frac_bits);
    append_bit_commitments(transcript, &commitments.qk_score_dot_output_frac_bits);
    append_bit_commitments(transcript, &commitments.qk_score_output_frac_bits);
    append_bit_commitments(transcript, &commitments.q_rope_output_frac_bits);
    append_bit_commitments(transcript, &commitments.k_rope_output_frac_bits);
    append_bit_commitments(transcript, &commitments.q_norm_norm_frac_bits);
    append_bit_commitments(transcript, &commitments.q_norm_output_frac_bits);
    append_bit_commitments(transcript, &commitments.k_norm_norm_frac_bits);
    append_bit_commitments(transcript, &commitments.k_norm_output_frac_bits);
    append_bit_commitments(transcript, &commitments.q_proj_output_frac_bits);
    append_bit_commitments(transcript, &commitments.k_proj_output_frac_bits);
    append_bit_commitments(transcript, &commitments.v_proj_output_frac_bits);
    append_bit_commitments(transcript, &commitments.rms_norm_atten_norm_frac_bits);
    append_bit_commitments(transcript, &commitments.rms_norm_atten_output_frac_bits);
}

fn append_bit_commitments<T>(
    transcript: &mut T,
    commitments: &[ChunkedCommitments<HyperKZGCommitment<Bn254>>; FRAC_BITS],
) where
    T: Transcript,
{
    for bit_commitments in commitments {
        append_chunked_commitments(transcript, bit_commitments);
    }
}

fn append_chunked_commitments<T>(
    transcript: &mut T,
    commitments: &ChunkedCommitments<HyperKZGCommitment<Bn254>>,
) where
    T: Transcript,
{
    for commitment in &commitments.commitments {
        transcript.append_point(&commitment.0.into_group());
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CommitLayerError {
    #[error("chunk size must be non-zero and a power of two: {0}")]
    InvalidChunkSize(usize),
    #[error("address space must be non-zero and a power of two: {0}")]
    InvalidAddressSpace(usize),
    #[error("PCS domain size {domain_size} is too small for address space {address_space}")]
    InvalidRaChunkDomain {
        domain_size: usize,
        address_space: usize,
    },
    #[error("one-hot commitment failed: {0}")]
    OneHotCommitment(String),
    #[error("invalid dense RA table")]
    InvalidDenseRaTable,
}

pub fn commit_layer(
    witness: &LayerRawWitness,
    params: CommitLayerParams,
    setup: &HyperKZGProverKey<Bn254>,
) -> Result<LayerCommitments<HyperKZGCommitment<Bn254>>, CommitLayerError> {
    let silu_lookup_ra =
        selected_entries_and_address_space(&witness.silu_lookup_ra, witness.silu.len())?;
    let softmax_lookup_ra =
        selected_entries_and_address_space(&witness.softmax_lookup_ra, witness.softmax.len())?;
    Ok(LayerCommitments {
        hidden_out: commit_i32_chunks::<HyperKZG<Bn254>>(
            &witness.hidden_out,
            params.pcs_domain_size,
            setup,
        )?,
        hidden_in_a: commit_i32_chunks::<HyperKZG<Bn254>>(
            &witness.residual_add_attn_a,
            params.pcs_domain_size,
            setup,
        )?,
        hidden_in_b: commit_i32_chunks::<HyperKZG<Bn254>>(
            &witness.hidden_in,
            params.pcs_domain_size,
            setup,
        )?,
        silu_lookup_ra: commit_ra_chunks(
            &silu_lookup_ra.selected,
            params.pcs_domain_size,
            silu_lookup_ra.address_space,
            setup,
        )?,
        softmax_lookup_ra: commit_ra_chunks(
            &softmax_lookup_ra.selected,
            params.pcs_domain_size,
            softmax_lookup_ra.address_space,
            setup,
        )?,
        bits: commit_layer_bits::<HyperKZG<Bn254>>(witness, params.pcs_domain_size, setup)?,
    })
}

pub fn commit_layer_openings(
    witness: &LayerOpeningWitnesses,
    shape: LayerShape,
    params: CommitLayerParams,
    setup: &HyperKZGProverKey<Bn254>,
) -> Result<LayerCommitments<HyperKZGCommitment<Bn254>>, CommitLayerError> {
    let hidden = commit_layer_hidden_openings(witness, params, setup)?;
    commit_layer_openings_with_hidden(hidden, witness, shape, params, setup)
}

pub fn commit_layer_hidden_openings(
    witness: &LayerOpeningWitnesses,
    params: CommitLayerParams,
    setup: &HyperKZGProverKey<Bn254>,
) -> Result<LayerHiddenCommitments<HyperKZGCommitment<Bn254>>, CommitLayerError> {
    Ok(LayerHiddenCommitments {
        hidden_out: commit_i32_chunks::<HyperKZG<Bn254>>(
            &witness.hidden_out,
            params.pcs_domain_size,
            setup,
        )?,
        hidden_in_a: commit_i32_chunks::<HyperKZG<Bn254>>(
            &witness.hidden_in_a,
            params.pcs_domain_size,
            setup,
        )?,
        hidden_in_b: commit_i32_chunks::<HyperKZG<Bn254>>(
            &witness.hidden_in_b,
            params.pcs_domain_size,
            setup,
        )?,
    })
}

pub fn commit_layer_openings_with_hidden(
    hidden: LayerHiddenCommitments<HyperKZGCommitment<Bn254>>,
    witness: &LayerOpeningWitnesses,
    shape: LayerShape,
    params: CommitLayerParams,
    setup: &HyperKZGProverKey<Bn254>,
) -> Result<LayerCommitments<HyperKZGCommitment<Bn254>>, CommitLayerError> {
    let shape = shape.padded();
    let silu_tensor_len = shape
        .seq
        .checked_mul(shape.intermediate)
        .ok_or(CommitLayerError::InvalidDenseRaTable)?;
    let softmax_tensor_len = shape
        .q_heads
        .checked_mul(shape.seq)
        .and_then(|len| len.checked_mul(shape.seq))
        .ok_or(CommitLayerError::InvalidDenseRaTable)?;
    let silu_lookup_ra =
        selected_entries_and_address_space(&witness.silu_lookup_ra, silu_tensor_len)?;
    let softmax_lookup_ra =
        selected_entries_and_address_space(&witness.softmax_lookup_ra, softmax_tensor_len)?;

    Ok(LayerCommitments {
        hidden_out: hidden.hidden_out,
        hidden_in_a: hidden.hidden_in_a,
        hidden_in_b: hidden.hidden_in_b,
        silu_lookup_ra: commit_ra_chunks(
            &silu_lookup_ra.selected,
            params.pcs_domain_size,
            silu_lookup_ra.address_space,
            setup,
        )?,
        softmax_lookup_ra: commit_ra_chunks(
            &softmax_lookup_ra.selected,
            params.pcs_domain_size,
            softmax_lookup_ra.address_space,
            setup,
        )?,
        bits: commit_layer_opening_bits::<HyperKZG<Bn254>>(witness, params.pcs_domain_size, setup)?,
    })
}

fn commit_layer_opening_bits<PCS>(
    witness: &LayerOpeningWitnesses,
    chunk_size: usize,
    setup: &PCS::ProverSetup,
) -> Result<LayerBitCommitments<PCS::Commitment>, CommitLayerError>
where
    PCS: CommitmentScheme<Field = Fr>,
{
    Ok(LayerBitCommitments {
        down_proj_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.down_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        silu_up_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.silu_up_output_frac_bits,
            chunk_size,
            setup,
        )?,
        silu_input_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.silu_input_frac_bits,
            chunk_size,
            setup,
        )?,
        silu_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.silu_output_frac_bits,
            chunk_size,
            setup,
        )?,
        gate_proj_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.gate_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        up_proj_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.up_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        rms_norm_mlp_norm_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.rms_norm_mlp_norm_frac_bits,
            chunk_size,
            setup,
        )?,
        rms_norm_mlp_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.rms_norm_mlp_output_frac_bits,
            chunk_size,
            setup,
        )?,
        o_proj_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.o_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        pv_matmul_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.pv_matmul_output_frac_bits,
            chunk_size,
            setup,
        )?,
        softmax_floor_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.softmax_floor_frac_bits,
            chunk_size,
            setup,
        )?,
        softmax_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.softmax_output_frac_bits,
            chunk_size,
            setup,
        )?,
        softmax_exp_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.softmax_exp_frac_bits,
            chunk_size,
            setup,
        )?,
        qk_score_dot_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.qk_score_dot_output_frac_bits,
            chunk_size,
            setup,
        )?,
        qk_score_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.qk_score_output_frac_bits,
            chunk_size,
            setup,
        )?,
        q_rope_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.q_rope_output_frac_bits,
            chunk_size,
            setup,
        )?,
        k_rope_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.k_rope_output_frac_bits,
            chunk_size,
            setup,
        )?,
        q_norm_norm_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.q_norm_norm_frac_bits,
            chunk_size,
            setup,
        )?,
        q_norm_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.q_norm_output_frac_bits,
            chunk_size,
            setup,
        )?,
        k_norm_norm_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.k_norm_norm_frac_bits,
            chunk_size,
            setup,
        )?,
        k_norm_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.k_norm_output_frac_bits,
            chunk_size,
            setup,
        )?,
        q_proj_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.q_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        k_proj_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.k_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        v_proj_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.v_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        rms_norm_atten_norm_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.rms_norm_atten_norm_frac_bits,
            chunk_size,
            setup,
        )?,
        rms_norm_atten_output_frac_bits: commit_bool_bit_tables::<PCS>(
            &witness.rms_norm_atten_output_frac_bits,
            chunk_size,
            setup,
        )?,
    })
}

struct SelectedRa {
    selected: Vec<u16>,
    address_space: usize,
}

fn selected_entries_and_address_space(
    table: &[u8],
    tensor_len: usize,
) -> Result<SelectedRa, CommitLayerError> {
    if tensor_len == 0 || !table.len().is_multiple_of(tensor_len) {
        return Err(CommitLayerError::InvalidDenseRaTable);
    }
    let entries = table.len() / tensor_len;
    let address_space = entries.next_power_of_two();
    validate_address_space(address_space)?;
    let mut selected = vec![None; tensor_len];
    for entry in 0..entries {
        let offset = entry * tensor_len;
        for tensor_index in 0..tensor_len {
            match table[offset + tensor_index] {
                0 => {}
                1 if selected[tensor_index].is_none() && entry <= u16::MAX as usize => {
                    selected[tensor_index] = Some(entry as u16);
                }
                _ => return Err(CommitLayerError::InvalidDenseRaTable),
            }
        }
    }
    let selected = selected
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(CommitLayerError::InvalidDenseRaTable)?;
    Ok(SelectedRa {
        selected,
        address_space,
    })
}

pub fn commit_ra_chunks(
    values: &[u16],
    pcs_domain_size: usize,
    address_space: usize,
    setup: &HyperKZGProverKey<Bn254>,
) -> Result<ChunkedCommitments<HyperKZGCommitment<Bn254>>, CommitLayerError> {
    let chunk_size = ra_tensor_chunk_size(pcs_domain_size, address_space)?;
    validate_chunk_size(chunk_size)?;
    let polys = values
        .chunks(chunk_size)
        .map(|chunk| one_hot_chunk_indices(chunk, chunk_size, address_space))
        .collect::<Result<Vec<_>, _>>()?;

    let commitments = commit_address_cycle_one_hot_chunks(&polys, address_space, setup)?;

    Ok(ChunkedCommitments {
        chunk_size,
        address_space: Some(address_space),
        len: values.len(),
        commitments,
    })
}

fn one_hot_chunk_indices(
    chunk: &[u16],
    chunk_size: usize,
    address_space: usize,
) -> Result<Vec<Option<u16>>, CommitLayerError> {
    let mut indices = Vec::with_capacity(chunk_size);
    for &index in chunk {
        if usize::from(index) >= address_space {
            return Err(CommitLayerError::InvalidDenseRaTable);
        }
        indices.push(Some(index));
    }
    indices.resize(chunk_size, None);
    Ok(indices)
}

fn commit_address_cycle_one_hot_chunks(
    chunks: &[Vec<Option<u16>>],
    address_space: usize,
    setup: &HyperKZGProverKey<Bn254>,
) -> Result<Vec<HyperKZGCommitment<Bn254>>, CommitLayerError> {
    let Some(chunk_size) = chunks.first().map(Vec::len) else {
        return Ok(Vec::new());
    };
    let required_size = address_space
        .checked_mul(chunk_size)
        .ok_or(CommitLayerError::InvalidDenseRaTable)?;
    if setup.kzg_pk.g1_powers().len() < required_size {
        return Err(CommitLayerError::InvalidRaChunkDomain {
            domain_size: setup.kzg_pk.g1_powers().len(),
            address_space,
        });
    }

    let powers = setup.kzg_pk.g1_powers();
    Ok(chunks
        .iter()
        .map(|chunk| {
            let commitment = chunk
                .iter()
                .enumerate()
                .filter_map(|(cycle, address)| {
                    address.map(|address| {
                        let index = cycle * address_space + usize::from(address);
                        powers[index].into_group()
                    })
                })
                .sum::<G1Projective>();
            HyperKZGCommitment(commitment.into_affine())
        })
        .collect())
}

pub fn commit_layer_bits<PCS>(
    witness: &LayerRawWitness,
    chunk_size: usize,
    setup: &PCS::ProverSetup,
) -> Result<LayerBitCommitments<PCS::Commitment>, CommitLayerError>
where
    PCS: CommitmentScheme<Field = Fr>,
{
    Ok(LayerBitCommitments {
        down_proj_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.down_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        silu_up_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.silu_up_output_frac_bits,
            chunk_size,
            setup,
        )?,
        silu_input_frac_bits: commit_bit_tables::<PCS>(
            &witness.silu_input_frac_bits,
            chunk_size,
            setup,
        )?,
        silu_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.silu_output_frac_bits,
            chunk_size,
            setup,
        )?,
        gate_proj_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.gate_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        up_proj_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.up_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        rms_norm_mlp_norm_frac_bits: commit_bit_tables::<PCS>(
            &witness.rms_norm_mlp_norm_frac_bits,
            chunk_size,
            setup,
        )?,
        rms_norm_mlp_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.rms_norm_mlp_output_frac_bits,
            chunk_size,
            setup,
        )?,
        o_proj_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.o_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        pv_matmul_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.pv_matmul_output_frac_bits,
            chunk_size,
            setup,
        )?,
        softmax_floor_frac_bits: commit_bit_tables::<PCS>(
            &witness.softmax_floor_frac_bits,
            chunk_size,
            setup,
        )?,
        softmax_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.softmax_output_frac_bits,
            chunk_size,
            setup,
        )?,
        softmax_exp_frac_bits: commit_bit_tables::<PCS>(
            &witness.softmax_exp_frac_bits,
            chunk_size,
            setup,
        )?,
        qk_score_dot_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.qk_score_dot_output_frac_bits,
            chunk_size,
            setup,
        )?,
        qk_score_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.qk_score_output_frac_bits,
            chunk_size,
            setup,
        )?,
        q_rope_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.q_rope_output_frac_bits,
            chunk_size,
            setup,
        )?,
        k_rope_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.k_rope_output_frac_bits,
            chunk_size,
            setup,
        )?,
        q_norm_norm_frac_bits: commit_bit_tables::<PCS>(
            &witness.q_norm_norm_frac_bits,
            chunk_size,
            setup,
        )?,
        q_norm_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.q_norm_output_frac_bits,
            chunk_size,
            setup,
        )?,
        k_norm_norm_frac_bits: commit_bit_tables::<PCS>(
            &witness.k_norm_norm_frac_bits,
            chunk_size,
            setup,
        )?,
        k_norm_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.k_norm_output_frac_bits,
            chunk_size,
            setup,
        )?,
        q_proj_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.q_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        k_proj_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.k_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        v_proj_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.v_proj_output_frac_bits,
            chunk_size,
            setup,
        )?,
        rms_norm_atten_norm_frac_bits: commit_bit_tables::<PCS>(
            &witness.rms_norm_atten_norm_frac_bits,
            chunk_size,
            setup,
        )?,
        rms_norm_atten_output_frac_bits: commit_bit_tables::<PCS>(
            &witness.rms_norm_atten_output_frac_bits,
            chunk_size,
            setup,
        )?,
    })
}

pub fn commit_bit_tables<PCS>(
    bit_tables: &[Vec<u8>; FRAC_BITS],
    chunk_size: usize,
    setup: &PCS::ProverSetup,
) -> Result<[ChunkedCommitments<PCS::Commitment>; FRAC_BITS], CommitLayerError>
where
    PCS: CommitmentScheme<Field = Fr>,
{
    Ok([
        commit_u8_chunks::<PCS>(&bit_tables[0], chunk_size, setup)?,
        commit_u8_chunks::<PCS>(&bit_tables[1], chunk_size, setup)?,
        commit_u8_chunks::<PCS>(&bit_tables[2], chunk_size, setup)?,
        commit_u8_chunks::<PCS>(&bit_tables[3], chunk_size, setup)?,
        commit_u8_chunks::<PCS>(&bit_tables[4], chunk_size, setup)?,
        commit_u8_chunks::<PCS>(&bit_tables[5], chunk_size, setup)?,
        commit_u8_chunks::<PCS>(&bit_tables[6], chunk_size, setup)?,
        commit_u8_chunks::<PCS>(&bit_tables[7], chunk_size, setup)?,
    ])
}

fn commit_bool_bit_tables<PCS>(
    bit_tables: &BitOpeningWitnesses,
    chunk_size: usize,
    setup: &PCS::ProverSetup,
) -> Result<[ChunkedCommitments<PCS::Commitment>; FRAC_BITS], CommitLayerError>
where
    PCS: CommitmentScheme<Field = Fr>,
{
    let values = std::array::from_fn(|bit| {
        bit_tables[bit]
            .iter()
            .copied()
            .map(u8::from)
            .collect::<Vec<_>>()
    });
    commit_bit_tables::<PCS>(&values, chunk_size, setup)
}

pub fn commit_i32_chunks<PCS>(
    values: &[i32],
    chunk_size: usize,
    setup: &PCS::ProverSetup,
) -> Result<ChunkedCommitments<PCS::Commitment>, CommitLayerError>
where
    PCS: CommitmentScheme<Field = Fr>,
{
    validate_chunk_size(chunk_size)?;
    let polys = values
        .chunks(chunk_size)
        .map(|chunk| {
            let mut padded = Vec::with_capacity(chunk_size);
            padded.extend_from_slice(chunk);
            padded.resize(chunk_size, 0);
            MultilinearPolynomial::from(padded)
        })
        .collect::<Vec<_>>();

    Ok(ChunkedCommitments {
        chunk_size,
        address_space: None,
        len: values.len(),
        commitments: PCS::batch_commit(&polys, setup)
            .into_iter()
            .map(|(commitment, _)| commitment)
            .collect(),
    })
}

pub fn commit_u8_chunks<PCS>(
    values: &[u8],
    chunk_size: usize,
    setup: &PCS::ProverSetup,
) -> Result<ChunkedCommitments<PCS::Commitment>, CommitLayerError>
where
    PCS: CommitmentScheme<Field = Fr>,
{
    validate_chunk_size(chunk_size)?;
    let polys = values
        .chunks(chunk_size)
        .map(|chunk| {
            let mut padded = Vec::with_capacity(chunk_size);
            padded.extend_from_slice(chunk);
            padded.resize(chunk_size, 0);
            MultilinearPolynomial::from(padded)
        })
        .collect::<Vec<_>>();

    Ok(ChunkedCommitments {
        chunk_size,
        address_space: None,
        len: values.len(),
        commitments: PCS::batch_commit(&polys, setup)
            .into_iter()
            .map(|(commitment, _)| commitment)
            .collect(),
    })
}

fn validate_chunk_size(chunk_size: usize) -> Result<(), CommitLayerError> {
    if chunk_size != 0 && chunk_size.is_power_of_two() {
        Ok(())
    } else {
        Err(CommitLayerError::InvalidChunkSize(chunk_size))
    }
}

fn validate_address_space(address_space: usize) -> Result<(), CommitLayerError> {
    if address_space != 0 && address_space.is_power_of_two() && address_space <= u16::MAX as usize {
        Ok(())
    } else {
        Err(CommitLayerError::InvalidAddressSpace(address_space))
    }
}

fn ra_tensor_chunk_size(
    pcs_domain_size: usize,
    address_space: usize,
) -> Result<usize, CommitLayerError> {
    validate_chunk_size(pcs_domain_size)?;
    validate_address_space(address_space)?;
    if pcs_domain_size < address_space || !pcs_domain_size.is_multiple_of(address_space) {
        return Err(CommitLayerError::InvalidRaChunkDomain {
            domain_size: pcs_domain_size,
            address_space,
        });
    }
    let chunk_size = pcs_domain_size / address_space;
    validate_chunk_size(chunk_size)?;
    Ok(chunk_size)
}

#[cfg(test)]
mod tests {
    use ark_bn254::Bn254;
    use joltworks::poly::commitment::{commitment_scheme::CommitmentScheme, hyperkzg::HyperKZG};

    use super::*;

    #[test]
    fn commits_i32_chunks_with_padding() {
        type Pcs = HyperKZG<Bn254>;
        let setup = Pcs::setup_prover(2);

        let committed = commit_i32_chunks::<Pcs>(&[1, -2, 3, 4, 5], 4, &setup).unwrap();

        assert_eq!(committed.chunk_size, 4);
        assert_eq!(committed.len, 5);
        assert_eq!(committed.commitments.len(), 2);
    }

    #[test]
    fn commits_u8_chunks_with_padding() {
        type Pcs = HyperKZG<Bn254>;
        let setup = Pcs::setup_prover(3);

        let committed = commit_u8_chunks::<Pcs>(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 8, &setup).unwrap();

        assert_eq!(committed.chunk_size, 8);
        assert_eq!(committed.len, 9);
        assert_eq!(committed.commitments.len(), 2);
    }

    #[test]
    fn commits_ra_chunks_as_one_hot() {
        type Pcs = HyperKZG<Bn254>;
        let setup = Pcs::setup_prover(10);

        let committed = commit_ra_chunks(&[0, 7, 255, 3, 9], 1024, 256, &setup).unwrap();

        assert_eq!(committed.chunk_size, 4);
        assert_eq!(committed.address_space, Some(256));
        assert_eq!(committed.len, 5);
        assert_eq!(committed.commitments.len(), 2);
    }

    #[test]
    fn commits_all_bit_tables() {
        type Pcs = HyperKZG<Bn254>;
        let setup = Pcs::setup_prover(2);
        let bit_tables = std::array::from_fn(|bit| vec![bit as u8; 5]);

        let committed = commit_bit_tables::<Pcs>(&bit_tables, 4, &setup).unwrap();

        assert_eq!(committed.len(), FRAC_BITS);
        assert!(committed.iter().all(|bits| bits.chunk_size == 4));
        assert!(committed.iter().all(|bits| bits.len == 5));
        assert!(committed.iter().all(|bits| bits.commitments.len() == 2));
    }

    #[test]
    fn rejects_invalid_chunk_size() {
        type Pcs = HyperKZG<Bn254>;
        let setup = Pcs::setup_prover(2);

        assert_eq!(
            commit_i32_chunks::<Pcs>(&[1, 2, 3], 3, &setup).unwrap_err(),
            CommitLayerError::InvalidChunkSize(3)
        );
    }
}
