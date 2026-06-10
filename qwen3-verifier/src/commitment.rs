use ark_bn254::Bn254;
use ark_ec::AffineRepr;
use joltworks::{poly::commitment::hyperkzg::HyperKZGCommitment, transcripts::Transcript};
use qwen3_common::{ChunkedCommitments, FRAC_BITS, LayerBitCommitments, LayerCommitments};

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
