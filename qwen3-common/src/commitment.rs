use crate::FRAC_BITS;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkedCommitments<C> {
    pub chunk_size: usize,
    pub address_space: Option<usize>,
    pub len: usize,
    pub commitments: Vec<C>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerCommitments<C> {
    pub hidden_out: ChunkedCommitments<C>,
    pub hidden_in_a: ChunkedCommitments<C>,
    pub hidden_in_b: ChunkedCommitments<C>,
    pub silu_lookup_ra: ChunkedCommitments<C>,
    pub softmax_lookup_ra: ChunkedCommitments<C>,
    pub bits: LayerBitCommitments<C>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerHiddenCommitments<C> {
    pub hidden_out: ChunkedCommitments<C>,
    pub hidden_in_a: ChunkedCommitments<C>,
    pub hidden_in_b: ChunkedCommitments<C>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerBitCommitments<C> {
    pub down_proj_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub silu_up_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub silu_input_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub silu_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub gate_proj_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub up_proj_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub rms_norm_mlp_norm_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub rms_norm_mlp_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub o_proj_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub pv_matmul_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub softmax_floor_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub softmax_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub softmax_exp_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub qk_score_dot_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub qk_score_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub q_rope_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub k_rope_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub q_norm_norm_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub q_norm_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub k_norm_norm_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub k_norm_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub q_proj_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub k_proj_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub v_proj_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub rms_norm_atten_norm_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
    pub rms_norm_atten_output_frac_bits: [ChunkedCommitments<C>; FRAC_BITS],
}
