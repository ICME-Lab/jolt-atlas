use joltworks::{field::JoltField, poly::multilinear_polynomial::MultilinearPolynomial};

use crate::claim::{Poly, Shape};

use super::{
    commitments::CommittedLayerPolys,
    tensors::LayerTensorIds,
    types::{LayerShape, LayerWeights},
    witness::LayerWitness,
};

/// All polynomial material needed by one independent layer IOP.
///
/// This is created immediately after trace witness materialization. `iop.rs`
/// must not build polynomials from witness vectors; it receives this struct and
/// only threads `Claim` and `Poly` values through op provers.
#[derive(Debug, Clone)]
pub struct LayerPolys<F: JoltField, C = ()> {
    pub hidden_in: Poly<F, C>,
    pub hidden_out: Poly<F, C>,

    pub residual_add_attn_a: Poly<F, C>,
    pub residual_add_attn_b: Poly<F, C>,
    pub down_proj: Poly<F, C>,
    pub o_proj: Poly<F, C>,

    pub rms_norm_atten_a: Poly<F, C>,
    pub rms_norm_atten_b: Poly<F, C>,
    pub rms_norm_atten_c: Poly<F, C>,
    pub rms_norm_mlp_a: Poly<F, C>,
    pub rms_norm_mlp_b: Poly<F, C>,

    pub q_proj: Poly<F, C>,
    pub k_proj: Poly<F, C>,
    pub v_proj: Poly<F, C>,
    pub q_norm: Poly<F, C>,
    pub k_norm: Poly<F, C>,
    pub q_rope: Poly<F, C>,
    pub k_rope: Poly<F, C>,
    pub qk_score: Poly<F, C>,
    pub softmax: Poly<F, C>,
    pub context: Poly<F, C>,

    pub gate_proj: Poly<F, C>,
    pub up_proj: Poly<F, C>,
    pub silu: Poly<F, C>,
    pub silu_up: Poly<F, C>,

    pub w_q_proj: Poly<F, C>,
    pub w_k_proj: Poly<F, C>,
    pub w_v_proj: Poly<F, C>,
    pub w_o_proj: Poly<F, C>,
    pub w_gate_proj: Poly<F, C>,
    pub w_up_proj: Poly<F, C>,
    pub w_down_proj: Poly<F, C>,
    pub w_rms_norm_atten: Poly<F, C>,
    pub w_q_norm: Poly<F, C>,
    pub w_k_norm: Poly<F, C>,
    pub w_rms_norm_mlp: Poly<F, C>,
    pub rope_cos: Poly<F, C>,
    pub rope_sin: Poly<F, C>,

    pub down_proj_round_ra: Vec<Poly<F, C>>,
    pub silu_up_round_ra: Vec<Poly<F, C>>,
    pub gate_proj_round_ra: Vec<Poly<F, C>>,
    pub up_proj_round_ra: Vec<Poly<F, C>>,
    pub rms_norm_mlp_round_ra: Vec<Poly<F, C>>,
    pub rms_norm_mlp_norm_round_ra: Vec<Poly<F, C>>,
    pub o_proj_round_ra: Vec<Poly<F, C>>,
    pub pv_matmul_round_ra: Vec<Poly<F, C>>,
    pub qk_score_round_ra: Vec<Poly<F, C>>,
    pub qk_score_dot_round_ra: Vec<Poly<F, C>>,
    pub q_rope_round_ra: Vec<Poly<F, C>>,
    pub k_rope_round_ra: Vec<Poly<F, C>>,
    pub q_norm_round_ra: Vec<Poly<F, C>>,
    pub q_norm_norm_round_ra: Vec<Poly<F, C>>,
    pub k_norm_round_ra: Vec<Poly<F, C>>,
    pub k_norm_norm_round_ra: Vec<Poly<F, C>>,
    pub q_proj_round_ra: Vec<Poly<F, C>>,
    pub k_proj_round_ra: Vec<Poly<F, C>>,
    pub v_proj_round_ra: Vec<Poly<F, C>>,
    pub rms_norm_atten_round_ra: Vec<Poly<F, C>>,
    pub rms_norm_atten_norm_round_ra: Vec<Poly<F, C>>,
    pub silu_gate_round_ra: Vec<Poly<F, C>>,
    pub silu_round_ra: Vec<Poly<F, C>>,
    pub silu_ra: Vec<Poly<F, C>>,
    pub softmax_round_ra: Vec<Poly<F, C>>,
    pub softmax_floor_round_ra: Vec<Poly<F, C>>,
    pub softmax_exp_round_ra: Vec<Poly<F, C>>,
    pub softmax_ra: Vec<Poly<F, C>>,
}

impl<F: JoltField> LayerPolys<F, ()> {
    pub fn from_witness(
        hidden_out: &[i32],
        witness: &LayerWitness,
        weights: &LayerWeights,
        shape: &LayerShape,
        _tensors: &LayerTensorIds,
    ) -> Self {
        Self {
            hidden_in: i32_poly(&witness.hidden_in, &shape.hidden_shape()),
            hidden_out: i32_poly(hidden_out, &shape.hidden_shape()),

            residual_add_attn_a: i32_poly(&witness.residual_add_attn_a, &shape.hidden_shape()),
            residual_add_attn_b: i32_poly(&witness.residual_add_attn_b, &shape.hidden_shape()),
            down_proj: i32_poly(&witness.down_proj, &shape.hidden_shape()),
            o_proj: i32_poly(&witness.o_proj, &shape.hidden_shape()),

            rms_norm_atten_a: i32_poly(&witness.rms_norm_atten_a, &shape.hidden_shape()),
            rms_norm_atten_b: i32_poly(&witness.rms_norm_atten_b, &shape.hidden_shape()),
            rms_norm_atten_c: i32_poly(&witness.rms_norm_atten_c, &shape.hidden_shape()),
            rms_norm_mlp_a: i32_poly(&witness.rms_norm_mlp_a, &shape.hidden_shape()),
            rms_norm_mlp_b: i32_poly(&witness.rms_norm_mlp_b, &shape.hidden_shape()),

            q_proj: i32_poly(
                &witness.q_proj,
                &Shape::new(vec![shape.seq, shape.attention_width()]),
            ),
            k_proj: i32_poly(
                &witness.k_proj,
                &Shape::new(vec![shape.seq, shape.kv_heads * shape.head_dim]),
            ),
            v_proj: i32_poly(
                &witness.v_proj,
                &Shape::new(vec![shape.seq, shape.kv_heads * shape.head_dim]),
            ),
            q_norm: i32_poly(
                &witness.q_norm,
                &Shape::new(vec![shape.seq, shape.q_heads, shape.head_dim]),
            ),
            k_norm: i32_poly(
                &witness.k_norm,
                &Shape::new(vec![shape.seq, shape.kv_heads, shape.head_dim]),
            ),
            q_rope: i32_poly(
                &witness.q_rope,
                &Shape::new(vec![shape.seq, shape.q_heads, shape.head_dim]),
            ),
            k_rope: i32_poly(
                &witness.k_rope,
                &Shape::new(vec![shape.seq, shape.kv_heads, shape.head_dim]),
            ),
            qk_score: i32_poly(
                &witness.qk_score,
                &Shape::new(vec![shape.q_heads, shape.seq, shape.seq]),
            ),
            softmax: i32_poly(
                &witness.softmax,
                &Shape::new(vec![shape.q_heads, shape.seq, shape.seq]),
            ),
            context: i32_poly(
                &witness.context,
                &Shape::new(vec![shape.seq, shape.q_heads, shape.head_dim]),
            ),

            gate_proj: i32_poly(&witness.gate_proj, &shape.intermediate_shape()),
            up_proj: i32_poly(&witness.up_proj, &shape.intermediate_shape()),
            silu: i32_poly(&witness.silu, &shape.intermediate_shape()),
            silu_up: i32_poly(&witness.silu_up, &shape.intermediate_shape()),

            w_q_proj: i32_poly(
                &weights.q_proj,
                &Shape::new(vec![shape.hidden, shape.attention_width()]),
            ),
            w_k_proj: i32_poly(
                &weights.k_proj,
                &Shape::new(vec![shape.hidden, shape.kv_heads * shape.head_dim]),
            ),
            w_v_proj: i32_poly(
                &weights.v_proj,
                &Shape::new(vec![shape.hidden, shape.kv_heads * shape.head_dim]),
            ),
            w_o_proj: i32_poly(
                &weights.o_proj,
                &Shape::new(vec![shape.attention_width(), shape.hidden]),
            ),
            w_gate_proj: i32_poly(
                &weights.gate_proj,
                &Shape::new(vec![shape.hidden, shape.intermediate]),
            ),
            w_up_proj: i32_poly(
                &weights.up_proj,
                &Shape::new(vec![shape.hidden, shape.intermediate]),
            ),
            w_down_proj: i32_poly(
                &weights.down_proj,
                &Shape::new(vec![shape.intermediate, shape.hidden]),
            ),
            w_rms_norm_atten: i32_poly(
                &weights.rms_norm_atten,
                &Shape::new(vec![shape.hidden]),
            ),
            w_q_norm: i32_poly(&weights.q_norm, &Shape::new(vec![shape.head_dim])),
            w_k_norm: i32_poly(&weights.k_norm, &Shape::new(vec![shape.head_dim])),
            w_rms_norm_mlp: i32_poly(&weights.rms_norm_mlp, &Shape::new(vec![shape.hidden])),
            rope_cos: i32_poly(
                &weights.rope_cos,
                &Shape::new(vec![shape.seq, shape.head_dim / 2]),
            ),
            rope_sin: i32_poly(
                &weights.rope_sin,
                &Shape::new(vec![shape.seq, shape.head_dim / 2]),
            ),

            down_proj_round_ra: Vec::new(),
            silu_up_round_ra: Vec::new(),
            gate_proj_round_ra: Vec::new(),
            up_proj_round_ra: Vec::new(),
            rms_norm_mlp_round_ra: Vec::new(),
            rms_norm_mlp_norm_round_ra: Vec::new(),
            o_proj_round_ra: Vec::new(),
            pv_matmul_round_ra: Vec::new(),
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
            silu_gate_round_ra: Vec::new(),
            silu_round_ra: Vec::new(),
            silu_ra: Vec::new(),
            softmax_round_ra: Vec::new(),
            softmax_floor_round_ra: Vec::new(),
            softmax_exp_round_ra: Vec::new(),
            softmax_ra: Vec::new(),
        }
    }
}

impl<F: JoltField, C: Clone> LayerPolys<F, C> {
    pub fn from_committed_polys(_committed: &CommittedLayerPolys<F, C>) -> Self {
        todo!("wire committed layer polys into LayerPolys")
    }
}

pub fn i32_poly<F: JoltField>(values: &[i32], shape: &Shape) -> Poly<F> {
    Poly::new(MultilinearPolynomial::from(padded_i32(values, shape)), None)
}

fn padded_i32(values: &[i32], shape: &Shape) -> Vec<i32> {
    let padded_dims = shape.padded_power_of_two().0;
    let mut out = vec![0; padded_dims.iter().product()];
    let strides = row_major_strides(shape.dims());
    let padded_strides = row_major_strides(&padded_dims);
    for (flat, &value) in values.iter().enumerate() {
        let mut padded_flat = 0;
        for (dim, (&stride, &padded_stride)) in strides.iter().zip(&padded_strides).enumerate() {
            let coord = (flat / stride) % shape.dims()[dim];
            padded_flat += coord * padded_stride;
        }
        out[padded_flat] = value;
    }
    out
}

fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    let mut stride = 1;
    for (idx, &dim) in dims.iter().enumerate().rev() {
        strides[idx] = stride;
        stride *= dim;
    }
    strides
}
