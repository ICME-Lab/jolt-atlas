use joltworks::{field::JoltField, poly::multilinear_polynomial::MultilinearPolynomial};

use crate::claim::{Poly, Shape};

use super::{
    commitments::LayerPolySet,
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
    pub softmax_input_frac_ra: Vec<Poly<F, C>>,
    pub softmax_ra: Vec<Poly<F, C>>,
}

impl<F: JoltField> LayerPolys<F, ()> {
    pub fn from_witness(
        witness: &LayerWitness,
        weights: &LayerWeights,
        shape: &LayerShape,
        tensors: &LayerTensorIds,
    ) -> Self {
        let pcs_polys = LayerPolySet::<F>::from_layer(witness, weights, shape, tensors);
        Self {
            hidden_in: i32_poly(&witness.hidden_in, &shape.hidden_shape()),

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
            w_rms_norm_atten: i32_poly(&weights.rms_norm_atten, &Shape::new(vec![shape.hidden])),
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

            down_proj_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.down_proj_acc),
            ),
            silu_up_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.silu_up_acc),
            ),
            gate_proj_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.gate_proj_acc),
            ),
            up_proj_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.up_proj_acc),
            ),
            rms_norm_mlp_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.rms_norm_mlp_acc),
            ),
            rms_norm_mlp_norm_round_ra: onehot_group(
                &pcs_polys,
                "rms_norm_mlp_norm_acc_round_ra.rad.",
            ),
            o_proj_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.o_proj_acc),
            ),
            pv_matmul_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.context_acc),
            ),
            qk_score_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.qk_score_scale_acc),
            ),
            qk_score_dot_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.qk_score_acc),
            ),
            q_rope_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.q_rope_acc),
            ),
            k_rope_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.k_rope_acc),
            ),
            q_norm_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.q_norm_acc),
            ),
            q_norm_norm_round_ra: onehot_group(&pcs_polys, "q_norm_norm_acc_round_ra.rad."),
            k_norm_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.k_norm_acc),
            ),
            k_norm_norm_round_ra: onehot_group(&pcs_polys, "k_norm_norm_acc_round_ra.rad."),
            q_proj_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.q_proj_acc),
            ),
            k_proj_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.k_proj_acc),
            ),
            v_proj_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.v_proj_acc),
            ),
            rms_norm_atten_round_ra: onehot_group(
                &pcs_polys,
                &format!("{}_round_ra.rad.", tensors.rms_norm_atten_acc),
            ),
            rms_norm_atten_norm_round_ra: onehot_group(
                &pcs_polys,
                "rms_norm_atten_norm_acc_round_ra.rad.",
            ),
            silu_gate_round_ra: onehot_group(&pcs_polys, "silu_round_ra.rad."),
            silu_round_ra: onehot_group(&pcs_polys, "silu_output_round_ra.rad."),
            silu_ra: [
                onehot_group(&pcs_polys, "silu_ra.rad."),
                onehot_group(&pcs_polys, "silu_slope_ra.rad."),
            ]
            .concat(),
            softmax_round_ra: onehot_group(&pcs_polys, "softmax_output_round_ra.rad."),
            softmax_floor_round_ra: onehot_group(&pcs_polys, "softmax_floor_round_ra.rad."),
            softmax_exp_round_ra: onehot_group(&pcs_polys, "softmax_exp_acc_round_ra.rad."),
            softmax_input_frac_ra: onehot_group(&pcs_polys, "softmax_input_frac_ra.rad."),
            softmax_ra: onehot_group(&pcs_polys, "softmax_ra.rad."),
        }
    }
}

impl<F: JoltField, C: Clone> LayerPolys<F, C> {
    pub fn from_witness_with_boundary(
        hidden_in: Poly<F, C>,
        witness: &LayerWitness,
        weights: &LayerWeights,
        shape: &LayerShape,
        tensors: &LayerTensorIds,
    ) -> Self {
        let base = LayerPolys::<F, ()>::from_witness(witness, weights, shape, tensors);

        Self {
            hidden_in,

            residual_add_attn_a: without_commitment(base.residual_add_attn_a),
            residual_add_attn_b: without_commitment(base.residual_add_attn_b),
            down_proj: without_commitment(base.down_proj),
            o_proj: without_commitment(base.o_proj),

            rms_norm_atten_a: without_commitment(base.rms_norm_atten_a),
            rms_norm_atten_b: without_commitment(base.rms_norm_atten_b),
            rms_norm_atten_c: without_commitment(base.rms_norm_atten_c),
            rms_norm_mlp_a: without_commitment(base.rms_norm_mlp_a),
            rms_norm_mlp_b: without_commitment(base.rms_norm_mlp_b),

            q_proj: without_commitment(base.q_proj),
            k_proj: without_commitment(base.k_proj),
            v_proj: without_commitment(base.v_proj),
            q_norm: without_commitment(base.q_norm),
            k_norm: without_commitment(base.k_norm),
            q_rope: without_commitment(base.q_rope),
            k_rope: without_commitment(base.k_rope),
            qk_score: without_commitment(base.qk_score),
            softmax: without_commitment(base.softmax),
            context: without_commitment(base.context),

            gate_proj: without_commitment(base.gate_proj),
            up_proj: without_commitment(base.up_proj),
            silu: without_commitment(base.silu),
            silu_up: without_commitment(base.silu_up),

            w_q_proj: without_commitment(base.w_q_proj),
            w_k_proj: without_commitment(base.w_k_proj),
            w_v_proj: without_commitment(base.w_v_proj),
            w_o_proj: without_commitment(base.w_o_proj),
            w_gate_proj: without_commitment(base.w_gate_proj),
            w_up_proj: without_commitment(base.w_up_proj),
            w_down_proj: without_commitment(base.w_down_proj),
            w_rms_norm_atten: without_commitment(base.w_rms_norm_atten),
            w_q_norm: without_commitment(base.w_q_norm),
            w_k_norm: without_commitment(base.w_k_norm),
            w_rms_norm_mlp: without_commitment(base.w_rms_norm_mlp),
            rope_cos: without_commitment(base.rope_cos),
            rope_sin: without_commitment(base.rope_sin),

            down_proj_round_ra: without_commitment_vec(base.down_proj_round_ra),
            silu_up_round_ra: without_commitment_vec(base.silu_up_round_ra),
            gate_proj_round_ra: without_commitment_vec(base.gate_proj_round_ra),
            up_proj_round_ra: without_commitment_vec(base.up_proj_round_ra),
            rms_norm_mlp_round_ra: without_commitment_vec(base.rms_norm_mlp_round_ra),
            rms_norm_mlp_norm_round_ra: without_commitment_vec(base.rms_norm_mlp_norm_round_ra),
            o_proj_round_ra: without_commitment_vec(base.o_proj_round_ra),
            pv_matmul_round_ra: without_commitment_vec(base.pv_matmul_round_ra),
            qk_score_round_ra: without_commitment_vec(base.qk_score_round_ra),
            qk_score_dot_round_ra: without_commitment_vec(base.qk_score_dot_round_ra),
            q_rope_round_ra: without_commitment_vec(base.q_rope_round_ra),
            k_rope_round_ra: without_commitment_vec(base.k_rope_round_ra),
            q_norm_round_ra: without_commitment_vec(base.q_norm_round_ra),
            q_norm_norm_round_ra: without_commitment_vec(base.q_norm_norm_round_ra),
            k_norm_round_ra: without_commitment_vec(base.k_norm_round_ra),
            k_norm_norm_round_ra: without_commitment_vec(base.k_norm_norm_round_ra),
            q_proj_round_ra: without_commitment_vec(base.q_proj_round_ra),
            k_proj_round_ra: without_commitment_vec(base.k_proj_round_ra),
            v_proj_round_ra: without_commitment_vec(base.v_proj_round_ra),
            rms_norm_atten_round_ra: without_commitment_vec(base.rms_norm_atten_round_ra),
            rms_norm_atten_norm_round_ra: without_commitment_vec(base.rms_norm_atten_norm_round_ra),
            silu_gate_round_ra: without_commitment_vec(base.silu_gate_round_ra),
            silu_round_ra: without_commitment_vec(base.silu_round_ra),
            silu_ra: without_commitment_vec(base.silu_ra),
            softmax_round_ra: without_commitment_vec(base.softmax_round_ra),
            softmax_floor_round_ra: without_commitment_vec(base.softmax_floor_round_ra),
            softmax_exp_round_ra: without_commitment_vec(base.softmax_exp_round_ra),
            softmax_input_frac_ra: without_commitment_vec(base.softmax_input_frac_ra),
            softmax_ra: without_commitment_vec(base.softmax_ra),
        }
    }
}

pub fn i32_poly<F: JoltField>(values: &[i32], shape: &Shape) -> Poly<F> {
    Poly::new(MultilinearPolynomial::from(padded_i32(values, shape)), None)
}

fn onehot_group<F: JoltField>(set: &LayerPolySet<F>, prefix: &str) -> Vec<Poly<F>> {
    let mut entries = set
        .entries
        .iter()
        .filter(|entry| entry.name.starts_with(prefix))
        .map(|entry| {
            let chunk = entry
                .name
                .rsplit_once('.')
                .and_then(|(_, suffix)| suffix.parse::<usize>().ok())
                .unwrap_or(0);
            (chunk, Poly::new(entry.poly.clone(), None))
        })
        .collect::<Vec<_>>();
    entries.sort_by_key(|(chunk, _)| *chunk);
    entries.into_iter().map(|(_, poly)| poly).collect()
}

fn without_commitment<F: JoltField, C>(poly: Poly<F, ()>) -> Poly<F, C> {
    Poly::new(poly.data, None)
}

fn without_commitment_vec<F: JoltField, C>(polys: Vec<Poly<F, ()>>) -> Vec<Poly<F, C>> {
    polys.into_iter().map(without_commitment).collect()
}

pub(crate) fn hidden_state_poly<F: JoltField, C>(values: &[i32], shape: &LayerShape) -> Poly<F, C> {
    let poly = i32_poly(values, &shape.hidden_shape());
    Poly::new(poly.data, None)
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
