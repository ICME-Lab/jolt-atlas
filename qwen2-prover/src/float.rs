use std::{error::Error, path::Path};

use half::{bf16, f16};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

pub fn tensor_f32(t: &TensorView<'_>) -> Result<Vec<f32>, Box<dyn Error>> {
    match t.dtype() {
        Dtype::BF16 => Ok(t
            .data()
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect()),
        Dtype::F16 => Ok(t
            .data()
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect()),
        Dtype::F32 => Ok(t
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        dt => Err(crate::weights::err(format!(
            "unsupported tensor dtype {dt:?}"
        ))),
    }
}

pub fn transpose(xs: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(xs.len(), rows * cols);
    let mut ys = vec![0.0; xs.len()];
    for r in 0..rows {
        for c in 0..cols {
            ys[c * rows + r] = xs[r * cols + c];
        }
    }
    ys
}

fn need_shape(t: &TensorView<'_>, shape: &[usize], name: &str) -> Result<(), Box<dyn Error>> {
    if t.shape() != shape {
        return Err(crate::weights::err(format!(
            "{name}: expected shape {:?}, got {:?}",
            shape,
            t.shape()
        )));
    }
    Ok(())
}

fn vec_f32(st: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>, Box<dyn Error>> {
    let t = st.tensor(name)?;
    need_shape(&t, shape, name)?;
    tensor_f32(&t)
}

fn linear_f32(
    st: &SafeTensors<'_>,
    name: &str,
    out: usize,
    input: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let t = st.tensor(name)?;
    need_shape(&t, &[out, input], name)?;
    Ok(transpose(&tensor_f32(&t)?, out, input))
}

fn row_f32(t: &TensorView<'_>, row: usize, out: &mut Vec<f32>) -> Result<(), Box<dyn Error>> {
    let n = crate::HIDDEN;
    match t.dtype() {
        Dtype::BF16 => {
            let start = row * n * 2;
            for b in t.data()[start..start + n * 2].chunks_exact(2) {
                out.push(bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32());
            }
        }
        Dtype::F16 => {
            let start = row * n * 2;
            for b in t.data()[start..start + n * 2].chunks_exact(2) {
                out.push(f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32());
            }
        }
        Dtype::F32 => {
            let start = row * n * 4;
            for b in t.data()[start..start + n * 4].chunks_exact(4) {
                out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            }
        }
        dt => {
            return Err(crate::weights::err(format!(
                "unsupported embedding dtype {dt:?}"
            )));
        }
    }
    Ok(())
}

pub fn embed_from_safetensors(
    st: &SafeTensors<'_>,
    ids: &[u32],
) -> Result<Vec<f32>, Box<dyn Error>> {
    if ids.len() != crate::SEQ {
        return Err(crate::weights::err(format!(
            "expected {} token ids, got {}",
            crate::SEQ,
            ids.len()
        )));
    }
    let t = st.tensor("model.embed_tokens.weight")?;
    need_shape(
        &t,
        &[crate::VOCAB, crate::HIDDEN],
        "model.embed_tokens.weight",
    )?;

    let mut out = Vec::with_capacity(crate::X_LEN);
    for &id in ids {
        let id = id as usize;
        if id >= crate::VOCAB {
            return Err(crate::weights::err(format!("token id {id} exceeds vocab")));
        }
        row_f32(&t, id, &mut out)?;
    }
    Ok(out)
}

pub fn rot(seq: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let mut xs = vec![0.0; seq * heads * head_dim];
    let half = head_dim / 2;
    for pos in 0..seq {
        for h in 0..heads {
            for p in 0..half {
                let f = crate::ROPE_THETA.powf(-((2 * p) as f64) / head_dim as f64);
                let t = pos as f64 * f;
                let i = (pos * heads + h) * head_dim;
                xs[i + p] = t.cos() as f32;
                xs[i + half + p] = t.sin() as f32;
            }
        }
    }
    xs
}

pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut y = vec![0.0; m * n];
    y.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
        for c in 0..n {
            let mut s = 0.0f32;
            for i in 0..k {
                s += a[r * k + i] * b[i * n + c];
            }
            row[c] = s;
        }
    });
    y
}

pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.par_iter().zip(b).map(|(&x, &y)| x + y).collect()
}

pub fn add_rows(x: &mut [f32], b: &[f32]) {
    assert_eq!(x.len() % b.len(), 0);
    x.par_chunks_mut(b.len()).for_each(|row| {
        for (x, &b) in row.iter_mut().zip(b) {
            *x += b;
        }
    });
}

pub fn rms_norm(x: &[f32], w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * cols);
    assert_eq!(w.len(), cols);
    let mut y = vec![0.0; x.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let xs = &x[r * cols..(r + 1) * cols];
        let ss = xs.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ss + 1e-6).sqrt();
        for c in 0..cols {
            row[c] = xs[c] * inv * w[c];
        }
    });
    y
}

pub fn rope(x: &[f32], r: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), r.len());
    let mut y = vec![0.0; x.len()];
    let hd = if x.len() % crate::HEAD_DIM == 0 {
        crate::HEAD_DIM
    } else {
        x.len()
    };
    y.par_chunks_mut(hd).enumerate().for_each(|(i, out)| {
        let base = i * hd;
        let half = hd / 2;
        for d in 0..half {
            let a = x[base + d];
            let b = x[base + half + d];
            let c = r[base + d];
            let s = r[base + half + d];
            out[d] = a * c - b * s;
            out[half + d] = b * c + a * s;
        }
    });
    y
}

pub fn score_qk(q: &[f32], k: &[f32]) -> Vec<f32> {
    assert_eq!(q.len(), crate::Q_LEN);
    assert_eq!(k.len(), crate::KV_LEN);
    let mut y = vec![0.0; crate::SCORE_LEN];
    y.par_chunks_mut(crate::SEQ * crate::SEQ)
        .enumerate()
        .for_each(|(h, ys)| {
            let kh = h / crate::KV_GROUP;
            for i in 0..crate::SEQ {
                for j in 0..crate::SEQ {
                    let o = i * crate::SEQ + j;
                    if j > i {
                        ys[o] = f32::NEG_INFINITY;
                        continue;
                    }
                    let mut s = 0.0f32;
                    for d in 0..crate::HEAD_DIM {
                        let qi = (i * crate::HEADS + h) * crate::HEAD_DIM + d;
                        let ki = (j * crate::KV_HEADS + kh) * crate::HEAD_DIM + d;
                        s += q[qi] * k[ki];
                    }
                    ys[o] = s / (crate::HEAD_DIM as f32).sqrt();
                }
            }
        });
    y
}

pub fn softmax(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * cols);
    let mut y = vec![0.0; x.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let xs = &x[r * cols..(r + 1) * cols];
        let mx = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for c in 0..cols {
            row[c] = (xs[c] - mx).exp();
            sum += row[c];
        }
        for value in row {
            *value /= sum;
        }
    });
    y
}

pub fn attn_v(p: &[f32], v: &[f32]) -> Vec<f32> {
    assert_eq!(p.len(), crate::SCORE_LEN);
    assert_eq!(v.len(), crate::KV_LEN);
    let mut y = vec![0.0; crate::X_LEN];
    y.par_chunks_mut(crate::HEAD_DIM)
        .enumerate()
        .for_each(|(oh, out)| {
            let pos = oh / crate::HEADS;
            let h = oh % crate::HEADS;
            let kh = h / crate::KV_GROUP;
            for (d, slot) in out.iter_mut().enumerate() {
                let mut s = 0.0f32;
                for j in 0..crate::SEQ {
                    let pi = (h * crate::SEQ + pos) * crate::SEQ + j;
                    let vi = (j * crate::KV_HEADS + kh) * crate::HEAD_DIM + d;
                    s += p[pi] * v[vi];
                }
                *slot = s;
            }
        });
    y
}

pub fn silu_mul(g: &[f32], u: &[f32]) -> Vec<f32> {
    assert_eq!(g.len(), u.len());
    g.par_iter()
        .zip(u)
        .map(|(&a, &b)| (a / (1.0 + (-a).exp())) * b)
        .collect()
}

pub struct LayerWeights {
    pub ln1: Vec<f32>,
    pub ln2: Vec<f32>,
    pub wq: Vec<f32>,
    pub bq: Vec<f32>,
    pub wk: Vec<f32>,
    pub bk: Vec<f32>,
    pub wv: Vec<f32>,
    pub bv: Vec<f32>,
    pub wo: Vec<f32>,
    pub wg: Vec<f32>,
    pub wu: Vec<f32>,
    pub wd: Vec<f32>,
}

pub fn load_layer_from_safetensors(
    st: &SafeTensors<'_>,
    layer: usize,
) -> Result<LayerWeights, Box<dyn Error>> {
    let p = format!("model.layers.{layer}");
    Ok(LayerWeights {
        ln1: vec_f32(st, &format!("{p}.input_layernorm.weight"), &[crate::HIDDEN])?,
        ln2: vec_f32(
            st,
            &format!("{p}.post_attention_layernorm.weight"),
            &[crate::HIDDEN],
        )?,
        wq: linear_f32(
            st,
            &format!("{p}.self_attn.q_proj.weight"),
            crate::HIDDEN,
            crate::HIDDEN,
        )?,
        bq: vec_f32(st, &format!("{p}.self_attn.q_proj.bias"), &[crate::HIDDEN])?,
        wk: linear_f32(
            st,
            &format!("{p}.self_attn.k_proj.weight"),
            crate::KV_HEADS * crate::HEAD_DIM,
            crate::HIDDEN,
        )?,
        bk: vec_f32(
            st,
            &format!("{p}.self_attn.k_proj.bias"),
            &[crate::KV_HEADS * crate::HEAD_DIM],
        )?,
        wv: linear_f32(
            st,
            &format!("{p}.self_attn.v_proj.weight"),
            crate::KV_HEADS * crate::HEAD_DIM,
            crate::HIDDEN,
        )?,
        bv: vec_f32(
            st,
            &format!("{p}.self_attn.v_proj.bias"),
            &[crate::KV_HEADS * crate::HEAD_DIM],
        )?,
        wo: linear_f32(
            st,
            &format!("{p}.self_attn.o_proj.weight"),
            crate::HIDDEN,
            crate::HIDDEN,
        )?,
        wg: linear_f32(
            st,
            &format!("{p}.mlp.gate_proj.weight"),
            crate::INTERMEDIATE,
            crate::HIDDEN,
        )?,
        wu: linear_f32(
            st,
            &format!("{p}.mlp.up_proj.weight"),
            crate::INTERMEDIATE,
            crate::HIDDEN,
        )?,
        wd: linear_f32(
            st,
            &format!("{p}.mlp.down_proj.weight"),
            crate::HIDDEN,
            crate::INTERMEDIATE,
        )?,
    })
}

pub struct Rotary {
    pub rq: Vec<f32>,
    pub rk: Vec<f32>,
}

impl Rotary {
    pub fn new() -> Self {
        Self {
            rq: rot(crate::SEQ, crate::HEADS, crate::HEAD_DIM),
            rk: rot(crate::SEQ, crate::KV_HEADS, crate::HEAD_DIM),
        }
    }
}

impl Default for Rotary {
    fn default() -> Self {
        Self::new()
    }
}

pub fn layer(x: &[f32], w: &LayerWeights, r: &Rotary) -> Vec<f32> {
    let n1 = rms_norm(x, &w.ln1, crate::SEQ, crate::HIDDEN);
    let mut q = matmul(&n1, &w.wq, crate::SEQ, crate::HIDDEN, crate::HIDDEN);
    add_rows(&mut q, &w.bq);
    let mut k = matmul(
        &n1,
        &w.wk,
        crate::SEQ,
        crate::HIDDEN,
        crate::KV_HEADS * crate::HEAD_DIM,
    );
    add_rows(&mut k, &w.bk);
    let mut v = matmul(
        &n1,
        &w.wv,
        crate::SEQ,
        crate::HIDDEN,
        crate::KV_HEADS * crate::HEAD_DIM,
    );
    add_rows(&mut v, &w.bv);
    let q = rope(&q, &r.rq);
    let k = rope(&k, &r.rk);
    let s = score_qk(&q, &k);
    let p = softmax(&s, crate::HEADS * crate::SEQ, crate::SEQ);
    let c = attn_v(&p, &v);
    let a = matmul(&c, &w.wo, crate::SEQ, crate::HIDDEN, crate::HIDDEN);
    let h = add(x, &a);
    let n2 = rms_norm(&h, &w.ln2, crate::SEQ, crate::HIDDEN);
    let g = matmul(&n2, &w.wg, crate::SEQ, crate::HIDDEN, crate::INTERMEDIATE);
    let u = matmul(&n2, &w.wu, crate::SEQ, crate::HIDDEN, crate::INTERMEDIATE);
    let m = silu_mul(&g, &u);
    let d = matmul(&m, &w.wd, crate::SEQ, crate::INTERMEDIATE, crate::HIDDEN);
    add(&h, &d)
}

pub fn layer_with_di_softmax(x: &[f32], w: &LayerWeights, r: &Rotary) -> Vec<f32> {
    layer_with_di_softmax_config(x, w, r, crate::illm::DiSoftmaxConfig::paper_default())
}

pub fn layer_with_di_softmax_config(
    x: &[f32],
    w: &LayerWeights,
    r: &Rotary,
    softmax_cfg: crate::illm::DiSoftmaxConfig,
) -> Vec<f32> {
    let n1 = rms_norm(x, &w.ln1, crate::SEQ, crate::HIDDEN);
    let mut q = matmul(&n1, &w.wq, crate::SEQ, crate::HIDDEN, crate::HIDDEN);
    add_rows(&mut q, &w.bq);
    let mut k = matmul(
        &n1,
        &w.wk,
        crate::SEQ,
        crate::HIDDEN,
        crate::KV_HEADS * crate::HEAD_DIM,
    );
    add_rows(&mut k, &w.bk);
    let mut v = matmul(
        &n1,
        &w.wv,
        crate::SEQ,
        crate::HIDDEN,
        crate::KV_HEADS * crate::HEAD_DIM,
    );
    add_rows(&mut v, &w.bv);
    let q = rope(&q, &r.rq);
    let k = rope(&k, &r.rk);
    let s = score_qk(&q, &k);
    let p =
        crate::illm::di_clipped_softmax_f32(&s, crate::HEADS * crate::SEQ, crate::SEQ, softmax_cfg)
            .dequantize();
    let c = attn_v(&p, &v);
    let a = matmul(&c, &w.wo, crate::SEQ, crate::HIDDEN, crate::HIDDEN);
    let h = add(x, &a);
    let n2 = rms_norm(&h, &w.ln2, crate::SEQ, crate::HIDDEN);
    let g = matmul(&n2, &w.wg, crate::SEQ, crate::HIDDEN, crate::INTERMEDIATE);
    let u = matmul(&n2, &w.wu, crate::SEQ, crate::HIDDEN, crate::INTERMEDIATE);
    let m = silu_mul(&g, &u);
    let d = matmul(&m, &w.wd, crate::SEQ, crate::INTERMEDIATE, crate::HIDDEN);
    add(&h, &d)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HybridOp {
    Add,
    AddBias,
    AddResidual,
    Matmul,
    MatmulQkv,
    MatmulO,
    MatmulAttention,
    MatmulGate,
    MatmulUp,
    MatmulGateUp,
    MatmulGateUpGroup1024,
    MatmulDown,
    MatmulDownGroup1024,
    MatmulMlpGroup1024,
    MatmulTokenChannel,
    MatmulTokenChannelPaper,
    Rope,
    Silu,
    SoftmaxPaper,
    SoftmaxLut,
}

pub fn layer_with_hybrid_op(
    x: &[f32],
    w: &LayerWeights,
    r: &Rotary,
    op: HybridOp,
    cfg: crate::illm::DiConfig,
) -> Vec<f32> {
    let n1 = rms_norm(x, &w.ln1, crate::SEQ, crate::HIDDEN);
    let mut q = matmul_hybrid_site(
        &n1,
        &w.wq,
        crate::SEQ,
        crate::HIDDEN,
        crate::HIDDEN,
        op,
        cfg,
        MatmulSite::Qkv,
    );
    q = add_rows_hybrid(&q, &w.bq, crate::SEQ, crate::HIDDEN, op, cfg);
    let mut k = matmul_hybrid_site(
        &n1,
        &w.wk,
        crate::SEQ,
        crate::HIDDEN,
        crate::KV_HEADS * crate::HEAD_DIM,
        op,
        cfg,
        MatmulSite::Qkv,
    );
    k = add_rows_hybrid(
        &k,
        &w.bk,
        crate::SEQ,
        crate::KV_HEADS * crate::HEAD_DIM,
        op,
        cfg,
    );
    let mut v = matmul_hybrid_site(
        &n1,
        &w.wv,
        crate::SEQ,
        crate::HIDDEN,
        crate::KV_HEADS * crate::HEAD_DIM,
        op,
        cfg,
        MatmulSite::Qkv,
    );
    v = add_rows_hybrid(
        &v,
        &w.bv,
        crate::SEQ,
        crate::KV_HEADS * crate::HEAD_DIM,
        op,
        cfg,
    );
    let q = rope_hybrid(&q, &r.rq, op, cfg);
    let k = rope_hybrid(&k, &r.rk, op, cfg);
    let s = score_qk(&q, &k);
    let p = softmax_hybrid(&s, crate::HEADS * crate::SEQ, crate::SEQ, op);
    let c = attn_v(&p, &v);
    let a = matmul_hybrid_site(
        &c,
        &w.wo,
        crate::SEQ,
        crate::HIDDEN,
        crate::HIDDEN,
        op,
        cfg,
        MatmulSite::O,
    );
    let h = add_hybrid(x, &a, op, cfg);
    let n2 = rms_norm(&h, &w.ln2, crate::SEQ, crate::HIDDEN);
    let g = matmul_hybrid_site(
        &n2,
        &w.wg,
        crate::SEQ,
        crate::HIDDEN,
        crate::INTERMEDIATE,
        op,
        cfg,
        MatmulSite::Gate,
    );
    let u = matmul_hybrid_site(
        &n2,
        &w.wu,
        crate::SEQ,
        crate::HIDDEN,
        crate::INTERMEDIATE,
        op,
        cfg,
        MatmulSite::Up,
    );
    let m = silu_mul_hybrid(&g, &u, op, cfg);
    let d = matmul_hybrid_site(
        &m,
        &w.wd,
        crate::SEQ,
        crate::INTERMEDIATE,
        crate::HIDDEN,
        op,
        cfg,
        MatmulSite::Down,
    );
    add_hybrid(&h, &d, op, cfg)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MatmulSite {
    Qkv,
    O,
    Gate,
    Up,
    Down,
}

fn matmul_hybrid_site(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    op: HybridOp,
    cfg: crate::illm::DiConfig,
    site: MatmulSite,
) -> Vec<f32> {
    let site_enabled = match op {
        HybridOp::MatmulQkv => site == MatmulSite::Qkv,
        HybridOp::MatmulO => site == MatmulSite::O,
        HybridOp::MatmulAttention => site == MatmulSite::Qkv || site == MatmulSite::O,
        HybridOp::MatmulGate => site == MatmulSite::Gate,
        HybridOp::MatmulUp => site == MatmulSite::Up,
        HybridOp::MatmulGateUp => site == MatmulSite::Gate || site == MatmulSite::Up,
        HybridOp::MatmulGateUpGroup1024 => site == MatmulSite::Gate || site == MatmulSite::Up,
        HybridOp::MatmulDown => site == MatmulSite::Down,
        HybridOp::MatmulDownGroup1024 => site == MatmulSite::Down,
        HybridOp::MatmulMlpGroup1024 => {
            site == MatmulSite::Gate || site == MatmulSite::Up || site == MatmulSite::Down
        }
        _ => false,
    };
    if site_enabled {
        if op == HybridOp::MatmulGateUpGroup1024
            || op == HybridOp::MatmulDownGroup1024
            || op == HybridOp::MatmulMlpGroup1024
        {
            return matmul_token_channel_group_output(a, b, m, k, n, cfg.bits, 1024);
        }
        return matmul_token_channel(a, b, m, k, n, cfg.bits);
    }
    matmul_hybrid(a, b, m, k, n, op, cfg)
}

fn matmul_hybrid(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    op: HybridOp,
    cfg: crate::illm::DiConfig,
) -> Vec<f32> {
    if op == HybridOp::MatmulTokenChannel {
        return matmul_token_channel(a, b, m, k, n, cfg.bits);
    }
    if op == HybridOp::MatmulTokenChannelPaper {
        return matmul_token_channel_paper(a, b, m, k, n, cfg.bits);
    }
    if op != HybridOp::Matmul {
        return matmul(a, b, m, k, n);
    }
    let aq = crate::illm::quantize_f32_observed(a, cfg);
    let bq = crate::illm::quantize_f32_observed(b, cfg);
    crate::illm::di_matmul(&aq, &bq, m, k, n, cfg).dequantize()
}

pub fn matmul_hybrid_for_debug(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    op: HybridOp,
    cfg: crate::illm::DiConfig,
) -> Vec<f32> {
    matmul_hybrid(a, b, m, k, n, op, cfg)
}

fn matmul_token_channel(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, bits: u8) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let q = crate::rebase::q_for_bits(bits);

    let mut a_int = vec![0i128; a.len()];
    let mut a_zp = vec![0i128; m];
    let mut a_scale = vec![0.0f64; m];
    for row in 0..m {
        let xs = &a[row * k..(row + 1) * k];
        let (scale, zp) = quant_params_f32(xs, q);
        a_scale[row] = scale;
        a_zp[row] = zp;
        for col in 0..k {
            a_int[row * k + col] = quantize_value(a[row * k + col], scale, zp, q);
        }
    }

    let mut b_int = vec![0i128; b.len()];
    let mut b_zp = vec![0i128; n];
    let mut b_scale = vec![0.0f64; n];
    for col in 0..n {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for row in 0..k {
            let value = b[row * n + col];
            min = min.min(value);
            max = max.max(value);
        }
        let (scale, zp) = quant_params_min_max(min, max, q);
        b_scale[col] = scale;
        b_zp[col] = zp;
        for row in 0..k {
            b_int[row * n + col] = quantize_value(b[row * n + col], scale, zp, q);
        }
    }

    let mut y_real = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i128;
            for inner in 0..k {
                let lhs = a_int[row * k + inner] - a_zp[row];
                let rhs = b_int[inner * n + col] - b_zp[col];
                acc += lhs * rhs;
            }
            y_real[row * n + col] = (acc as f64 * a_scale[row] * b_scale[col]) as f32;
        }
    }

    let mut y = vec![0.0f32; m * n];
    for row in 0..m {
        let xs = &y_real[row * n..(row + 1) * n];
        let (scale, zp) = quant_params_f32(xs, q);
        for col in 0..n {
            let value = quantize_value(y_real[row * n + col], scale, zp, q);
            y[row * n + col] = ((value - zp) as f64 * scale) as f32;
        }
    }
    y
}

fn matmul_token_channel_group_output(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bits: u8,
    group: usize,
) -> Vec<f32> {
    let y_real = matmul_token_channel_no_output_quant(a, b, m, k, n, bits);
    quantize_rows_groupwise(&y_real, m, n, bits, group)
}

fn matmul_token_channel_no_output_quant(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bits: u8,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let q = crate::rebase::q_for_bits(bits);

    let mut a_int = vec![0i128; a.len()];
    let mut a_zp = vec![0i128; m];
    let mut a_scale = vec![0.0f64; m];
    for row in 0..m {
        let xs = &a[row * k..(row + 1) * k];
        let (scale, zp) = quant_params_f32(xs, q);
        a_scale[row] = scale;
        a_zp[row] = zp;
        for col in 0..k {
            a_int[row * k + col] = quantize_value(a[row * k + col], scale, zp, q);
        }
    }

    let mut b_int = vec![0i128; b.len()];
    let mut b_zp = vec![0i128; n];
    let mut b_scale = vec![0.0f64; n];
    for col in 0..n {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for row in 0..k {
            let value = b[row * n + col];
            min = min.min(value);
            max = max.max(value);
        }
        let (scale, zp) = quant_params_min_max(min, max, q);
        b_scale[col] = scale;
        b_zp[col] = zp;
        for row in 0..k {
            b_int[row * n + col] = quantize_value(b[row * n + col], scale, zp, q);
        }
    }

    let mut y_real = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i128;
            for inner in 0..k {
                let lhs = a_int[row * k + inner] - a_zp[row];
                let rhs = b_int[inner * n + col] - b_zp[col];
                acc += lhs * rhs;
            }
            y_real[row * n + col] = (acc as f64 * a_scale[row] * b_scale[col]) as f32;
        }
    }
    y_real
}

fn quantize_rows_groupwise(
    xs: &[f32],
    rows: usize,
    cols: usize,
    bits: u8,
    group: usize,
) -> Vec<f32> {
    assert_eq!(xs.len(), rows * cols);
    assert!(group > 0);
    let q = crate::rebase::q_for_bits(bits);
    let mut y = vec![0.0f32; xs.len()];
    for row in 0..rows {
        let mut start = 0;
        while start < cols {
            let end = (start + group).min(cols);
            let slice = &xs[row * cols + start..row * cols + end];
            let (scale, zp) = quant_params_f32(slice, q);
            for col in start..end {
                let value = quantize_value(xs[row * cols + col], scale, zp, q);
                y[row * cols + col] = ((value - zp) as f64 * scale) as f32;
            }
            start = end;
        }
    }
    y
}

fn matmul_token_channel_paper(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bits: u8,
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let q = crate::rebase::q_for_bits(bits);

    let mut a_int = vec![0i128; a.len()];
    let mut a_zp = vec![0i128; m];
    let mut a_scale = vec![0.0f64; m];
    for row in 0..m {
        let xs = &a[row * k..(row + 1) * k];
        let (scale, zp) = quant_params_f32(xs, q);
        a_scale[row] = scale;
        a_zp[row] = zp;
        for col in 0..k {
            a_int[row * k + col] = quantize_value(a[row * k + col], scale, zp, q);
        }
    }

    let mut b_int = vec![0i128; b.len()];
    let mut b_zp = vec![0i128; n];
    let mut b_scale = vec![0.0f64; n];
    for col in 0..n {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for row in 0..k {
            let value = b[row * n + col];
            min = min.min(value);
            max = max.max(value);
        }
        let (scale, zp) = quant_params_min_max(min, max, q);
        b_scale[col] = scale;
        b_zp[col] = zp;
        for row in 0..k {
            b_int[row * n + col] = quantize_value(b[row * n + col], scale, zp, q);
        }
    }

    let mut y = vec![0.0f32; m * n];
    let multiplier_shift = 32;
    for row in 0..m {
        let row_scale = b_scale
            .iter()
            .map(|&scale| a_scale[row] * scale)
            .fold(f64::INFINITY, f64::min);
        let multipliers = b_scale
            .iter()
            .map(|&scale| {
                crate::rebase::make_shift_multiplier_from_ratio(
                    a_scale[row] * scale / row_scale,
                    multiplier_shift,
                    crate::rebase::Rounding::Floor,
                )
            })
            .collect::<Vec<_>>();

        let mut acc_row = vec![0i128; n];
        for col in 0..n {
            let mut acc = 0i128;
            for inner in 0..k {
                let lhs = a_int[row * k + inner] - a_zp[row];
                let rhs = b_int[inner * n + col] - b_zp[col];
                acc += lhs * rhs;
            }
            acc_row[col] = multipliers[col].apply(acc);
        }

        let p_min = *acc_row.iter().min().unwrap();
        let p_max = *acc_row.iter().max().unwrap();
        if p_min == p_max {
            continue;
        }
        let values = crate::rebase::di_rebase_shift_slice(
            &acc_row,
            p_min,
            p_max,
            bits,
            crate::rebase::di_rebase_shift_multiplier(
                p_min,
                p_max,
                bits,
                multiplier_shift,
                crate::rebase::Rounding::Floor,
            ),
        );
        let params = crate::rebase::di_rebase_output_params(row_scale, p_min, p_max, bits);
        for col in 0..n {
            y[row * n + col] = ((values[col] - params.zero_point) as f64 * params.scale) as f32;
        }
    }
    y
}

fn quant_params_f32(xs: &[f32], q: i128) -> (f64, i128) {
    let min = xs.iter().copied().fold(f32::INFINITY, f32::min);
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    quant_params_min_max(min, max, q)
}

fn quant_params_min_max(min: f32, max: f32, q: i128) -> (f64, i128) {
    if min == max {
        return (1.0 / q as f64, (-(min as f64) * q as f64).round() as i128);
    }
    let scale = (max as f64 - min as f64) / q as f64;
    let zero_point = (-(min as f64) / scale).round() as i128;
    (scale, zero_point)
}

fn quantize_value(value: f32, scale: f64, zero_point: i128, q: i128) -> i128 {
    let y = ((value as f64 / scale) + zero_point as f64).floor() as i128;
    debug_assert!(0 <= y && y <= q, "quantized value out of range: {y}");
    y
}

fn rope_hybrid(x: &[f32], r: &[f32], op: HybridOp, cfg: crate::illm::DiConfig) -> Vec<f32> {
    if op != HybridOp::Rope {
        return rope(x, r);
    }
    let xq = crate::illm::quantize_f32_observed(x, cfg);
    crate::illm::di_rope(&xq, r, cfg).dequantize()
}

fn add_hybrid(a: &[f32], b: &[f32], op: HybridOp, cfg: crate::illm::DiConfig) -> Vec<f32> {
    if op != HybridOp::Add && op != HybridOp::AddResidual {
        return add(a, b);
    }
    let aq = crate::illm::quantize_f32_observed(a, cfg);
    let bq = crate::illm::quantize_f32_observed(b, cfg);
    crate::illm::di_add(&aq, &bq, 32, cfg).dequantize()
}

fn add_rows_hybrid(
    input: &[f32],
    bias: &[f32],
    rows: usize,
    cols: usize,
    op: HybridOp,
    cfg: crate::illm::DiConfig,
) -> Vec<f32> {
    if op != HybridOp::Add && op != HybridOp::AddBias {
        let mut out = input.to_vec();
        add_rows(&mut out, bias);
        return out;
    }
    let input = crate::illm::quantize_f32_observed(input, cfg);
    crate::illm::di_add_rows(&input, bias, rows, cols, cfg).dequantize()
}

fn softmax_hybrid(x: &[f32], rows: usize, cols: usize, op: HybridOp) -> Vec<f32> {
    match op {
        HybridOp::SoftmaxPaper => crate::illm::di_clipped_softmax_f32(
            x,
            rows,
            cols,
            crate::illm::DiSoftmaxConfig::paper_default(),
        )
        .dequantize(),
        HybridOp::SoftmaxLut => crate::illm::di_lut_softmax_f32(
            x,
            rows,
            cols,
            crate::illm::DiSoftmaxConfig::paper_default(),
        )
        .dequantize(),
        _ => softmax(x, rows, cols),
    }
}

fn silu_mul_hybrid(g: &[f32], u: &[f32], op: HybridOp, cfg: crate::illm::DiConfig) -> Vec<f32> {
    if op != HybridOp::Silu {
        return silu_mul(g, u);
    }
    let gq = crate::illm::quantize_f32_observed(g, cfg);
    let uq = crate::illm::quantize_f32_observed(u, cfg);
    let sg = crate::illm::di_silu(&gq, cfg);
    crate::illm::di_mul(&sg, &uq, cfg).dequantize()
}

pub fn final_norm_from_safetensors(st: &SafeTensors<'_>) -> Result<Vec<f32>, Box<dyn Error>> {
    vec_f32(st, "model.norm.weight", &[crate::HIDDEN])
}

pub fn forward_from_safetensors(
    st: &SafeTensors<'_>,
    ids: &[u32],
) -> Result<Vec<f32>, Box<dyn Error>> {
    let r = Rotary::new();
    let mut x = embed_from_safetensors(st, ids)?;
    for i in 0..crate::LAYERS {
        let w = load_layer_from_safetensors(st, i)?;
        x = layer(&x, &w, &r);
    }
    let w = final_norm_from_safetensors(st)?;
    Ok(rms_norm(&x, &w, crate::SEQ, crate::HIDDEN))
}

pub fn forward_with_di_softmax_from_safetensors(
    st: &SafeTensors<'_>,
    ids: &[u32],
) -> Result<Vec<f32>, Box<dyn Error>> {
    forward_with_di_softmax_config_from_safetensors(
        st,
        ids,
        crate::illm::DiSoftmaxConfig::paper_default(),
    )
}

pub fn forward_with_di_softmax_config_from_safetensors(
    st: &SafeTensors<'_>,
    ids: &[u32],
    softmax_cfg: crate::illm::DiSoftmaxConfig,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let r = Rotary::new();
    let mut x = embed_from_safetensors(st, ids)?;
    for i in 0..crate::LAYERS {
        let w = load_layer_from_safetensors(st, i)?;
        x = layer_with_di_softmax_config(&x, &w, &r, softmax_cfg);
    }
    let w = final_norm_from_safetensors(st)?;
    Ok(rms_norm(&x, &w, crate::SEQ, crate::HIDDEN))
}

pub fn forward_with_hybrid_op_from_safetensors(
    st: &SafeTensors<'_>,
    ids: &[u32],
    op: HybridOp,
    cfg: crate::illm::DiConfig,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let r = Rotary::new();
    let mut x = embed_from_safetensors(st, ids)?;
    for i in 0..crate::LAYERS {
        let w = load_layer_from_safetensors(st, i)?;
        x = layer_with_hybrid_op(&x, &w, &r, op, cfg);
    }
    let w = final_norm_from_safetensors(st)?;
    Ok(rms_norm(&x, &w, crate::SEQ, crate::HIDDEN))
}

pub fn forward(path: impl AsRef<Path>, ids: &[u32]) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    forward_from_safetensors(&st, ids)
}

pub fn lm_head_scores_from_safetensors(
    st: &SafeTensors<'_>,
    x: &[f32],
) -> Result<Vec<f32>, Box<dyn Error>> {
    if x.len() != crate::HIDDEN {
        return Err(crate::weights::err(format!(
            "expected hidden vector length {}, got {}",
            crate::HIDDEN,
            x.len()
        )));
    }
    let t = st.tensor("model.embed_tokens.weight")?;
    need_shape(
        &t,
        &[crate::VOCAB, crate::HIDDEN],
        "model.embed_tokens.weight",
    )?;

    let mut row = Vec::with_capacity(crate::HIDDEN);
    let mut scores = Vec::with_capacity(crate::VOCAB);
    for id in 0..crate::VOCAB {
        row.clear();
        row_f32(&t, id, &mut row)?;
        let score = x.iter().zip(&row).map(|(&a, &b)| a * b).sum::<f32>();
        scores.push(score);
    }
    Ok(scores)
}

pub fn nll_from_scores(scores: &[f32], target: usize) -> f64 {
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp = scores
        .iter()
        .map(|&s| ((s - max_score) as f64).exp())
        .sum::<f64>();
    max_score as f64 + sum_exp.ln() - scores[target] as f64
}

pub fn nll_tied_lm_head_from_safetensors(
    st: &SafeTensors<'_>,
    x: &[f32],
    target: u32,
) -> Result<f64, Box<dyn Error>> {
    if target as usize >= crate::VOCAB {
        return Err(crate::weights::err(format!(
            "target token id {target} exceeds vocab"
        )));
    }
    let scores = lm_head_scores_from_safetensors(st, x)?;
    Ok(nll_from_scores(&scores, target as usize))
}

pub fn perplexity_tied_lm_head_prefix_from_safetensors(
    st: &SafeTensors<'_>,
    hidden: &[f32],
    ids: &[u32],
    max_targets: usize,
) -> Result<f64, Box<dyn Error>> {
    if hidden.len() != crate::X_LEN {
        return Err(crate::weights::err(format!(
            "expected hidden length {}, got {}",
            crate::X_LEN,
            hidden.len()
        )));
    }
    if ids.len() != crate::SEQ {
        return Err(crate::weights::err(format!(
            "expected {} token ids, got {}",
            crate::SEQ,
            ids.len()
        )));
    }

    let mut nll = 0.0;
    let mut n = 0usize;
    for pos in 0..crate::SEQ - 1 {
        if n >= max_targets {
            break;
        }
        if ids[pos + 1] == crate::text::EOS {
            break;
        }
        let x = &hidden[pos * crate::HIDDEN..(pos + 1) * crate::HIDDEN];
        nll += nll_tied_lm_head_from_safetensors(st, x, ids[pos + 1])?;
        n += 1;
    }
    if n == 0 {
        return Err(crate::weights::err(
            "no target tokens for perplexity".to_string(),
        ));
    }
    Ok((nll / n as f64).exp())
}
