use std::{error::Error, path::Path};

use rayon::prelude::*;
use safetensors::SafeTensors;

pub const S: i32 = crate::lut::S;
pub const ONE: i32 = crate::lut::ONE;

fn qm_s(a: i32, b: i32, s: i32) -> i32 {
    ((a as i64 * b as i64) >> s) as i32
}

pub fn rot(seq: usize, heads: usize, head_dim: usize) -> Vec<i32> {
    rot_with_one(seq, heads, head_dim, ONE)
}

pub fn rot_with_one(seq: usize, heads: usize, head_dim: usize, one: i32) -> Vec<i32> {
    let mut xs = vec![0; seq * heads * head_dim];
    let half = head_dim / 2;
    for pos in 0..seq {
        for h in 0..heads {
            for p in 0..half {
                let f = crate::ROPE_THETA.powf(-((2 * p) as f64) / head_dim as f64);
                let t = pos as f64 * f;
                let i = (pos * heads + h) * head_dim;
                xs[i + p] = (t.cos() * one as f64).round() as i32;
                xs[i + half + p] = (t.sin() * one as f64).round() as i32;
            }
        }
    }
    xs
}

pub fn matmul(a: &[i32], b: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
    matmul_with_scale(a, b, m, k, n, S)
}

pub fn matmul_with_scale(
    a: &[i32],
    b: &[i32],
    m: usize,
    k: usize,
    n: usize,
    s_bits: i32,
) -> Vec<i32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut y = vec![0; m * n];
    y.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
        for c in 0..n {
            let mut s = 0i64;
            for i in 0..k {
                s += a[r * k + i] as i64 * b[i * n + c] as i64;
            }
            row[c] = (s >> s_bits) as i32;
        }
    });
    y
}

pub fn add(a: &[i32], b: &[i32]) -> Vec<i32> {
    assert_eq!(a.len(), b.len());
    a.par_iter().zip(b).map(|(&x, &y)| x + y).collect()
}

pub fn add_rows(x: &mut [i32], b: &[i32]) {
    assert_eq!(x.len() % b.len(), 0);
    x.par_chunks_mut(b.len()).for_each(|row| {
        for (x, &b) in row.iter_mut().zip(b) {
            *x += b;
        }
    });
}

pub fn rms_norm(
    x: &[i32],
    w: &[i32],
    _rsqrt: &[i32],
    _rz: i32,
    rows: usize,
    cols: usize,
) -> Vec<i32> {
    rms_norm_with_scale(x, w, rows, cols, ONE, S)
}

pub fn rms_norm_with_scale(
    x: &[i32],
    w: &[i32],
    rows: usize,
    cols: usize,
    one: i32,
    s_bits: i32,
) -> Vec<i32> {
    assert_eq!(x.len(), rows * cols);
    assert_eq!(w.len(), cols);
    let mut y = vec![0; x.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let xs = &x[r * cols..(r + 1) * cols];
        let mut ss = 0i64;
        for &v in xs {
            ss += v as i64 * v as i64;
        }
        let eps = cols as f64 * (one as f64 * one as f64) * 1e-6;
        let inv = (one as f64 * (cols as f64).sqrt() * one as f64 / ((ss as f64 + eps).sqrt()))
            .round() as i32;
        for c in 0..cols {
            row[c] = qm_s(qm_s(xs[c], inv, s_bits), w[c], s_bits);
        }
    });
    y
}

pub fn rope(x: &[i32], r: &[i32]) -> Vec<i32> {
    rope_with_scale(x, r, S)
}

pub fn rope_with_scale(x: &[i32], r: &[i32], s_bits: i32) -> Vec<i32> {
    assert_eq!(x.len(), r.len());
    let mut y = vec![0; x.len()];
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
            out[d] = qm_s(a, c, s_bits) - qm_s(b, s, s_bits);
            out[half + d] = qm_s(b, c, s_bits) + qm_s(a, s, s_bits);
        }
    });
    y
}

pub fn score_qk(q: &[i32], k: &[i32]) -> Vec<i32> {
    score_qk_with_scale(q, k, S)
}

pub fn score_qk_with_scale(q: &[i32], k: &[i32], s_bits: i32) -> Vec<i32> {
    assert_eq!(q.len(), crate::Q_LEN);
    assert_eq!(k.len(), crate::KV_LEN);
    let mut y = vec![0; crate::SCORE_LEN];
    y.par_chunks_mut(crate::SEQ * crate::SEQ)
        .enumerate()
        .for_each(|(h, ys)| {
            let kh = h / crate::KV_GROUP;
            for i in 0..crate::SEQ {
                for j in 0..crate::SEQ {
                    let o = i * crate::SEQ + j;
                    if j > i {
                        ys[o] = -32768;
                        continue;
                    }
                    let mut s = 0i64;
                    for d in 0..crate::HEAD_DIM {
                        let qi = (i * crate::HEADS + h) * crate::HEAD_DIM + d;
                        let ki = (j * crate::KV_HEADS + kh) * crate::HEAD_DIM + d;
                        s += q[qi] as i64 * k[ki] as i64;
                    }
                    ys[o] = ((s >> s_bits) as i32) / 8;
                }
            }
        });
    y
}

pub fn softmax(x: &[i32], _exp: &[i32], _ez: i32, rows: usize, cols: usize) -> Vec<i32> {
    softmax_with_scale(x, rows, cols, ONE)
}

pub fn softmax_with_scale(x: &[i32], rows: usize, cols: usize, one: i32) -> Vec<i32> {
    assert_eq!(x.len(), rows * cols);
    let mut y = vec![0; x.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let xs = &x[r * cols..(r + 1) * cols];
        let mx = xs.iter().copied().max().unwrap() as f64 / one as f64;
        let mut sum = 0.0;
        let mut tmp = vec![0.0; cols];
        for c in 0..cols {
            tmp[c] = ((xs[c] as f64 / one as f64) - mx).exp();
            sum += tmp[c];
        }
        if sum == 0.0 {
            return;
        }
        for c in 0..cols {
            row[c] = (tmp[c] / sum * one as f64).round() as i32;
        }
    });
    y
}

pub fn attn_v(p: &[i32], v: &[i32]) -> Vec<i32> {
    attn_v_with_scale(p, v, S)
}

pub fn attn_v_with_scale(p: &[i32], v: &[i32], s_bits: i32) -> Vec<i32> {
    assert_eq!(p.len(), crate::SCORE_LEN);
    assert_eq!(v.len(), crate::KV_LEN);
    let mut y = vec![0; crate::X_LEN];
    y.par_chunks_mut(crate::HEAD_DIM)
        .enumerate()
        .for_each(|(oh, out)| {
            let pos = oh / crate::HEADS;
            let h = oh % crate::HEADS;
            let kh = h / crate::KV_GROUP;
            for (d, slot) in out.iter_mut().enumerate() {
                let mut s = 0i64;
                for j in 0..crate::SEQ {
                    let pi = (h * crate::SEQ + pos) * crate::SEQ + j;
                    let vi = (j * crate::KV_HEADS + kh) * crate::HEAD_DIM + d;
                    s += p[pi] as i64 * v[vi] as i64;
                }
                *slot = (s >> s_bits) as i32;
            }
        });
    y
}

pub fn silu_mul(g: &[i32], u: &[i32], _silu: &[i32], _sz: i32) -> Vec<i32> {
    silu_mul_with_scale(g, u, ONE, S)
}

pub fn silu_mul_with_scale(g: &[i32], u: &[i32], one: i32, s_bits: i32) -> Vec<i32> {
    assert_eq!(g.len(), u.len());
    g.par_iter()
        .zip(u)
        .map(|(&a, &b)| {
            let x = a as f64 / one as f64;
            let y = x / (1.0 + (-x).exp());
            qm_s((y * one as f64).round() as i32, b, s_bits)
        })
        .collect()
}

pub struct Luts {
    rsqrt: Vec<i32>,
    rz: i32,
    silu: Vec<i32>,
    sz: i32,
    exp: Vec<i32>,
    ez: i32,
    rq: Vec<i32>,
    rk: Vec<i32>,
}

impl Luts {
    pub fn new() -> Self {
        let (rsqrt, rz) = crate::lut::rsqrt(0, 1_000_000);
        let (silu, sz) = crate::lut::silu(-32768, 32767);
        let (exp, ez) = crate::lut::exp(-32768, 0);
        Self {
            rsqrt,
            rz,
            silu,
            sz,
            exp,
            ez,
            rq: rot(crate::SEQ, crate::HEADS, crate::HEAD_DIM),
            rk: rot(crate::SEQ, crate::KV_HEADS, crate::HEAD_DIM),
        }
    }
}

impl Default for Luts {
    fn default() -> Self {
        Self::new()
    }
}

pub fn layer(x: &[i32], w: &[Vec<i32>], l: &Luts) -> Vec<i32> {
    let n1 = rms_norm(
        x,
        &w[crate::weights::LN1],
        &l.rsqrt,
        l.rz,
        crate::SEQ,
        crate::HIDDEN,
    );
    let mut q = matmul(
        &n1,
        &w[crate::weights::WQ],
        crate::SEQ,
        crate::HIDDEN,
        crate::HIDDEN,
    );
    add_rows(&mut q, &w[crate::weights::BQ]);
    let mut k = matmul(
        &n1,
        &w[crate::weights::WK],
        crate::SEQ,
        crate::HIDDEN,
        crate::KV_HEADS * crate::HEAD_DIM,
    );
    add_rows(&mut k, &w[crate::weights::BK]);
    let mut v = matmul(
        &n1,
        &w[crate::weights::WV],
        crate::SEQ,
        crate::HIDDEN,
        crate::KV_HEADS * crate::HEAD_DIM,
    );
    add_rows(&mut v, &w[crate::weights::BV]);
    let q = rope(&q, &l.rq);
    let k = rope(&k, &l.rk);
    let s = score_qk(&q, &k);
    let p = softmax(&s, &l.exp, l.ez, crate::HEADS * crate::SEQ, crate::SEQ);
    let c = attn_v(&p, &v);
    let a = matmul(
        &c,
        &w[crate::weights::WO],
        crate::SEQ,
        crate::HIDDEN,
        crate::HIDDEN,
    );
    let h = add(x, &a);
    let n2 = rms_norm(
        &h,
        &w[crate::weights::LN2],
        &l.rsqrt,
        l.rz,
        crate::SEQ,
        crate::HIDDEN,
    );
    let g = matmul(
        &n2,
        &w[crate::weights::WG],
        crate::SEQ,
        crate::HIDDEN,
        crate::INTERMEDIATE,
    );
    let u = matmul(
        &n2,
        &w[crate::weights::WU],
        crate::SEQ,
        crate::HIDDEN,
        crate::INTERMEDIATE,
    );
    let m = silu_mul(&g, &u, &l.silu, l.sz);
    let d = matmul(
        &m,
        &w[crate::weights::WD],
        crate::SEQ,
        crate::INTERMEDIATE,
        crate::HIDDEN,
    );
    add(&h, &d)
}

pub struct ScaleLuts {
    rq: Vec<i32>,
    rk: Vec<i32>,
}

impl ScaleLuts {
    pub fn new(one: i32) -> Self {
        Self {
            rq: rot_with_one(crate::SEQ, crate::HEADS, crate::HEAD_DIM, one),
            rk: rot_with_one(crate::SEQ, crate::KV_HEADS, crate::HEAD_DIM, one),
        }
    }
}

pub fn layer_with_scale(
    x: &[i32],
    w: &[Vec<i32>],
    l: &ScaleLuts,
    one: i32,
    s_bits: i32,
) -> Vec<i32> {
    let n1 = rms_norm_with_scale(
        x,
        &w[crate::weights::LN1],
        crate::SEQ,
        crate::HIDDEN,
        one,
        s_bits,
    );
    let mut q = matmul_with_scale(
        &n1,
        &w[crate::weights::WQ],
        crate::SEQ,
        crate::HIDDEN,
        crate::HIDDEN,
        s_bits,
    );
    add_rows(&mut q, &w[crate::weights::BQ]);
    let mut k = matmul_with_scale(
        &n1,
        &w[crate::weights::WK],
        crate::SEQ,
        crate::HIDDEN,
        crate::KV_HEADS * crate::HEAD_DIM,
        s_bits,
    );
    add_rows(&mut k, &w[crate::weights::BK]);
    let mut v = matmul_with_scale(
        &n1,
        &w[crate::weights::WV],
        crate::SEQ,
        crate::HIDDEN,
        crate::KV_HEADS * crate::HEAD_DIM,
        s_bits,
    );
    add_rows(&mut v, &w[crate::weights::BV]);
    let q = rope_with_scale(&q, &l.rq, s_bits);
    let k = rope_with_scale(&k, &l.rk, s_bits);
    let s = score_qk_with_scale(&q, &k, s_bits);
    let p = softmax_with_scale(&s, crate::HEADS * crate::SEQ, crate::SEQ, one);
    let c = attn_v_with_scale(&p, &v, s_bits);
    let a = matmul_with_scale(
        &c,
        &w[crate::weights::WO],
        crate::SEQ,
        crate::HIDDEN,
        crate::HIDDEN,
        s_bits,
    );
    let h = add(x, &a);
    let n2 = rms_norm_with_scale(
        &h,
        &w[crate::weights::LN2],
        crate::SEQ,
        crate::HIDDEN,
        one,
        s_bits,
    );
    let g = matmul_with_scale(
        &n2,
        &w[crate::weights::WG],
        crate::SEQ,
        crate::HIDDEN,
        crate::INTERMEDIATE,
        s_bits,
    );
    let u = matmul_with_scale(
        &n2,
        &w[crate::weights::WU],
        crate::SEQ,
        crate::HIDDEN,
        crate::INTERMEDIATE,
        s_bits,
    );
    let m = silu_mul_with_scale(&g, &u, one, s_bits);
    let d = matmul_with_scale(
        &m,
        &w[crate::weights::WD],
        crate::SEQ,
        crate::INTERMEDIATE,
        crate::HIDDEN,
        s_bits,
    );
    add(&h, &d)
}

pub fn forward_from_safetensors(
    st: &SafeTensors<'_>,
    ids: &[u32],
) -> Result<Vec<i32>, Box<dyn Error>> {
    let l = Luts::new();
    let mut x = crate::text::embed_from_safetensors_with_one(st, ids, ONE)?;
    for i in 0..crate::LAYERS {
        let w = crate::weights::load_layer_from_safetensors_with_one(st, i, ONE)?;
        x = layer(&x, &w, &l);
    }
    let w = crate::weights::final_norm_with_one(st, ONE)?;
    Ok(rms_norm(&x, &w, &l.rsqrt, l.rz, crate::SEQ, crate::HIDDEN))
}

pub fn forward_from_safetensors_with_scale(
    st: &SafeTensors<'_>,
    ids: &[u32],
    s_bits: i32,
) -> Result<Vec<i32>, Box<dyn Error>> {
    if !(0..=30).contains(&s_bits) {
        return Err(crate::weights::err(format!(
            "scale bits must be in 0..=30, got {s_bits}"
        )));
    }
    let one = 1i32 << s_bits;
    let l = ScaleLuts::new(one);
    let mut x = crate::text::embed_from_safetensors_with_one(st, ids, one)?;
    for i in 0..crate::LAYERS {
        let w = crate::weights::load_layer_from_safetensors_with_one(st, i, one)?;
        x = layer_with_scale(&x, &w, &l, one, s_bits);
    }
    let w = crate::weights::final_norm_with_one(st, one)?;
    Ok(rms_norm_with_scale(
        &x,
        &w,
        crate::SEQ,
        crate::HIDDEN,
        one,
        s_bits,
    ))
}

pub fn lm_head_scores_from_safetensors(
    st: &SafeTensors<'_>,
    x: &[i32],
) -> Result<Vec<i64>, Box<dyn Error>> {
    if x.len() != crate::HIDDEN {
        return Err(crate::weights::err(format!(
            "expected hidden vector length {}, got {}",
            crate::HIDDEN,
            x.len()
        )));
    }
    let t = st.tensor("model.embed_tokens.weight")?;
    let mut row = Vec::with_capacity(crate::HIDDEN);
    let mut scores = Vec::with_capacity(crate::VOCAB);
    for id in 0..crate::VOCAB {
        row.clear();
        crate::text::row_q(&t, id, &mut row, ONE)?;
        let mut s = 0i64;
        for (&a, &b) in x.iter().zip(&row) {
            s += a as i64 * b as i64;
        }
        scores.push(s >> S);
    }
    Ok(scores)
}

pub fn lm_head_scores_from_safetensors_with_scale(
    st: &SafeTensors<'_>,
    x: &[i32],
    s_bits: i32,
) -> Result<Vec<i64>, Box<dyn Error>> {
    if x.len() != crate::HIDDEN {
        return Err(crate::weights::err(format!(
            "expected hidden vector length {}, got {}",
            crate::HIDDEN,
            x.len()
        )));
    }
    let one = 1i32 << s_bits;
    let t = st.tensor("model.embed_tokens.weight")?;
    let mut row = Vec::with_capacity(crate::HIDDEN);
    let mut scores = Vec::with_capacity(crate::VOCAB);
    for id in 0..crate::VOCAB {
        row.clear();
        crate::text::row_q(&t, id, &mut row, one)?;
        let mut s = 0i64;
        for (&a, &b) in x.iter().zip(&row) {
            s += a as i64 * b as i64;
        }
        scores.push(s >> s_bits);
    }
    Ok(scores)
}

pub fn nll_tied_lm_head_from_safetensors(
    st: &SafeTensors<'_>,
    x: &[i32],
    target: u32,
) -> Result<f64, Box<dyn Error>> {
    let scores = lm_head_scores_from_safetensors(st, x)?
        .into_iter()
        .map(|s| s as f64 / ONE as f64)
        .collect::<Vec<_>>();
    Ok(crate::text::nll_from_scores(&scores, target as usize))
}

pub fn nll_tied_lm_head_from_safetensors_with_scale(
    st: &SafeTensors<'_>,
    x: &[i32],
    target: u32,
    s_bits: i32,
) -> Result<f64, Box<dyn Error>> {
    let one = 1i32 << s_bits;
    let scores = lm_head_scores_from_safetensors_with_scale(st, x, s_bits)?
        .into_iter()
        .map(|s| s as f64 / one as f64)
        .collect::<Vec<_>>();
    Ok(crate::text::nll_from_scores(&scores, target as usize))
}

pub fn perplexity_tied_lm_head_prefix_from_safetensors(
    st: &SafeTensors<'_>,
    hidden: &[i32],
    ids: &[u32],
    max_targets: usize,
) -> Result<f64, Box<dyn Error>> {
    let mut nll = 0.0;
    let mut n = 0usize;
    for pos in 0..crate::SEQ - 1 {
        if n >= max_targets || ids[pos + 1] == crate::text::EOS {
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

pub fn perplexity_tied_lm_head_prefix_from_safetensors_with_scale(
    st: &SafeTensors<'_>,
    hidden: &[i32],
    ids: &[u32],
    max_targets: usize,
    s_bits: i32,
) -> Result<f64, Box<dyn Error>> {
    let mut nll = 0.0;
    let mut n = 0usize;
    for pos in 0..crate::SEQ - 1 {
        if n >= max_targets || ids[pos + 1] == crate::text::EOS {
            break;
        }
        let x = &hidden[pos * crate::HIDDEN..(pos + 1) * crate::HIDDEN];
        nll += nll_tied_lm_head_from_safetensors_with_scale(st, x, ids[pos + 1], s_bits)?;
        n += 1;
    }
    if n == 0 {
        return Err(crate::weights::err(
            "no target tokens for perplexity".to_string(),
        ));
    }
    Ok((nll / n as f64).exp())
}

pub fn perplexity(
    tokenizer_path: impl AsRef<Path>,
    safetensors_path: impl AsRef<Path>,
    text: &str,
    max_targets: usize,
) -> Result<f64, Box<dyn Error>> {
    let ids = crate::text::tokenize(tokenizer_path, text)?;
    let bytes = std::fs::read(safetensors_path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let h = forward_from_safetensors(&st, &ids)?;
    perplexity_tied_lm_head_prefix_from_safetensors(&st, &h, &ids, max_targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_q8_8() {
        let a = vec![ONE, 2 * ONE, -ONE, ONE / 2];
        let b = vec![ONE, ONE / 2, 2 * ONE, -ONE];
        assert_eq!(
            matmul(&a, &b, 2, 2, 2),
            vec![5 * ONE, -3 * ONE / 2, 0, -ONE]
        );
    }

    #[test]
    fn rope_identity() {
        let x = vec![10, 20, -30, 40];
        let r = vec![ONE, ONE, 0, 0];
        assert_eq!(rope(&x, &r), x);
    }
}
