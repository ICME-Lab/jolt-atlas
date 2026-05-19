use std::{error::Error, time::Instant};

use half::{bf16, f16};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut max_targets = usize::MAX;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--max-targets" => {
                max_targets = args
                    .next()
                    .ok_or("--max-targets requires a value")?
                    .parse()?
            }
            "--first3" => max_targets = 3,
            "--full" => max_targets = usize::MAX,
            other => words.push(other.to_string()),
        }
    }

    let text = if words.is_empty() {
        "hello world this is a test".to_string()
    } else {
        words.join(" ")
    };

    let t = Instant::now();
    let ids = qwen2_prover::text::tokenize(&tok, &text)?;
    let bytes = std::fs::read(model)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let h = forward_f16(&st, &ids)?;
    let ppl = perplexity_tied_lm_head_f16(&st, &h, &ids, max_targets)?;

    println!("text: {text:?}");
    if max_targets == usize::MAX {
        println!("f16 ppl(full): {ppl}");
    } else {
        println!("f16 ppl(first {max_targets} targets): {ppl}");
    }
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn f16f(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

fn f16_vec(xs: &mut [f32]) {
    xs.par_iter_mut().for_each(|x| *x = f16f(*x));
}

fn forward_f16(st: &SafeTensors<'_>, ids: &[u32]) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut x = qwen2_prover::float::embed_from_safetensors(st, ids)?;
    f16_vec(&mut x);
    let mut r = qwen2_prover::float::Rotary::new();
    f16_vec(&mut r.rq);
    f16_vec(&mut r.rk);
    for layer in 0..qwen2_prover::LAYERS {
        let mut w = qwen2_prover::float::load_layer_from_safetensors(st, layer)?;
        round_layer(&mut w);
        x = layer_f16(&x, &w, &r);
    }
    let mut w = qwen2_prover::float::final_norm_from_safetensors(st)?;
    f16_vec(&mut w);
    Ok(rms_norm_f16(
        &x,
        &w,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
    ))
}

fn round_layer(w: &mut qwen2_prover::float::LayerWeights) {
    f16_vec(&mut w.ln1);
    f16_vec(&mut w.ln2);
    f16_vec(&mut w.wq);
    f16_vec(&mut w.bq);
    f16_vec(&mut w.wk);
    f16_vec(&mut w.bk);
    f16_vec(&mut w.wv);
    f16_vec(&mut w.bv);
    f16_vec(&mut w.wo);
    f16_vec(&mut w.wg);
    f16_vec(&mut w.wu);
    f16_vec(&mut w.wd);
}

fn layer_f16(
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &qwen2_prover::float::Rotary,
) -> Vec<f32> {
    let n1 = rms_norm_f16(x, &w.ln1, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
    let mut q = matmul_f16(
        &n1,
        &w.wq,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
    );
    add_rows_f16(&mut q, &w.bq);
    let mut k = matmul_f16(
        &n1,
        &w.wk,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
    );
    add_rows_f16(&mut k, &w.bk);
    let mut v = matmul_f16(
        &n1,
        &w.wv,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
    );
    add_rows_f16(&mut v, &w.bv);
    let q = rope_f16(&q, &r.rq);
    let k = rope_f16(&k, &r.rk);
    let s = score_qk_f16(&q, &k);
    let p = softmax_f16(
        &s,
        qwen2_prover::HEADS * qwen2_prover::SEQ,
        qwen2_prover::SEQ,
    );
    let c = attn_v_f16(&p, &v);
    let a = matmul_f16(
        &c,
        &w.wo,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
    );
    let h = add_f16(x, &a);
    let n2 = rms_norm_f16(&h, &w.ln2, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
    let g = matmul_f16(
        &n2,
        &w.wg,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
    );
    let u = matmul_f16(
        &n2,
        &w.wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
    );
    let m = silu_mul_f16(&g, &u);
    let d = matmul_f16(
        &m,
        &w.wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
    );
    add_f16(&h, &d)
}

fn matmul_f16(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0; m * n];
    y.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
        for c in 0..n {
            let mut s = 0.0f32;
            for i in 0..k {
                s += f16f(a[r * k + i]) * f16f(b[i * n + c]);
            }
            row[c] = f16f(s);
        }
    });
    y
}

fn add_f16(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.par_iter()
        .zip(b)
        .map(|(&x, &y)| f16f(f16f(x) + f16f(y)))
        .collect()
}

fn add_rows_f16(x: &mut [f32], b: &[f32]) {
    x.par_chunks_mut(b.len()).for_each(|row| {
        for (x, &b) in row.iter_mut().zip(b) {
            *x = f16f(f16f(*x) + f16f(b));
        }
    });
}

fn rms_norm_f16(x: &[f32], w: &[f32], _rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let xs = &x[r * cols..(r + 1) * cols];
        let ss = xs.iter().map(|v| f16f(*v) * f16f(*v)).sum::<f32>() / cols as f32;
        let inv = f16f(1.0 / (ss + 1e-6).sqrt());
        for c in 0..cols {
            row[c] = f16f(f16f(xs[c]) * inv * f16f(w[c]));
        }
    });
    y
}

fn rope_f16(x: &[f32], r: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    let hd = if x.len() % qwen2_prover::HEAD_DIM == 0 {
        qwen2_prover::HEAD_DIM
    } else {
        x.len()
    };
    y.par_chunks_mut(hd).enumerate().for_each(|(i, out)| {
        let base = i * hd;
        let half = hd / 2;
        for d in 0..half {
            let a = f16f(x[base + d]);
            let b = f16f(x[base + half + d]);
            let c = f16f(r[base + d]);
            let s = f16f(r[base + half + d]);
            out[d] = f16f(a * c - b * s);
            out[half + d] = f16f(b * c + a * s);
        }
    });
    y
}

fn score_qk_f16(q: &[f32], k: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0; qwen2_prover::SCORE_LEN];
    y.par_chunks_mut(qwen2_prover::SEQ * qwen2_prover::SEQ)
        .enumerate()
        .for_each(|(h, ys)| {
            let kh = h / qwen2_prover::KV_GROUP;
            for i in 0..qwen2_prover::SEQ {
                for j in 0..qwen2_prover::SEQ {
                    let o = i * qwen2_prover::SEQ + j;
                    if j > i {
                        ys[o] = f32::NEG_INFINITY;
                        continue;
                    }
                    let mut s = 0.0f32;
                    for d in 0..qwen2_prover::HEAD_DIM {
                        let qi = (i * qwen2_prover::HEADS + h) * qwen2_prover::HEAD_DIM + d;
                        let ki = (j * qwen2_prover::KV_HEADS + kh) * qwen2_prover::HEAD_DIM + d;
                        s += f16f(q[qi]) * f16f(k[ki]);
                    }
                    ys[o] = f16f(s / (qwen2_prover::HEAD_DIM as f32).sqrt());
                }
            }
        });
    y
}

fn softmax_f16(x: &[f32], _rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0; x.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, row)| {
        let xs = &x[r * cols..(r + 1) * cols];
        let mx = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for c in 0..cols {
            row[c] = f16f((f16f(xs[c] - mx)).exp());
            sum += row[c];
        }
        let sum = f16f(sum);
        for value in row {
            *value = f16f(*value / sum);
        }
    });
    y
}

fn attn_v_f16(p: &[f32], v: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0; qwen2_prover::X_LEN];
    y.par_chunks_mut(qwen2_prover::HEAD_DIM)
        .enumerate()
        .for_each(|(oh, out)| {
            let pos = oh / qwen2_prover::HEADS;
            let h = oh % qwen2_prover::HEADS;
            let kh = h / qwen2_prover::KV_GROUP;
            for (d, slot) in out.iter_mut().enumerate() {
                let mut s = 0.0f32;
                for j in 0..qwen2_prover::SEQ {
                    let pi = (h * qwen2_prover::SEQ + pos) * qwen2_prover::SEQ + j;
                    let vi = (j * qwen2_prover::KV_HEADS + kh) * qwen2_prover::HEAD_DIM + d;
                    s += f16f(p[pi]) * f16f(v[vi]);
                }
                *slot = f16f(s);
            }
        });
    y
}

fn silu_mul_f16(g: &[f32], u: &[f32]) -> Vec<f32> {
    g.par_iter()
        .zip(u)
        .map(|(&a, &b)| {
            let a = f16f(a);
            let b = f16f(b);
            f16f(f16f(a / f16f(1.0 + f16f(-a).exp())) * b)
        })
        .collect()
}

fn perplexity_tied_lm_head_f16(
    st: &SafeTensors<'_>,
    hidden: &[f32],
    ids: &[u32],
    max_targets: usize,
) -> Result<f64, Box<dyn Error>> {
    let mut nll = 0.0;
    let mut n = 0usize;
    for pos in 0..qwen2_prover::SEQ - 1 {
        if n >= max_targets {
            break;
        }
        if ids[pos + 1] == qwen2_prover::text::EOS {
            break;
        }
        let x = &hidden[pos * qwen2_prover::HIDDEN..(pos + 1) * qwen2_prover::HIDDEN];
        let scores = lm_head_scores_f16(st, x)?;
        nll += qwen2_prover::float::nll_from_scores(&scores, ids[pos + 1] as usize);
        n += 1;
    }
    if n == 0 {
        return Err("no target tokens for perplexity".into());
    }
    Ok((nll / n as f64).exp())
}

fn lm_head_scores_f16(st: &SafeTensors<'_>, x: &[f32]) -> Result<Vec<f32>, Box<dyn Error>> {
    let t = st.tensor("model.embed_tokens.weight")?;
    let mut row = Vec::with_capacity(qwen2_prover::HIDDEN);
    let mut scores = Vec::with_capacity(qwen2_prover::VOCAB);
    for id in 0..qwen2_prover::VOCAB {
        row.clear();
        row_f32(&t, id, &mut row)?;
        let mut score = 0.0f32;
        for (&a, &b) in x.iter().zip(&row) {
            score += f16f(a) * f16f(b);
        }
        scores.push(f16f(score));
    }
    Ok(scores)
}

fn row_f32(t: &TensorView<'_>, row: usize, out: &mut Vec<f32>) -> Result<(), Box<dyn Error>> {
    let n = qwen2_prover::HIDDEN;
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
            return Err(format!("unsupported embedding dtype {dt:?}").into());
        }
    }
    Ok(())
}
