use std::{error::Error, time::Instant};

use qwen2_prover::{
    float::{HybridOp, Rotary},
    illm::{DiConfig, DiRebaseMethod},
    rebase::Rounding,
};

const CALIB_TEXTS: &[&str] = &[
    "hello world this is a test",
    "the quick brown fox jumps over the lazy dog",
    "mathematics and proofs require careful reasoning",
    "large language models use attention and feed forward layers",
    "zero knowledge proofs need precise integer arithmetic",
];

const EVAL_TEXTS: &[&str] = &[
    "hello world this is a test",
    "the quick brown fox jumps over the lazy dog",
    "mathematics and proofs require careful reasoning",
];

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut bits = 8u8;
    let mut n2_top = 16usize;
    let mut mid_top = 4usize;
    let mut exponent = 0.5f32;
    let mut percentile = 0.99f32;
    let mut eval_texts: Vec<String> = EVAL_TEXTS.iter().map(|s| s.to_string()).collect();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bits" => bits = args.next().ok_or("--bits requires a value")?.parse()?,
            "--n2-top" => n2_top = args.next().ok_or("--n2-top requires a value")?.parse()?,
            "--mid-top" => mid_top = args.next().ok_or("--mid-top requires a value")?.parse()?,
            "--exponent" => exponent = args.next().ok_or("--exponent requires a value")?.parse()?,
            "--percentile" => {
                percentile = args
                    .next()
                    .ok_or("--percentile requires a value")?
                    .parse()?
            }
            "--eval" => eval_texts.push(args.next().ok_or("--eval requires a value")?),
            "--clear-eval" => eval_texts.clear(),
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }

    let cfg = DiConfig {
        bits,
        rounding: Rounding::Floor,
        rebase: DiRebaseMethod::Shift {
            multiplier_shift: 32,
        },
    };

    let t = Instant::now();
    let bytes = std::fs::read(model)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let r = Rotary::new();
    let coeffs = calibrate(&tok, &st, &r, cfg, n2_top, mid_top, exponent, percentile)?;

    println!("bits: {bits}");
    println!("n2_top: {n2_top}");
    println!("mid_top: {mid_top}");
    println!("exponent: {exponent}");
    println!("percentile: {percentile}");
    print_hot_channels("n2", &coeffs.n2, 4);
    print_hot_channels("mid", &coeffs.mid, 4);

    let mut sum = 0.0f64;
    for text in &eval_texts {
        let ids = qwen2_prover::text::tokenize(&tok, text)?;
        let h_float = qwen2_prover::float::forward_from_safetensors(&st, &ids)?;
        let float_ppl = qwen2_prover::float::perplexity_tied_lm_head_prefix_from_safetensors(
            &st,
            &h_float,
            &ids,
            usize::MAX,
        )?;
        let h = forward_with_coeffs(&st, &ids, &r, cfg, &coeffs)?;
        let ppl = qwen2_prover::float::perplexity_tied_lm_head_prefix_from_safetensors(
            &st,
            &h,
            &ids,
            usize::MAX,
        )?;
        sum += ppl;
        println!("eval: {text:?}");
        println!("  float ppl(full): {float_ppl}");
        println!("  fsbr  ppl(full): {ppl}");
    }
    println!("avg fsbr ppl(full): {}", sum / eval_texts.len() as f64);
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

struct Coeffs {
    n2: Vec<Vec<f32>>,
    mid: Vec<Vec<f32>>,
}

fn calibrate(
    tok: &std::path::Path,
    st: &safetensors::SafeTensors,
    r: &Rotary,
    cfg: DiConfig,
    n2_top: usize,
    mid_top: usize,
    exponent: f32,
    percentile: f32,
) -> Result<Coeffs, Box<dyn Error>> {
    let mut n2_max = vec![vec![0.0f32; qwen2_prover::HIDDEN]; qwen2_prover::LAYERS];
    let mut unused_mid_max = vec![vec![0.0f32; qwen2_prover::INTERMEDIATE]; qwen2_prover::LAYERS];
    for text in CALIB_TEXTS {
        let ids = qwen2_prover::text::tokenize(tok, text)?;
        let mut x = qwen2_prover::float::embed_from_safetensors(st, &ids)?;
        for layer in 0..qwen2_prover::LAYERS {
            let w = qwen2_prover::float::load_layer_from_safetensors(st, layer)?;
            x = collect_layer(layer, &x, &w, r, &mut n2_max, &mut unused_mid_max);
        }
    }
    let n2 = make_alphas(&n2_max, n2_top, exponent, percentile);

    let mut mid_max = vec![vec![0.0f32; qwen2_prover::INTERMEDIATE]; qwen2_prover::LAYERS];
    for text in CALIB_TEXTS {
        let ids = qwen2_prover::text::tokenize(tok, text)?;
        let mut x = qwen2_prover::float::embed_from_safetensors(st, &ids)?;
        for layer in 0..qwen2_prover::LAYERS {
            let w = qwen2_prover::float::load_layer_from_safetensors(st, layer)?;
            x = collect_mid_layer(layer, &x, &w, r, cfg, &n2, &mut mid_max);
        }
    }
    Ok(Coeffs {
        n2,
        mid: make_alphas(&mid_max, mid_top, exponent, percentile),
    })
}

fn collect_mid_layer(
    layer: usize,
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &Rotary,
    cfg: DiConfig,
    n2_alphas: &[Vec<f32>],
    mid_max: &mut [Vec<f32>],
) -> Vec<f32> {
    let n1 = qwen2_prover::float::rms_norm(x, &w.ln1, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
    let mut q = qwen2_prover::float::matmul(
        &n1,
        &w.wq,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
    );
    qwen2_prover::float::add_rows(&mut q, &w.bq);
    let mut k = qwen2_prover::float::matmul(
        &n1,
        &w.wk,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
    );
    qwen2_prover::float::add_rows(&mut k, &w.bk);
    let mut v = qwen2_prover::float::matmul(
        &n1,
        &w.wv,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
    );
    qwen2_prover::float::add_rows(&mut v, &w.bv);
    let q = qwen2_prover::float::rope(&q, &r.rq);
    let k = qwen2_prover::float::rope(&k, &r.rk);
    let s = qwen2_prover::float::score_qk(&q, &k);
    let p = qwen2_prover::float::softmax(
        &s,
        qwen2_prover::HEADS * qwen2_prover::SEQ,
        qwen2_prover::SEQ,
    );
    let c = qwen2_prover::float::attn_v(&p, &v);
    let a = qwen2_prover::float::matmul(
        &c,
        &w.wo,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
    );
    let h = qwen2_prover::float::add(x, &a);
    let mut n2 = qwen2_prover::float::rms_norm(&h, &w.ln2, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
    let mut wg = w.wg.clone();
    let mut wu = w.wu.clone();
    apply_alphas(
        &mut n2,
        &mut [&mut wg, &mut wu],
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        &n2_alphas[layer],
    );
    let g = qwen2_prover::float::matmul_hybrid_for_debug(
        &n2,
        &wg,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    let u = qwen2_prover::float::matmul_hybrid_for_debug(
        &n2,
        &wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    let m = qwen2_prover::float::silu_mul(&g, &u);
    update_channel_max(
        &mut mid_max[layer],
        &m,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
    );
    let d = qwen2_prover::float::matmul_hybrid_for_debug(
        &m,
        &w.wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    qwen2_prover::float::add(&h, &d)
}

fn collect_layer(
    layer: usize,
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &Rotary,
    n2_max: &mut [Vec<f32>],
    mid_max: &mut [Vec<f32>],
) -> Vec<f32> {
    let n1 = qwen2_prover::float::rms_norm(x, &w.ln1, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
    let mut q = qwen2_prover::float::matmul(
        &n1,
        &w.wq,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
    );
    qwen2_prover::float::add_rows(&mut q, &w.bq);
    let mut k = qwen2_prover::float::matmul(
        &n1,
        &w.wk,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
    );
    qwen2_prover::float::add_rows(&mut k, &w.bk);
    let mut v = qwen2_prover::float::matmul(
        &n1,
        &w.wv,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
    );
    qwen2_prover::float::add_rows(&mut v, &w.bv);
    let q = qwen2_prover::float::rope(&q, &r.rq);
    let k = qwen2_prover::float::rope(&k, &r.rk);
    let s = qwen2_prover::float::score_qk(&q, &k);
    let p = qwen2_prover::float::softmax(
        &s,
        qwen2_prover::HEADS * qwen2_prover::SEQ,
        qwen2_prover::SEQ,
    );
    let c = qwen2_prover::float::attn_v(&p, &v);
    let a = qwen2_prover::float::matmul(
        &c,
        &w.wo,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
    );
    let h = qwen2_prover::float::add(x, &a);
    let n2 = qwen2_prover::float::rms_norm(&h, &w.ln2, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
    update_channel_max(
        &mut n2_max[layer],
        &n2,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
    );
    let g = qwen2_prover::float::matmul(
        &n2,
        &w.wg,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
    );
    let u = qwen2_prover::float::matmul(
        &n2,
        &w.wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
    );
    let m = qwen2_prover::float::silu_mul(&g, &u);
    update_channel_max(
        &mut mid_max[layer],
        &m,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
    );
    let d = qwen2_prover::float::matmul(
        &m,
        &w.wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
    );
    qwen2_prover::float::add(&h, &d)
}

fn forward_with_coeffs(
    st: &safetensors::SafeTensors,
    ids: &[u32],
    r: &Rotary,
    cfg: DiConfig,
    coeffs: &Coeffs,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut x = qwen2_prover::float::embed_from_safetensors(st, ids)?;
    for layer in 0..qwen2_prover::LAYERS {
        let w = qwen2_prover::float::load_layer_from_safetensors(st, layer)?;
        x = layer_with_coeffs(layer, &x, &w, r, cfg, coeffs);
    }
    let norm = qwen2_prover::float::final_norm_from_safetensors(st)?;
    Ok(qwen2_prover::float::rms_norm(
        &x,
        &norm,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
    ))
}

fn layer_with_coeffs(
    layer: usize,
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &Rotary,
    cfg: DiConfig,
    coeffs: &Coeffs,
) -> Vec<f32> {
    let n1 = qwen2_prover::float::rms_norm(x, &w.ln1, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
    let mut q = qwen2_prover::float::matmul(
        &n1,
        &w.wq,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
    );
    qwen2_prover::float::add_rows(&mut q, &w.bq);
    let mut k = qwen2_prover::float::matmul(
        &n1,
        &w.wk,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
    );
    qwen2_prover::float::add_rows(&mut k, &w.bk);
    let mut v = qwen2_prover::float::matmul(
        &n1,
        &w.wv,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
    );
    qwen2_prover::float::add_rows(&mut v, &w.bv);
    let q = qwen2_prover::float::rope(&q, &r.rq);
    let k = qwen2_prover::float::rope(&k, &r.rk);
    let s = qwen2_prover::float::score_qk(&q, &k);
    let p = qwen2_prover::float::softmax(
        &s,
        qwen2_prover::HEADS * qwen2_prover::SEQ,
        qwen2_prover::SEQ,
    );
    let c = qwen2_prover::float::attn_v(&p, &v);
    let a = qwen2_prover::float::matmul(
        &c,
        &w.wo,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
    );
    let h = qwen2_prover::float::add(x, &a);
    let mut n2 = qwen2_prover::float::rms_norm(&h, &w.ln2, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
    let mut wg = w.wg.clone();
    let mut wu = w.wu.clone();
    apply_alphas(
        &mut n2,
        &mut [&mut wg, &mut wu],
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        &coeffs.n2[layer],
    );
    let g = qwen2_prover::float::matmul_hybrid_for_debug(
        &n2,
        &wg,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    let u = qwen2_prover::float::matmul_hybrid_for_debug(
        &n2,
        &wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    let mut m = qwen2_prover::float::silu_mul(&g, &u);
    let mut wd = w.wd.clone();
    apply_alphas(
        &mut m,
        &mut [&mut wd],
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        &coeffs.mid[layer],
    );
    let d = qwen2_prover::float::matmul_hybrid_for_debug(
        &m,
        &wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    qwen2_prover::float::add(&h, &d)
}

fn update_channel_max(channel_max: &mut [f32], xs: &[f32], rows: usize, cols: usize) {
    for row in 0..rows {
        for col in 0..cols {
            channel_max[col] = channel_max[col].max(xs[row * cols + col].abs());
        }
    }
}

fn make_alphas(maxima: &[Vec<f32>], top: usize, exponent: f32, percentile: f32) -> Vec<Vec<f32>> {
    maxima
        .iter()
        .map(|layer_max| {
            let mut sorted = layer_max.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let target = percentile_sorted(&sorted, percentile).max(1e-12);
            let mut order: Vec<usize> = (0..layer_max.len()).collect();
            order.sort_by(|&a, &b| layer_max[b].total_cmp(&layer_max[a]));
            let mut alphas = vec![1.0f32; layer_max.len()];
            for &channel in order.iter().take(top.min(layer_max.len())) {
                alphas[channel] = (layer_max[channel] / target).max(1.0).powf(exponent);
            }
            alphas
        })
        .collect()
}

fn apply_alphas(
    x: &mut [f32],
    weights: &mut [&mut Vec<f32>],
    rows: usize,
    input_cols: usize,
    output_cols: usize,
    alphas: &[f32],
) {
    for channel in 0..input_cols {
        let alpha = alphas[channel];
        if alpha <= 1.0 {
            continue;
        }
        for row in 0..rows {
            x[row * input_cols + channel] /= alpha;
        }
        for weight in weights.iter_mut() {
            for col in 0..output_cols {
                weight[channel * output_cols + col] *= alpha;
            }
        }
    }
}

fn print_hot_channels(name: &str, alphas: &[Vec<f32>], top: usize) {
    println!("{name} hot channels:");
    for (layer, layer_alphas) in alphas.iter().enumerate() {
        let mut order: Vec<usize> = (0..layer_alphas.len()).collect();
        order.sort_by(|&a, &b| layer_alphas[b].total_cmp(&layer_alphas[a]));
        let desc = order
            .iter()
            .take(top)
            .filter(|&&channel| layer_alphas[channel] > 1.0)
            .map(|&channel| format!("c{}:x{:.2}", channel, layer_alphas[channel]))
            .collect::<Vec<_>>()
            .join(", ");
        if !desc.is_empty() {
            println!("  L{layer}: {desc}");
        }
    }
}

fn percentile_sorted(xs: &[f32], p: f32) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let idx = ((xs.len() - 1) as f32 * p).round() as usize;
    xs[idx]
}
