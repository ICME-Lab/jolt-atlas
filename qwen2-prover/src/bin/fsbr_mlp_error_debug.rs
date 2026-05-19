use std::{error::Error, time::Instant};

use qwen2_prover::{
    float::{HybridOp, Rotary},
    illm::{DiConfig, DiRebaseMethod},
    rebase::Rounding,
};

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut bits = 8u8;
    let mut n2_top = 128usize;
    let mut mid_top = 32usize;
    let mut exponent = 0.35f32;
    let mut layer_limit = qwen2_prover::LAYERS;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bits" => bits = args.next().ok_or("--bits requires a value")?.parse()?,
            "--n2-top" => n2_top = args.next().ok_or("--n2-top requires a value")?.parse()?,
            "--mid-top" => mid_top = args.next().ok_or("--mid-top requires a value")?.parse()?,
            "--exponent" => exponent = args.next().ok_or("--exponent requires a value")?.parse()?,
            "--layers" => {
                layer_limit = args
                    .next()
                    .ok_or("--layers requires a value")?
                    .parse::<usize>()?
                    .min(qwen2_prover::LAYERS)
            }
            other => words.push(other.to_string()),
        }
    }

    let text = if words.is_empty() {
        "hello world this is a test".to_string()
    } else {
        words.join(" ")
    };

    let cfg = DiConfig {
        bits,
        rounding: Rounding::Floor,
        rebase: DiRebaseMethod::Shift {
            multiplier_shift: 32,
        },
    };

    let t = Instant::now();
    let ids = qwen2_prover::text::tokenize(&tok, &text)?;
    let bytes = std::fs::read(model)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let r = Rotary::new();
    let mut x = qwen2_prover::float::embed_from_safetensors(&st, &ids)?;

    println!("text: {text:?}");
    println!("bits: {bits}");
    println!("n2_top: {n2_top}");
    println!("mid_top: {mid_top}");
    println!("exponent: {exponent}");
    println!(
        "{:<18} {:>8} {:>12} {:>10} {:>10} {:>23} {:>23}",
        "site", "cos", "mse", "max_abs", "zero%", "float_range", "quant_range"
    );

    let mut summary = Summary::default();
    for layer in 0..layer_limit {
        let w = qwen2_prover::float::load_layer_from_safetensors(&st, layer)?;
        x = inspect_layer(
            layer,
            &x,
            &w,
            &r,
            cfg,
            n2_top,
            mid_top,
            exponent,
            &mut summary,
        );
    }
    summary.print();
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn inspect_layer(
    layer: usize,
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &Rotary,
    cfg: DiConfig,
    n2_top: usize,
    mid_top: usize,
    exponent: f32,
    summary: &mut Summary,
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
    smooth_matmul_input(
        &mut n2,
        &mut [&mut wg, &mut wu],
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        n2_top,
        exponent,
    );

    let gate_f = qwen2_prover::float::matmul(
        &n2,
        &wg,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
    );
    let gate_q = qwen2_prover::float::matmul_hybrid_for_debug(
        &n2,
        &wg,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    summary.push(
        "gate",
        print_stats(&format!("L{layer}.gate"), &gate_f, &gate_q),
    );

    let up_f = qwen2_prover::float::matmul(
        &n2,
        &wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
    );
    let up_q = qwen2_prover::float::matmul_hybrid_for_debug(
        &n2,
        &wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    summary.push("up", print_stats(&format!("L{layer}.up"), &up_f, &up_q));

    let mid_f = qwen2_prover::float::silu_mul(&gate_f, &up_f);
    let mut mid_q = qwen2_prover::float::silu_mul(&gate_q, &up_q);
    summary.push(
        "mid_pre_fsbr",
        print_stats(&format!("L{layer}.mid_pre"), &mid_f, &mid_q),
    );

    let mut wd = w.wd.clone();
    smooth_matmul_input(
        &mut mid_q,
        &mut [&mut wd],
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        mid_top,
        exponent,
    );
    let mut mid_f_smooth = mid_f.clone();
    let mut wd_f = w.wd.clone();
    smooth_matmul_input(
        &mut mid_f_smooth,
        &mut [&mut wd_f],
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        mid_top,
        exponent,
    );
    summary.push(
        "mid_post_fsbr",
        print_stats(&format!("L{layer}.mid_post"), &mid_f_smooth, &mid_q),
    );

    let down_f = qwen2_prover::float::matmul(
        &mid_f_smooth,
        &wd_f,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
    );
    let down_q = qwen2_prover::float::matmul_hybrid_for_debug(
        &mid_q,
        &wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    summary.push(
        "down",
        print_stats(&format!("L{layer}.down"), &down_f, &down_q),
    );
    let down_q_from_float_mid = qwen2_prover::float::matmul_hybrid_for_debug(
        &mid_f_smooth,
        &wd_f,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    summary.push(
        "down_mat_only",
        print_stats(
            &format!("L{layer}.down_mat_only"),
            &down_f,
            &down_q_from_float_mid,
        ),
    );
    let down_from_mid_error = qwen2_prover::float::matmul(
        &mid_q,
        &wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
    );
    summary.push(
        "down_input_only",
        print_stats(
            &format!("L{layer}.down_input_only"),
            &down_f,
            &down_from_mid_error,
        ),
    );
    qwen2_prover::float::add(&h, &down_q)
}

fn smooth_matmul_input(
    x: &mut [f32],
    weights: &mut [&mut Vec<f32>],
    rows: usize,
    input_cols: usize,
    output_cols: usize,
    top: usize,
    exponent: f32,
) {
    let mut channel_max = vec![0.0f32; input_cols];
    for row in 0..rows {
        for col in 0..input_cols {
            channel_max[col] = channel_max[col].max(x[row * input_cols + col].abs());
        }
    }
    let mut sorted = channel_max.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let target = percentile_sorted(&sorted, 0.99).max(1e-12);
    let mut order: Vec<usize> = (0..input_cols).collect();
    order.sort_by(|&a, &b| channel_max[b].total_cmp(&channel_max[a]));
    for &channel in order.iter().take(top.min(input_cols)) {
        let alpha = (channel_max[channel] / target).max(1.0).powf(exponent);
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

#[derive(Clone, Copy, Default)]
struct Stats {
    cosine: f64,
    mse: f64,
    max_abs: f64,
    zero_pct: f64,
}

#[derive(Default)]
struct Summary {
    gate: Vec<Stats>,
    up: Vec<Stats>,
    mid_pre_fsbr: Vec<Stats>,
    mid_post_fsbr: Vec<Stats>,
    down: Vec<Stats>,
    down_mat_only: Vec<Stats>,
    down_input_only: Vec<Stats>,
}

impl Summary {
    fn push(&mut self, name: &str, stats: Stats) {
        match name {
            "gate" => self.gate.push(stats),
            "up" => self.up.push(stats),
            "mid_pre_fsbr" => self.mid_pre_fsbr.push(stats),
            "mid_post_fsbr" => self.mid_post_fsbr.push(stats),
            "down" => self.down.push(stats),
            "down_mat_only" => self.down_mat_only.push(stats),
            "down_input_only" => self.down_input_only.push(stats),
            _ => unreachable!(),
        }
    }

    fn print(&self) {
        println!();
        println!(
            "{:<14} {:>8} {:>12} {:>10} {:>10}",
            "summary", "cos", "mse", "max_abs", "zero%"
        );
        print_avg("gate", &self.gate);
        print_avg("up", &self.up);
        print_avg("mid_pre", &self.mid_pre_fsbr);
        print_avg("mid_post", &self.mid_post_fsbr);
        print_avg("down", &self.down);
        print_avg("down_mat_only", &self.down_mat_only);
        print_avg("down_input_only", &self.down_input_only);
    }
}

fn print_avg(name: &str, xs: &[Stats]) {
    if xs.is_empty() {
        return;
    }
    let n = xs.len() as f64;
    let avg = Stats {
        cosine: xs.iter().map(|s| s.cosine).sum::<f64>() / n,
        mse: xs.iter().map(|s| s.mse).sum::<f64>() / n,
        max_abs: xs.iter().map(|s| s.max_abs).sum::<f64>() / n,
        zero_pct: xs.iter().map(|s| s.zero_pct).sum::<f64>() / n,
    };
    println!(
        "{name:<14} {:>8.5} {:>12.4e} {:>10.4} {:>9.3}%",
        avg.cosine, avg.mse, avg.max_abs, avg.zero_pct
    );
}

fn print_stats(name: &str, reference: &[f32], actual: &[f32]) -> Stats {
    assert_eq!(reference.len(), actual.len());
    let mut dot = 0.0f64;
    let mut ref_norm = 0.0f64;
    let mut act_norm = 0.0f64;
    let mut mse = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut ref_min = f32::INFINITY;
    let mut ref_max = f32::NEG_INFINITY;
    let mut act_min = f32::INFINITY;
    let mut act_max = f32::NEG_INFINITY;
    let mut zeroish = 0usize;
    for (&r, &a) in reference.iter().zip(actual) {
        let r64 = r as f64;
        let a64 = a as f64;
        dot += r64 * a64;
        ref_norm += r64 * r64;
        act_norm += a64 * a64;
        let diff = r64 - a64;
        mse += diff * diff;
        max_abs = max_abs.max(diff.abs());
        ref_min = ref_min.min(r);
        ref_max = ref_max.max(r);
        act_min = act_min.min(a);
        act_max = act_max.max(a);
        if a.abs() < 1e-12 {
            zeroish += 1;
        }
    }
    mse /= reference.len() as f64;
    let cosine = dot / (ref_norm.sqrt() * act_norm.sqrt()).max(f64::MIN_POSITIVE);
    let zero_pct = 100.0 * zeroish as f64 / actual.len() as f64;
    println!(
        "{name:<18} {cosine:>8.5} {mse:>12.4e} {max_abs:>10.4} {zero_pct:>9.3}% [{ref_min:>9.3},{ref_max:>9.3}] [{act_min:>9.3},{act_max:>9.3}]"
    );
    Stats {
        cosine,
        mse,
        max_abs,
        zero_pct,
    }
}

fn percentile_sorted(xs: &[f32], p: f32) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let idx = ((xs.len() - 1) as f32 * p).round() as usize;
    xs[idx]
}
