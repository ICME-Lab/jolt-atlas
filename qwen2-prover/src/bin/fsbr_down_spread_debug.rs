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
    let mut percentile = 0.99f32;
    let mut layer_limit = qwen2_prover::LAYERS;
    let mut words = Vec::new();
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
    println!("percentile: {percentile}");
    println!(
        "{:<12} {:<5} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "site",
        "kind",
        "min",
        "max",
        "p50",
        "p90",
        "p99",
        "p999",
        "max_abs",
        ">2p99",
        ">4p99",
        ">8p99",
        ">16",
        ">32",
        ">64",
        ">100"
    );

    let mut float_summary = Vec::new();
    let mut quant_summary = Vec::new();
    for layer in 0..layer_limit {
        let w = qwen2_prover::float::load_layer_from_safetensors(&st, layer)?;
        let (next, down_f, down_q) =
            inspect_layer(&x, &w, &r, cfg, n2_top, mid_top, exponent, percentile);
        x = next;

        let name = format!("L{layer}.down");
        let fs = print_spread(&name, "float", &down_f);
        let qs = print_spread(&name, "di", &down_q);
        float_summary.push(fs);
        quant_summary.push(qs);
    }

    println!();
    print_summary("float", &float_summary);
    print_summary("di", &quant_summary);
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn inspect_layer(
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &Rotary,
    cfg: DiConfig,
    n2_top: usize,
    mid_top: usize,
    exponent: f32,
    percentile: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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
        percentile,
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
    let up_q = qwen2_prover::float::matmul_hybrid_for_debug(
        &n2,
        &wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        HybridOp::MatmulTokenChannel,
        cfg,
    );

    let gate_f = qwen2_prover::float::matmul(
        &n2,
        &wg,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
    );
    let up_f = qwen2_prover::float::matmul(
        &n2,
        &wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
    );

    let mut mid_q = qwen2_prover::float::silu_mul(&gate_q, &up_q);
    let mut mid_f = qwen2_prover::float::silu_mul(&gate_f, &up_f);
    let mut wd = w.wd.clone();
    let mut wd_f = w.wd.clone();
    smooth_matmul_input(
        &mut mid_q,
        &mut [&mut wd],
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        mid_top,
        exponent,
        percentile,
    );
    smooth_matmul_input(
        &mut mid_f,
        &mut [&mut wd_f],
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        mid_top,
        exponent,
        percentile,
    );

    let down_f = qwen2_prover::float::matmul(
        &mid_f,
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
    let next = qwen2_prover::float::add(&h, &down_q);
    (next, down_f, down_q)
}

fn smooth_matmul_input(
    x: &mut [f32],
    weights: &mut [&mut Vec<f32>],
    rows: usize,
    input_cols: usize,
    output_cols: usize,
    top: usize,
    exponent: f32,
    percentile: f32,
) {
    let mut channel_max = vec![0.0f32; input_cols];
    for row in 0..rows {
        for col in 0..input_cols {
            channel_max[col] = channel_max[col].max(x[row * input_cols + col].abs());
        }
    }
    let mut sorted = channel_max.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let target = percentile_sorted(&sorted, percentile).max(1e-12);
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

#[derive(Clone, Copy)]
struct Spread {
    min: f32,
    max: f32,
    p50: f32,
    p90: f32,
    p99: f32,
    p999: f32,
    max_abs: f32,
    gt_2p99: usize,
    gt_4p99: usize,
    gt_8p99: usize,
    gt_16: usize,
    gt_32: usize,
    gt_64: usize,
    gt_100: usize,
}

fn print_spread(name: &str, kind: &str, xs: &[f32]) -> Spread {
    let spread = spread(xs);
    println!(
        "{name:<12} {kind:<5} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>9.3} {:>9.3} {:>8.3} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        spread.min,
        spread.max,
        spread.p50,
        spread.p90,
        spread.p99,
        spread.p999,
        spread.max_abs,
        spread.gt_2p99,
        spread.gt_4p99,
        spread.gt_8p99,
        spread.gt_16,
        spread.gt_32,
        spread.gt_64,
        spread.gt_100
    );
    spread
}

fn spread(xs: &[f32]) -> Spread {
    let min = xs.iter().copied().fold(f32::INFINITY, f32::min);
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut abs = xs.iter().map(|x| x.abs()).collect::<Vec<_>>();
    abs.sort_by(|a, b| a.total_cmp(b));
    let p50 = percentile_sorted(&abs, 0.50);
    let p90 = percentile_sorted(&abs, 0.90);
    let p99 = percentile_sorted(&abs, 0.99);
    let p999 = percentile_sorted(&abs, 0.999);
    let max_abs = *abs.last().unwrap_or(&0.0);
    let gt_2p99 = abs.iter().filter(|&&x| x > 2.0 * p99).count();
    let gt_4p99 = abs.iter().filter(|&&x| x > 4.0 * p99).count();
    let gt_8p99 = abs.iter().filter(|&&x| x > 8.0 * p99).count();
    let gt_16 = abs.iter().filter(|&&x| x > 16.0).count();
    let gt_32 = abs.iter().filter(|&&x| x > 32.0).count();
    let gt_64 = abs.iter().filter(|&&x| x > 64.0).count();
    let gt_100 = abs.iter().filter(|&&x| x > 100.0).count();
    Spread {
        min,
        max,
        p50,
        p90,
        p99,
        p999,
        max_abs,
        gt_2p99,
        gt_4p99,
        gt_8p99,
        gt_16,
        gt_32,
        gt_64,
        gt_100,
    }
}

fn print_summary(kind: &str, xs: &[Spread]) {
    let n = xs.len().max(1) as f32;
    let avg_p99 = xs.iter().map(|s| s.p99).sum::<f32>() / n;
    let avg_p999 = xs.iter().map(|s| s.p999).sum::<f32>() / n;
    let max_abs = xs.iter().map(|s| s.max_abs).fold(0.0f32, f32::max);
    let gt_2p99 = xs.iter().map(|s| s.gt_2p99).sum::<usize>();
    let gt_4p99 = xs.iter().map(|s| s.gt_4p99).sum::<usize>();
    let gt_8p99 = xs.iter().map(|s| s.gt_8p99).sum::<usize>();
    let gt_16 = xs.iter().map(|s| s.gt_16).sum::<usize>();
    let gt_32 = xs.iter().map(|s| s.gt_32).sum::<usize>();
    let gt_64 = xs.iter().map(|s| s.gt_64).sum::<usize>();
    let gt_100 = xs.iter().map(|s| s.gt_100).sum::<usize>();
    println!(
        "summary {kind}: avg_p99={avg_p99:.3} avg_p999={avg_p999:.3} max_abs={max_abs:.3} total>2p99={gt_2p99} total>4p99={gt_4p99} total>8p99={gt_8p99} total>16={gt_16} total>32={gt_32} total>64={gt_64} total>100={gt_100}"
    );
}

fn percentile_sorted(xs: &[f32], p: f32) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let idx = ((xs.len() - 1) as f32 * p).round() as usize;
    xs[idx]
}
