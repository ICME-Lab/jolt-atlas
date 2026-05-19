use std::{error::Error, time::Instant};

use qwen2_prover::float::Rotary;

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        words.push(arg);
    }
    let text = if words.is_empty() {
        "hello world this is a test".to_string()
    } else {
        words.join(" ")
    };

    let t = Instant::now();
    let ids = qwen2_prover::text::tokenize(&tok, &text)?;
    let bytes = std::fs::read(model)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let r = Rotary::new();
    let mut x = qwen2_prover::float::embed_from_safetensors(&st, &ids)?;

    let mut x_stats = Vec::new();
    let mut n1_stats = Vec::new();
    let mut h_stats = Vec::new();
    let mut n2_stats = Vec::new();

    println!("text: {text:?}");
    println!(
        "{:<5} {:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}",
        "layer", "tensor", "rms", "p99", "p999", "max_abs", ">16", ">32", "kurtosis"
    );

    for layer in 0..qwen2_prover::LAYERS {
        let w = qwen2_prover::float::load_layer_from_safetensors(&st, layer)?;
        let x_s = SpreadStats::from(&x);
        let n1 = qwen2_prover::float::rms_norm(&x, &w.ln1, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
        let n1_s = SpreadStats::from(&n1);

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
        let h = qwen2_prover::float::add(&x, &a);
        let h_s = SpreadStats::from(&h);
        let n2 = qwen2_prover::float::rms_norm(&h, &w.ln2, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
        let n2_s = SpreadStats::from(&n2);

        print_row(layer, "x", &x_s);
        print_row(layer, "n1", &n1_s);
        print_row(layer, "h", &h_s);
        print_row(layer, "n2", &n2_s);

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
        let d = qwen2_prover::float::matmul(
            &m,
            &w.wd,
            qwen2_prover::SEQ,
            qwen2_prover::INTERMEDIATE,
            qwen2_prover::HIDDEN,
        );
        x = qwen2_prover::float::add(&h, &d);

        x_stats.push(x_s);
        n1_stats.push(n1_s);
        h_stats.push(h_s);
        n2_stats.push(n2_s);
    }

    println!();
    println!("summary over layers:");
    print_summary("x", &x_stats);
    print_summary("n1", &n1_stats);
    print_summary("h", &h_stats);
    print_summary("n2", &n2_stats);
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn print_row(layer: usize, name: &str, s: &SpreadStats) {
    println!(
        "{layer:<5} {name:<8} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10} {:>10} {:>12.2}",
        s.rms, s.p99, s.p999, s.max_abs, s.gt_16, s.gt_32, s.kurtosis
    );
}

fn print_summary(name: &str, xs: &[SpreadStats]) {
    let n = xs.len().max(1) as f64;
    let avg = |f: fn(&SpreadStats) -> f64| xs.iter().map(f).sum::<f64>() / n;
    let max = |f: fn(&SpreadStats) -> f64| xs.iter().map(f).fold(0.0, f64::max);
    let gt_16 = xs.iter().map(|s| s.gt_16).sum::<usize>();
    let gt_32 = xs.iter().map(|s| s.gt_32).sum::<usize>();
    println!(
        "{name:<8} avg_rms={:.4} avg_p99={:.4} avg_p999={:.4} max_abs={:.4} total>16={} total>32={} avg_kurtosis={:.2} max_kurtosis={:.2}",
        avg(|s| s.rms),
        avg(|s| s.p99),
        avg(|s| s.p999),
        max(|s| s.max_abs),
        gt_16,
        gt_32,
        avg(|s| s.kurtosis),
        max(|s| s.kurtosis),
    );
}

#[derive(Clone)]
struct SpreadStats {
    rms: f64,
    p99: f64,
    p999: f64,
    max_abs: f64,
    gt_16: usize,
    gt_32: usize,
    kurtosis: f64,
}

impl SpreadStats {
    fn from(xs: &[f32]) -> Self {
        let mut abs = xs.iter().map(|x| x.abs()).collect::<Vec<_>>();
        abs.sort_by(|a, b| a.total_cmp(b));
        let mut sum2 = 0.0f64;
        let mut sum4 = 0.0f64;
        let mut gt_16 = 0usize;
        let mut gt_32 = 0usize;
        for &x in xs {
            let x = x as f64;
            let x2 = x * x;
            sum2 += x2;
            sum4 += x2 * x2;
            if x.abs() > 16.0 {
                gt_16 += 1;
            }
            if x.abs() > 32.0 {
                gt_32 += 1;
            }
        }
        let n = xs.len().max(1) as f64;
        let mean2 = sum2 / n;
        let mean4 = sum4 / n;
        Self {
            rms: mean2.sqrt(),
            p99: percentile_sorted(&abs, 0.99) as f64,
            p999: percentile_sorted(&abs, 0.999) as f64,
            max_abs: abs.last().copied().unwrap_or(0.0) as f64,
            gt_16,
            gt_32,
            kurtosis: mean4 / (mean2 * mean2).max(f64::MIN_POSITIVE),
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
