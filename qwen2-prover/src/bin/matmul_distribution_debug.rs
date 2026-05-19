use std::{error::Error, time::Instant};

use qwen2_prover::float::Rotary;

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut bits = 8u8;
    let mut layer_limit = qwen2_prover::LAYERS;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bits" => {
                let value = args.next().ok_or("--bits requires a value")?;
                bits = value.parse()?;
            }
            "--layers" => {
                let value = args.next().ok_or("--layers requires a value")?;
                layer_limit = value.parse::<usize>()?.min(qwen2_prover::LAYERS);
            }
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
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let r = Rotary::new();
    let mut x = qwen2_prover::float::embed_from_safetensors(&st, &ids)?;

    println!("text: {text:?}");
    println!("bits: {bits}");
    println!(
        "{:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "site", "in_p99", "in_max", "w_p99", "w_max", "out_p99", "out_max", "y_step", "outlier"
    );

    for layer in 0..layer_limit {
        let w = qwen2_prover::float::load_layer_from_safetensors(&st, layer)?;
        x = inspect_layer(layer, &x, &w, &r, bits);
    }

    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn inspect_layer(
    layer: usize,
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &Rotary,
    bits: u8,
) -> Vec<f32> {
    let n1 = qwen2_prover::float::rms_norm(x, &w.ln1, qwen2_prover::SEQ, qwen2_prover::HIDDEN);

    let mut q = inspect_matmul(
        layer,
        "q_proj",
        &n1,
        &w.wq,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
        bits,
    );
    qwen2_prover::float::add_rows(&mut q, &w.bq);
    let mut k = inspect_matmul(
        layer,
        "k_proj",
        &n1,
        &w.wk,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
        bits,
    );
    qwen2_prover::float::add_rows(&mut k, &w.bk);
    let mut v = inspect_matmul(
        layer,
        "v_proj",
        &n1,
        &w.wv,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
        bits,
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
    let a = inspect_matmul(
        layer,
        "o_proj",
        &c,
        &w.wo,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
        bits,
    );
    let h = qwen2_prover::float::add(x, &a);
    let n2 = qwen2_prover::float::rms_norm(&h, &w.ln2, qwen2_prover::SEQ, qwen2_prover::HIDDEN);

    let g = inspect_matmul(
        layer,
        "gate_proj",
        &n2,
        &w.wg,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        bits,
    );
    let u = inspect_matmul(
        layer,
        "up_proj",
        &n2,
        &w.wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        bits,
    );
    let m = qwen2_prover::float::silu_mul(&g, &u);
    let d = inspect_matmul(
        layer,
        "down_proj",
        &m,
        &w.wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        bits,
    );
    qwen2_prover::float::add(&h, &d)
}

fn inspect_matmul(
    layer: usize,
    name: &str,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bits: u8,
) -> Vec<f32> {
    let y = qwen2_prover::float::matmul(a, b, m, k, n);
    let input = abs_stats(a);
    let weight = column_abs_stats(b, k, n);
    let output = abs_stats(&y);
    let y_step = mean_row_step(&y, m, n, bits);
    let outlier = output.max / output.p99.max(f32::MIN_POSITIVE);
    println!(
        "L{layer}.{name:<10} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.4} {:>8.2}",
        input.p99, input.max, weight.p99, weight.max, output.p99, output.max, y_step, outlier
    );
    y
}

#[derive(Clone, Copy)]
struct AbsStats {
    p99: f32,
    max: f32,
}

fn abs_stats(xs: &[f32]) -> AbsStats {
    let mut values: Vec<f32> = xs.iter().map(|x| x.abs()).collect();
    values.sort_by(|a, b| a.total_cmp(b));
    let p99 = percentile_sorted(&values, 0.99);
    let max = *values.last().unwrap_or(&0.0);
    AbsStats { p99, max }
}

fn column_abs_stats(xs: &[f32], rows: usize, cols: usize) -> AbsStats {
    let mut maxima = Vec::with_capacity(cols);
    for col in 0..cols {
        let mut max_abs = 0.0f32;
        for row in 0..rows {
            max_abs = max_abs.max(xs[row * cols + col].abs());
        }
        maxima.push(max_abs);
    }
    maxima.sort_by(|a, b| a.total_cmp(b));
    let p99 = percentile_sorted(&maxima, 0.99);
    let max = *maxima.last().unwrap_or(&0.0);
    AbsStats { p99, max }
}

fn mean_row_step(xs: &[f32], rows: usize, cols: usize, bits: u8) -> f32 {
    let q = ((1u32 << bits) - 1) as f32;
    let mut sum = 0.0f32;
    for row in 0..rows {
        let row = &xs[row * cols..(row + 1) * cols];
        let min = row.iter().copied().fold(f32::INFINITY, f32::min);
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        sum += (max - min) / q;
    }
    sum / rows as f32
}

fn percentile_sorted(xs: &[f32], p: f32) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let idx = ((xs.len() - 1) as f32 * p).round() as usize;
    xs[idx]
}
