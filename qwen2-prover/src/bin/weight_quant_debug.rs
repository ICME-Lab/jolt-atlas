use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut layer_limit = qwen2_prover::LAYERS;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--layers" => {
                layer_limit = args
                    .next()
                    .ok_or("--layers requires a value")?
                    .parse::<usize>()?
                    .min(qwen2_prover::LAYERS)
            }
            _ => return Err(format!("unknown arg {arg}").into()),
        }
    }

    let t = Instant::now();
    let bytes = std::fs::read(model)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;

    println!(
        "{:<12} {:>4} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10} {:>10} {:>10}",
        "site",
        "bits",
        "col_p50",
        "col_p90",
        "col_p99",
        "col_max",
        "step_p99",
        "step_max",
        "cos",
        "mse",
        "max_err"
    );

    for site in ["gate", "up", "down"] {
        for bits in [6u8, 8, 10, 12, 14, 16] {
            let mut all = Vec::new();
            let mut col_ranges = Vec::new();
            let mut col_steps = Vec::new();
            for layer in 0..layer_limit {
                let w = qwen2_prover::float::load_layer_from_safetensors(&st, layer)?;
                let (xs, rows, cols) = match site {
                    "gate" => (&w.wg[..], qwen2_prover::HIDDEN, qwen2_prover::INTERMEDIATE),
                    "up" => (&w.wu[..], qwen2_prover::HIDDEN, qwen2_prover::INTERMEDIATE),
                    "down" => (&w.wd[..], qwen2_prover::INTERMEDIATE, qwen2_prover::HIDDEN),
                    _ => unreachable!(),
                };
                let q = (1i128 << bits) - 1;
                let deq = quantize_weight_cols(xs, rows, cols, q, &mut col_ranges, &mut col_steps);
                all.push(error_metrics(xs, &deq));
            }
            col_ranges.sort_by(|a, b| a.total_cmp(b));
            col_steps.sort_by(|a, b| a.total_cmp(b));
            let avg = avg_metrics(&all);
            println!(
                "{site:<12} {bits:>4} {:>9.5} {:>9.5} {:>9.5} {:>9.5} {:>9.6} {:>9.6} {:>10.7} {:>10.3e} {:>10.5}",
                percentile_sorted(&col_ranges, 0.50),
                percentile_sorted(&col_ranges, 0.90),
                percentile_sorted(&col_ranges, 0.99),
                *col_ranges.last().unwrap_or(&0.0),
                percentile_sorted(&col_steps, 0.99),
                *col_steps.last().unwrap_or(&0.0),
                avg.cosine,
                avg.mse,
                avg.max_err
            );
        }
    }

    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn quantize_weight_cols(
    xs: &[f32],
    rows: usize,
    cols: usize,
    q: i128,
    col_ranges: &mut Vec<f32>,
    col_steps: &mut Vec<f32>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; xs.len()];
    for col in 0..cols {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for row in 0..rows {
            let value = xs[row * cols + col];
            min = min.min(value);
            max = max.max(value);
        }
        let range = max - min;
        let scale = if range == 0.0 {
            1.0 / q as f32
        } else {
            range / q as f32
        };
        let zp = (-(min / scale)).round() as i128;
        col_ranges.push(range);
        col_steps.push(scale);
        for row in 0..rows {
            let idx = row * cols + col;
            let y = ((xs[idx] / scale) as f64 + zp as f64).floor() as i128;
            out[idx] = ((y - zp) as f32) * scale;
        }
    }
    out
}

#[derive(Clone, Copy, Default)]
struct Metrics {
    cosine: f64,
    mse: f64,
    max_err: f64,
}

fn error_metrics(reference: &[f32], actual: &[f32]) -> Metrics {
    let mut dot = 0.0;
    let mut rn = 0.0;
    let mut an = 0.0;
    let mut sq = 0.0;
    let mut max_err = 0.0f64;
    for (&r, &a) in reference.iter().zip(actual) {
        let r = r as f64;
        let a = a as f64;
        dot += r * a;
        rn += r * r;
        an += a * a;
        let diff = r - a;
        sq += diff * diff;
        max_err = max_err.max(diff.abs());
    }
    Metrics {
        cosine: dot / (rn.sqrt() * an.sqrt()).max(f64::MIN_POSITIVE),
        mse: sq / reference.len() as f64,
        max_err,
    }
}

fn avg_metrics(xs: &[Metrics]) -> Metrics {
    let n = xs.len().max(1) as f64;
    Metrics {
        cosine: xs.iter().map(|x| x.cosine).sum::<f64>() / n,
        mse: xs.iter().map(|x| x.mse).sum::<f64>() / n,
        max_err: xs.iter().map(|x| x.max_err).sum::<f64>() / n,
    }
}

fn percentile_sorted(xs: &[f32], p: f32) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let idx = ((xs.len() - 1) as f32 * p).round() as usize;
    xs[idx]
}
