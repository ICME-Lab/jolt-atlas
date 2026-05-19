use std::{error::Error, time::Instant};

use qwen2_prover::float::Rotary;

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut bits = 8u8;
    let mut mode = "none".to_string();
    let mut threshold = 32.0f32;
    let mut frac_bits = 16u32;
    let mut a_outlier_threshold = None;
    let mut a_frac_bits = 16u32;
    let mut n2_top = 256usize;
    let mut exponent = 0.75f32;
    let mut percentile = 0.95f32;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bits" => bits = args.next().ok_or("--bits requires a value")?.parse()?,
            "--mode" => mode = args.next().ok_or("--mode requires a value")?,
            "--threshold" => {
                threshold = args.next().ok_or("--threshold requires a value")?.parse()?
            }
            "--frac-bits" => {
                frac_bits = args.next().ok_or("--frac-bits requires a value")?.parse()?
            }
            "--a-outlier-threshold" => {
                a_outlier_threshold = Some(
                    args.next()
                        .ok_or("--a-outlier-threshold requires a value")?
                        .parse()?,
                )
            }
            "--a-frac-bits" => {
                a_frac_bits = args
                    .next()
                    .ok_or("--a-frac-bits requires a value")?
                    .parse()?
            }
            "--n2-top" => n2_top = args.next().ok_or("--n2-top requires a value")?.parse()?,
            "--exponent" => exponent = args.next().ok_or("--exponent requires a value")?.parse()?,
            "--percentile" => {
                percentile = args
                    .next()
                    .ok_or("--percentile requires a value")?
                    .parse()?
            }
            other => words.push(other.to_string()),
        }
    }
    if mode != "none" && mode != "n2" {
        return Err("--mode must be none or n2".into());
    }

    let text = if words.is_empty() {
        "mathematics and proofs require careful reasoning".to_string()
    } else {
        words.join(" ")
    };

    let t = Instant::now();
    let ids = qwen2_prover::text::tokenize(&tok, &text)?;
    let bytes = std::fs::read(model)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let r = Rotary::new();
    let mut x = qwen2_prover::float::embed_from_safetensors(&st, &ids)?;
    let q = (1i128 << bits) - 1;

    let mut m_no_yq_vs_float = Vec::new();
    let mut m_y8_vs_float = Vec::new();
    let mut m_split_vs_float = Vec::new();
    let mut m_y8_vs_no_yq = Vec::new();
    let mut m_split_vs_no_yq = Vec::new();
    let mut outlier_counts = Vec::new();

    for layer in 0..qwen2_prover::LAYERS {
        let w = qwen2_prover::float::load_layer_from_safetensors(&st, layer)?;
        let (next, mut n2) = layer_to_n2_and_next(&x, &w, &r);
        x = next;
        let mut wg = w.wg.clone();
        if mode == "n2" {
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
        }

        let y_float = qwen2_prover::float::matmul(
            &n2,
            &wg,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
        );
        let y_no_yq = if let Some(a_threshold) = a_outlier_threshold {
            matmul_token_channel_a_i32_no_yq(
                &n2,
                &wg,
                qwen2_prover::SEQ,
                qwen2_prover::HIDDEN,
                qwen2_prover::INTERMEDIATE,
                q,
                a_threshold,
                a_frac_bits,
            )
        } else {
            matmul_token_channel_no_yq(
                &n2,
                &wg,
                qwen2_prover::SEQ,
                qwen2_prover::HIDDEN,
                qwen2_prover::INTERMEDIATE,
                q,
            )
        };
        let y8 = quantize_rows_to_float(&y_no_yq, qwen2_prover::SEQ, qwen2_prover::INTERMEDIATE, q);
        let (ysplit, outliers) = quantize_output_with_i32_outliers(
            &y_no_yq,
            qwen2_prover::SEQ,
            qwen2_prover::INTERMEDIATE,
            q,
            threshold,
            frac_bits,
        );

        m_no_yq_vs_float.push(error_metrics(&y_float, &y_no_yq));
        m_y8_vs_float.push(error_metrics(&y_float, &y8));
        m_split_vs_float.push(error_metrics(&y_float, &ysplit));
        m_y8_vs_no_yq.push(error_metrics(&y_no_yq, &y8));
        m_split_vs_no_yq.push(error_metrics(&y_no_yq, &ysplit));
        outlier_counts.push(outliers);
    }

    println!("text: {text:?}");
    println!("bits: {bits}");
    println!("mode: {mode}");
    println!("threshold: {threshold}");
    println!("frac_bits: {frac_bits}");
    println!("a_outlier_threshold: {a_outlier_threshold:?}");
    println!("a_frac_bits: {a_frac_bits}");
    println!("n2_top: {n2_top}");
    println!("exponent: {exponent}");
    println!("percentile: {percentile}");
    println!();
    println!(
        "{:<20} {:>10} {:>12} {:>10}",
        "comparison", "cosine", "mse", "max_err"
    );
    print_avg("noYQ vs float", &m_no_yq_vs_float);
    print_avg("Y8 vs float", &m_y8_vs_float);
    print_avg("Ysplit vs float", &m_split_vs_float);
    print_avg("Y8 vs noYQ", &m_y8_vs_no_yq);
    print_avg("Ysplit vs noYQ", &m_split_vs_no_yq);
    let total_outliers = outlier_counts.iter().sum::<usize>();
    let total = qwen2_prover::LAYERS * qwen2_prover::SEQ * qwen2_prover::INTERMEDIATE;
    println!();
    println!(
        "outliers: {total_outliers}/{total} ({:.5}%)",
        total_outliers as f64 * 100.0 / total as f64
    );
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn layer_to_n2_and_next(
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &Rotary,
) -> (Vec<f32>, Vec<f32>) {
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
    (qwen2_prover::float::add(&h, &d), n2)
}

fn matmul_token_channel_no_yq(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    q: i128,
) -> Vec<f32> {
    let mut a_int = vec![0i128; a.len()];
    let mut a_zp = vec![0i128; m];
    let mut a_scale = vec![0.0f64; m];
    for row in 0..m {
        let xs = &a[row * k..(row + 1) * k];
        let (scale, zp) = quant_params_f32(xs, q);
        a_scale[row] = scale;
        a_zp[row] = zp;
        for col in 0..k {
            a_int[row * k + col] = quantize_value(a[row * k + col], scale, zp);
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
            b_int[row * n + col] = quantize_value(b[row * n + col], scale, zp);
        }
    }

    let mut y = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i128;
            for inner in 0..k {
                let lhs = a_int[row * k + inner] - a_zp[row];
                let rhs = b_int[inner * n + col] - b_zp[col];
                acc += lhs * rhs;
            }
            y[row * n + col] = (acc as f64 * a_scale[row] * b_scale[col]) as f32;
        }
    }
    y
}

fn matmul_token_channel_a_i32_no_yq(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    q: i128,
    a_threshold: f32,
    a_frac_bits: u32,
) -> Vec<f32> {
    let fixed_scale = (1u64 << a_frac_bits) as f64;
    let mut a_small = vec![0.0f32; a.len()];
    let mut a_outlier = vec![0i32; a.len()];
    for idx in 0..a.len() {
        if a[idx].abs() > a_threshold {
            a_outlier[idx] = (a[idx] as f64 * fixed_scale).round() as i32;
        } else {
            a_small[idx] = a[idx];
        }
    }

    let mut a_int = vec![0i128; a.len()];
    let mut a_zp = vec![0i128; m];
    let mut a_scale = vec![0.0f64; m];
    for row in 0..m {
        let xs = &a_small[row * k..(row + 1) * k];
        let (scale, zp) = quant_params_f32(xs, q);
        a_scale[row] = scale;
        a_zp[row] = zp;
        for col in 0..k {
            a_int[row * k + col] = quantize_value(a_small[row * k + col], scale, zp);
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
            b_int[row * n + col] = quantize_value(b[row * n + col], scale, zp);
        }
    }

    let mut y = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut small_acc = 0i128;
            let mut outlier_acc = 0.0f64;
            for inner in 0..k {
                let rhs_int = b_int[inner * n + col] - b_zp[col];
                small_acc += (a_int[row * k + inner] - a_zp[row]) * rhs_int;
                let a_out = a_outlier[row * k + inner];
                if a_out != 0 {
                    outlier_acc += (a_out as f64 / fixed_scale) * (rhs_int as f64 * b_scale[col]);
                }
            }
            y[row * n + col] =
                (small_acc as f64 * a_scale[row] * b_scale[col] + outlier_acc) as f32;
        }
    }
    y
}

fn quantize_output_with_i32_outliers(
    y_real: &[f32],
    rows: usize,
    cols: usize,
    q: i128,
    threshold: f32,
    frac_bits: u32,
) -> (Vec<f32>, usize) {
    let mut small = vec![0.0f32; rows * cols];
    let mut outlier = vec![0i32; rows * cols];
    let scale = (1u64 << frac_bits) as f64;
    let mut count = 0usize;
    for idx in 0..y_real.len() {
        let value = y_real[idx];
        if value.abs() > threshold {
            count += 1;
            outlier[idx] = (value as f64 * scale).round() as i32;
        } else {
            small[idx] = value;
        }
    }
    let mut y = quantize_rows_to_float(&small, rows, cols, q);
    for idx in 0..y.len() {
        if outlier[idx] != 0 {
            y[idx] += (outlier[idx] as f64 / scale) as f32;
        }
    }
    (y, count)
}

fn quantize_rows_to_float(xs: &[f32], rows: usize, cols: usize, q: i128) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let row_xs = &xs[row * cols..(row + 1) * cols];
        let (scale, zp) = quant_params_f32(row_xs, q);
        for col in 0..cols {
            let value = quantize_value(xs[row * cols + col], scale, zp);
            y[row * cols + col] = ((value - zp) as f64 * scale) as f32;
        }
    }
    y
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

fn print_avg(label: &str, xs: &[Metrics]) {
    let n = xs.len().max(1) as f64;
    let cosine = xs.iter().map(|x| x.cosine).sum::<f64>() / n;
    let mse = xs.iter().map(|x| x.mse).sum::<f64>() / n;
    let max_err = xs.iter().map(|x| x.max_err).sum::<f64>() / n;
    println!("{label:<20} {cosine:>10.7} {mse:>12.5e} {max_err:>10.5}");
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

fn quantize_value(value: f32, scale: f64, zero_point: i128) -> i128 {
    ((value as f64 / scale) + zero_point as f64).floor() as i128
}

fn percentile_sorted(xs: &[f32], p: f32) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let idx = ((xs.len() - 1) as f32 * p).round() as usize;
    xs[idx]
}
