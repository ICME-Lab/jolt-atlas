use std::{error::Error, time::Instant};

use rayon::prelude::*;

use qwen2_prover::float::Rotary;

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut seeds = 4096u64;
    let mut groups = vec![32usize, 64, 128];
    let mut rounds_list = vec![1usize, 2, 3];
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seeds" => seeds = args.next().ok_or("--seeds requires a value")?.parse()?,
            "--groups" => {
                let value = args.next().ok_or("--groups requires a value")?;
                groups = value
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--rounds" => {
                let value = args.next().ok_or("--rounds requires a value")?;
                rounds_list = value
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<Vec<_>, _>>()?;
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
    let mut down_y =
        Vec::with_capacity(qwen2_prover::LAYERS * qwen2_prover::SEQ * qwen2_prover::HIDDEN);

    for layer in 0..qwen2_prover::LAYERS {
        let w = qwen2_prover::float::load_layer_from_safetensors(&st, layer)?;
        let (h, n2) = layer_to_h_and_n2(&x, &w, &r);
        let gate = qwen2_prover::float::matmul(
            &n2,
            &w.wg,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
        );
        let up = qwen2_prover::float::matmul(
            &n2,
            &w.wu,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
        );
        let mid = qwen2_prover::float::silu_mul(&gate, &up);
        let down = qwen2_prover::float::matmul(
            &mid,
            &w.wd,
            qwen2_prover::SEQ,
            qwen2_prover::INTERMEDIATE,
            qwen2_prover::HIDDEN,
        );
        down_y.extend_from_slice(&down);
        x = qwen2_prover::float::add(&h, &down);
    }

    let none = stats_plain(&down_y).metrics(0, 0, 0);
    let mut configs = Vec::new();
    for &group in &groups {
        if !group.is_power_of_two()
            || group > qwen2_prover::HIDDEN
            || qwen2_prover::HIDDEN % group != 0
        {
            return Err(format!(
                "invalid group {group}; must divide {}",
                qwen2_prover::HIDDEN
            )
            .into());
        }
        for &rounds in &rounds_list {
            if rounds == 0 {
                return Err("rounds must be positive".into());
            }
            for seed in 0..seeds {
                configs.push((group, rounds, seed));
            }
        }
    }

    let mut results = configs
        .par_iter()
        .map(|&(group, rounds, seed)| {
            stats_rotated(&down_y, group, rounds, seed).metrics(group, rounds, seed)
        })
        .collect::<Vec<_>>();

    let mut by_kurtosis = results.clone();
    by_kurtosis.sort_by(|a, b| a.kurtosis.total_cmp(&b.kurtosis));
    let mut by_max_abs = std::mem::take(&mut results);
    by_max_abs.sort_by(|a, b| a.max_abs.total_cmp(&b.max_abs));

    println!("text: {text:?}");
    println!("seeds_per_group: {seeds}");
    println!("groups: {groups:?}");
    println!("rounds: {rounds_list:?}");
    println!("count: {}", none.count);
    println!(
        "none: kurtosis={:.5} max_abs={:.5} rms={:.5}",
        none.kurtosis, none.max_abs, none.rms
    );
    println!();
    println!("best by kurtosis:");
    print_results(&by_kurtosis, 16);
    println!();
    println!("best by max_abs:");
    print_results(&by_max_abs, 16);
    println!();
    println!("Y8 quantization error:");
    let direct = quant_error_direct(
        &down_y,
        qwen2_prover::SEQ * qwen2_prover::LAYERS,
        qwen2_prover::HIDDEN,
    );
    println!(
        "  {:<18} cosine={:.8} mse={:.8} max_err={:.5}",
        "direct", direct.cosine, direct.mse, direct.max_err
    );
    if let Some(best) = by_kurtosis.first() {
        let err = quant_error_rotated(
            &down_y,
            qwen2_prover::SEQ * qwen2_prover::LAYERS,
            qwen2_prover::HIDDEN,
            best.group,
            best.rounds,
            best.seed,
        );
        println!(
            "  {:<18} cosine={:.8} mse={:.8} max_err={:.5}  group={} rounds={} seed={}",
            "best_kurtosis", err.cosine, err.mse, err.max_err, best.group, best.rounds, best.seed
        );
    }
    if let Some(best) = by_max_abs.first() {
        let err = quant_error_rotated(
            &down_y,
            qwen2_prover::SEQ * qwen2_prover::LAYERS,
            qwen2_prover::HIDDEN,
            best.group,
            best.rounds,
            best.seed,
        );
        println!(
            "  {:<18} cosine={:.8} mse={:.8} max_err={:.5}  group={} rounds={} seed={}",
            "best_max_abs", err.cosine, err.mse, err.max_err, best.group, best.rounds, best.seed
        );
    }
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn print_results(xs: &[Metrics], n: usize) {
    println!(
        "  {:>6} {:>6} {:>8} {:>12} {:>12} {:>12}",
        "group", "rounds", "seed", "kurtosis", "max_abs", "rms"
    );
    for x in xs.iter().take(n) {
        println!(
            "  {:>6} {:>6} {:>8} {:>12.5} {:>12.5} {:>12.5}",
            x.group, x.rounds, x.seed, x.kurtosis, x.max_abs, x.rms
        );
    }
}

fn stats_plain(xs: &[f32]) -> RunningStats {
    let mut stat = RunningStats::default();
    stat.extend(xs);
    stat
}

fn stats_rotated(xs: &[f32], group: usize, rounds: usize, seed: u64) -> RunningStats {
    let mut stat = RunningStats::default();
    let mut row_buf = vec![0.0f32; qwen2_prover::HIDDEN];
    let mut perm_buf = vec![0.0f32; qwen2_prover::HIDDEN];
    for row in xs.chunks_exact(qwen2_prover::HIDDEN) {
        row_buf.copy_from_slice(row);
        for round in 0..rounds {
            for (block, chunk) in row_buf.chunks_exact_mut(group).enumerate() {
                apply_deterministic_signs(chunk, seed, round as u64, block as u64);
                hadamard_orthonormal_in_place(chunk);
            }
            if round + 1 < rounds {
                permute_stride(&row_buf, &mut perm_buf, seed, round as u64);
                row_buf.copy_from_slice(&perm_buf);
            }
        }
        stat.extend(&row_buf);
    }
    stat
}

fn quant_error_direct(xs: &[f32], rows: usize, cols: usize) -> ErrorMetrics {
    let quantized = quantize_rows_to_float(xs, rows, cols);
    error_metrics(xs, &quantized)
}

fn quant_error_rotated(
    xs: &[f32],
    rows: usize,
    cols: usize,
    group: usize,
    rounds: usize,
    seed: u64,
) -> ErrorMetrics {
    let mut rotated = xs.to_vec();
    apply_rotation_rows(&mut rotated, rows, cols, group, rounds, seed);
    let mut quantized = quantize_rows_to_float(&rotated, rows, cols);
    apply_rotation_rows_inverse(&mut quantized, rows, cols, group, rounds, seed);
    error_metrics(xs, &quantized)
}

fn apply_rotation_rows(
    xs: &mut [f32],
    rows: usize,
    cols: usize,
    group: usize,
    rounds: usize,
    seed: u64,
) {
    let mut perm_buf = vec![0.0f32; cols];
    for row in 0..rows {
        let row_xs = &mut xs[row * cols..(row + 1) * cols];
        for round in 0..rounds {
            for (block, chunk) in row_xs.chunks_exact_mut(group).enumerate() {
                apply_deterministic_signs(chunk, seed, round as u64, block as u64);
                hadamard_orthonormal_in_place(chunk);
            }
            if round + 1 < rounds {
                permute_stride(row_xs, &mut perm_buf, seed, round as u64);
                row_xs.copy_from_slice(&perm_buf);
            }
        }
    }
}

fn apply_rotation_rows_inverse(
    xs: &mut [f32],
    rows: usize,
    cols: usize,
    group: usize,
    rounds: usize,
    seed: u64,
) {
    let mut perm_buf = vec![0.0f32; cols];
    for row in 0..rows {
        let row_xs = &mut xs[row * cols..(row + 1) * cols];
        for round in (0..rounds).rev() {
            if round + 1 < rounds {
                inverse_permute_stride(row_xs, &mut perm_buf, seed, round as u64);
                row_xs.copy_from_slice(&perm_buf);
            }
            for (block, chunk) in row_xs.chunks_exact_mut(group).enumerate() {
                hadamard_orthonormal_in_place(chunk);
                apply_deterministic_signs(chunk, seed, round as u64, block as u64);
            }
        }
    }
}

fn quantize_rows_to_float(xs: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; xs.len()];
    let q = 255.0f64;
    for row in 0..rows {
        let row_xs = &xs[row * cols..(row + 1) * cols];
        let min = row_xs.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        let max = row_xs.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let scale = ((max - min) / q).max(f64::MIN_POSITIVE);
        let zp = (-min / scale).round();
        for col in 0..cols {
            let idx = row * cols + col;
            let quant = ((xs[idx] as f64 / scale) + zp).floor();
            y[idx] = ((quant - zp) * scale) as f32;
        }
    }
    y
}

fn error_metrics(a: &[f32], b: &[f32]) -> ErrorMetrics {
    let mut dot = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    let mut mse = 0.0f64;
    let mut max_err = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let x = x as f64;
        let y = y as f64;
        let e = x - y;
        dot += x * y;
        aa += x * x;
        bb += y * y;
        mse += e * e;
        max_err = max_err.max(e.abs());
    }
    ErrorMetrics {
        cosine: dot / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE),
        mse: mse / a.len().max(1) as f64,
        max_err,
    }
}

struct ErrorMetrics {
    cosine: f64,
    mse: f64,
    max_err: f64,
}

#[derive(Clone, Default)]
struct RunningStats {
    count: usize,
    sum2: f64,
    sum4: f64,
    max_abs: f64,
}

impl RunningStats {
    fn extend(&mut self, xs: &[f32]) {
        for &x in xs {
            let x = x as f64;
            let x2 = x * x;
            self.count += 1;
            self.sum2 += x2;
            self.sum4 += x2 * x2;
            self.max_abs = self.max_abs.max(x.abs());
        }
    }

    fn metrics(&self, group: usize, rounds: usize, seed: u64) -> Metrics {
        let n = self.count.max(1) as f64;
        let mean2 = self.sum2 / n;
        let mean4 = self.sum4 / n;
        Metrics {
            group,
            rounds,
            seed,
            count: self.count,
            rms: mean2.sqrt(),
            kurtosis: mean4 / (mean2 * mean2).max(f64::MIN_POSITIVE),
            max_abs: self.max_abs,
        }
    }
}

#[derive(Clone)]
struct Metrics {
    group: usize,
    rounds: usize,
    seed: u64,
    count: usize,
    rms: f64,
    kurtosis: f64,
    max_abs: f64,
}

fn layer_to_h_and_n2(
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
    (h, n2)
}

fn apply_deterministic_signs(xs: &mut [f32], seed: u64, round: u64, block: u64) {
    for (i, x) in xs.iter_mut().enumerate() {
        let key = seed
            ^ round.wrapping_mul(0x9E3779B97F4A7C15)
            ^ block.wrapping_mul(0xD1B54A32D192ED03)
            ^ i as u64;
        if splitmix_sign(key) < 0 {
            *x = -*x;
        }
    }
}

fn permute_stride(src: &[f32], dst: &mut [f32], seed: u64, round: u64) {
    let n = src.len();
    let strides = [31usize, 95, 191, 255, 383, 447, 639, 831];
    let stride =
        strides[((seed ^ round.wrapping_mul(0xBF58476D1CE4E5B9)) as usize) % strides.len()];
    debug_assert_eq!(gcd(stride, n), 1);
    let offset = splitmix64(seed ^ round.wrapping_mul(0x94D049BB133111EB)) as usize % n;
    for i in 0..n {
        dst[i] = src[(offset + i * stride) % n];
    }
}

fn inverse_permute_stride(src: &[f32], dst: &mut [f32], seed: u64, round: u64) {
    let n = src.len();
    let strides = [31usize, 95, 191, 255, 383, 447, 639, 831];
    let stride =
        strides[((seed ^ round.wrapping_mul(0xBF58476D1CE4E5B9)) as usize) % strides.len()];
    debug_assert_eq!(gcd(stride, n), 1);
    let offset = splitmix64(seed ^ round.wrapping_mul(0x94D049BB133111EB)) as usize % n;
    for i in 0..n {
        dst[(offset + i * stride) % n] = src[i];
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

fn splitmix_sign(mut x: u64) -> i32 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    if ((x ^ (x >> 31)) & 1) == 0 { 1 } else { -1 }
}

fn hadamard_orthonormal_in_place(xs: &mut [f32]) {
    debug_assert!(xs.len().is_power_of_two());
    let n = xs.len();
    let mut step = 1;
    while step < n {
        for base in (0..n).step_by(step * 2) {
            for i in 0..step {
                let a = xs[base + i];
                let b = xs[base + step + i];
                xs[base + i] = a + b;
                xs[base + step + i] = a - b;
            }
        }
        step *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for x in xs {
        *x *= scale;
    }
}
