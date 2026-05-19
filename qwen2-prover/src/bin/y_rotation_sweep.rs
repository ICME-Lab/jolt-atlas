use std::{error::Error, time::Instant};

use qwen2_prover::float::Rotary;

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut seeds = 16u64;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seeds" => seeds = args.next().ok_or("--seeds requires a value")?.parse()?,
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

    let mut gate = vec![RunningStats::default(); seeds as usize + 1];
    let mut up = vec![RunningStats::default(); seeds as usize + 1];
    let mut down = vec![RunningStats::default(); seeds as usize + 1];

    for layer in 0..qwen2_prover::LAYERS {
        let w = qwen2_prover::float::load_layer_from_safetensors(&st, layer)?;
        let (h, n2) = layer_to_h_and_n2(&x, &w, &r);

        let gate_y = qwen2_prover::float::matmul(
            &n2,
            &w.wg,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
        );
        let up_y = qwen2_prover::float::matmul(
            &n2,
            &w.wu,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
        );
        let mid = qwen2_prover::float::silu_mul(&gate_y, &up_y);
        let down_y = qwen2_prover::float::matmul(
            &mid,
            &w.wd,
            qwen2_prover::SEQ,
            qwen2_prover::INTERMEDIATE,
            qwen2_prover::HIDDEN,
        );

        gate[0].extend(&gate_y);
        up[0].extend(&up_y);
        down[0].extend(&down_y);
        for seed in 0..seeds {
            let idx = seed as usize + 1;
            accumulate_rotated_rows(
                &mut gate[idx],
                &gate_y,
                qwen2_prover::SEQ,
                qwen2_prover::INTERMEDIATE,
                seed,
            );
            accumulate_rotated_rows(
                &mut up[idx],
                &up_y,
                qwen2_prover::SEQ,
                qwen2_prover::INTERMEDIATE,
                seed,
            );
            accumulate_rotated_rows(
                &mut down[idx],
                &down_y,
                qwen2_prover::SEQ,
                qwen2_prover::HIDDEN,
                seed,
            );
        }

        x = qwen2_prover::float::add(&h, &down_y);
    }

    println!("text: {text:?}");
    println!("seeds: {seeds}");
    println!("rotation: output-side signed Hadamard, Y' = YR");
    println!("gate/up group: 256, down group: 1024");
    println!();
    print_site("gate_y", &gate);
    print_site("up_y", &up);
    print_site("down_y", &down);
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn print_site(name: &str, stats: &[RunningStats]) {
    let mut rows = stats
        .iter()
        .enumerate()
        .map(|(idx, stat)| {
            let label = if idx == 0 {
                "none".to_string()
            } else {
                format!("seed{}", idx - 1)
            };
            (label, stat.metrics())
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| a.1.kurtosis.total_cmp(&b.1.kurtosis));
    println!("{name}:");
    println!(
        "  {:<8} {:>12} {:>12} {:>12} {:>12}",
        "seed", "kurtosis", "max_abs", "rms", "count"
    );
    for (label, metric) in rows.iter().take(8) {
        println!(
            "  {label:<8} {:>12.5} {:>12.5} {:>12.5} {:>12}",
            metric.kurtosis, metric.max_abs, metric.rms, metric.count
        );
    }
    let none = stats[0].metrics();
    let best = &rows[0];
    println!(
        "  best vs none: kurtosis {:.5} -> {:.5}, max_abs {:.5} -> {:.5}",
        none.kurtosis, best.1.kurtosis, none.max_abs, best.1.max_abs
    );
    println!();
}

fn accumulate_rotated_rows(
    stat: &mut RunningStats,
    xs: &[f32],
    rows: usize,
    cols: usize,
    seed: u64,
) {
    let group = hadamard_group_len(cols);
    let mut tmp = vec![0.0f32; group];
    for row in 0..rows {
        let row_xs = &xs[row * cols..(row + 1) * cols];
        for (block, chunk) in row_xs.chunks_exact(group).enumerate() {
            tmp.copy_from_slice(chunk);
            apply_deterministic_signs(&mut tmp, seed, block as u64);
            hadamard_orthonormal_in_place(&mut tmp);
            stat.extend(&tmp);
        }
    }
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

    fn metrics(&self) -> Metrics {
        let n = self.count.max(1) as f64;
        let mean2 = self.sum2 / n;
        let mean4 = self.sum4 / n;
        Metrics {
            count: self.count,
            rms: mean2.sqrt(),
            kurtosis: mean4 / (mean2 * mean2).max(f64::MIN_POSITIVE),
            max_abs: self.max_abs,
        }
    }
}

struct Metrics {
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

fn hadamard_group_len(k: usize) -> usize {
    if k.is_power_of_two() {
        return k;
    }
    let lowbit = k & k.wrapping_neg();
    lowbit.max(1)
}

fn apply_deterministic_signs(xs: &mut [f32], seed: u64, block: u64) {
    for (i, x) in xs.iter_mut().enumerate() {
        if splitmix_sign(seed ^ block.wrapping_mul(0xD1B54A32D192ED03) ^ i as u64) < 0 {
            *x = -*x;
        }
    }
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
