use std::{error::Error, time::Instant};

use qwen2_prover::float::Rotary;

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut bits = 8u8;
    let mut n2_top = 256usize;
    let mut exponent = 0.75f32;
    let mut percentile = 0.95f32;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bits" => bits = args.next().ok_or("--bits requires a value")?.parse()?,
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

    let mut before_scales = Vec::new();
    let mut after_scales = Vec::new();
    let mut ratios = Vec::new();
    let mut before_ranges = Vec::new();
    let mut after_ranges = Vec::new();

    println!("text: {text:?}");
    println!("tokens: {}", ids.len());
    println!("bits: {bits}");
    println!("n2_top: {n2_top}");
    println!("exponent: {exponent}");
    println!("percentile: {percentile}");
    println!(
        "{:<7} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "layer", "scale_p50", "scale_max", "after_p50", "after_max", "ratio_p50", "ratio_max"
    );

    for layer in 0..qwen2_prover::LAYERS {
        let w = qwen2_prover::float::load_layer_from_safetensors(&st, layer)?;
        let (next, mut n2) = layer_to_n2(&x, &w, &r);
        x = next;

        let before = row_scales(&n2, qwen2_prover::SEQ, qwen2_prover::HIDDEN, bits);
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
        let after = row_scales(&n2, qwen2_prover::SEQ, qwen2_prover::HIDDEN, bits);
        let layer_ratios = before
            .iter()
            .zip(&after)
            .map(|(b, a)| b.scale / a.scale.max(f32::MIN_POSITIVE))
            .collect::<Vec<_>>();

        for item in &before {
            before_scales.push(item.scale);
            before_ranges.push(item.range);
        }
        for item in &after {
            after_scales.push(item.scale);
            after_ranges.push(item.range);
        }
        ratios.extend(layer_ratios.iter().copied());

        let mut before_layer = before.iter().map(|s| s.scale).collect::<Vec<_>>();
        let mut after_layer = after.iter().map(|s| s.scale).collect::<Vec<_>>();
        before_layer.sort_by(|a, b| a.total_cmp(b));
        after_layer.sort_by(|a, b| a.total_cmp(b));
        let mut layer_ratios_sorted = layer_ratios;
        layer_ratios_sorted.sort_by(|a, b| a.total_cmp(b));
        println!(
            "L{layer:<6} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>10.3} {:>10.3}",
            percentile_sorted(&before_layer, 0.50),
            *before_layer.last().unwrap_or(&0.0),
            percentile_sorted(&after_layer, 0.50),
            *after_layer.last().unwrap_or(&0.0),
            percentile_sorted(&layer_ratios_sorted, 0.50),
            *layer_ratios_sorted.last().unwrap_or(&0.0),
        );
    }

    before_scales.sort_by(|a, b| a.total_cmp(b));
    after_scales.sort_by(|a, b| a.total_cmp(b));
    ratios.sort_by(|a, b| a.total_cmp(b));
    before_ranges.sort_by(|a, b| a.total_cmp(b));
    after_ranges.sort_by(|a, b| a.total_cmp(b));
    println!();
    println!("summary:");
    println!(
        "scale before p50/p90/p99/max: {:.6} {:.6} {:.6} {:.6}",
        percentile_sorted(&before_scales, 0.50),
        percentile_sorted(&before_scales, 0.90),
        percentile_sorted(&before_scales, 0.99),
        *before_scales.last().unwrap_or(&0.0),
    );
    println!(
        "scale after  p50/p90/p99/max: {:.6} {:.6} {:.6} {:.6}",
        percentile_sorted(&after_scales, 0.50),
        percentile_sorted(&after_scales, 0.90),
        percentile_sorted(&after_scales, 0.99),
        *after_scales.last().unwrap_or(&0.0),
    );
    println!(
        "before/after scale ratio p50/p90/p99/max: {:.3} {:.3} {:.3} {:.3}",
        percentile_sorted(&ratios, 0.50),
        percentile_sorted(&ratios, 0.90),
        percentile_sorted(&ratios, 0.99),
        *ratios.last().unwrap_or(&0.0),
    );
    println!(
        "range before p50/p90/p99/max: {:.3} {:.3} {:.3} {:.3}",
        percentile_sorted(&before_ranges, 0.50),
        percentile_sorted(&before_ranges, 0.90),
        percentile_sorted(&before_ranges, 0.99),
        *before_ranges.last().unwrap_or(&0.0),
    );
    println!(
        "range after  p50/p90/p99/max: {:.3} {:.3} {:.3} {:.3}",
        percentile_sorted(&after_ranges, 0.50),
        percentile_sorted(&after_ranges, 0.90),
        percentile_sorted(&after_ranges, 0.99),
        *after_ranges.last().unwrap_or(&0.0),
    );
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn layer_to_n2(
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

#[derive(Clone, Copy)]
struct RowScale {
    scale: f32,
    range: f32,
}

fn row_scales(xs: &[f32], rows: usize, cols: usize, bits: u8) -> Vec<RowScale> {
    let q = ((1u32 << bits) - 1) as f32;
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_xs = &xs[row * cols..(row + 1) * cols];
        let min = row_xs.iter().copied().fold(f32::INFINITY, f32::min);
        let max = row_xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        out.push(RowScale {
            scale: range / q,
            range,
        });
    }
    out
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

fn percentile_sorted(xs: &[f32], p: f32) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let idx = ((xs.len() - 1) as f32 * p).round() as usize;
    xs[idx]
}
