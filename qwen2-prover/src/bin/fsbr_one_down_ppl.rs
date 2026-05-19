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
    let mut layer = 22usize;
    let mut top = 4usize;
    let mut sqrt_alpha = false;
    let mut max_targets = usize::MAX;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bits" => {
                let value = args.next().ok_or("--bits requires a value")?;
                bits = value.parse()?;
            }
            "--layer" => {
                let value = args.next().ok_or("--layer requires a value")?;
                layer = value.parse::<usize>()?.min(qwen2_prover::LAYERS - 1);
            }
            "--top" => {
                let value = args.next().ok_or("--top requires a value")?;
                top = value.parse()?;
            }
            "--sqrt-alpha" => sqrt_alpha = true,
            "--max-targets" => {
                let value = args.next().ok_or("--max-targets requires a value")?;
                max_targets = value.parse()?;
            }
            "--first3" => max_targets = 3,
            "--full" => max_targets = usize::MAX,
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

    let h_float = qwen2_prover::float::forward_from_safetensors(&st, &ids)?;
    let ppl_float = qwen2_prover::float::perplexity_tied_lm_head_prefix_from_safetensors(
        &st,
        &h_float,
        &ids,
        max_targets,
    )?;
    let (h_plain, plain_stats) = forward_one_down(&st, &ids, &r, layer, cfg, None)?;
    let ppl_plain = qwen2_prover::float::perplexity_tied_lm_head_prefix_from_safetensors(
        &st,
        &h_plain,
        &ids,
        max_targets,
    )?;
    let (h_fsbr, fsbr_stats) = forward_one_down(
        &st,
        &ids,
        &r,
        layer,
        cfg,
        Some(FsbrConfig { top, sqrt_alpha }),
    )?;
    let ppl_fsbr = qwen2_prover::float::perplexity_tied_lm_head_prefix_from_safetensors(
        &st,
        &h_fsbr,
        &ids,
        max_targets,
    )?;

    println!("text: {text:?}");
    println!("bits: {bits}");
    println!("target: L{layer}.down_proj");
    println!("fsbr top channels: {top}");
    println!("sqrt alpha: {sqrt_alpha}");
    println!(
        "plain selected: {}",
        plain_stats
            .selected
            .iter()
            .map(|s| format!("c{}:{:.1}/x{:.1}", s.channel, s.max_abs, s.alpha))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "fsbr selected: {}",
        fsbr_stats
            .selected
            .iter()
            .map(|s| format!("c{}:{:.1}/x{:.1}", s.channel, s.max_abs, s.alpha))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "mid max before/after: {:.3} -> {:.3}",
        fsbr_stats.mid_max_before, fsbr_stats.mid_max_after
    );
    println!(
        "mid p99 before/after: {:.3} -> {:.3}",
        fsbr_stats.mid_p99_before, fsbr_stats.mid_p99_after
    );
    println!(
        "down y_step before/after: {:.6} -> {:.6}",
        plain_stats.down_y_step, fsbr_stats.down_y_step
    );
    if max_targets == usize::MAX {
        println!("float ppl(full): {ppl_float}");
        println!("plain one-down ppl(full): {ppl_plain}");
        println!("fsbr one-down ppl(full): {ppl_fsbr}");
    } else {
        println!("float ppl(first {max_targets}): {ppl_float}");
        println!("plain one-down ppl(first {max_targets}): {ppl_plain}");
        println!("fsbr one-down ppl(first {max_targets}): {ppl_fsbr}");
    }
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn forward_one_down(
    st: &safetensors::SafeTensors,
    ids: &[u32],
    r: &Rotary,
    target_layer: usize,
    cfg: DiConfig,
    fsbr: Option<FsbrConfig>,
) -> Result<(Vec<f32>, FsbrStats), Box<dyn Error>> {
    let mut x = qwen2_prover::float::embed_from_safetensors(st, ids)?;
    let mut stats = FsbrStats::default();
    for layer in 0..qwen2_prover::LAYERS {
        let w = qwen2_prover::float::load_layer_from_safetensors(st, layer)?;
        x = if layer == target_layer {
            layer_one_down(&x, &w, r, cfg, fsbr, &mut stats)
        } else {
            qwen2_prover::float::layer(&x, &w, r)
        };
    }
    let norm = qwen2_prover::float::final_norm_from_safetensors(st)?;
    Ok((
        qwen2_prover::float::rms_norm(&x, &norm, qwen2_prover::SEQ, qwen2_prover::HIDDEN),
        stats,
    ))
}

fn layer_one_down(
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &Rotary,
    cfg: DiConfig,
    fsbr: Option<FsbrConfig>,
    stats: &mut FsbrStats,
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
    let mut m = qwen2_prover::float::silu_mul(&g, &u);
    let mut wd = w.wd.clone();
    apply_simple_fsbr(&mut m, &mut wd, fsbr, stats);
    let d = qwen2_prover::float::matmul_hybrid_for_debug(
        &m,
        &wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    stats.down_y_step = mean_row_step(&d, qwen2_prover::SEQ, qwen2_prover::HIDDEN, cfg.bits);
    qwen2_prover::float::add(&h, &d)
}

fn apply_simple_fsbr(
    m: &mut [f32],
    wd: &mut [f32],
    fsbr: Option<FsbrConfig>,
    stats: &mut FsbrStats,
) {
    let rows = qwen2_prover::SEQ;
    let cols = qwen2_prover::INTERMEDIATE;
    let out = qwen2_prover::HIDDEN;
    let mut channel_max = vec![0.0f32; cols];
    for row in 0..rows {
        for col in 0..cols {
            channel_max[col] = channel_max[col].max(m[row * cols + col].abs());
        }
    }
    let mut sorted = channel_max.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let target = percentile_sorted(&sorted, 0.99).max(1e-12);
    stats.mid_max_before = *sorted.last().unwrap_or(&0.0);
    stats.mid_p99_before = target;

    let mut order: Vec<usize> = (0..cols).collect();
    order.sort_by(|&a, &b| channel_max[b].total_cmp(&channel_max[a]));
    let take = fsbr.map(|cfg| cfg.top).unwrap_or(0).min(cols);
    for &channel in order.iter().take(take) {
        let ratio = (channel_max[channel] / target).max(1.0);
        let alpha = if fsbr.is_some_and(|cfg| cfg.sqrt_alpha) {
            ratio.sqrt()
        } else {
            ratio
        };
        stats.selected.push(SelectedChannel {
            channel,
            max_abs: channel_max[channel],
            alpha,
        });
        if alpha <= 1.0 {
            continue;
        }
        for row in 0..rows {
            m[row * cols + channel] /= alpha;
        }
        for col in 0..out {
            wd[channel * out + col] *= alpha;
        }
    }

    let after = abs_stats(m);
    stats.mid_p99_after = after.0;
    stats.mid_max_after = after.1;
}

#[derive(Clone, Copy)]
struct FsbrConfig {
    top: usize,
    sqrt_alpha: bool,
}

#[derive(Default)]
struct FsbrStats {
    selected: Vec<SelectedChannel>,
    mid_p99_before: f32,
    mid_max_before: f32,
    mid_p99_after: f32,
    mid_max_after: f32,
    down_y_step: f32,
}

struct SelectedChannel {
    channel: usize,
    max_abs: f32,
    alpha: f32,
}

fn abs_stats(xs: &[f32]) -> (f32, f32) {
    let mut values: Vec<f32> = xs.iter().map(|x| x.abs()).collect();
    values.sort_by(|a, b| a.total_cmp(b));
    (
        percentile_sorted(&values, 0.99),
        *values.last().unwrap_or(&0.0),
    )
}

fn percentile_sorted(xs: &[f32], p: f32) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let idx = ((xs.len() - 1) as f32 * p).round() as usize;
    xs[idx]
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
