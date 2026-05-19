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

    let mut gate_up_bits = 8u8;
    let mut down_a_bits = 8u8;
    let mut down_w_bits = 8u8;
    let mut down_y_bits = 8u8;
    let mut no_down_yq = false;
    let mut down_y_residual_tensor = false;
    let mut down_sparse_threshold = None;
    let mut down_y_two_tensor_threshold = None;
    let mut all_y_two_tensor_threshold = None;
    let mut down_residual_lambda = 1.0f32;
    let mut diagnose = false;
    let mut n2_top = 128usize;
    let mut mid_top = 32usize;
    let mut exponent = 0.35f32;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--gate-up-bits" => {
                gate_up_bits = args
                    .next()
                    .ok_or("--gate-up-bits requires a value")?
                    .parse()?
            }
            "--down-a-bits" => {
                down_a_bits = args
                    .next()
                    .ok_or("--down-a-bits requires a value")?
                    .parse()?
            }
            "--down-w-bits" => {
                down_w_bits = args
                    .next()
                    .ok_or("--down-w-bits requires a value")?
                    .parse()?
            }
            "--down-y-bits" => {
                down_y_bits = args
                    .next()
                    .ok_or("--down-y-bits requires a value")?
                    .parse()?
            }
            "--no-down-yq" => no_down_yq = true,
            "--down-y-residual-tensor" => down_y_residual_tensor = true,
            "--down-sparse-threshold" => {
                down_sparse_threshold = Some(
                    args.next()
                        .ok_or("--down-sparse-threshold requires a value")?
                        .parse()?,
                )
            }
            "--down-y-two-tensor-threshold" => {
                down_y_two_tensor_threshold = Some(
                    args.next()
                        .ok_or("--down-y-two-tensor-threshold requires a value")?
                        .parse()?,
                )
            }
            "--all-y-two-tensor-threshold" => {
                let value = args
                    .next()
                    .ok_or("--all-y-two-tensor-threshold requires a value")?
                    .parse()?;
                all_y_two_tensor_threshold = Some(value);
                down_y_two_tensor_threshold = Some(value);
            }
            "--down-residual-lambda" => {
                down_residual_lambda = args
                    .next()
                    .ok_or("--down-residual-lambda requires a value")?
                    .parse()?
            }
            "--diagnose" => diagnose = true,
            "--n2-top" => n2_top = args.next().ok_or("--n2-top requires a value")?.parse()?,
            "--mid-top" => mid_top = args.next().ok_or("--mid-top requires a value")?.parse()?,
            "--exponent" => exponent = args.next().ok_or("--exponent requires a value")?.parse()?,
            other => words.push(other.to_string()),
        }
    }

    let text = if words.is_empty() {
        "hello world this is a test".to_string()
    } else {
        words.join(" ")
    };

    let cfg = DiConfig {
        bits: gate_up_bits,
        rounding: Rounding::Floor,
        rebase: DiRebaseMethod::Shift {
            multiplier_shift: 32,
        },
    };
    let down_cfg = DownQuant {
        a_bits: down_a_bits,
        w_bits: down_w_bits,
        y_bits: down_y_bits,
        no_yq: no_down_yq,
        y_residual_tensor: down_y_residual_tensor,
        sparse_threshold: down_sparse_threshold,
        y_two_tensor_threshold: down_y_two_tensor_threshold,
    };

    let t = Instant::now();
    let ids = qwen2_prover::text::tokenize(&tok, &text)?;
    let bytes = std::fs::read(model)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let r = Rotary::new();
    let mut diagnostics = Diagnostics::default();
    let h = forward(
        &st,
        &ids,
        &r,
        cfg,
        down_cfg,
        n2_top,
        mid_top,
        exponent,
        all_y_two_tensor_threshold,
        down_residual_lambda,
        if diagnose {
            Some(&mut diagnostics)
        } else {
            None
        },
    )?;
    let ppl = qwen2_prover::float::perplexity_tied_lm_head_prefix_from_safetensors(
        &st,
        &h,
        &ids,
        usize::MAX,
    )?;

    println!("text: {text:?}");
    println!("gate_up_bits: {gate_up_bits}");
    println!("down_a_bits: {down_a_bits}");
    println!("down_w_bits: {down_w_bits}");
    println!("down_y_bits: {down_y_bits}");
    println!("no_down_yq: {no_down_yq}");
    println!("down_y_residual_tensor: {down_y_residual_tensor}");
    println!("down_sparse_threshold: {down_sparse_threshold:?}");
    println!("down_y_two_tensor_threshold: {down_y_two_tensor_threshold:?}");
    println!("all_y_two_tensor_threshold: {all_y_two_tensor_threshold:?}");
    println!("down_residual_lambda: {down_residual_lambda}");
    println!("n2_top: {n2_top}");
    println!("mid_top: {mid_top}");
    println!("exponent: {exponent}");
    println!("ppl(full): {ppl}");
    if diagnose {
        diagnostics.print();
    }
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

#[derive(Clone, Copy)]
struct DownQuant {
    a_bits: u8,
    w_bits: u8,
    y_bits: u8,
    no_yq: bool,
    y_residual_tensor: bool,
    sparse_threshold: Option<f32>,
    y_two_tensor_threshold: Option<f32>,
}

fn forward(
    st: &safetensors::SafeTensors,
    ids: &[u32],
    r: &Rotary,
    cfg: DiConfig,
    down_cfg: DownQuant,
    n2_top: usize,
    mid_top: usize,
    exponent: f32,
    all_y_two_tensor_threshold: Option<f32>,
    down_residual_lambda: f32,
    mut diagnostics: Option<&mut Diagnostics>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut x = qwen2_prover::float::embed_from_safetensors(st, ids)?;
    for layer in 0..qwen2_prover::LAYERS {
        let w = qwen2_prover::float::load_layer_from_safetensors(st, layer)?;
        x = layer_forward(
            &x,
            &w,
            r,
            cfg,
            down_cfg,
            n2_top,
            mid_top,
            exponent,
            all_y_two_tensor_threshold,
            down_residual_lambda,
            diagnostics.as_deref_mut(),
        );
    }
    let norm = qwen2_prover::float::final_norm_from_safetensors(st)?;
    Ok(qwen2_prover::float::rms_norm(
        &x,
        &norm,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
    ))
}

fn layer_forward(
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &Rotary,
    cfg: DiConfig,
    down_cfg: DownQuant,
    n2_top: usize,
    mid_top: usize,
    exponent: f32,
    all_y_two_tensor_threshold: Option<f32>,
    down_residual_lambda: f32,
    diagnostics: Option<&mut Diagnostics>,
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
    let gate_up_quant = DownQuant {
        a_bits: cfg.bits,
        w_bits: cfg.bits,
        y_bits: cfg.bits,
        no_yq: false,
        y_residual_tensor: false,
        sparse_threshold: None,
        y_two_tensor_threshold: all_y_two_tensor_threshold,
    };
    let g = if all_y_two_tensor_threshold.is_some() {
        matmul_token_channel_split(
            &n2,
            &wg,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
            gate_up_quant,
        )
    } else {
        qwen2_prover::float::matmul_hybrid_for_debug(
            &n2,
            &wg,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
            HybridOp::MatmulTokenChannel,
            cfg,
        )
    };
    let u = if all_y_two_tensor_threshold.is_some() {
        matmul_token_channel_split(
            &n2,
            &wu,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
            gate_up_quant,
        )
    } else {
        qwen2_prover::float::matmul_hybrid_for_debug(
            &n2,
            &wu,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
            HybridOp::MatmulTokenChannel,
            cfg,
        )
    };
    let mut m = qwen2_prover::float::silu_mul(&g, &u);
    let mut wd = w.wd.clone();
    smooth_matmul_input(
        &mut m,
        &mut [&mut wd],
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        mid_top,
        exponent,
    );
    if let Some(diagnostics) = diagnostics {
        let d_ref = qwen2_prover::float::matmul(
            &m,
            &wd,
            qwen2_prover::SEQ,
            qwen2_prover::INTERMEDIATE,
            qwen2_prover::HIDDEN,
        );
        diagnostics.push(
            &d_ref,
            &matmul_token_channel_split(
                &m,
                &wd,
                qwen2_prover::SEQ,
                qwen2_prover::INTERMEDIATE,
                qwen2_prover::HIDDEN,
                DownQuant {
                    y_bits: 8,
                    y_residual_tensor: false,
                    sparse_threshold: None,
                    y_two_tensor_threshold: None,
                    ..down_cfg
                },
            ),
            "Y8",
        );
        diagnostics.push(
            &d_ref,
            &matmul_token_channel_split(
                &m,
                &wd,
                qwen2_prover::SEQ,
                qwen2_prover::INTERMEDIATE,
                qwen2_prover::HIDDEN,
                DownQuant {
                    y_bits: 10,
                    y_residual_tensor: false,
                    sparse_threshold: None,
                    y_two_tensor_threshold: None,
                    ..down_cfg
                },
            ),
            "Y10",
        );
        diagnostics.push(
            &d_ref,
            &matmul_token_channel_split(
                &m,
                &wd,
                qwen2_prover::SEQ,
                qwen2_prover::INTERMEDIATE,
                qwen2_prover::HIDDEN,
                DownQuant {
                    y_bits: 12,
                    y_residual_tensor: false,
                    sparse_threshold: None,
                    y_two_tensor_threshold: None,
                    ..down_cfg
                },
            ),
            "Y12",
        );
        diagnostics.push(
            &d_ref,
            &matmul_token_channel_split(
                &m,
                &wd,
                qwen2_prover::SEQ,
                qwen2_prover::INTERMEDIATE,
                qwen2_prover::HIDDEN,
                DownQuant {
                    no_yq: true,
                    y_residual_tensor: false,
                    sparse_threshold: None,
                    y_two_tensor_threshold: None,
                    ..down_cfg
                },
            ),
            "noYQ",
        );
        diagnostics.push(
            &d_ref,
            &matmul_token_channel_split(
                &m,
                &wd,
                qwen2_prover::SEQ,
                qwen2_prover::INTERMEDIATE,
                qwen2_prover::HIDDEN,
                DownQuant {
                    y_residual_tensor: true,
                    sparse_threshold: None,
                    y_two_tensor_threshold: None,
                    ..down_cfg
                },
            ),
            "Y8+R8",
        );
        diagnostics.push(
            &d_ref,
            &matmul_token_channel_split(
                &m,
                &wd,
                qwen2_prover::SEQ,
                qwen2_prover::INTERMEDIATE,
                qwen2_prover::HIDDEN,
                DownQuant {
                    y_two_tensor_threshold: down_cfg.y_two_tensor_threshold,
                    y_residual_tensor: false,
                    sparse_threshold: None,
                    ..down_cfg
                },
            ),
            "Y2T",
        );
    }
    let d = matmul_token_channel_split(
        &m,
        &wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        down_cfg,
    );
    let d = if down_residual_lambda == 1.0 {
        d
    } else {
        d.into_iter()
            .map(|value| value * down_residual_lambda)
            .collect()
    };
    qwen2_prover::float::add(&h, &d)
}

fn matmul_token_channel_split(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    cfg: DownQuant,
) -> Vec<f32> {
    let qa = q_for_bits(cfg.a_bits);
    let qw = q_for_bits(cfg.w_bits);
    let qy = q_for_bits(cfg.y_bits);

    let mut a_int = vec![0i128; a.len()];
    let mut a_zp = vec![0i128; m];
    let mut a_scale = vec![0.0f64; m];
    for row in 0..m {
        let xs = &a[row * k..(row + 1) * k];
        let (scale, zp) = quant_params_f32(xs, qa);
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
        let (scale, zp) = quant_params_min_max(min, max, qw);
        b_scale[col] = scale;
        b_zp[col] = zp;
        for row in 0..k {
            b_int[row * n + col] = quantize_value(b[row * n + col], scale, zp);
        }
    }

    let mut y_real = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0i128;
            for inner in 0..k {
                let lhs = a_int[row * k + inner] - a_zp[row];
                let rhs = b_int[inner * n + col] - b_zp[col];
                acc += lhs * rhs;
            }
            y_real[row * n + col] = (acc as f64 * a_scale[row] * b_scale[col]) as f32;
        }
    }
    if cfg.no_yq {
        return y_real;
    }

    if let Some(threshold) = cfg.y_two_tensor_threshold {
        return quantize_two_output_tensors(&y_real, m, n, qy, threshold);
    }

    let y_base_real = if let Some(threshold) = cfg.sparse_threshold {
        y_real
            .iter()
            .map(|&value| if value.abs() > threshold { 0.0 } else { value })
            .collect::<Vec<_>>()
    } else {
        y_real.clone()
    };

    let mut y = quantize_rows_to_float(&y_base_real, m, n, qy);
    if cfg.y_residual_tensor {
        let residual = y_base_real
            .iter()
            .zip(&y)
            .map(|(&real, &base)| real - base)
            .collect::<Vec<_>>();
        let residual_q = quantize_rows_to_float(&residual, m, n, qy);
        for idx in 0..y.len() {
            y[idx] += residual_q[idx];
        }
    }
    if let Some(threshold) = cfg.sparse_threshold {
        for idx in 0..y.len() {
            if y_real[idx].abs() > threshold {
                y[idx] += y_real[idx];
            }
        }
    }
    y
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

fn quantize_two_output_tensors(
    y_real: &[f32],
    rows: usize,
    cols: usize,
    q: i128,
    threshold: f32,
) -> Vec<f32> {
    let mut small = vec![0.0f32; rows * cols];
    let mut large = vec![0.0f32; rows * cols];
    for idx in 0..y_real.len() {
        let value = y_real[idx];
        if value.abs() > threshold {
            large[idx] = value;
        } else {
            small[idx] = value;
        }
    }

    let small = quantize_rows_to_float(&small, rows, cols, q);
    let large = quantize_rows_to_float(&large, rows, cols, q);
    small.iter().zip(&large).map(|(&a, &b)| a + b).collect()
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

fn q_for_bits(bits: u8) -> i128 {
    (1i128 << bits) - 1
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

#[derive(Default)]
struct Diagnostics {
    y8: Vec<ErrorMetrics>,
    y10: Vec<ErrorMetrics>,
    y12: Vec<ErrorMetrics>,
    no_yq: Vec<ErrorMetrics>,
    y8_r8: Vec<ErrorMetrics>,
    y2t: Vec<ErrorMetrics>,
}

impl Diagnostics {
    fn push(&mut self, reference: &[f32], actual: &[f32], label: &str) {
        let value = error_metrics(reference, actual);
        match label {
            "Y8" => self.y8.push(value),
            "Y10" => self.y10.push(value),
            "Y12" => self.y12.push(value),
            "noYQ" => self.no_yq.push(value),
            "Y8+R8" => self.y8_r8.push(value),
            "Y2T" => self.y2t.push(value),
            _ => unreachable!(),
        }
    }

    fn print(&self) {
        println!("down local error against float down:");
        println!("  label      cosine        mse");
        print_metric("Y8", &self.y8);
        print_metric("Y10", &self.y10);
        print_metric("Y12", &self.y12);
        print_metric("noYQ", &self.no_yq);
        print_metric("Y8+R8", &self.y8_r8);
        print_metric("Y2T", &self.y2t);
    }
}

#[derive(Clone, Copy)]
struct ErrorMetrics {
    cosine: f64,
    mse: f64,
}

fn error_metrics(reference: &[f32], actual: &[f32]) -> ErrorMetrics {
    assert_eq!(reference.len(), actual.len());
    let mut dot = 0.0f64;
    let mut ref_norm = 0.0f64;
    let mut act_norm = 0.0f64;
    let mut sq = 0.0f64;
    for (&r, &a) in reference.iter().zip(actual) {
        let r = r as f64;
        let a = a as f64;
        dot += r * a;
        ref_norm += r * r;
        act_norm += a * a;
        let diff = r - a;
        sq += diff * diff;
    }
    ErrorMetrics {
        cosine: dot / (ref_norm.sqrt() * act_norm.sqrt()).max(f64::MIN_POSITIVE),
        mse: sq / reference.len() as f64,
    }
}

fn print_metric(label: &str, xs: &[ErrorMetrics]) {
    let n = xs.len().max(1) as f64;
    let cosine = xs.iter().map(|x| x.cosine).sum::<f64>() / n;
    let mse = xs.iter().map(|x| x.mse).sum::<f64>() / n;
    println!("  {label:<8} {cosine:>8.5} {mse:>12.5e}");
}
