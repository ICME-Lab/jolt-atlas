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
    let mut a_bits = None;
    let mut w_bits = None;
    let mut y_bits = None;
    let mut mode = FsbrMode::None;
    let mut site = MlpSite::All;
    let mut n2_top = 16usize;
    let mut mid_top = 4usize;
    let mut exponent = 0.5f32;
    let mut n2_exponent = None;
    let mut mid_exponent = None;
    let mut percentile = 0.99f32;
    let mut rotation = RotationMode::None;
    let mut rotation_before_fsbr = false;
    let mut a_two_tensor = false;
    let mut a_outlier_i32_threshold = None;
    let mut a_outlier_frac_bits = 16u32;
    let mut y_two_tensor_threshold = None;
    let mut y_outlier_i32_threshold = None;
    let mut y_outlier_frac_bits = 16u32;
    let mut down_y_rotation = None;
    let mut max_targets = usize::MAX;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bits" => {
                let value = args.next().ok_or("--bits requires a value")?;
                bits = value.parse()?;
            }
            "--a-bits" => {
                let value = args.next().ok_or("--a-bits requires a value")?;
                a_bits = Some(value.parse()?);
            }
            "--w-bits" => {
                let value = args.next().ok_or("--w-bits requires a value")?;
                w_bits = Some(value.parse()?);
            }
            "--y-bits" => {
                let value = args.next().ok_or("--y-bits requires a value")?;
                y_bits = Some(value.parse()?);
            }
            "--mode" => {
                let value = args.next().ok_or("--mode requires a value")?;
                mode = parse_mode(&value)?;
            }
            "--site" => {
                let value = args.next().ok_or("--site requires a value")?;
                site = parse_site(&value)?;
            }
            "--n2-top" => {
                let value = args.next().ok_or("--n2-top requires a value")?;
                n2_top = value.parse()?;
            }
            "--mid-top" => {
                let value = args.next().ok_or("--mid-top requires a value")?;
                mid_top = value.parse()?;
            }
            "--exponent" => {
                let value = args.next().ok_or("--exponent requires a value")?;
                exponent = value.parse()?;
            }
            "--n2-exponent" => {
                let value = args.next().ok_or("--n2-exponent requires a value")?;
                n2_exponent = Some(value.parse()?);
            }
            "--mid-exponent" => {
                let value = args.next().ok_or("--mid-exponent requires a value")?;
                mid_exponent = Some(value.parse()?);
            }
            "--percentile" => {
                let value = args.next().ok_or("--percentile requires a value")?;
                percentile = value.parse()?;
            }
            "--rotation" => {
                let value = args.next().ok_or("--rotation requires a value")?;
                rotation = parse_rotation(&value)?;
            }
            "--rotation-before-fsbr" => {
                rotation_before_fsbr = true;
            }
            "--a-two-tensor" => {
                a_two_tensor = true;
            }
            "--a-outlier-i32-threshold" => {
                let value = args
                    .next()
                    .ok_or("--a-outlier-i32-threshold requires a value")?;
                a_outlier_i32_threshold = Some(value.parse()?);
            }
            "--a-outlier-frac-bits" => {
                let value = args
                    .next()
                    .ok_or("--a-outlier-frac-bits requires a value")?;
                a_outlier_frac_bits = value.parse()?;
            }
            "--y-two-tensor-threshold" => {
                let value = args
                    .next()
                    .ok_or("--y-two-tensor-threshold requires a value")?;
                y_two_tensor_threshold = Some(value.parse()?);
            }
            "--y-outlier-i32-threshold" => {
                let value = args
                    .next()
                    .ok_or("--y-outlier-i32-threshold requires a value")?;
                y_outlier_i32_threshold = Some(value.parse()?);
            }
            "--y-outlier-frac-bits" => {
                let value = args
                    .next()
                    .ok_or("--y-outlier-frac-bits requires a value")?;
                y_outlier_frac_bits = value.parse()?;
            }
            "--down-y-rotation" => {
                let value = args
                    .next()
                    .ok_or("--down-y-rotation requires group,rounds,seed")?;
                down_y_rotation = Some(parse_output_rotation(&value)?);
            }
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
    if y_two_tensor_threshold.is_some() && y_outlier_i32_threshold.is_some() {
        return Err(
            "--y-two-tensor-threshold and --y-outlier-i32-threshold are mutually exclusive".into(),
        );
    }
    if a_two_tensor && a_outlier_i32_threshold.is_some() {
        return Err("--a-two-tensor and --a-outlier-i32-threshold are mutually exclusive".into());
    }

    let cfg = DiConfig {
        bits,
        rounding: Rounding::Floor,
        rebase: DiRebaseMethod::Shift {
            multiplier_shift: 32,
        },
    };
    let matmul_bits = MatmulBits {
        a: a_bits.unwrap_or(bits),
        w: w_bits.unwrap_or(bits),
        y: y_bits.unwrap_or(bits),
    };

    let t = Instant::now();
    let ids = qwen2_prover::text::tokenize(&tok, &text)?;
    let bytes = std::fs::read(model)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let r = Rotary::new();

    let n2_exponent = n2_exponent.unwrap_or(exponent);
    let mid_exponent = mid_exponent.unwrap_or(exponent);
    let h = forward_mlp_ablation(
        &st,
        &ids,
        &r,
        cfg,
        matmul_bits,
        mode,
        n2_top,
        mid_top,
        n2_exponent,
        mid_exponent,
        percentile,
        rotation,
        rotation_before_fsbr,
        site,
        a_two_tensor,
        a_outlier_i32_threshold,
        a_outlier_frac_bits,
        y_two_tensor_threshold,
        y_outlier_i32_threshold,
        y_outlier_frac_bits,
        down_y_rotation,
    )?;
    let ppl = qwen2_prover::float::perplexity_tied_lm_head_prefix_from_safetensors(
        &st,
        &h,
        &ids,
        max_targets,
    )?;

    println!("text: {text:?}");
    println!("bits: {bits}");
    println!("a_bits: {}", matmul_bits.a);
    println!("w_bits: {}", matmul_bits.w);
    println!("y_bits: {}", matmul_bits.y);
    println!("mode: {mode:?}");
    println!("site: {site:?}");
    println!("n2_top: {n2_top}");
    println!("mid_top: {mid_top}");
    println!("n2_exponent: {n2_exponent}");
    println!("mid_exponent: {mid_exponent}");
    println!("percentile: {percentile}");
    println!("rotation: {rotation:?}");
    println!("rotation_before_fsbr: {rotation_before_fsbr}");
    println!("a_two_tensor: {a_two_tensor}");
    println!("a_outlier_i32_threshold: {a_outlier_i32_threshold:?}");
    println!("a_outlier_frac_bits: {a_outlier_frac_bits}");
    println!("y_two_tensor_threshold: {y_two_tensor_threshold:?}");
    println!("y_outlier_i32_threshold: {y_outlier_i32_threshold:?}");
    println!("y_outlier_frac_bits: {y_outlier_frac_bits}");
    println!("down_y_rotation: {down_y_rotation:?}");
    if max_targets == usize::MAX {
        println!("ppl(full): {ppl}");
    } else {
        println!("ppl(first {max_targets}): {ppl}");
    }
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FsbrMode {
    None,
    N2,
    Down,
    Both,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MlpSite {
    Gate,
    Up,
    GateUp,
    Down,
    All,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RotationMode {
    None,
    Hadamard,
}

#[derive(Clone, Copy, Debug)]
struct OutputRotation {
    group: usize,
    rounds: usize,
    seed: u64,
}

#[derive(Clone, Copy)]
struct MatmulBits {
    a: u8,
    w: u8,
    y: u8,
}

fn parse_mode(value: &str) -> Result<FsbrMode, Box<dyn Error>> {
    match value {
        "none" => Ok(FsbrMode::None),
        "n2" => Ok(FsbrMode::N2),
        "down" => Ok(FsbrMode::Down),
        "both" => Ok(FsbrMode::Both),
        _ => Err(format!("unknown --mode {value:?}; expected none, n2, down, both").into()),
    }
}

fn parse_site(value: &str) -> Result<MlpSite, Box<dyn Error>> {
    match value {
        "gate" => Ok(MlpSite::Gate),
        "up" => Ok(MlpSite::Up),
        "gate-up" => Ok(MlpSite::GateUp),
        "down" => Ok(MlpSite::Down),
        "all" | "mlp" => Ok(MlpSite::All),
        _ => Err(format!("unknown --site {value:?}; expected gate, up, gate-up, down, all").into()),
    }
}

fn parse_rotation(value: &str) -> Result<RotationMode, Box<dyn Error>> {
    match value {
        "none" => Ok(RotationMode::None),
        "hadamard" | "quarot" => Ok(RotationMode::Hadamard),
        _ => Err(format!("unknown --rotation {value:?}; expected none, hadamard").into()),
    }
}

fn parse_output_rotation(value: &str) -> Result<OutputRotation, Box<dyn Error>> {
    let parts = value.split(',').collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err("--down-y-rotation expects group,rounds,seed".into());
    }
    Ok(OutputRotation {
        group: parts[0].parse()?,
        rounds: parts[1].parse()?,
        seed: parts[2].parse()?,
    })
}

fn forward_mlp_ablation(
    st: &safetensors::SafeTensors,
    ids: &[u32],
    r: &Rotary,
    cfg: DiConfig,
    matmul_bits: MatmulBits,
    mode: FsbrMode,
    n2_top: usize,
    mid_top: usize,
    n2_exponent: f32,
    mid_exponent: f32,
    percentile: f32,
    rotation: RotationMode,
    rotation_before_fsbr: bool,
    site: MlpSite,
    a_two_tensor: bool,
    a_outlier_i32_threshold: Option<f32>,
    a_outlier_frac_bits: u32,
    y_two_tensor_threshold: Option<f32>,
    y_outlier_i32_threshold: Option<f32>,
    y_outlier_frac_bits: u32,
    down_y_rotation: Option<OutputRotation>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut x = qwen2_prover::float::embed_from_safetensors(st, ids)?;
    for layer in 0..qwen2_prover::LAYERS {
        let w = qwen2_prover::float::load_layer_from_safetensors(st, layer)?;
        x = layer_mlp_ablation(
            &x,
            &w,
            r,
            cfg,
            matmul_bits,
            mode,
            n2_top,
            mid_top,
            n2_exponent,
            mid_exponent,
            percentile,
            rotation,
            rotation_before_fsbr,
            site,
            a_two_tensor,
            a_outlier_i32_threshold,
            a_outlier_frac_bits,
            y_two_tensor_threshold,
            y_outlier_i32_threshold,
            y_outlier_frac_bits,
            down_y_rotation,
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

fn layer_mlp_ablation(
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &Rotary,
    cfg: DiConfig,
    matmul_bits: MatmulBits,
    mode: FsbrMode,
    n2_top: usize,
    mid_top: usize,
    n2_exponent: f32,
    mid_exponent: f32,
    percentile: f32,
    rotation: RotationMode,
    rotation_before_fsbr: bool,
    site: MlpSite,
    a_two_tensor: bool,
    a_outlier_i32_threshold: Option<f32>,
    a_outlier_frac_bits: u32,
    y_two_tensor_threshold: Option<f32>,
    y_outlier_i32_threshold: Option<f32>,
    y_outlier_frac_bits: u32,
    down_y_rotation: Option<OutputRotation>,
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
    let matmul_rotation = if rotation_before_fsbr {
        if rotation == RotationMode::Hadamard
            && (site == MlpSite::Gate
                || site == MlpSite::Up
                || site == MlpSite::GateUp
                || site == MlpSite::All)
        {
            let (rot_n2, rot_wg) = rotate_matmul_inputs_hadamard(
                &n2,
                &wg,
                qwen2_prover::SEQ,
                qwen2_prover::HIDDEN,
                qwen2_prover::INTERMEDIATE,
            );
            let (_, rot_wu) = rotate_matmul_inputs_hadamard(
                &n2,
                &wu,
                qwen2_prover::SEQ,
                qwen2_prover::HIDDEN,
                qwen2_prover::INTERMEDIATE,
            );
            n2 = rot_n2;
            wg = rot_wg;
            wu = rot_wu;
        }
        RotationMode::None
    } else {
        rotation
    };
    if (site == MlpSite::Gate
        || site == MlpSite::Up
        || site == MlpSite::GateUp
        || site == MlpSite::All)
        && (mode == FsbrMode::N2 || mode == FsbrMode::Both)
    {
        smooth_matmul_input(
            &mut n2,
            &mut [&mut wg, &mut wu],
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
            n2_top,
            n2_exponent,
            percentile,
        );
    }
    let g = if site == MlpSite::Gate || site == MlpSite::GateUp || site == MlpSite::All {
        matmul_token_channel_optional_two_tensor(
            &n2,
            &wg,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
            cfg,
            matmul_bits,
            matmul_rotation,
            a_two_tensor,
            a_outlier_i32_threshold,
            a_outlier_frac_bits,
            y_two_tensor_threshold,
            y_outlier_i32_threshold,
            y_outlier_frac_bits,
            None,
        )
    } else {
        qwen2_prover::float::matmul(
            &n2,
            &wg,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
        )
    };
    let u = if site == MlpSite::Up || site == MlpSite::GateUp || site == MlpSite::All {
        matmul_token_channel_optional_two_tensor(
            &n2,
            &wu,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
            cfg,
            matmul_bits,
            matmul_rotation,
            a_two_tensor,
            a_outlier_i32_threshold,
            a_outlier_frac_bits,
            y_two_tensor_threshold,
            y_outlier_i32_threshold,
            y_outlier_frac_bits,
            None,
        )
    } else {
        qwen2_prover::float::matmul(
            &n2,
            &wu,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
        )
    };
    let mut m = qwen2_prover::float::silu_mul(&g, &u);
    let mut wd = w.wd.clone();
    let down_rotation = if rotation_before_fsbr {
        if rotation == RotationMode::Hadamard && (site == MlpSite::Down || site == MlpSite::All) {
            let (rot_m, rot_wd) = rotate_matmul_inputs_hadamard(
                &m,
                &wd,
                qwen2_prover::SEQ,
                qwen2_prover::INTERMEDIATE,
                qwen2_prover::HIDDEN,
            );
            m = rot_m;
            wd = rot_wd;
        }
        RotationMode::None
    } else {
        rotation
    };
    if (site == MlpSite::Down || site == MlpSite::All)
        && (mode == FsbrMode::Down || mode == FsbrMode::Both)
    {
        smooth_matmul_input(
            &mut m,
            &mut [&mut wd],
            qwen2_prover::SEQ,
            qwen2_prover::INTERMEDIATE,
            qwen2_prover::HIDDEN,
            mid_top,
            mid_exponent,
            percentile,
        );
    }
    let d = if site == MlpSite::Down || site == MlpSite::All {
        matmul_token_channel_optional_two_tensor(
            &m,
            &wd,
            qwen2_prover::SEQ,
            qwen2_prover::INTERMEDIATE,
            qwen2_prover::HIDDEN,
            cfg,
            matmul_bits,
            down_rotation,
            a_two_tensor,
            a_outlier_i32_threshold,
            a_outlier_frac_bits,
            y_two_tensor_threshold,
            y_outlier_i32_threshold,
            y_outlier_frac_bits,
            down_y_rotation,
        )
    } else {
        qwen2_prover::float::matmul(
            &m,
            &wd,
            qwen2_prover::SEQ,
            qwen2_prover::INTERMEDIATE,
            qwen2_prover::HIDDEN,
        )
    };
    qwen2_prover::float::add(&h, &d)
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

fn rotate_matmul_inputs_hadamard(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let group = hadamard_group_len(k);
    let mut rot_a = a.to_vec();
    for row in 0..m {
        let row = &mut rot_a[row * k..(row + 1) * k];
        for chunk in row.chunks_exact_mut(group) {
            apply_deterministic_signs(chunk);
            hadamard_orthonormal_in_place(chunk);
        }
    }

    let mut rot_b = b.to_vec();
    let mut tmp = vec![0.0f32; group];
    for base in (0..k).step_by(group) {
        for col in 0..n {
            for i in 0..group {
                tmp[i] = rot_b[(base + i) * n + col];
            }
            apply_deterministic_signs(&mut tmp);
            hadamard_orthonormal_in_place(&mut tmp);
            for i in 0..group {
                rot_b[(base + i) * n + col] = tmp[i];
            }
        }
    }
    (rot_a, rot_b)
}

fn hadamard_group_len(k: usize) -> usize {
    if k.is_power_of_two() {
        return k;
    }
    let lowbit = k & k.wrapping_neg();
    lowbit.max(1)
}

fn apply_deterministic_signs(xs: &mut [f32]) {
    for (i, x) in xs.iter_mut().enumerate() {
        if splitmix_sign(i as u64) < 0 {
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

fn apply_output_rotation_rows(xs: &mut [f32], rows: usize, cols: usize, rotation: OutputRotation) {
    assert_eq!(xs.len(), rows * cols);
    assert!(rotation.group.is_power_of_two());
    assert!(rotation.group <= cols);
    assert_eq!(cols % rotation.group, 0);
    assert!(rotation.rounds > 0);
    let mut perm_buf = vec![0.0f32; cols];
    for row in 0..rows {
        let row_xs = &mut xs[row * cols..(row + 1) * cols];
        for round in 0..rotation.rounds {
            for (block, chunk) in row_xs.chunks_exact_mut(rotation.group).enumerate() {
                apply_seeded_signs(chunk, rotation.seed, round as u64, block as u64);
                hadamard_orthonormal_in_place(chunk);
            }
            if round + 1 < rotation.rounds {
                permute_stride(row_xs, &mut perm_buf, rotation.seed, round as u64);
                row_xs.copy_from_slice(&perm_buf);
            }
        }
    }
}

fn apply_output_rotation_rows_inverse(
    xs: &mut [f32],
    rows: usize,
    cols: usize,
    rotation: OutputRotation,
) {
    assert_eq!(xs.len(), rows * cols);
    assert!(rotation.group.is_power_of_two());
    assert!(rotation.group <= cols);
    assert_eq!(cols % rotation.group, 0);
    assert!(rotation.rounds > 0);
    let mut perm_buf = vec![0.0f32; cols];
    for row in 0..rows {
        let row_xs = &mut xs[row * cols..(row + 1) * cols];
        for round in (0..rotation.rounds).rev() {
            if round + 1 < rotation.rounds {
                inverse_permute_stride(row_xs, &mut perm_buf, rotation.seed, round as u64);
                row_xs.copy_from_slice(&perm_buf);
            }
            for (block, chunk) in row_xs.chunks_exact_mut(rotation.group).enumerate() {
                hadamard_orthonormal_in_place(chunk);
                apply_seeded_signs(chunk, rotation.seed, round as u64, block as u64);
            }
        }
    }
}

fn apply_seeded_signs(xs: &mut [f32], seed: u64, round: u64, block: u64) {
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
    let stride = output_rotation_stride(seed, round);
    debug_assert_eq!(gcd(stride, n), 1);
    let offset = splitmix64(seed ^ round.wrapping_mul(0x94D049BB133111EB)) as usize % n;
    for i in 0..n {
        dst[i] = src[(offset + i * stride) % n];
    }
}

fn inverse_permute_stride(src: &[f32], dst: &mut [f32], seed: u64, round: u64) {
    let n = src.len();
    let stride = output_rotation_stride(seed, round);
    debug_assert_eq!(gcd(stride, n), 1);
    let offset = splitmix64(seed ^ round.wrapping_mul(0x94D049BB133111EB)) as usize % n;
    for i in 0..n {
        dst[(offset + i * stride) % n] = src[i];
    }
}

fn output_rotation_stride(seed: u64, round: u64) -> usize {
    let strides = [31usize, 95, 191, 255, 383, 447, 639, 831];
    strides[((seed ^ round.wrapping_mul(0xBF58476D1CE4E5B9)) as usize) % strides.len()]
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

fn matmul_token_channel_optional_two_tensor(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    cfg: DiConfig,
    matmul_bits: MatmulBits,
    rotation: RotationMode,
    a_two_tensor: bool,
    a_outlier_i32_threshold: Option<f32>,
    a_outlier_frac_bits: u32,
    y_two_tensor_threshold: Option<f32>,
    y_outlier_i32_threshold: Option<f32>,
    y_outlier_frac_bits: u32,
    output_rotation: Option<OutputRotation>,
) -> Vec<f32> {
    if rotation == RotationMode::Hadamard {
        let (rot_a, rot_b) = rotate_matmul_inputs_hadamard(a, b, m, k, n);
        return matmul_token_channel_optional_two_tensor(
            &rot_a,
            &rot_b,
            m,
            k,
            n,
            cfg,
            matmul_bits,
            RotationMode::None,
            a_two_tensor,
            a_outlier_i32_threshold,
            a_outlier_frac_bits,
            y_two_tensor_threshold,
            y_outlier_i32_threshold,
            y_outlier_frac_bits,
            output_rotation,
        );
    }
    if !a_two_tensor
        && a_outlier_i32_threshold.is_none()
        && y_two_tensor_threshold.is_none()
        && y_outlier_i32_threshold.is_none()
        && output_rotation.is_none()
    {
        if matmul_bits.a == cfg.bits && matmul_bits.w == cfg.bits && matmul_bits.y == cfg.bits {
            return qwen2_prover::float::matmul_hybrid_for_debug(
                a,
                b,
                m,
                k,
                n,
                HybridOp::MatmulTokenChannel,
                cfg,
            );
        }
        return matmul_token_channel_split_bits(a, b, m, k, n, matmul_bits);
    }
    if let Some(threshold) = a_outlier_i32_threshold {
        return matmul_token_channel_a_i32_outliers(
            a,
            b,
            m,
            k,
            n,
            matmul_bits,
            threshold,
            a_outlier_frac_bits,
            y_two_tensor_threshold,
            y_outlier_i32_threshold,
            y_outlier_frac_bits,
            output_rotation,
        );
    }
    if !a_two_tensor {
        return matmul_token_channel_split_output(
            a,
            b,
            m,
            k,
            n,
            matmul_bits,
            y_two_tensor_threshold,
            y_outlier_i32_threshold,
            y_outlier_frac_bits,
            output_rotation,
        );
    }

    matmul_token_channel_a_two_tensor(
        a,
        b,
        m,
        k,
        n,
        matmul_bits,
        y_two_tensor_threshold,
        y_outlier_i32_threshold,
        y_outlier_frac_bits,
        output_rotation,
    )
}

fn matmul_token_channel_a_i32_outliers(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bits: MatmulBits,
    a_threshold: f32,
    a_frac_bits: u32,
    y_two_tensor_threshold: Option<f32>,
    y_outlier_i32_threshold: Option<f32>,
    y_outlier_frac_bits: u32,
    output_rotation: Option<OutputRotation>,
) -> Vec<f32> {
    let qa = (1i128 << bits.a) - 1;
    let qw = (1i128 << bits.w) - 1;
    let qy = (1i128 << bits.y) - 1;
    let a_fixed_scale = (1u64 << a_frac_bits) as f64;

    let mut a_small = vec![0.0f32; a.len()];
    let mut a_outlier = vec![0i32; a.len()];
    for idx in 0..a.len() {
        if a[idx].abs() > a_threshold {
            a_outlier[idx] = (a[idx] as f64 * a_fixed_scale).round() as i32;
        } else {
            a_small[idx] = a[idx];
        }
    }

    let mut a_small_int = vec![0i128; a.len()];
    let mut a_small_zp = vec![0i128; m];
    let mut a_small_scale = vec![0.0f64; m];
    for row in 0..m {
        let xs = &a_small[row * k..(row + 1) * k];
        let (scale, zp) = quant_params_f32(xs, qa);
        a_small_scale[row] = scale;
        a_small_zp[row] = zp;
        for col in 0..k {
            let idx = row * k + col;
            a_small_int[idx] = quantize_value(a_small[idx], scale, zp);
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
            let mut small_acc = 0i128;
            let mut outlier_acc = 0f64;
            for inner in 0..k {
                let rhs_int = b_int[inner * n + col] - b_zp[col];
                small_acc += (a_small_int[row * k + inner] - a_small_zp[row]) * rhs_int;
                let a_out = a_outlier[row * k + inner];
                if a_out != 0 {
                    let a_real = a_out as f64 / a_fixed_scale;
                    let b_real = rhs_int as f64 * b_scale[col];
                    outlier_acc += a_real * b_real;
                }
            }
            let small = small_acc as f64 * a_small_scale[row] * b_scale[col];
            y_real[row * n + col] = (small + outlier_acc) as f32;
        }
    }

    quantize_output_with_optional_split(
        &y_real,
        m,
        n,
        qy,
        y_two_tensor_threshold,
        y_outlier_i32_threshold,
        y_outlier_frac_bits,
        output_rotation,
    )
}

fn matmul_token_channel_split_output(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    matmul_bits: MatmulBits,
    y_two_tensor_threshold: Option<f32>,
    y_outlier_i32_threshold: Option<f32>,
    y_outlier_frac_bits: u32,
    output_rotation: Option<OutputRotation>,
) -> Vec<f32> {
    let qa = (1i128 << matmul_bits.a) - 1;
    let qw = (1i128 << matmul_bits.w) - 1;
    let qy = (1i128 << matmul_bits.y) - 1;
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

    quantize_output_with_optional_split(
        &y_real,
        m,
        n,
        qy,
        y_two_tensor_threshold,
        y_outlier_i32_threshold,
        y_outlier_frac_bits,
        output_rotation,
    )
}

fn matmul_token_channel_a_two_tensor(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bits: MatmulBits,
    y_two_tensor_threshold: Option<f32>,
    y_outlier_i32_threshold: Option<f32>,
    y_outlier_frac_bits: u32,
    output_rotation: Option<OutputRotation>,
) -> Vec<f32> {
    let qa = (1i128 << bits.a) - 1;
    let qw = (1i128 << bits.w) - 1;
    let qy = (1i128 << bits.y) - 1;

    let mut a_main_int = vec![0i128; a.len()];
    let mut a_main_zp = vec![0i128; m];
    let mut a_main_scale = vec![0.0f64; m];
    let mut a_main_real = vec![0.0f32; a.len()];
    for row in 0..m {
        let xs = &a[row * k..(row + 1) * k];
        let (scale, zp) = quant_params_f32(xs, qa);
        a_main_scale[row] = scale;
        a_main_zp[row] = zp;
        for col in 0..k {
            let idx = row * k + col;
            let value = quantize_value(a[idx], scale, zp);
            a_main_int[idx] = value;
            a_main_real[idx] = ((value - zp) as f64 * scale) as f32;
        }
    }

    let mut a_residual = vec![0.0f32; a.len()];
    for idx in 0..a.len() {
        a_residual[idx] = a[idx] - a_main_real[idx];
    }

    let mut a_res_int = vec![0i128; a.len()];
    let mut a_res_zp = vec![0i128; m];
    let mut a_res_scale = vec![0.0f64; m];
    for row in 0..m {
        let xs = &a_residual[row * k..(row + 1) * k];
        let (scale, zp) = quant_params_f32(xs, qa);
        a_res_scale[row] = scale;
        a_res_zp[row] = zp;
        for col in 0..k {
            let idx = row * k + col;
            a_res_int[idx] = quantize_value(a_residual[idx], scale, zp);
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
            let mut main_acc = 0i128;
            let mut res_acc = 0i128;
            for inner in 0..k {
                let rhs = b_int[inner * n + col] - b_zp[col];
                main_acc += (a_main_int[row * k + inner] - a_main_zp[row]) * rhs;
                res_acc += (a_res_int[row * k + inner] - a_res_zp[row]) * rhs;
            }
            let main = main_acc as f64 * a_main_scale[row] * b_scale[col];
            let residual = res_acc as f64 * a_res_scale[row] * b_scale[col];
            y_real[row * n + col] = (main + residual) as f32;
        }
    }

    quantize_output_with_optional_split(
        &y_real,
        m,
        n,
        qy,
        y_two_tensor_threshold,
        y_outlier_i32_threshold,
        y_outlier_frac_bits,
        output_rotation,
    )
}

fn quantize_output_with_optional_split(
    y_real: &[f32],
    rows: usize,
    cols: usize,
    q: i128,
    y_two_tensor_threshold: Option<f32>,
    y_outlier_i32_threshold: Option<f32>,
    y_outlier_frac_bits: u32,
    output_rotation: Option<OutputRotation>,
) -> Vec<f32> {
    if let Some(rotation) = output_rotation {
        let mut rotated = y_real.to_vec();
        apply_output_rotation_rows(&mut rotated, rows, cols, rotation);
        let mut quantized = quantize_output_with_optional_split(
            &rotated,
            rows,
            cols,
            q,
            y_two_tensor_threshold,
            y_outlier_i32_threshold,
            y_outlier_frac_bits,
            None,
        );
        apply_output_rotation_rows_inverse(&mut quantized, rows, cols, rotation);
        return quantized;
    }
    if let Some(threshold) = y_two_tensor_threshold {
        return quantize_two_output_tensors(y_real, rows, cols, q, threshold);
    }
    if let Some(threshold) = y_outlier_i32_threshold {
        return quantize_output_with_i32_outliers(
            y_real,
            rows,
            cols,
            q,
            threshold,
            y_outlier_frac_bits,
        );
    }
    quantize_rows_to_float(y_real, rows, cols, q)
}

fn quantize_output_with_i32_outliers(
    y_real: &[f32],
    rows: usize,
    cols: usize,
    q: i128,
    threshold: f32,
    frac_bits: u32,
) -> Vec<f32> {
    let mut small = vec![0.0f32; rows * cols];
    let mut outlier = vec![0i32; rows * cols];
    let scale = (1u64 << frac_bits) as f64;
    for idx in 0..y_real.len() {
        let value = y_real[idx];
        if value.abs() > threshold {
            outlier[idx] = (value as f64 * scale).round() as i32;
        } else {
            small[idx] = value;
        }
    }
    let mut y = quantize_rows_to_float(&small, rows, cols, q);
    for idx in 0..y.len() {
        if outlier[idx] == 0 {
            continue;
        }
        y[idx] += (outlier[idx] as f64 / scale) as f32;
    }
    y
}

fn matmul_token_channel_split_bits(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bits: MatmulBits,
) -> Vec<f32> {
    let qa = (1i128 << bits.a) - 1;
    let qw = (1i128 << bits.w) - 1;
    let qy = (1i128 << bits.y) - 1;

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
    quantize_rows_to_float(&y_real, m, n, qy)
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
