use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

use qwen2_prover::float::Rotary;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RotationMode {
    None,
    Hadamard,
}

fn parse_rotation(value: &str) -> Result<RotationMode, Box<dyn Error>> {
    match value {
        "none" => Ok(RotationMode::None),
        "hadamard" | "quarot" => Ok(RotationMode::Hadamard),
        _ => Err(format!("unknown --rotation {value:?}; expected none, hadamard").into()),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut out = PathBuf::from("target/qwen_mlp_awy_values.csv");
    let mut max_points = 50_000usize;
    let mut n2_top = 256usize;
    let mut mid_top = 64usize;
    let mut n2_exponent = 0.95f32;
    let mut mid_exponent = 0.20f32;
    let mut percentile = 0.95f32;
    let mut rotation = RotationMode::None;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = args.next().ok_or("--out requires a value")?.into(),
            "--max-points" => {
                max_points = args
                    .next()
                    .ok_or("--max-points requires a value")?
                    .parse()?
            }
            "--n2-top" => n2_top = args.next().ok_or("--n2-top requires a value")?.parse()?,
            "--mid-top" => mid_top = args.next().ok_or("--mid-top requires a value")?.parse()?,
            "--n2-exponent" => {
                n2_exponent = args
                    .next()
                    .ok_or("--n2-exponent requires a value")?
                    .parse()?
            }
            "--mid-exponent" => {
                mid_exponent = args
                    .next()
                    .ok_or("--mid-exponent requires a value")?
                    .parse()?
            }
            "--percentile" => {
                percentile = args
                    .next()
                    .ok_or("--percentile requires a value")?
                    .parse()?
            }
            "--rotation" => {
                let value = args.next().ok_or("--rotation requires a value")?;
                rotation = parse_rotation(&value)?;
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

    let mut writer = BufWriter::new(File::create(&out)?);
    writeln!(writer, "site,tensor,index,value")?;

    let totals = Totals::new();
    let mut offsets = Offsets::default();

    for layer in 0..qwen2_prover::LAYERS {
        let w = qwen2_prover::float::load_layer_from_safetensors(&st, layer)?;
        let (h, mut n2) = layer_to_h_and_n2(&x, &w, &r);

        let mut wg = w.wg.clone();
        let mut wu = w.wu.clone();
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

        let (gate_a, gate_w) = rotate_for_plot(
            &n2,
            &wg,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
            rotation,
        );
        let gate = qwen2_prover::float::matmul(
            &gate_a,
            &gate_w,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
        );
        let (up_a, up_w) = rotate_for_plot(
            &n2,
            &wu,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
            rotation,
        );
        let up = qwen2_prover::float::matmul(
            &up_a,
            &up_w,
            qwen2_prover::SEQ,
            qwen2_prover::HIDDEN,
            qwen2_prover::INTERMEDIATE,
        );

        write_sampled(
            &mut writer,
            "gate",
            "A",
            &gate_a,
            &mut offsets.gate_a,
            totals.gate_a,
            max_points,
        )?;
        write_sampled(
            &mut writer,
            "gate",
            "W",
            &gate_w,
            &mut offsets.gate_w,
            totals.gate_w,
            max_points,
        )?;
        write_sampled(
            &mut writer,
            "gate",
            "Y",
            &gate,
            &mut offsets.gate_y,
            totals.gate_y,
            max_points,
        )?;
        write_sampled(
            &mut writer,
            "up",
            "A",
            &up_a,
            &mut offsets.up_a,
            totals.up_a,
            max_points,
        )?;
        write_sampled(
            &mut writer,
            "up",
            "W",
            &up_w,
            &mut offsets.up_w,
            totals.up_w,
            max_points,
        )?;
        write_sampled(
            &mut writer,
            "up",
            "Y",
            &up,
            &mut offsets.up_y,
            totals.up_y,
            max_points,
        )?;

        let mut mid = qwen2_prover::float::silu_mul(&gate, &up);
        let mut wd = w.wd.clone();
        smooth_matmul_input(
            &mut mid,
            &mut [&mut wd],
            qwen2_prover::SEQ,
            qwen2_prover::INTERMEDIATE,
            qwen2_prover::HIDDEN,
            mid_top,
            mid_exponent,
            percentile,
        );
        let (down_a, down_w) = rotate_for_plot(
            &mid,
            &wd,
            qwen2_prover::SEQ,
            qwen2_prover::INTERMEDIATE,
            qwen2_prover::HIDDEN,
            rotation,
        );
        let down = qwen2_prover::float::matmul(
            &down_a,
            &down_w,
            qwen2_prover::SEQ,
            qwen2_prover::INTERMEDIATE,
            qwen2_prover::HIDDEN,
        );
        write_sampled(
            &mut writer,
            "down",
            "A",
            &down_a,
            &mut offsets.down_a,
            totals.down_a,
            max_points,
        )?;
        write_sampled(
            &mut writer,
            "down",
            "W",
            &down_w,
            &mut offsets.down_w,
            totals.down_w,
            max_points,
        )?;
        write_sampled(
            &mut writer,
            "down",
            "Y",
            &down,
            &mut offsets.down_y,
            totals.down_y,
            max_points,
        )?;

        x = qwen2_prover::float::add(&h, &down);
    }

    writer.flush()?;
    println!("text: {text:?}");
    println!("out: {}", out.display());
    println!("max_points_per_series: {max_points}");
    println!("n2_top: {n2_top}");
    println!("mid_top: {mid_top}");
    println!("n2_exponent: {n2_exponent}");
    println!("mid_exponent: {mid_exponent}");
    println!("percentile: {percentile}");
    println!("rotation: {rotation:?}");
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn write_sampled(
    writer: &mut BufWriter<File>,
    site: &str,
    tensor: &str,
    xs: &[f32],
    offset: &mut usize,
    total: usize,
    max_points: usize,
) -> Result<(), Box<dyn Error>> {
    let stride = total.div_ceil(max_points.max(1)).max(1);
    for (local, &value) in xs.iter().enumerate() {
        let index = *offset + local;
        if index % stride == 0 {
            writeln!(writer, "{site},{tensor},{index},{value}")?;
        }
    }
    *offset += xs.len();
    Ok(())
}

fn rotate_for_plot(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    rotation: RotationMode,
) -> (Vec<f32>, Vec<f32>) {
    match rotation {
        RotationMode::None => (a.to_vec(), b.to_vec()),
        RotationMode::Hadamard => rotate_matmul_inputs_hadamard(a, b, m, k, n),
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

#[derive(Default)]
struct Offsets {
    gate_a: usize,
    gate_w: usize,
    gate_y: usize,
    up_a: usize,
    up_w: usize,
    up_y: usize,
    down_a: usize,
    down_w: usize,
    down_y: usize,
}

struct Totals {
    gate_a: usize,
    gate_w: usize,
    gate_y: usize,
    up_a: usize,
    up_w: usize,
    up_y: usize,
    down_a: usize,
    down_w: usize,
    down_y: usize,
}

impl Totals {
    fn new() -> Self {
        Self {
            gate_a: qwen2_prover::LAYERS * qwen2_prover::SEQ * qwen2_prover::HIDDEN,
            gate_w: qwen2_prover::LAYERS * qwen2_prover::HIDDEN * qwen2_prover::INTERMEDIATE,
            gate_y: qwen2_prover::LAYERS * qwen2_prover::SEQ * qwen2_prover::INTERMEDIATE,
            up_a: qwen2_prover::LAYERS * qwen2_prover::SEQ * qwen2_prover::HIDDEN,
            up_w: qwen2_prover::LAYERS * qwen2_prover::HIDDEN * qwen2_prover::INTERMEDIATE,
            up_y: qwen2_prover::LAYERS * qwen2_prover::SEQ * qwen2_prover::INTERMEDIATE,
            down_a: qwen2_prover::LAYERS * qwen2_prover::SEQ * qwen2_prover::INTERMEDIATE,
            down_w: qwen2_prover::LAYERS * qwen2_prover::INTERMEDIATE * qwen2_prover::HIDDEN,
            down_y: qwen2_prover::LAYERS * qwen2_prover::SEQ * qwen2_prover::HIDDEN,
        }
    }
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
