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
    let mut x = qwen2_prover::float::embed_from_safetensors(&st, &ids)?;

    println!("text: {text:?}");
    println!("bits: {bits}");
    println!(
        "{:<16} {:>8} {:>12} {:>10} {:>10} {:>23} {:>23}",
        "site", "cos", "mse", "max_abs", "zero%", "float_range", "quant_range"
    );

    for layer in 0..layer_limit {
        let w = qwen2_prover::float::load_layer_from_safetensors(&st, layer)?;
        x = inspect_layer(layer, &x, &w, &r, cfg);
    }

    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn inspect_layer(
    layer: usize,
    x: &[f32],
    w: &qwen2_prover::float::LayerWeights,
    r: &Rotary,
    cfg: DiConfig,
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

    let gate_f = qwen2_prover::float::matmul(
        &n2,
        &w.wg,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
    );
    let gate_q = qwen2_prover::float::matmul_hybrid_for_debug(
        &n2,
        &w.wg,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    print_stats(&format!("L{layer}.gate"), &gate_f, &gate_q);

    let up_f = qwen2_prover::float::matmul(
        &n2,
        &w.wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
    );
    let up_q = qwen2_prover::float::matmul_hybrid_for_debug(
        &n2,
        &w.wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    print_stats(&format!("L{layer}.up"), &up_f, &up_q);

    let silu_f = silu_only(&gate_f);
    let silu_q = silu_only(&gate_q);
    print_stats(&format!("L{layer}.silu"), &silu_f, &silu_q);

    let mid_f = mul(&silu_f, &up_f);
    let mid_gate_q = mul(&silu_q, &up_f);
    let mid_up_q = mul(&silu_f, &up_q);
    let mid_both_q = mul(&silu_q, &up_q);
    print_stats(&format!("L{layer}.mid_gate"), &mid_f, &mid_gate_q);
    print_stats(&format!("L{layer}.mid_up"), &mid_f, &mid_up_q);
    print_stats(&format!("L{layer}.mid_both"), &mid_f, &mid_both_q);

    let down_f = qwen2_prover::float::matmul(
        &mid_f,
        &w.wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
    );
    let down_from_mid_q = qwen2_prover::float::matmul(
        &mid_both_q,
        &w.wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
    );
    print_stats(&format!("L{layer}.down_in"), &down_f, &down_from_mid_q);
    let down_q = qwen2_prover::float::matmul_hybrid_for_debug(
        &mid_f,
        &w.wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    print_stats(&format!("L{layer}.down_mat"), &down_f, &down_q);

    qwen2_prover::float::add(&h, &down_q)
}

fn silu_only(xs: &[f32]) -> Vec<f32> {
    xs.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
}

fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(&x, &y)| x * y).collect()
}

fn print_stats(name: &str, reference: &[f32], actual: &[f32]) {
    assert_eq!(reference.len(), actual.len());
    let mut dot = 0.0f64;
    let mut ref_norm = 0.0f64;
    let mut act_norm = 0.0f64;
    let mut mse = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut ref_min = f32::INFINITY;
    let mut ref_max = f32::NEG_INFINITY;
    let mut act_min = f32::INFINITY;
    let mut act_max = f32::NEG_INFINITY;
    let mut zeroish = 0usize;
    for (&r, &a) in reference.iter().zip(actual) {
        let r64 = r as f64;
        let a64 = a as f64;
        dot += r64 * a64;
        ref_norm += r64 * r64;
        act_norm += a64 * a64;
        let diff = r64 - a64;
        mse += diff * diff;
        max_abs = max_abs.max(diff.abs());
        ref_min = ref_min.min(r);
        ref_max = ref_max.max(r);
        act_min = act_min.min(a);
        act_max = act_max.max(a);
        if a.abs() < 1e-12 {
            zeroish += 1;
        }
    }
    mse /= reference.len() as f64;
    let cosine = dot / (ref_norm.sqrt() * act_norm.sqrt()).max(f64::MIN_POSITIVE);
    let zero_pct = 100.0 * zeroish as f64 / actual.len() as f64;
    println!(
        "{name:<16} {cosine:>8.5} {mse:>12.4e} {max_abs:>10.4} {zero_pct:>9.3}% [{ref_min:>9.3},{ref_max:>9.3}] [{act_min:>9.3},{act_max:>9.3}]"
    );
}
