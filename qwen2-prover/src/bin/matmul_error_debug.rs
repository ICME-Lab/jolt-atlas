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
        "{:<18} {:>8} {:>12} {:>10} {:>10} {:>23} {:>23}",
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

    let mut q = compare_matmul(
        layer,
        "q_proj",
        &n1,
        &w.wq,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
        cfg,
    );
    qwen2_prover::float::add_rows(&mut q, &w.bq);
    let mut k = compare_matmul(
        layer,
        "k_proj",
        &n1,
        &w.wk,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
        cfg,
    );
    qwen2_prover::float::add_rows(&mut k, &w.bk);
    let mut v = compare_matmul(
        layer,
        "v_proj",
        &n1,
        &w.wv,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::KV_HEADS * qwen2_prover::HEAD_DIM,
        cfg,
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
    let a = compare_matmul(
        layer,
        "o_proj",
        &c,
        &w.wo,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::HIDDEN,
        cfg,
    );
    let h = qwen2_prover::float::add(x, &a);
    let n2 = qwen2_prover::float::rms_norm(&h, &w.ln2, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
    let g = compare_matmul(
        layer,
        "gate_proj",
        &n2,
        &w.wg,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        cfg,
    );
    let u = compare_matmul(
        layer,
        "up_proj",
        &n2,
        &w.wu,
        qwen2_prover::SEQ,
        qwen2_prover::HIDDEN,
        qwen2_prover::INTERMEDIATE,
        cfg,
    );
    let m = qwen2_prover::float::silu_mul(&g, &u);
    let d = compare_matmul(
        layer,
        "down_proj",
        &m,
        &w.wd,
        qwen2_prover::SEQ,
        qwen2_prover::INTERMEDIATE,
        qwen2_prover::HIDDEN,
        cfg,
    );
    qwen2_prover::float::add(&h, &d)
}

fn compare_matmul(
    layer: usize,
    name: &str,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    cfg: DiConfig,
) -> Vec<f32> {
    let reference = qwen2_prover::float::matmul(a, b, m, k, n);
    let actual = qwen2_prover::float::matmul_hybrid_for_debug(
        a,
        b,
        m,
        k,
        n,
        HybridOp::MatmulTokenChannel,
        cfg,
    );
    print_stats(&format!("L{layer}.{name}"), &reference, &actual);
    actual
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
        "{name:<18} {cosine:>8.5} {mse:>12.4e} {max_abs:>10.4} {zero_pct:>9.3}% [{ref_min:>9.3},{ref_max:>9.3}] [{act_min:>9.3},{act_max:>9.3}]"
    );
}
