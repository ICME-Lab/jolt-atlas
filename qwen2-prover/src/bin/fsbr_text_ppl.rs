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

    let mut max_targets = 3usize;
    let mut bits = 8u8;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bits" => {
                let value = args.next().ok_or("--bits requires a value")?;
                bits = value.parse()?;
            }
            "--max-targets" => {
                let value = args.next().ok_or("--max-targets requires a value")?;
                max_targets = value.parse()?;
            }
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

    let mut x_ref = qwen2_prover::float::embed_from_safetensors(&st, &ids)?;
    let mut x_quant = x_ref.clone();
    let mut avg_mse_before = 0.0f64;
    let mut avg_mse_after = 0.0f64;

    for layer in 0..qwen2_prover::LAYERS {
        let w = qwen2_prover::float::load_layer_from_safetensors(&st, layer)?;
        let y_ref = qwen2_prover::float::layer(&x_ref, &w, &r);
        let y_quant = qwen2_prover::float::layer_with_hybrid_op(
            &x_quant,
            &w,
            &r,
            HybridOp::MatmulTokenChannel,
            cfg,
        );
        avg_mse_before += mse(&y_ref, &y_quant);
        let correction =
            ChannelAffine::fit(&y_quant, &y_ref, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
        x_quant = correction.apply(&y_quant, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
        avg_mse_after += mse(&y_ref, &x_quant);
        x_ref = y_ref;
    }

    avg_mse_before /= qwen2_prover::LAYERS as f64;
    avg_mse_after /= qwen2_prover::LAYERS as f64;

    let norm = qwen2_prover::float::final_norm_from_safetensors(&st)?;
    let h = qwen2_prover::float::rms_norm(&x_quant, &norm, qwen2_prover::SEQ, qwen2_prover::HIDDEN);
    let ppl = qwen2_prover::float::perplexity_tied_lm_head_prefix_from_safetensors(
        &st,
        &h,
        &ids,
        max_targets,
    )?;

    println!("text: {text:?}");
    println!("bits: {bits}");
    println!("fsbr: text-specific channel affine over block outputs");
    println!("avg layer mse before: {avg_mse_before}");
    println!("avg layer mse after: {avg_mse_after}");
    if max_targets == usize::MAX {
        println!("ppl(full): {ppl}");
    } else {
        println!("ppl(first {max_targets} targets): {ppl}");
    }
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

struct ChannelAffine {
    alpha: Vec<f32>,
    beta: Vec<f32>,
}

impl ChannelAffine {
    fn fit(input: &[f32], target: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(input.len(), rows * cols);
        assert_eq!(target.len(), rows * cols);
        let mut alpha = vec![1.0f32; cols];
        let mut beta = vec![0.0f32; cols];
        for col in 0..cols {
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            for row in 0..rows {
                sx += input[row * cols + col] as f64;
                sy += target[row * cols + col] as f64;
            }
            let mx = sx / rows as f64;
            let my = sy / rows as f64;
            let mut cov = 0.0f64;
            let mut var = 0.0f64;
            for row in 0..rows {
                let x = input[row * cols + col] as f64 - mx;
                let y = target[row * cols + col] as f64 - my;
                cov += x * y;
                var += x * x;
            }
            let a = if var > 1e-24 { cov / var } else { 1.0 };
            let b = my - a * mx;
            alpha[col] = a as f32;
            beta[col] = b as f32;
        }
        Self { alpha, beta }
    }

    fn apply(&self, input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        assert_eq!(input.len(), rows * cols);
        let mut out = vec![0.0f32; input.len()];
        for row in 0..rows {
            for col in 0..cols {
                out[row * cols + col] = input[row * cols + col] * self.alpha[col] + self.beta[col];
            }
        }
        out
    }
}

fn mse(reference: &[f32], actual: &[f32]) -> f64 {
    assert_eq!(reference.len(), actual.len());
    reference
        .iter()
        .zip(actual)
        .map(|(&r, &a)| {
            let diff = r as f64 - a as f64;
            diff * diff
        })
        .sum::<f64>()
        / reference.len() as f64
}
