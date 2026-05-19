use std::{error::Error, time::Instant};

use qwen2_prover::{
    illm::{self, DiConfig, DiRebaseMethod},
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
            "--max-targets" => {
                let value = args.next().ok_or("--max-targets requires a value")?;
                max_targets = value.parse()?;
            }
            "--full" => max_targets = usize::MAX,
            "--bits" => {
                let value = args.next().ok_or("--bits requires a value")?;
                bits = value.parse()?;
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
    let h = illm::forward_from_safetensors(&st, &ids, cfg)?;
    let ppl =
        illm::perplexity_tied_float_lm_head_prefix_from_safetensors(&st, &h, &ids, max_targets)?;

    println!("text: {text:?}");
    println!("bits: {bits}");
    if max_targets == usize::MAX {
        println!("ppl(full, float lm head): {ppl}");
    } else {
        println!("ppl(first {max_targets} targets, float lm head): {ppl}");
    }
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}
