use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut max_targets = 3usize;
    let mut clip_input = true;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--max-targets" => {
                let value = args.next().ok_or("--max-targets requires a value")?;
                max_targets = value.parse()?;
            }
            "--full" => max_targets = usize::MAX,
            "--no-clip" => clip_input = false,
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
    let mut softmax_cfg = qwen2_prover::illm::DiSoftmaxConfig::paper_default();
    if !clip_input {
        softmax_cfg = softmax_cfg.without_input_clip();
    }
    let h = qwen2_prover::float::forward_with_di_softmax_config_from_safetensors(
        &st,
        &ids,
        softmax_cfg,
    )?;
    let ppl = qwen2_prover::float::perplexity_tied_lm_head_prefix_from_safetensors(
        &st,
        &h,
        &ids,
        max_targets,
    )?;

    println!("text: {text:?}");
    println!("clip: {clip_input}");
    if max_targets == usize::MAX {
        println!("ppl(full, float with DI softmax): {ppl}");
    } else {
        println!("ppl(first {max_targets} targets, float with DI softmax): {ppl}");
    }
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}
