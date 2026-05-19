use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");

    let mut scales = vec![4, 6, 8, 10, 12, 14];
    let mut max_targets = 3usize;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scales" => {
                let value = args
                    .next()
                    .ok_or("--scales requires comma-separated bits")?;
                scales = value
                    .split(',')
                    .map(|s| s.parse::<i32>())
                    .collect::<Result<Vec<_>, _>>()?;
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

    let ids = qwen2_prover::text::tokenize(&tok, &text)?;
    let bytes = std::fs::read(model)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;

    println!("text: {text:?}");
    if max_targets == usize::MAX {
        println!("targets: full");
    } else {
        println!("targets: first {max_targets}");
    }
    println!(
        "{:<6} {:>14} {:>8} {:>14} {:>8} {:>14} {:>10}",
        "S", "PPL", "rank", "target_score", "top", "top_score", "elapsed"
    );

    for s_bits in scales {
        let t = Instant::now();
        let h = qwen2_prover::cpu::forward_from_safetensors_with_scale(&st, &ids, s_bits)?;
        let ppl = qwen2_prover::cpu::perplexity_tied_lm_head_prefix_from_safetensors_with_scale(
            &st,
            &h,
            &ids,
            max_targets,
            s_bits,
        )?;
        let x = &h[..qwen2_prover::HIDDEN];
        let scores = qwen2_prover::cpu::lm_head_scores_from_safetensors_with_scale(&st, x, s_bits)?;
        let target = ids[1] as usize;
        let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
        ranked.sort_unstable_by_key(|&(_, score)| std::cmp::Reverse(score));
        let rank = ranked.iter().position(|&(id, _)| id == target).unwrap() + 1;
        println!(
            "{:<6} {:>14.6} {:>8} {:>14} {:>8} {:>14} {:>10.2?}",
            s_bits,
            ppl,
            rank,
            scores[target],
            ranked[0].0,
            ranked[0].1,
            t.elapsed()
        );
    }

    Ok(())
}
