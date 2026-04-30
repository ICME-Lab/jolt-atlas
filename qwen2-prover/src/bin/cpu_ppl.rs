use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let tok = root.join("atlas-onnx-tracer/models/qwen/tokenizer.json");
    let model = root.join("atlas-onnx-tracer/models/qwen/model.safetensors");
    let text = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let text = if text.is_empty() {
        "hello world this is a test".to_string()
    } else {
        text
    };

    let t = Instant::now();
    let ids = qwen2_prover::text::tokenize(&tok, &text)?;
    let bytes = std::fs::read(model)?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let h = qwen2_prover::cpu::forward_from_safetensors(&st, &ids)?;
    stat("final norm", &h);
    let full_ppl = qwen2_prover::cpu::perplexity_tied_lm_head_prefix_from_safetensors(
        &st,
        &h,
        &ids,
        usize::MAX,
    )?;
    let ppl = qwen2_prover::cpu::perplexity_tied_lm_head_prefix_from_safetensors(&st, &h, &ids, 3)?;
    let x = &h[..qwen2_prover::HIDDEN];
    let scores = qwen2_prover::cpu::lm_head_scores_from_safetensors(&st, x)?;
    let target = ids[1] as usize;
    let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
    let rank = ranked.iter().position(|&(id, _)| id == target).unwrap() + 1;
    println!("text: {text:?}");
    println!("ppl: {full_ppl}");
    println!("ppl(first 3 targets): {ppl}");
    println!(
        "first target: id={} rank={} score={} top={} top_score={}",
        target, rank, scores[target], ranked[0].0, ranked[0].1
    );
    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn stat(name: &str, xs: &[i32]) {
    if std::env::var_os("QWEN2_CPU_PPL_STATS").is_none() {
        return;
    }
    let min = xs.iter().copied().min().unwrap();
    let max = xs.iter().copied().max().unwrap();
    let nz = xs.iter().filter(|&&x| x != 0).count();
    println!("{name}: min={min} max={max} nz={nz}/{}", xs.len());
}
