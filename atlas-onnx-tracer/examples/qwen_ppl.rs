use std::{env, error::Error, panic, time::Instant};

use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use tokenizers::Tokenizer;

const ONNX: &str = "atlas-onnx-tracer/models/qwen/network.onnx";
const TOK: &str = "atlas-onnx-tracer/models/qwen/tokenizer.json";
const VOCAB: usize = 151_936;

fn main() -> Result<(), Box<dyn Error>> {
    let text = env::args().skip(1).collect::<Vec<_>>().join(" ");
    let text = if text.is_empty() {
        "hello world this is a test".to_string()
    } else {
        text
    };

    let tok = Tokenizer::from_file(TOK).map_err(|e| e.to_string())?;
    let enc = tok.encode(text.as_str(), true).map_err(|e| e.to_string())?;
    let ids = enc.get_ids().to_vec();
    if ids.len() < 2 {
        return Err("need at least two tokens to measure perplexity".into());
    }

    let t = Instant::now();
    let logits = run_tract(&ids);
    let ppl = ppl(&logits, &ids)?;
    let ppl3 = ppl_prefix(&logits, &ids, 3)?;
    let f32_diag = diag_f32(&logits[..VOCAB], ids[1] as usize);

    println!("text: {text:?}");
    println!("tokens: {}", ids.len());
    println!("tract f32 ppl: {ppl}");
    println!("tract f32 ppl(first 3 targets): {ppl3}");
    println!(
        "tract f32 first target: id={} rank={} score={} top={} top_score={}",
        f32_diag.target, f32_diag.rank, f32_diag.score, f32_diag.top, f32_diag.top_score
    );

    for scale in [7, 8, 12] {
        let qt = Instant::now();
        match catch_quiet(|| run_quant(&ids, scale)) {
            Ok(logits) => {
                let ppl = ppl_i32(&logits, &ids, scale)?;
                let ppl3 = ppl_i32_prefix(&logits, &ids, scale, 3)?;
                let diag = diag_i32(&logits[..VOCAB], ids[1] as usize);
                println!("quantized scale={scale} ppl: {ppl}");
                println!("quantized scale={scale} ppl(first 3 targets): {ppl3}");
                println!(
                    "quantized scale={scale} first target: id={} rank={} score={} top={} top_score={}",
                    diag.target, diag.rank, diag.score, diag.top, diag.top_score
                );
            }
            Err(_) => {
                println!("quantized scale={scale} failed: panic during quantized Model::forward");
            }
        }
        println!("quantized scale={scale} elapsed: {:?}", qt.elapsed());
    }

    println!("elapsed: {:?}", t.elapsed());
    Ok(())
}

fn run_tract(ids: &[u32]) -> Vec<f32> {
    let n = ids.len();
    let args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", n),
        ("past_sequence_length", 0),
    ])
    .with_padding(false);

    let input_ids = ids.iter().map(|&x| x as f32).collect::<Vec<_>>();
    let attention_mask = vec![1.0; n];
    let position_ids = (0..n).map(|x| x as f32).collect::<Vec<_>>();

    let input_ids = Tensor::new(Some(&input_ids), &[1, n]).unwrap();
    let attention_mask = Tensor::new(Some(&attention_mask), &[1, n]).unwrap();
    let position_ids = Tensor::new(Some(&position_ids), &[1, n]).unwrap();

    let out = Model::run_tract_forward(
        ONNX,
        &args,
        &[
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
            ("position_ids", position_ids),
        ],
    );
    out[0].data().to_vec()
}

fn catch_quiet<T>(
    f: impl FnOnce() -> T + panic::UnwindSafe,
) -> Result<T, Box<dyn std::any::Any + Send>> {
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(f);
    panic::set_hook(hook);
    result
}

fn run_quant(ids: &[u32], scale: i32) -> Vec<i32> {
    let n = ids.len();
    let args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", n),
        ("past_sequence_length", 0),
    ])
    .set_scale(scale)
    .with_padding(false);

    let model = Model::load(ONNX, &args);
    let input_ids = ids.iter().map(|&x| x as i32).collect::<Vec<_>>();
    let position_ids = (0..n).map(|x| x as i32).collect::<Vec<_>>();
    let attention_mask = vec![1 << scale; n];

    let input_ids = Tensor::new(Some(&input_ids), &[1, n]).unwrap();
    let position_ids = Tensor::new(Some(&position_ids), &[1, n]).unwrap();
    let attention_mask = Tensor::new(Some(&attention_mask), &[1, n]).unwrap();

    let out = model.forward(&[input_ids, position_ids, attention_mask]);
    out[0].data().to_vec()
}

fn ppl(logits: &[f32], ids: &[u32]) -> Result<f64, Box<dyn Error>> {
    ppl_prefix(logits, ids, usize::MAX)
}

fn ppl_prefix(logits: &[f32], ids: &[u32], max_targets: usize) -> Result<f64, Box<dyn Error>> {
    let n = ids.len();
    if logits.len() != n * VOCAB {
        return Err(format!("expected {} logits, got {}", n * VOCAB, logits.len()).into());
    }

    let mut nll = 0.0;
    let mut count = 0usize;
    for pos in 0..n - 1 {
        if count >= max_targets {
            break;
        }
        let next = ids[pos + 1] as usize;
        let row = &logits[pos * VOCAB..(pos + 1) * VOCAB];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum = row.iter().map(|&x| ((x as f64) - mx).exp()).sum::<f64>();
        nll += mx + sum.ln() - row[next] as f64;
        count += 1;
    }
    Ok((nll / count as f64).exp())
}

fn ppl_i32(logits: &[i32], ids: &[u32], scale: i32) -> Result<f64, Box<dyn Error>> {
    ppl_i32_prefix(logits, ids, scale, usize::MAX)
}

fn ppl_i32_prefix(
    logits: &[i32],
    ids: &[u32],
    scale: i32,
    max_targets: usize,
) -> Result<f64, Box<dyn Error>> {
    let n = ids.len();
    if logits.len() != n * VOCAB {
        return Err(format!("expected {} logits, got {}", n * VOCAB, logits.len()).into());
    }

    let one = (1 << scale) as f64;
    let mut nll = 0.0;
    let mut count = 0usize;
    for pos in 0..n - 1 {
        if count >= max_targets {
            break;
        }
        let next = ids[pos + 1] as usize;
        let row = &logits[pos * VOCAB..(pos + 1) * VOCAB];
        let mx = row.iter().copied().max().unwrap() as f64 / one;
        let sum = row
            .iter()
            .map(|&x| ((x as f64 / one) - mx).exp())
            .sum::<f64>();
        nll += mx + sum.ln() - row[next] as f64 / one;
        count += 1;
    }
    Ok((nll / count as f64).exp())
}

struct Diag<T> {
    target: usize,
    rank: usize,
    score: T,
    top: usize,
    top_score: T,
}

fn diag_f32(row: &[f32], target: usize) -> Diag<f32> {
    let mut ranked = row.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rank = ranked.iter().position(|&(id, _)| id == target).unwrap() + 1;
    Diag {
        target,
        rank,
        score: row[target],
        top: ranked[0].0,
        top_score: ranked[0].1,
    }
}

fn diag_i32(row: &[i32], target: usize) -> Diag<i32> {
    let mut ranked = row.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_unstable_by_key(|&(_, s)| std::cmp::Reverse(s));
    let rank = ranked.iter().position(|&(id, _)| id == target).unwrap() + 1;
    Diag {
        target,
        rank,
        score: row[target],
        top: ranked[0].0,
        top_score: ranked[0].1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppl_uniform() {
        let logits = vec![0.0; VOCAB * 2];
        let ids = vec![0, 1];
        let got = ppl(&logits, &ids).unwrap();
        assert!((got - VOCAB as f64).abs() < 1e-6);
    }
}
