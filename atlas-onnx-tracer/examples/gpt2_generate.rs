use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use std::{
    env,
    io::{self, Write},
};
use tokenizers::Tokenizer;

const ONNX_PATH: &str = "atlas-onnx-tracer/models/gpt2/network.onnx";
const TOKENIZER_PATH: &str = "atlas-onnx-tracer/models/gpt2/tokenizer.json";
const VOCAB_SIZE: usize = 50257;
const DEFAULT_NUM_GENERATE: usize = 15;

/// Autoregressive GPT-2 generation: quantized i32 vs Tract f32.
///
/// Generates `NUM_GENERATE` tokens with greedy decoding from both the
/// quantized model and the Tract f32 reference, then prints both
/// completions side-by-side for a quick coherence check.
///
/// # Setup
///
/// ```sh
/// pip install 'optimum[exporters]' 'optimum[onnxruntime]'
/// python -m optimum.exporters.onnx --model gpt2 atlas-onnx-tracer/models/gpt2
/// ```
///
/// # Usage
///
/// ```sh
/// cargo run --release -p atlas-onnx-tracer --features fused-ops --example gpt2_generate
/// cargo run --release -p atlas-onnx-tracer --features fused-ops --example gpt2_generate -- "Once upon a time"
/// cargo run --release -p atlas-onnx-tracer --features fused-ops --example gpt2_generate -- "Once upon a time" 30
/// ```
fn main() {
    let args: Vec<String> = env::args().collect();

    let prompt = if args.len() > 1 {
        args[1].clone()
    } else {
        "The white man worked as a".to_string()
    };

    let num_generate: usize = if args.len() > 2 {
        args[2]
            .parse()
            .expect("second argument (num_generate) must be a positive integer")
    } else {
        DEFAULT_NUM_GENERATE
    };

    let tokenizer = Tokenizer::from_file(TOKENIZER_PATH)
        .expect("failed to load tokenizer – see doc-comment for setup");

    println!("Prompt: \"{prompt}\"");
    println!("Generating {num_generate} tokens …\n");

    let encoding = tokenizer
        .encode(prompt.clone(), false)
        .expect("tokenization failed");
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();

    // ── Quantized i32 generation (streamed) ──────────────────────────────
    let scale = 12;
    println!("Quantized i32 (scale={scale}):");
    print!("  {prompt}");
    io::stdout().flush().unwrap();
    let quant_ids = generate_quantized_streaming(&prompt_ids, scale, num_generate, &tokenizer);
    let quant_text = tokenizer
        .decode(&quant_ids, false)
        .unwrap_or_else(|_| "<decode error>".to_string());
    println!("\n");

    // ── Tract f32 generation ─────────────────────────────────────────────
    println!("Running Tract f32 reference …");
    let tract_ids = generate_tract(&prompt_ids, num_generate);
    let tract_text = tokenizer
        .decode(&tract_ids, false)
        .unwrap_or_else(|_| "<decode error>".to_string());

    // ── Print results ────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  Quantized i32 scale={scale} ({} tokens):",
        quant_ids.len()
    );
    println!("  \"{quant_text}\"");
    println!();
    println!("  Tract f32 ({} tokens):", tract_ids.len());
    println!("  \"{tract_text}\"");
    println!("═══════════════════════════════════════════════════════════");

    // ── Token-level comparison ───────────────────────────────────────────
    let gen_start = prompt_ids.len();
    let tract_gen = &tract_ids[gen_start..];
    let quant_gen = &quant_ids[gen_start..];
    let matching = tract_gen
        .iter()
        .zip(quant_gen.iter())
        .filter(|(a, b)| a == b)
        .count();

    println!();
    println!(
        "Token match: {matching}/{num_generate} ({:.0}%)",
        matching as f64 / num_generate as f64 * 100.0
    );

    println!();
    println!(
        "{:<5} {:<12} {:<12} {:<5}",
        "Step", "Tract", "Quantized", "Match"
    );
    println!("{}", "-".repeat(40));
    for i in 0..num_generate {
        let t_id = tract_gen[i];
        let q_id = quant_gen[i];
        let t_word = tokenizer
            .decode(&[t_id], false)
            .unwrap_or_else(|_| "?".into());
        let q_word = tokenizer
            .decode(&[q_id], false)
            .unwrap_or_else(|_| "?".into());
        let m = if t_id == q_id { "✓" } else { "✗" };
        println!(
            "{:<5} {:<12} {:<12} {}",
            i + 1,
            format!("\"{}\"", t_word),
            format!("\"{}\"", q_word),
            m
        );
    }
}

/// Greedy generation using Tract f32 reference.
fn generate_tract(prompt_ids: &[u32], num_generate: usize) -> Vec<u32> {
    let mut ids = prompt_ids.to_vec();

    for _ in 0..num_generate {
        let cur_len = ids.len();
        let run_args = RunArgs::new([
            ("batch_size", 1),
            ("sequence_length", cur_len),
            ("past_sequence_length", 0),
        ])
        .with_padding(false);

        let f32_ids: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        let f32_mask: Vec<f32> = vec![1.0; cur_len];
        let f32_pos: Vec<f32> = (0..cur_len).map(|i| i as f32).collect();

        let t_ids = Tensor::new(Some(&f32_ids), &[1, cur_len]).unwrap();
        let t_mask = Tensor::new(Some(&f32_mask), &[1, cur_len]).unwrap();
        let t_pos = Tensor::new(Some(&f32_pos), &[1, cur_len]).unwrap();

        let outs = Model::run_tract_forward(
            ONNX_PATH,
            &run_args,
            &[
                ("input_ids", t_ids),
                ("attention_mask", t_mask),
                ("position_ids", t_pos),
            ],
        );

        let logits = outs[0].data();
        let last_start = (cur_len - 1) * VOCAB_SIZE;
        let last_logits = &logits[last_start..last_start + VOCAB_SIZE];
        let next = argmax_f32(last_logits);
        ids.push(next as u32);
    }
    ids
}

/// Greedy generation using the quantized i32 model, streaming tokens to stdout.
fn generate_quantized_streaming(
    prompt_ids: &[u32],
    scale: i32,
    num_generate: usize,
    tokenizer: &Tokenizer,
) -> Vec<u32> {
    let mut ids: Vec<u32> = prompt_ids.to_vec();

    for _ in 0..num_generate {
        let cur_len = ids.len();
        let run_args = RunArgs::new([
            ("batch_size", 1),
            ("sequence_length", cur_len),
            ("past_sequence_length", 0),
        ])
        .set_scale(scale)
        .with_padding(false);

        let model = Model::load(ONNX_PATH, &run_args);

        let i32_ids: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        let i32_pos: Vec<i32> = (0..cur_len as i32).collect();
        let i32_mask: Vec<i32> = vec![1 << scale; cur_len];

        let input_ids = Tensor::new(Some(&i32_ids), &[1, cur_len]).unwrap();
        let position_ids = Tensor::new(Some(&i32_pos), &[1, cur_len]).unwrap();
        let attention_mask = Tensor::new(Some(&i32_mask), &[1, cur_len]).unwrap();

        let outputs = model.forward(&[input_ids, position_ids, attention_mask]);
        let data = outputs[0].data();
        let last_start = (cur_len - 1) * VOCAB_SIZE;
        let last_logits = &data[last_start..last_start + VOCAB_SIZE];
        let next = argmax_i32(last_logits) as u32;
        ids.push(next);

        // Stream the token to stdout
        let word = tokenizer
            .decode(&[next], false)
            .unwrap_or_else(|_| "?".into());
        print!("{word}");
        io::stdout().flush().unwrap();
    }
    ids
}

fn argmax_f32(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

fn argmax_i32(data: &[i32]) -> usize {
    data.iter().enumerate().max_by_key(|&(_, &v)| v).unwrap().0
}
