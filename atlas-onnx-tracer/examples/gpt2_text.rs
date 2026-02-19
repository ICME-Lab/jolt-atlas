use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use common::utils::logging::setup_tracing;
use tokenizers::Tokenizer;

/// Runs GPT-2 next-token prediction on real text.
///
/// # Setup
///
/// Export the model with Optimum (one-time):
///
/// ```sh
/// pip install 'optimum[exporters]' 'optimum[onnxruntime]'
/// python -m optimum.exporters.onnx --model gpt2 atlas-onnx-tracer/models/gpt2
/// ```
///
/// # Usage
///
/// ```sh
/// cargo run --release --example gpt2_text
/// ```
fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("gpt2_text");

    // ── 1. Load the HuggingFace BPE tokenizer ───────────────────────────
    let tokenizer = Tokenizer::from_file("./models/gpt2/tokenizer.json")
        .expect("failed to load tokenizer.json – see doc-comment for setup instructions");

    // ── 2. Encode the input text ─────────────────────────────────────────
    let text = "The white man worked as a";
    println!("Input text : \"{text}\"");

    let encoding = tokenizer.encode(text, false).expect("tokenization failed");
    let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    let seq_len = token_ids.len();

    println!("Tokens     : {:?}", encoding.get_tokens());
    println!("Token IDs  : {token_ids:?}");
    println!("Seq length : {seq_len}");
    println!();

    // ── 3. Load the ONNX model ───────────────────────────────────────────
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ])
    .with_pre_rebase_nonlinear(true);

    let model = Model::load("atlas-onnx-tracer/models/gpt2/model.onnx", &run_args);

    // ── 4. Build model inputs ────────────────────────────────────────────
    // Input 0: input_ids   [1, seq_len]  — token IDs from the tokenizer
    let input_ids = Tensor::new(Some(&token_ids), &[1, seq_len]).unwrap();

    // Input 1: position_ids [1, seq_len] — 0, 1, 2, …
    let position_ids_data: Vec<i32> = (0..seq_len as i32).collect();
    let position_ids = Tensor::new(Some(&position_ids_data), &[1, seq_len]).unwrap();

    // Input 2: attention_mask [1, seq_len] — all-ones in quantized form
    let scale = run_args.scale;
    let attention_mask_data: Vec<i32> = vec![1 << scale; seq_len];
    let attention_mask = Tensor::new(Some(&attention_mask_data), &[1, seq_len]).unwrap();

    // ── 5. Run the forward pass ──────────────────────────────────────────
    println!("Running GPT-2 forward pass …");
    let outputs = model.forward(&[input_ids, position_ids, attention_mask]);

    let data = outputs[0].data(); // &[i32] — quantized logits
    let dims = outputs[0].dims(); // expected: [1, seq_len, 50257]
    println!("Output shape: {dims:?}");
    println!();

    // ── 6. Extract logits for the last position ──────────────────────────
    let vocab_size: usize = 50257;
    let last_pos_start = (seq_len - 1) * vocab_size;
    let last_pos_logits = &data[last_pos_start..last_pos_start + vocab_size];

    // ── 7. Decode: top-5 predictions ─────────────────────────────────────
    let mut indexed_logits: Vec<(usize, i32)> = last_pos_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_logits.sort_by(|(_, a), (_, b)| b.cmp(a));

    let scale_divisor = (1i64 << scale) as f64;

    println!("Top-10 predicted next tokens:");
    println!("{:<6} {:<10} {:<12} Word", "Rank", "Token ID", "Logit");
    println!("{}", "-".repeat(50));
    for (rank, &(token_id, logit)) in indexed_logits.iter().take(10).enumerate() {
        let word = tokenizer
            .decode(&[token_id as u32], false)
            .unwrap_or_else(|_| "<unk>".to_string());
        let logit_f = logit as f64 / scale_divisor;
        println!(
            "{:<6} {:<10} {:<12.4} \"{}\"",
            rank + 1,
            token_id,
            logit_f,
            word
        );
    }

    // ── 8. Final answer ──────────────────────────────────────────────────
    let (best_token_id, _) = indexed_logits[0];
    let predicted_word = tokenizer
        .decode(&[best_token_id as u32], false)
        .unwrap_or_else(|_| "<unk>".to_string());

    println!();
    println!("Predicted next word: \"{predicted_word}\"");
    println!("Full completion    : \"{text}{predicted_word}\"");
}
