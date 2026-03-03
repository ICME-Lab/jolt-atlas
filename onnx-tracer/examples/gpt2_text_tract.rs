use common::utils::logging::setup_tracing;
use onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use tokenizers::Tokenizer;

/// Runs GPT-2 next-token prediction via **Tract** (native f32, no quantization).
///
/// This is the unquantized counterpart of `gpt2_text` — it bypasses the
/// fixed-point computation graph and instead runs the ONNX model directly
/// through the Tract inference engine at full floating-point precision.
///
/// # Setup
///
/// Export the model with Optimum (one-time):
///
/// ```sh
/// pip install 'optimum[exporters]' 'optimum[onnxruntime]'
/// python -m optimum.exporters.onnx --model gpt2 onnx-tracer/models/gpt2
/// ```
///
/// # Usage
///
/// ```sh
/// cargo run --release --example gpt2_text_tract
/// ```
fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("gpt2_text_tract");

    // ── 1. Load the HuggingFace BPE tokenizer ───────────────────────────
    let tokenizer = Tokenizer::from_file("onnx-tracer/models/gpt2/tokenizer.json")
        .expect("failed to load tokenizer.json – see doc-comment for setup instructions");

    // ── 2. Encode the input text ─────────────────────────────────────────
    let text = "The white man worked as a";
    println!("Input text : \"{text}\"");

    let encoding = tokenizer.encode(text, false).expect("tokenization failed");
    let token_ids: Vec<f32> = encoding.get_ids().iter().map(|&id| id as f32).collect();
    let seq_len = token_ids.len();

    println!("Tokens     : {:?}", encoding.get_tokens());
    println!("Token IDs  : {:?}", encoding.get_ids());
    println!("Seq length : {seq_len}");
    println!();

    // ── 3. Prepare RunArgs (variables only – scale is irrelevant here) ───
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ]);

    // ── 4. Build f32 model inputs ────────────────────────────────────────
    // Input 0: input_ids   [1, seq_len]
    let input_ids = Tensor::new(Some(&token_ids), &[1, seq_len]).unwrap();

    // Input 1: attention_mask [1, seq_len] — all-ones
    let attention_mask_data: Vec<f32> = vec![1.0; seq_len];
    let attention_mask = Tensor::new(Some(&attention_mask_data), &[1, seq_len]).unwrap();

    // Input 2: position_ids [1, seq_len] — 0, 1, 2, …
    let position_ids_data: Vec<f32> = (0..seq_len as i64).map(|i| i as f32).collect();
    let position_ids = Tensor::new(Some(&position_ids_data), &[1, seq_len]).unwrap();

    // ── 5. Run inference through Tract (unquantized) ─────────────────────
    println!("Running GPT-2 forward pass via Tract (f32) …");
    let outputs = Model::run_tract_forward(
        "onnx-tracer/models/gpt2/network.onnx",
        &run_args,
        &[
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
            ("position_ids", position_ids),
        ],
    );

    let data = outputs[0].data(); // &[f32] — raw logits
    let dims = outputs[0].dims(); // expected: [1, seq_len, 50257]
    println!("Output shape: {dims:?}");
    println!();

    // ── 6. Extract logits for the last position ──────────────────────────
    let vocab_size: usize = 50257;
    let last_pos_start = (seq_len - 1) * vocab_size;
    let last_pos_logits = &data[last_pos_start..last_pos_start + vocab_size];

    // ── 7. Decode: top-10 predictions ────────────────────────────────────
    let mut indexed_logits: Vec<(usize, f32)> = last_pos_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed_logits.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    println!("Top-10 predicted next tokens (Tract f32):");
    println!("{:<6} {:<10} {:<12} Word", "Rank", "Token ID", "Logit");
    println!("{}", "-".repeat(50));
    for (rank, &(token_id, logit)) in indexed_logits.iter().take(10).enumerate() {
        let word = tokenizer
            .decode(&[token_id as u32], false)
            .unwrap_or_else(|_| "<unk>".to_string());
        println!(
            "{:<6} {:<10} {:<12.4} \"{}\"",
            rank + 1,
            token_id,
            logit,
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
