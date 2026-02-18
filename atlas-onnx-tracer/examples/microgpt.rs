use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    println!("Minimal GPT with RMSNorm + Multi-Head Attention + ReLU MLP\n");

    let microgpt = Model::load("atlas-onnx-tracer/models/microgpt/network.onnx", &Default::default());
    println!("model: {}", microgpt.pretty_print());

    let mut rng = StdRng::seed_from_u64(0x42);

    // Model hyperparameters (matching the microGPT Python script by @karpathy)
    // vocab_size=32, n_embd=16, n_head=4, n_layer=1, block_size=16
    let vocab_size: i32 = 32;
    let block_size = 16;

    // Generate random token IDs in [0, vocab_size)
    let input_data: Vec<i32> = (0..block_size)
        .map(|_| rng.gen_range(0..vocab_size))
        .collect();

    // Shape [1, 16] — (batch=1, seq_len=16)
    let input = Tensor::new(Some(&input_data), &[1, block_size]).unwrap();

    let outputs = microgpt.forward(&[input.clone()]);
    println!("outputs {outputs:?}");
    println!("Output shape: {:?}", outputs[0].dims());

    let data = outputs[0].data();

    // The output is (1, 16, 32): logits for each position in the sequence.
    // Print the argmax prediction for the last position (next-token prediction).
    let last_pos_start = (block_size - 1) * (vocab_size as usize);
    let last_pos_logits = &data[last_pos_start..last_pos_start + vocab_size as usize];

    let (pred_token, pred_logit) = last_pos_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!("Predicted next token ID: {pred_token}, logit: {pred_logit}");
}
