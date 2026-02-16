use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    let minigpt = Model::load("./models/minigpt/network.onnx", &Default::default());
    println!("model: {}", minigpt.pretty_print());

    let mut rng = StdRng::seed_from_u64(0x42);

    // Model hyperparameters (matching the minigpt Python script by @karpathy)
    // vocab_size=1024, n_embd=32, n_head=8, n_layer=2, block_size=32
    let vocab_size: i32 = 1024;
    let block_size = 32;

    // Generate random token IDs in [0, vocab_size)
    let input_data: Vec<i32> = (0..block_size)
        .map(|_| rng.gen_range(0..vocab_size))
        .collect();

    // Shape [1, 32] — (batch=1, seq_len=32)
    let input = Tensor::new(Some(&input_data), &[1, block_size]).unwrap();

    let outputs = minigpt.forward(&[input.clone()]);
    // println!("outputs {outputs:?}");
    println!("Output shape: {:?}", outputs[0].dims());

    let data = outputs[0].data();

    // The output is (1, 32, 1024): logits for each position in the sequence.
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
