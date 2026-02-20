use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use common::utils::logging::setup_tracing;
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Traces GPT-2 [1] with dummy inputs.
///
/// # Setup
///
/// Install optimum:
///
/// ```sh
/// pip install 'optimum[exporters]'
/// pip install 'optimum[onnxruntime]'
/// ```
///
/// Export the model using Optimum [2]:
///
/// ```sh
/// python -m optimum.exporters.onnx --model gpt2 atlas-onnx-tracer/models/gpt2
/// ```
///
/// # Model inputs (with past_sequence_length=0, KV-cache inputs are pruned)
///
/// | Index | Node | Name            | Shape            | Description                         |
/// |-------|------|-----------------|------------------|-------------------------------------|
/// | 0     | 1    | input_ids       | [1, seq_len]     | Token IDs (Gather indices)          |
/// | 1     | 4    | position_ids    | [1, seq_len]     | Position indices (Gather indices)   |
/// | 2     | 57   | attention_mask  | [1, seq_len]     | Mask (quantized: 1.0 = 1<<scale)   |
///
/// [1] https://openai.com/research/better-language-models
/// [2] https://huggingface.co/docs/optimum/index
fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("gpt2");

    // Reduce sequence_length for faster tracing; increase as needed (max 1024).
    let seq_len: usize = 16;
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ])
    .with_pre_rebase_nonlinear(true);
    let model = Model::load("atlas-onnx-tracer/models/gpt2/network.onnx", &run_args);
    println!("{}", model.pretty_print());
    println!("max num vars: {}", model.max_num_vars());

    let mut rng = StdRng::seed_from_u64(42);
    let vocab_size: i32 = 50257;

    // Input 0 → node 1: input_ids — random token IDs used as Gather indices
    let input_ids_data: Vec<i32> = (0..seq_len).map(|_| rng.gen_range(0..vocab_size)).collect();
    let input_ids = Tensor::new(Some(&input_ids_data), &[1, seq_len]).unwrap();

    // Input 1 → node 4: position_ids — sequential positions used as Gather indices
    let position_ids_data: Vec<i32> = (0..seq_len as i32).collect();
    let position_ids = Tensor::new(Some(&position_ids_data), &[1, seq_len]).unwrap();

    // Input 2 → node 57: attention_mask — all 1s (attend everywhere)
    // The model's Cast handler divides by scale to de-quantize, so we provide
    // the mask in quantized form: 1.0 in fixed-point = 1 << scale.
    let scale = run_args.scale;
    let attention_mask_data: Vec<i32> = vec![1 << scale; seq_len];
    let attention_mask = Tensor::new(Some(&attention_mask_data), &[1, seq_len]).unwrap();

    let output = model.forward(&[input_ids, position_ids, attention_mask]);
    println!("Output shape: {:?}", output[0].dims());
}
