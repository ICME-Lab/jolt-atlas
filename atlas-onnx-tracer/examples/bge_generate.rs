/// Traces BGE-small-en-v1.5 with dummy inputs.
///
/// # Setup
///
/// Download the model:
///
/// ```sh
/// python scripts/download_bge_small_en_v1_5.py
/// ```
///
/// # Model inputs
///
/// | Index | Name            | Shape            | Description                           |
/// |-------|-----------------|------------------|---------------------------------------|
/// | 0     | input_ids       | [1, seq_len]     | Token IDs (Gather indices)            |
/// | 1     | token_type_ids  | [1, seq_len]     | Segment IDs (all zeros for single)    |
/// | 2     | attention_mask  | [1, seq_len]     | Binary mask (1 = attend)              |
use atlas_onnx_tracer::model::{Model, RunArgs};

fn main() {
    let seq_len: usize = 16;
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ]);
    let model = Model::load(
        "atlas-onnx-tracer/models/bge-small-en-v1.5/network.onnx",
        &run_args,
    );
    println!("{}", model.pretty_print());
    println!("max num vars: {}", model.max_num_vars());
}
