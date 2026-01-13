use atlas_onnx_tracer::model::{Model, RunArgs};

/// Generates text using GPT-2 [1] and a prompt.
///
/// Install optimum
///
/// ```sh
/// pip install 'optimum[exporters]'
/// ```
///
/// ```sh
/// pip install 'optimum[onnxruntime]'
/// ```
///
/// Then, export the model using Optimum [2]:
///
/// ```sh
/// python -m optimum.exporters.onnx --model gpt2 atlas-onnx-tracer/models/gpt2
/// ```
/// TODO: doc on how to prompt using atlas-onnx-tracer
///
/// [1] https://openai.com/research/better-language-models
/// [2] https://huggingface.co/docs/optimum/index
fn main() {
    // TODO: handle multiple inputs

    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", 512),
        ("past_sequence_length", 0),
    ]);
    let gpt2 = Model::load("./models/gpt2/model.onnx", &run_args);
    println!("{}", gpt2.pretty_print());
}
