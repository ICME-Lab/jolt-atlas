use atlas_onnx_tracer::model::{Model, RunArgs};

fn main() {
    // pretty print gpt-2
    let path = "atlas-onnx-tracer/models/gpt2/network.onnx";
    let run_args = RunArgs::default()
        .with_padding(false)
        .with("sequence_length", 512)
        .with("past_sequence_length", 0);
    let gpt2 = Model::load(path, &run_args);
    println!("{}", gpt2.pretty_print());
}
