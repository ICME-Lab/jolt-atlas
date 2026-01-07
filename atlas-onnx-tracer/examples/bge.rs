use atlas_onnx_tracer::model::{Model, RunArgs};

// TODO: add bge-small onnx file to .gitignore and add docs on how to download bge-small-en
fn main() {
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", 512),
        ("past_sequence_length", 0),
    ]);
    let bge_small = Model::load("./models/bge-small/network.onnx", &run_args);
    println!("{}", bge_small.pretty_print());
}
