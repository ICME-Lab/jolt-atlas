use atlas_onnx_tracer::model::Model;

fn main() {
    let nano_gpt = Model::load("./models/nanoGPT/network.onnx", &Default::default());
    println!("{}", nano_gpt.pretty_print());
}
