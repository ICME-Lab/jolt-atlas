use atlas_onnx_tracer::model::Model;

fn main() {
    let relu_nano_gpt = Model::load("./models/relu_nanoGPT/network.onnx", &Default::default());
    println!("{}", relu_nano_gpt.pretty_print());
}
