use atlas_onnx_tracer::{model::Model, tensor::Tensor};

fn main() {
    let transformer = Model::load("./models/transformer/network.onnx", &Default::default());
    println!("{}", transformer.pretty_print());

    let input_data = vec![1 << 7; 64 * 64];
    let input = Tensor::new(Some(&input_data), &[1, 64, 64]).unwrap();

    let outputs = transformer.forward(&[input.clone()]);
    println!("Outputs: {outputs:?}");
}
