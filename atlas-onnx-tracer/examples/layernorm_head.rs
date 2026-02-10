use atlas_onnx_tracer::{model::Model, tensor::Tensor};

fn main() {
    let layernorm_head = Model::load("./models/layernorm_head/network.onnx", &Default::default());
    println!("{}", layernorm_head.pretty_print());

    let input_data = vec![1 << 4; 16 * 16];
    let input = Tensor::new(Some(&input_data), &[16, 16]).unwrap();

    let outputs = layernorm_head.forward(&[input.clone()]);
    println!("Outputs: {outputs:?}");
    let trace = layernorm_head.trace(&[input]);
    println!("Trace: {trace:#?}");
}
