use atlas_onnx_tracer::{model::Model, tensor::Tensor};

fn main() {
    let multihead_attention = Model::load(
        "atlas-onnx-tracer/models/multihead_attention/network.onnx",
        &Default::default(),
    );
    println!("{}", multihead_attention.pretty_print());

    let input_data = vec![128; 16 * 128];
    let input = Tensor::new(Some(&input_data), &[1, 1, 16, 128]).unwrap();

    let outputs = multihead_attention.forward(&[input]);
    println!("Outputs: {outputs:?}");
}
