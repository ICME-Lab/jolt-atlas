use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};

fn main() {
    let run_args = RunArgs::default();
    let model = Model::load("models/self_attention_layer/network.onnx", &run_args);
    println!("{}", model.pretty_print());

    let input_data = vec![1; 64 * 64];
    let input = Tensor::new(Some(&input_data), &[1, 64, 64]).unwrap();

    let _outputs = model.forward(&[input]);
}
