use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};

fn main() {
    // Load the perceptron model with padding enabled (default)
    let run_args = RunArgs::default();
    let model = Model::load("atlas-onnx-tracer/models/perceptron/network.onnx", &run_args);

    println!("{}", model.pretty_print());

    // Create input tensor with shape [1, 4]
    let input_data = vec![1, 2, 3, 4];
    let input = Tensor::new(Some(&input_data), &[1, 4]).unwrap();

    // Execute the model
    let outputs = model.forward(&[input]);

    println!("\nOutput tensors:");
    for (i, output) in outputs.iter().enumerate() {
        println!("  Output {i}: {output:?}");
    }
}
