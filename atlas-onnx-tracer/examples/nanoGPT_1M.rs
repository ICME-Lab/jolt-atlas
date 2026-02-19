use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    let model = Model::load(
        "atlas-onnx-tracer/models/nanoGPT_1M/network.onnx",
        &Default::default(),
    );
    println!("{}", model.pretty_print());

    let mut rng = StdRng::seed_from_u64(42);
    let input_data: Vec<i32> = (0..128).map(|_| rng.gen_range(0..65)).collect();
    let input = Tensor::new(Some(&input_data), &[1, 128]).unwrap();

    let trace = model.trace(&[input]);
    println!("trace outputs: {}", trace.node_outputs.len());
}
