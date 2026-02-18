use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    let transformer = Model::load("atlas-onnx-tracer/models/transformer/network.onnx", &Default::default());
    println!("{}", transformer.pretty_print());

    let mut rng = StdRng::seed_from_u64(0x8096);
    let input_data: Vec<i32> = (0..64 * 64)
        .map(|_| (1 << 7) + rng.gen_range(-50..=50))
        .collect();
    let input = Tensor::new(Some(&input_data), &[1, 64, 64]).unwrap();

    let _trace = transformer.trace(&[input.clone()]);
}
