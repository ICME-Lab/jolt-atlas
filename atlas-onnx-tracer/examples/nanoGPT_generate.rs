use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use common::utils::logging::setup_tracing;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("nanoGPT");

    let nano_gpt = Model::load(
        "atlas-onnx-tracer/models/nanoGPT/network.onnx",
        &Default::default(),
    );
    println!("{}", nano_gpt.pretty_print());
    let mut rng = StdRng::seed_from_u64(0x1096);
    let input_data: Vec<i32> = (0..64)
        .map(|_| (1 << 5) + rng.gen_range(-20..=20))
        .collect();
    let input = Tensor::new(Some(&input_data), &[1, 64]).unwrap();

    let _trace = nano_gpt.trace(&[input.clone()]);
}
