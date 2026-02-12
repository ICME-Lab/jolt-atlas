use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    let layernorm_head = Model::load("./models/layernorm_head/network.onnx", &Default::default());
    println!("{}", layernorm_head.pretty_print());

    let mut rng = StdRng::seed_from_u64(0x8096);
    let input_data: Vec<i32> = (0..16 * 16)
        .map(|_| (1 << 7) + rng.gen_range(-50..=50))
        .collect();
    let input = Tensor::new(Some(&input_data), &[16, 16]).unwrap();

    let outputs = layernorm_head.forward(&[input.clone()]);
    println!("Outputs: {outputs:?}");
}
