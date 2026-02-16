use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    let transformer = Model::load("./models/transformer/network.onnx", &Default::default());
    println!("{}", transformer.pretty_print());

    let mut rng = StdRng::seed_from_u64(0x8096);
    let input_data: Vec<i32> = (0..64 * 64)
        .map(|_| (1 << 7) + rng.gen_range(-50..=50))
        .collect();
    let input = Tensor::new(Some(&input_data), &[1, 64, 64]).unwrap();

    let trace = transformer.trace(&[input.clone()]);
    let tanh_layer_data = trace.layer_data(&transformer[110]);
    println!(
        "tanh max:{:?}",
        tanh_layer_data.operands[0].iter().max().unwrap(),
    );
    println!(
        "tanh min:{:?}",
        tanh_layer_data.operands[0].iter().min().unwrap(),
    );
}
