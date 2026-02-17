use atlas_onnx_tracer::{model::Model, tensor::Tensor};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn main() {
    let nano_gpt = Model::load("./models/nanoGPT/network.onnx", &Default::default());
    println!("{}", nano_gpt.pretty_print());
    let mut rng = StdRng::seed_from_u64(0x1096);
    let input_data: Vec<i32> = (0..64)
        .map(|_| (1 << 5) + rng.gen_range(-20..=20))
        .collect();
    let input = Tensor::new(Some(&input_data), &[1, 64]).unwrap();

    let trace = nano_gpt.trace(&[input.clone()]);

    let tanh_layer_data = trace.layer_data(&nano_gpt[111]);
    println!(
        "tanh max:{:?}",
        tanh_layer_data.operands[0].iter().max().unwrap(),
    );
    println!(
        "tanh min:{:?}",
        tanh_layer_data.operands[0].iter().min().unwrap(),
    );
    println!("--------------");
    let tanh_layer_data = trace.layer_data(&nano_gpt[225]);
    println!(
        "tanh max:{:?}",
        tanh_layer_data.operands[0].iter().max().unwrap(),
    );
    println!(
        "tanh min:{:?}",
        tanh_layer_data.operands[0].iter().min().unwrap(),
    );
    println!("--------------");
    let tanh_layer_data = trace.layer_data(&nano_gpt[339]);
    println!(
        "tanh max:{:?}",
        tanh_layer_data.operands[0].iter().max().unwrap(),
    );
    println!(
        "tanh min:{:?}",
        tanh_layer_data.operands[0].iter().min().unwrap(),
    );
    println!("--------------");
    let tanh_layer_data = trace.layer_data(&nano_gpt[453]);
    println!(
        "tanh max:{:?}",
        tanh_layer_data.operands[0].iter().max().unwrap(),
    );
    println!(
        "tanh min:{:?}",
        tanh_layer_data.operands[0].iter().min().unwrap(),
    );
    println!("--------------");
}
