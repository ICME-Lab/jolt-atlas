use common::utils::logging::setup_tracing;
/// nanoGPT ONNX proof, selectable between the HyperKZG and Dory commitment
/// schemes so their execution profiles can be compared on the same workload.
///
/// ```bash
/// # HyperKZG (default), Chrome trace
/// cargo run --release --package jolt-atlas-core --example gpt2 -- --trace
///
/// # HyperKZG (default), terminal timing
/// cargo run --release --package jolt-atlas-core --example gpt2 -- --trace-terminal
///
/// # Dory, Chrome trace
/// cargo run --release --package jolt-atlas-core --example gpt2 -- --pcs dory --trace
///
/// # Both, back to back (one Chrome trace covering both for comparison)
/// cargo run --release --package jolt-atlas-core --example gpt2 -- --pcs both --trace
/// ```
use jolt_atlas_core::compat::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, Bn254, DoryScheme, Fr, HyperKZG, ONNXProof,
};
use joltworks::poly::commitment::commitment_scheme::CommitmentScheme;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Which commitment scheme(s) to profile.
#[derive(Clone, Copy, PartialEq)]
enum PcsChoice {
    HyperKzg,
    Dory,
    Both,
}

impl PcsChoice {
    /// Reads `--pcs <hyperkzg|dory|both>` from the process arguments, defaulting
    /// to HyperKZG (the historical behaviour of this example).
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let value = args
            .iter()
            .position(|a| a == "--pcs")
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str());
        match value {
            Some("dory") => PcsChoice::Dory,
            Some("both") => PcsChoice::Both,
            Some("hyperkzg") | None => PcsChoice::HyperKzg,
            Some(other) => panic!("unknown --pcs value {other:?} (expected hyperkzg|dory|both)"),
        }
    }
}

/// Prove + verify the model with the chosen PCS, reporting timings. Generic over
/// the commitment scheme so both share exactly one code path.
fn run<PCS: CommitmentScheme<Field = Fr>>(
    label: &str,
    shared: AtlasSharedPreprocessing,
    inputs: &[Tensor<i32>],
) {
    println!("\n=== {label} ===");
    let setup = std::time::Instant::now();
    let prover_preprocessing = AtlasProverPreprocessing::<Fr, PCS>::new(shared);
    println!("[{label}] preprocessing (SRS) took {:.2?}", setup.elapsed());

    let timing = std::time::Instant::now();
    let (proof, io, _debug_info) =
        ONNXProof::<Fr, Blake2bTranscript, PCS>::prove(&prover_preprocessing, inputs);
    println!("[{label}] proof generation took {:.2?}", timing.elapsed());

    let verifier_preprocessing = AtlasVerifierPreprocessing::<Fr, PCS>::from(&prover_preprocessing);

    let timing = std::time::Instant::now();
    proof.verify(&verifier_preprocessing, &io, None).unwrap();
    println!("[{label}] verification took {:.2?}", timing.elapsed());
    println!("[{label}] verified successfully!");
}

fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("nanoGPT ONNX Proof");
    let pcs = PcsChoice::from_args();

    // Reduce sequence_length for faster tracing; increase as needed (max 1024).
    let seq_len: usize = 16;
    let run_args = RunArgs::new([
        ("batch_size", 1),
        ("sequence_length", seq_len),
        ("past_sequence_length", 0),
    ]);
    let model = Model::load("models/gpt2/network.onnx", &run_args);
    println!("{}", model.pretty_print());
    println!("max num vars: {}", model.max_num_vars());

    let mut rng = StdRng::seed_from_u64(42);
    let vocab_size: i32 = 50257;

    // Input 0 → node 1: input_ids — random token IDs used as Gather indices
    let input_ids_data: Vec<i32> = (0..seq_len).map(|_| rng.gen_range(0..vocab_size)).collect();
    let input_ids = Tensor::from_slice(&input_ids_data, &[1, seq_len]).unwrap();

    // Input 1: attention_mask — all 1s, raw integers (it indexes the
    // extended-mask dictionary; the importer's descriptors say RawInt).
    let attention_mask_data: Vec<i32> = vec![1; seq_len];
    let attention_mask = Tensor::from_slice(&attention_mask_data, &[1, seq_len]).unwrap();

    // Input 2: position_ids — sequential positions used as Gather indices.
    let position_ids_data: Vec<i32> = (0..seq_len as i32).collect();
    let position_ids = Tensor::from_slice(&position_ids_data, &[1, seq_len]).unwrap();

    tracing::info!("Loaded model and generated input data");
    let shared = AtlasSharedPreprocessing::preprocess(model);
    let inputs = [input_ids, attention_mask, position_ids];

    // `shared` is cloned per run so `--pcs both` can profile them back to back.
    if matches!(pcs, PcsChoice::HyperKzg | PcsChoice::Both) {
        run::<HyperKZG<Bn254>>("HyperKZG", shared.clone(), &inputs);
    }
    if matches!(pcs, PcsChoice::Dory | PcsChoice::Both) {
        run::<DoryScheme>("Dory", shared.clone(), &inputs);
    }
}
