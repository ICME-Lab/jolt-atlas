use ark_bn254::Fr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::{poly::commitment::dory::DoryCommitmentScheme, transcripts::KeccakTranscript};
use onnx_tracer::{builder, tensor::Tensor};
use zkml_jolt_core::jolt::{JoltSNARK, JoltVerifierPreprocessing};

#[allow(clippy::upper_case_acronyms)]
type PCS = DoryCommitmentScheme;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Proof serialization round-trip test");
    println!("====================================\n");

    // Create input tensor
    let input_data = vec![1, 2, 3, 4];
    let shape = [1, 4];
    let input_tensor = Tensor::new(Some(&input_data), &shape)?;

    // Preprocess and generate proof
    let max_trace_length = 1 << 12;
    let preprocessing = JoltSNARK::<Fr, PCS, KeccakTranscript>::prover_preprocess(
        builder::simple_mlp_small_model,
        max_trace_length,
    );
    let verifier_preprocessing: JoltVerifierPreprocessing<Fr, PCS> = (&preprocessing).into();

    let (snark, program_io, _debug_info) = JoltSNARK::<Fr, PCS, KeccakTranscript>::prove(
        &preprocessing,
        builder::simple_mlp_small_model,
        &input_tensor,
    );

    // Verify original proof
    snark
        .clone()
        .verify(&verifier_preprocessing, program_io.clone(), None)
        .expect("original proof should verify");
    println!("✓ Original proof verified");

    // Serialize proof
    let mut buffer = Vec::new();
    snark
        .serialize_compressed(&mut buffer)
        .expect("serialization should succeed");
    println!("✓ Proof serialized ({} bytes)", buffer.len());

    // Deserialize proof
    let deserialized_snark =
        JoltSNARK::<Fr, PCS, KeccakTranscript>::deserialize_compressed(buffer.as_slice())
            .expect("deserialization should succeed");
    println!("✓ Proof deserialized");

    // Verify deserialized proof
    deserialized_snark
        .verify(&verifier_preprocessing, program_io, None)
        .expect("deserialized proof should verify");
    println!("✓ Deserialized proof verified\n");

    println!("Round-trip serialization successful!");
    Ok(())
}
