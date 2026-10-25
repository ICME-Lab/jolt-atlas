//! The shared end-to-end path.

use std::time::Instant;

use atlas_trace::{trace::ModelExecutionIO, Model, Tensor};
use joltworks::{
    poly::commitment::commitment_scheme::CommitmentScheme,
    transcripts::{Blake2bProver, Blake2bVerifier},
};

use crate::onnx_proof::{
    proof_serialization::{deserialize_proof, serialize_proof},
    prove, verify, AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    ONNXProof,
};

/// Knobs for [`prove_verify_roundtrip`]; the defaults are the full check.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoundtripOptions {
    /// Verify with the prover's debug info so the verifier's transcript and
    /// opening accumulator are compared step by step against the prover's.
    pub lockstep_debug: bool,
    /// Serialize → deserialize → verify again → re-serialize byte-identically.
    pub serialization_roundtrip: bool,
    /// Over-provision the SRS (Dory: wider columns).
    pub srs_num_vars: Option<usize>,
}

impl Default for RoundtripOptions {
    fn default() -> Self {
        Self {
            lockstep_debug: true,
            serialization_roundtrip: true,
            srs_num_vars: None,
        }
    }
}

/// Prove `model` on `inputs`, verify, and round-trip the proof through the
/// wire format. Panics on any failure. Returns the public IO.
pub(crate) fn prove_verify_roundtrip<PCS>(
    model: Model,
    inputs: &[Tensor<i32>],
    opts: RoundtripOptions,
) -> ModelExecutionIO
where
    PCS: CommitmentScheme,
{
    // Independent reference execution: the proven statement must be exactly
    // what the executor computes.
    let expected = model
        .trace(inputs)
        .expect("reference execution")
        .io(&model)
        .outputs;

    let shared = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = match opts.srs_num_vars {
        Some(n) => AtlasProverPreprocessing::<PCS::Field, PCS>::new_with_srs_num_vars(shared, n),
        None => AtlasProverPreprocessing::<PCS::Field, PCS>::new(shared),
    };

    let t = Instant::now();
    let (proof, io, debug) = prove::<PCS::Field, Blake2bProver, PCS>(&prover_pp, inputs);
    println!("prove: {:?}", t.elapsed());
    assert_eq!(
        io.outputs, expected,
        "prover IO diverged from the reference execution"
    );

    let verifier_pp = AtlasVerifierPreprocessing::<PCS::Field, PCS>::from(&prover_pp);
    let t = Instant::now();
    verify::<PCS::Field, Blake2bVerifier, PCS>(
        &proof,
        &verifier_pp,
        &io,
        if opts.lockstep_debug { debug } else { None },
    )
    .expect("honest proof must verify");
    println!("verify: {:?}", t.elapsed());

    if opts.serialization_roundtrip {
        let bytes = serialize_proof(&proof).expect("proof serializes");
        println!("proof size: {} bytes", bytes.len());
        let de: ONNXProof<PCS::Field, PCS> = deserialize_proof(&bytes).expect("proof deserializes");
        verify::<PCS::Field, Blake2bVerifier, PCS>(&de, &verifier_pp, &io, None)
            .expect("deserialized proof must verify");
        assert_eq!(
            serialize_proof(&de).expect("re-serializes"),
            bytes,
            "serialization must be idempotent"
        );
    }
    io
}
