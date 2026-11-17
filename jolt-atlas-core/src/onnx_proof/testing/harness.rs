//! The shared end-to-end path.

use std::time::Instant;

use atlas_pcs::commitment_scheme::CommitmentScheme;
use atlas_trace::{trace::ModelExecutionIO, Model, Tensor};
use atlas_transcript::{Blake2bProver, Blake2bVerifier};

use crate::onnx_proof::{
    prove, verify, AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    ONNXProof,
};

/// Knobs for [`prove_verify_roundtrip`]; the defaults are the full check.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoundtripOptions {
    /// Verify with the prover's debug info so the verifier's transcript and
    /// opening accumulator are compared step by step against the prover's.
    pub lockstep_debug: bool,
    /// Verify a proof rebuilt from the proof's bytes, and require the bytes followed by one
    /// more to be rejected.
    pub bytes_roundtrip: bool,
}

impl Default for RoundtripOptions {
    fn default() -> Self {
        Self {
            lockstep_debug: true,
            bytes_roundtrip: true,
        }
    }
}

/// Prove `model` on `inputs`, verify, and verify again from the proof's bytes.
/// Panics on any failure. Returns the public IO.
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
    let prover_pp = AtlasProverPreprocessing::<PCS::Field, PCS>::new(shared);

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

    if opts.bytes_roundtrip {
        println!("proof size: {} bytes", proof.0.len());
        verify::<PCS::Field, Blake2bVerifier, PCS>(
            &ONNXProof(proof.0.clone()),
            &verifier_pp,
            &io,
            None,
        )
        .expect("the proof's bytes must verify");
        let trailing = ONNXProof([proof.0.as_slice(), &[0]].concat());
        assert!(
            verify::<PCS::Field, Blake2bVerifier, PCS>(&trailing, &verifier_pp, &io, None).is_err(),
            "a proof followed by one more byte must not verify"
        );
    }
    io
}
