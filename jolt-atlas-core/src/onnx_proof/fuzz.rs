//! The `verify` fuzz target's input format and driver, shared by the cargo-fuzz harness in
//! `fuzz/` and the regression replay in the test suite.
//!
//! An input is an 8-byte header followed by a serialized proof. The header edits a copy of the
//! honest public IO (`[kind, tensor, index_lo, index_hi, delta, ..]`, see [`mutate_io`]); the
//! rest is handed to the proof deserializer. Whatever the bytes, the verifier must return: an
//! `Err` is the expected outcome, a panic is a finding.

use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_onnx_import::fixtures::{standard_bindings, synth_inputs, GOLDEN_SEED};
use atlas_pcs::commitment_scheme::CommitmentScheme;
use atlas_trace::{trace::ModelExecutionIO, Tensor};
use atlas_transcript::{Blake2bProver, Blake2bVerifier};

use crate::loader::{load_imported, RunArgs};
use crate::onnx_proof::{
    proof_serialization::{deserialize_proof, serialize_proof},
    prove, AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
};

/// Bytes of the IO-edit header in front of the serialized proof.
pub const HEADER_LEN: usize = 8;

/// The model the target verifies against, with an honest proof of it: the checked-in op
/// gauntlet on its golden inputs.
pub struct VerifyTarget<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    verifier_pp: AtlasVerifierPreprocessing<F, PCS>,
    io: ModelExecutionIO,
    seed: Vec<u8>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> VerifyTarget<F, PCS> {
    /// Load `models/op_gauntlet` from `repo_root`, prove it once, and keep the verifier's side.
    pub fn new(repo_root: &std::path::Path) -> Self {
        let path = repo_root.join("models/op_gauntlet/network.onnx");
        let imported = load_imported(
            path.to_str().expect("utf-8 model path"),
            &RunArgs::new(standard_bindings(16)),
        );
        let inputs = synth_inputs(&imported, GOLDEN_SEED);
        let prover_pp = AtlasProverPreprocessing::<F, PCS>::new(
            AtlasSharedPreprocessing::preprocess(imported.model),
        );
        let (proof, io, _) = prove::<F, Blake2bProver, PCS>(&prover_pp, &inputs);
        let mut seed = vec![0; HEADER_LEN];
        seed.extend(serialize_proof(&proof).expect("the honest proof serializes"));
        Self {
            verifier_pp: AtlasVerifierPreprocessing::from(&prover_pp),
            io,
            seed,
        }
    }

    /// The honest input: a zero header (no IO edit) and the honest proof. The corpus seed.
    pub fn seed(&self) -> &[u8] {
        &self.seed
    }

    /// Verify one fuzz input. `None` if the bytes are too short or do not deserialize to a
    /// proof; otherwise the verifier's verdict on the decoded proof and the edited IO.
    pub fn run(&self, data: &[u8]) -> Option<Result<(), ProofVerifyError>> {
        let (header, proof_bytes) = data.split_at_checked(HEADER_LEN)?;
        let proof = deserialize_proof::<F, PCS>(proof_bytes).ok()?;
        let mut io = self.io.clone();
        mutate_io(&mut io, header);
        Some(proof.verify::<Blake2bVerifier>(&self.verifier_pp, &io))
    }
}

/// Apply the header's IO edit: `kind` selects it, `tensor`/`index` address an element (both
/// reduced modulo the available range), `delta` is added to it.
///
/// | kind | edit |
/// |------|------|
/// | 1 | `inputs[tensor][index] += delta` |
/// | 2 | `outputs[tensor][index] += delta` |
/// | 3 | drop the last input tensor |
/// | 4 | drop every output tensor |
/// | 5 | truncate `outputs[tensor]` to `index` elements |
/// | 6 | `input_indices[tensor] += delta` |
/// | 7 | `output_indices[tensor] += delta` |
/// | other | none |
fn mutate_io(io: &mut ModelExecutionIO, header: &[u8]) {
    let &[kind, tensor, index_lo, index_hi, delta, ..] = header else {
        return;
    };
    let (tensor, index) = (
        tensor as usize,
        u16::from_le_bytes([index_lo, index_hi]) as usize,
    );
    let bump = |tensors: &mut Vec<Tensor<i32>>| {
        if tensors.is_empty() {
            return;
        }
        let slot = tensor % tensors.len();
        let data = tensors[slot].data_mut();
        if !data.is_empty() {
            let at = index % data.len();
            data[at] = data[at].wrapping_add(delta as i32);
        }
    };
    match kind {
        1 => bump(&mut io.inputs),
        2 => bump(&mut io.outputs),
        3 => {
            io.inputs.pop();
        }
        4 => io.outputs.clear(),
        5 if !io.outputs.is_empty() => {
            let slot = tensor % io.outputs.len();
            let data = io.outputs[slot].data();
            let len = index % (data.len() + 1);
            if let Ok(truncated) = Tensor::from_slice(&data[..len], &[len]) {
                io.outputs[slot] = truncated;
            }
        }
        6 if !io.input_indices.is_empty() => {
            let slot = tensor % io.input_indices.len();
            io.input_indices[slot] = io.input_indices[slot].wrapping_add(delta as usize);
        }
        7 if !io.output_indices.is_empty() => {
            let slot = tensor % io.output_indices.len();
            io.output_indices[slot] = io.output_indices[slot].wrapping_add(delta as usize);
        }
        _ => {}
    }
}
