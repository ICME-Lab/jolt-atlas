//! --- Jolt ONNX VM ---

use crate::jolt::{
    bytecode::{BytecodePreprocessing, JoltONNXBytecode},
    dag::{
        jolt_dag::JoltDAG,
        state_manager::{Proofs, StateManager, VerifierState},
    },
    pcs::{Openings, ProverOpeningAccumulator, VerifierOpeningAccumulator},
    trace::trace,
};
#[cfg(test)]
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::{
    field::JoltField,
    poly::{commitment::commitment_scheme::CommitmentScheme, opening_proof::OpeningPoint},
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
    zkvm::witness::DTH_ROOT_OF_K,
};
use onnx_tracer::{ProgramIO, graph::model::Model, tensor::Tensor};
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, rc::Rc};

pub mod bytecode;
pub mod dag;
pub mod executor;
pub mod lookup_table;
pub mod memory;
pub mod pcs;
pub mod r1cs;
pub mod sumcheck;
pub mod trace;
pub mod witness;

#[derive(Debug, Clone)]
pub struct Claims<F: JoltField>(Openings<F>);

/// A SNARK for ONNX model inference
#[derive(Clone, Debug)]
pub struct JoltSNARK<F: JoltField, PCS: CommitmentScheme<Field = F>, FS: Transcript> {
    opening_claims: Claims<F>,
    commitments: Vec<PCS::Commitment>,
    proofs: Proofs<F, PCS, FS>,
    pub trace_length: usize,
    memory_K: usize,
    _bytecode_d: usize,
    twist_sumcheck_switch_index: usize,
}

impl<F, PCS, FS> JoltSNARK<F, PCS, FS>
where
    FS: Transcript,
    PCS: CommitmentScheme<Field = F>,
    F: JoltField,
{
    /// Jolt DAG prover
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "Jolt::prove")]
    pub fn prove<ModelFunc>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        model: ModelFunc,
        input: &Tensor<i32>,
    ) -> (Self, ProgramIO, Option<ProverDebugInfo<F, FS, PCS>>)
    where
        ModelFunc: Fn() -> Model,
    {
        let (trace, program_io) = trace(model, input, &preprocessing.shared.bytecode);
        let state_manager: StateManager<'_, F, FS, PCS> =
            StateManager::new_prover(preprocessing, trace, program_io.clone());
        let (snark, debug_info) = JoltDAG::prove(state_manager).ok().unwrap();
        (snark, program_io, debug_info)
    }

    #[tracing::instrument(skip_all, name = "Jolt::verify")]
    pub fn verify(
        self,
        preprocessing: &JoltVerifierPreprocessing<F, PCS>,
        program_io: ProgramIO,
        _debug_info: Option<ProverDebugInfo<F, FS, PCS>>,
    ) -> Result<(), ProofVerifyError> {
        #[cfg(test)]
        let T = self.trace_length.next_power_of_two();
        // Need to initialize globals because the verifier computes commitments
        // in `VerifierOpeningAccumulator::append` inside of a `#[cfg(test)]` block
        #[cfg(test)]
        let _guard = DoryGlobals::initialize(DTH_ROOT_OF_K, T);

        let state_manager = self.to_verifier_state_manager(preprocessing, program_io);

        #[cfg(test)]
        {
            if let Some(debug_info) = _debug_info {
                let mut transcript = state_manager.transcript.borrow_mut();
                transcript.compare_to(debug_info.transcript);
                let opening_accumulator = state_manager.get_verifier_accumulator();
                opening_accumulator
                    .borrow_mut()
                    .compare_to(debug_info.opening_accumulator);
            }
        }

        JoltDAG::verify(state_manager).expect("Verification failed");

        Ok(())
    }

    pub fn from_prover_state_manager(mut state_manager: StateManager<'_, F, FS, PCS>) -> Self {
        let prover_state = state_manager.prover_state.as_mut().unwrap();
        let openings = std::mem::take(&mut prover_state.accumulator.borrow_mut().openings);
        let commitments = state_manager.commitments.take();
        let proofs = state_manager.proofs.take();
        let trace_length = prover_state.trace.len();
        let memory_K = state_manager.memory_K;
        let twist_sumcheck_switch_index = state_manager.twist_sumcheck_switch_index;

        Self {
            opening_claims: Claims(openings),
            commitments,
            proofs,
            trace_length,
            memory_K,
            _bytecode_d: prover_state.preprocessing.shared.bytecode.d,
            twist_sumcheck_switch_index,
        }
    }

    pub fn to_verifier_state_manager<'a>(
        self,
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        program_io: ProgramIO,
    ) -> StateManager<'a, F, FS, PCS> {
        let mut opening_accumulator = VerifierOpeningAccumulator::<F>::new();
        // Populate claims in the verifier accumulator
        for (key, (_, claim)) in self.opening_claims.0.iter() {
            opening_accumulator
                .openings_mut()
                .insert(*key, (OpeningPoint::default(), *claim));
        }

        let proofs = Rc::new(RefCell::new(self.proofs));
        let commitments = Rc::new(RefCell::new(self.commitments));
        let transcript = Rc::new(RefCell::new(FS::new(b"Jolt")));

        StateManager {
            transcript,
            proofs,
            commitments,
            program_io,
            memory_K: self.memory_K,
            twist_sumcheck_switch_index: self.twist_sumcheck_switch_index,
            prover_state: None,
            verifier_state: Some(VerifierState {
                preprocessing,
                trace_length: self.trace_length,
                accumulator: Rc::new(RefCell::new(opening_accumulator)),
            }),
        }
    }
}

#[allow(dead_code)]
pub struct ProverDebugInfo<F, ProofTranscript, PCS>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    pub transcript: ProofTranscript,
    pub opening_accumulator: ProverOpeningAccumulator<F>,
    pub prover_setup: PCS::ProverSetup,
}

/// Preprocessing data needed for both prover and verifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoltSharedPreprocessing {
    /// The preprocessed bytecode
    pub bytecode: BytecodePreprocessing,
    // pub precompiles: PrecompilePreprocessing,
}

/// Preprocessing data needed only for the prover
#[derive(Clone, Serialize, Deserialize)]
pub struct JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::ProverSetup,
    pub shared: JoltSharedPreprocessing,
}

impl<F, PCS> JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn memory_K(&self) -> usize {
        self.shared.bytecode.memory_K
    }

    pub fn bytecode_d(&self) -> usize {
        self.shared.bytecode.d
    }

    pub fn bytecode(&self) -> &[JoltONNXBytecode] {
        &self.shared.bytecode.bytecode
    }
}

/// Preprocessing data needed only for the verifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,
    pub shared: JoltSharedPreprocessing,
}

impl<F, PCS> JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn memory_K(&self) -> usize {
        self.shared.bytecode.memory_K
    }
}

impl<F, PCS, FS> JoltSNARK<F, PCS, FS>
where
    FS: Transcript,
    PCS: CommitmentScheme<Field = F>,
    F: JoltField,
{
    /// Preprocesses the ONNX model to produce shared preprocessing data
    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    pub fn shared_preprocess<ModelFunc>(model: ModelFunc) -> JoltSharedPreprocessing
    where
        ModelFunc: Fn() -> Model,
    {
        let bytecode_preprocessing = BytecodePreprocessing::preprocess(model);
        // let precompile_preprocessing = PrecompilePreprocessing::preprocess(&bytecode);
        JoltSharedPreprocessing {
            bytecode: bytecode_preprocessing,
            // precompiles: precompile_preprocessing,
        }
    }

    /// Preprocesses the ONNX model to produce prover preprocessing data.
    /// * Preproceses the bytecode
    /// * Sets up commitment key
    #[tracing::instrument(skip_all, name = "Jolt::preprocess")]
    pub fn prover_preprocess<ModelFunc>(
        model: ModelFunc,
        max_trace_length: usize,
    ) -> JoltProverPreprocessing<F, PCS>
    where
        ModelFunc: Fn() -> Model,
    {
        let shared = Self::shared_preprocess(model);
        let max_T: usize = max_trace_length.next_power_of_two();
        let generators = PCS::setup_prover(DTH_ROOT_OF_K.log_2() + max_T.log_2());
        JoltProverPreprocessing { shared, generators }
    }
}

impl<F, PCS> From<&JoltProverPreprocessing<F, PCS>> for JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn from(preprocessing: &JoltProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&preprocessing.generators);
        JoltVerifierPreprocessing {
            generators,
            shared: preprocessing.shared.clone(),
        }
    }
}

#[cfg(test)]
mod e2e_tests {
    use crate::jolt::JoltSNARK;
    use ark_bn254::Fr;
    use jolt_core::{
        poly::commitment::{dory::DoryCommitmentScheme, mock::MockCommitScheme},
        transcripts::KeccakTranscript,
    };
    use onnx_tracer::{builder, tensor::Tensor};
    use serial_test::serial;

    type PCS0 = DoryCommitmentScheme;
    type _PCS1 = MockCommitScheme<Fr>;

    #[test]
    #[serial]
    fn test_addsubmulconst() {
        let model = builder::addsubmulconst_model;
        let input = Tensor::new(Some(&[1, 2, 3, 4]), &[1, 4]).unwrap();
        let preprocessing =
            JoltSNARK::<Fr, PCS0, KeccakTranscript>::prover_preprocess(model, 1 << 10);
        let (snark, program_io, _debug_info) =
            JoltSNARK::<Fr, PCS0, KeccakTranscript>::prove(&preprocessing, model, &input);
        snark
            .verify(&(&preprocessing).into(), program_io, None)
            .unwrap();
    }

    #[test]
    #[serial]
    fn test_addsubmul() {
        let model = builder::addsubmul_model;
        let input = Tensor::new(Some(&[1, 2, 3, 4]), &[1, 4]).unwrap();
        let preprocessing =
            JoltSNARK::<Fr, PCS0, KeccakTranscript>::prover_preprocess(model, 1 << 10);
        let (snark, program_io, _debug_info) =
            JoltSNARK::<Fr, PCS0, KeccakTranscript>::prove(&preprocessing, model, &input);
        snark
            .verify(&(&preprocessing).into(), program_io, None)
            .unwrap();
    }

    #[test]
    #[serial]
    fn test_add() {
        let model = builder::add_model;
        let input = Tensor::new(Some(&[3, 4, 5, 0]), &[1, 4]).unwrap();
        let preprocessing =
            JoltSNARK::<Fr, PCS0, KeccakTranscript>::prover_preprocess(model, 1 << 10);
        let (snark, program_io, _debug_info) =
            JoltSNARK::<Fr, PCS0, KeccakTranscript>::prove(&preprocessing, model, &input);
        snark
            .verify(&(&preprocessing).into(), program_io, None)
            .unwrap();
    }

    #[test]
    #[serial]
    fn test_scalar_input_and_inference() {
        let model = builder::scalar_addsubmul_model;
        let input = Tensor::new(Some(&[10]), &[1, 1]).unwrap();
        let preprocessing =
            JoltSNARK::<Fr, PCS0, KeccakTranscript>::prover_preprocess(model, 1 << 10);
        let (snark, program_io, _debug_info) =
            JoltSNARK::<Fr, PCS0, KeccakTranscript>::prove(&preprocessing, model, &input);
        snark
            .verify(&(&preprocessing).into(), program_io, None)
            .unwrap();
    }

    #[test]
    #[serial]
    fn test_relu() {
        let model = builder::relu_model;
        let input = Tensor::new(Some(&[-3, -2, 0, 1]), &[1, 4]).unwrap();
        let preprocessing =
            JoltSNARK::<Fr, PCS0, KeccakTranscript>::prover_preprocess(model, 1 << 10);
        let (snark, program_io, _debug_info) =
            JoltSNARK::<Fr, PCS0, KeccakTranscript>::prove(&preprocessing, model, &input);
        snark
            .verify(&(&preprocessing).into(), program_io, None)
            .unwrap();
    }
}
