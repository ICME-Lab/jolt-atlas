//! State management for Jolt prover and verifier.
//!
//! This module provides unified state handling for both proving and verification,
//! managing transcripts, accumulators, commitments, and proof data throughout
//! the protocol execution.
//!
//! # Zero-Knowledge Support
//!
//! When `ZKConfig::enabled` is true, the proving process uses hiding commitments
//! and ZK sumcheck protocols to ensure zero-knowledge properties.

use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{BIG_ENDIAN, OpeningPoint},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::math::Math,
};
use num_derive::FromPrimitive;
use onnx_tracer::{ProgramIO, utils::normalize};

use crate::jolt::{
    JoltProverPreprocessing, JoltVerifierPreprocessing,
    bytecode::{BytecodePreprocessing, JoltONNXBytecode},
    fp_lookups::FpLookupProof,
    pcs::{ProverOpeningAccumulator, ReducedOpeningProof, SumcheckId, VerifierOpeningAccumulator},
    precompiles::{PrecompilePreprocessing, PrecompileSNARK},
    trace::JoltONNXCycle,
    witness::{CommittedPolynomial, VirtualPolynomial},
};

/// Configuration for zero-knowledge proof generation.
///
/// When ZK mode is enabled, the prover uses hiding commitments and
/// ZK sumcheck protocols to ensure the proof reveals nothing about
/// the polynomial coefficients or intermediate values.
#[derive(Clone, Debug, Default)]
pub struct ZKConfig {
    /// Whether zero-knowledge mode is enabled.
    ///
    /// When true:
    /// - Sumcheck proofs use hiding commitments instead of plaintext coefficients
    /// - The BlindFold approach with NIFS folding is used for ZK
    /// - Blinding factors are tracked throughout the protocol
    pub enabled: bool,
}

impl ZKConfig {
    /// Creates a new ZK config with ZK mode enabled.
    pub fn enabled() -> Self {
        Self { enabled: true }
    }

    /// Creates a new ZK config with ZK mode disabled (default).
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

/// Keys identifying different proof components in the proof map.
#[derive(PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, FromPrimitive)]
#[repr(u8)]
pub enum ProofKeys {
    Stage1Sumcheck,
    Stage2Sumcheck,
    Stage3Sumcheck,
    Stage4Sumcheck,
    Stage5Sumcheck,
    PrecompileProof,
    FpLookupProof,
    ReducedOpeningProof,
    /// ZK sumcheck proof for Stage 2 (when ZK mode is enabled)
    ZKStage2Sumcheck,
    /// ZK sumcheck proof for Stage 3 (when ZK mode is enabled)
    ZKStage3Sumcheck,
    /// ZK sumcheck proof for Stage 4 (when ZK mode is enabled)
    ZKStage4Sumcheck,
    /// ZK sumcheck proof for Stage 5 (when ZK mode is enabled)
    ZKStage5Sumcheck,
    /// Batch opening proof for Stage 2 ZK sumcheck
    ZKStage2BatchOpening,
    /// Batch opening proof for Stage 3 ZK sumcheck
    ZKStage3BatchOpening,
    /// Batch opening proof for Stage 4 ZK sumcheck
    ZKStage4BatchOpening,
    /// Batch opening proof for Stage 5 ZK sumcheck
    ZKStage5BatchOpening,
}

/// Wrapper enum for different proof types stored in the proof map.
#[derive(Clone, Debug)]
pub enum ProofData<F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript> {
    /// Standard sumcheck proof with plaintext round polynomials
    SumcheckProof(SumcheckInstanceProof<F, ProofTranscript>),
    /// Zero-knowledge sumcheck proof with committed round polynomials
    ZKSumcheckProof(crate::jolt::sumcheck::ZKSumcheckProof<F, PCS>),
    /// Reduced opening proof for batch polynomial openings
    ReducedOpeningProof(ReducedOpeningProof<F, PCS, ProofTranscript>),
    /// Proof for precompile operations
    PrecompileProof(PrecompileSNARK<F, ProofTranscript>),
    /// Proof for floating-point lookup activations
    FpLookupProof(FpLookupProof<F, ProofTranscript>),
    /// Batch opening proof for ZK sumcheck (full ZK mode)
    ZKBatchOpeningProof(crate::jolt::sumcheck::ZKSumcheckBatchOpeningProof<F, PCS>),
}

pub type Proofs<F, PCS, ProofTranscript> = BTreeMap<ProofKeys, ProofData<F, PCS, ProofTranscript>>;

/// Prover-specific state containing preprocessing data, execution trace, and opening accumulator.
pub struct ProverState<'a, F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub trace: Vec<JoltONNXCycle>,
    pub accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
    /// Final memory values after trace execution.
    pub val_final: Vec<i64>,
}

/// Verifier-specific state containing preprocessing data and opening accumulator.
pub struct VerifierState<'a, F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub trace_length: usize,
    pub accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
}

/// Central manager for prover/verifier state during protocol execution.
///
/// Holds shared state (transcript, proofs, commitments, program I/O) and
/// role-specific state (either `prover_state` or `verifier_state`).
pub struct StateManager<
    'a,
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
> {
    /// Fiat-Shamir transcript for non-interactive proofs.
    pub transcript: Rc<RefCell<ProofTranscript>>,
    /// Collected proofs from each protocol stage.
    pub proofs: Rc<RefCell<Proofs<F, PCS, ProofTranscript>>>,
    /// Polynomial commitments.
    pub commitments: Rc<RefCell<Vec<PCS::Commitment>>>,
    /// Memory size parameter (number of memory addresses).
    pub memory_K: usize,
    /// Index where sumcheck switches from binding low-to-high to high-to-low.
    pub twist_sumcheck_switch_index: usize,
    /// Public inputs and outputs of the program.
    pub program_io: ProgramIO,
    /// Zero-knowledge configuration.
    pub zk_config: ZKConfig,
    pub prover_state: Option<ProverState<'a, F, PCS>>,
    pub verifier_state: Option<VerifierState<'a, F, PCS>>,
}

impl<'a, F, ProofTranscript, PCS> StateManager<'a, F, ProofTranscript, PCS>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    /// Creates a new state manager configured for proving.
    ///
    /// Initializes the opening accumulator, transcript, and computes `val_final`
    /// (the final memory state) from the execution trace.
    pub fn new_prover(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        trace: Vec<JoltONNXCycle>,
        program_io: ProgramIO,
    ) -> Self {
        let opening_accumulator = ProverOpeningAccumulator::new();
        let opening_accumulator = Rc::new(RefCell::new(opening_accumulator));
        let transcript = Rc::new(RefCell::new(ProofTranscript::new(b"Jolt")));
        let proofs = Rc::new(RefCell::new(BTreeMap::new()));
        let commitments = Rc::new(RefCell::new(vec![]));

        //  Calculate K for DoryGlobals initialization
        let memory_K = preprocessing.shared.bytecode.memory_K;

        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = T / num_chunks;
        let twist_sumcheck_switch_index = chunk_size.log_2();
        let mut val_final = vec![0i64; memory_K];
        trace
            .iter()
            .zip(preprocessing.bytecode())
            .for_each(|(cycle, instr)| {
                // TODO(AntoineF4C5): total memory is built anyway, may use earlier create a memory, and allow ts1 .. td values to be read from there
                val_final[instr.td as usize] = cycle.td_write().1 as u32 as i32 as i64
            });
        Self {
            transcript,
            proofs,
            commitments,
            program_io,
            memory_K,
            twist_sumcheck_switch_index,
            zk_config: ZKConfig::disabled(),
            prover_state: Some(ProverState {
                preprocessing,
                trace,
                accumulator: opening_accumulator,
                val_final,
            }),
            verifier_state: None,
        }
    }

    /// Creates a new state manager configured for proving with ZK mode.
    ///
    /// Initializes the opening accumulator, transcript, and computes `val_final`
    /// (the final memory state) from the execution trace, with ZK mode enabled.
    pub fn new_prover_zk(
        preprocessing: &'a JoltProverPreprocessing<F, PCS>,
        trace: Vec<JoltONNXCycle>,
        program_io: ProgramIO,
    ) -> Self {
        let mut state_manager = Self::new_prover(preprocessing, trace, program_io);
        state_manager.zk_config = ZKConfig::enabled();
        state_manager
    }

    /// Creates a new state manager configured for verification.
    ///
    /// Only used in tests; in practice, the verifier state manager is
    /// constructed using `JoltProof::to_verifier_state_manager`.
    #[cfg(test)]
    pub fn new_verifier(
        preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
        program_io: ProgramIO,
        trace_length: usize,
        memory_K: usize,
        twist_sumcheck_switch_index: usize,
    ) -> Self {
        let opening_accumulator = VerifierOpeningAccumulator::new();
        let opening_accumulator = Rc::new(RefCell::new(opening_accumulator));
        let transcript = Rc::new(RefCell::new(ProofTranscript::new(b"Jolt")));
        let proofs = Rc::new(RefCell::new(BTreeMap::new()));
        let commitments = Rc::new(RefCell::new(vec![]));
        StateManager {
            transcript,
            proofs,
            commitments,
            program_io,
            memory_K,
            twist_sumcheck_switch_index,
            zk_config: ZKConfig::disabled(),
            prover_state: None,
            verifier_state: Some(VerifierState {
                preprocessing,
                trace_length,
                accumulator: opening_accumulator,
            }),
        }
    }

    /// Returns true if ZK mode is enabled.
    pub fn is_zk_enabled(&self) -> bool {
        self.zk_config.enabled
    }

    /// Returns prover preprocessing, trace, and program I/O.
    ///
    /// # Panics
    /// Panics if called on a verifier state manager.
    pub fn get_prover_data(
        &self,
    ) -> (
        &'a JoltProverPreprocessing<F, PCS>,
        &Vec<JoltONNXCycle>,
        &ProgramIO,
    ) {
        if let Some(ref prover_state) = self.prover_state {
            (
                prover_state.preprocessing,
                &prover_state.trace,
                &self.program_io,
            )
        } else {
            panic!("Prover state not initialized");
        }
    }

    /// Returns verifier preprocessing, program I/O, and trace length.
    ///
    /// # Panics
    /// Panics if called on a prover state manager.
    pub fn get_verifier_data(&self) -> (&'a JoltVerifierPreprocessing<F, PCS>, &ProgramIO, usize) {
        if let Some(ref verifier_state) = self.verifier_state {
            (
                verifier_state.preprocessing,
                &self.program_io,
                verifier_state.trace_length,
            )
        } else {
            panic!("Verifier state not initialized");
        }
    }

    /// Returns the bytecode from whichever state (prover or verifier) is initialized.
    pub fn get_bytecode(&self) -> &[JoltONNXBytecode] {
        if let Some(ref verifier_state) = self.verifier_state {
            &verifier_state.preprocessing.shared.bytecode.bytecode
        } else if let Some(ref prover_state) = self.prover_state {
            &prover_state.preprocessing.shared.bytecode.bytecode
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    /// Returns the prover's opening accumulator.
    pub fn get_prover_accumulator(&self) -> Rc<RefCell<ProverOpeningAccumulator<F>>> {
        if let Some(ref prover_state) = self.prover_state {
            prover_state.accumulator.clone()
        } else {
            panic!("Prover state not initialized");
        }
    }

    /// Returns the shared Fiat-Shamir transcript.
    pub fn get_transcript(&self) -> Rc<RefCell<ProofTranscript>> {
        self.transcript.clone()
    }

    /// Returns the verifier's opening accumulator.
    pub fn get_verifier_accumulator(&self) -> Rc<RefCell<VerifierOpeningAccumulator<F>>> {
        if let Some(ref verifier_state) = self.verifier_state {
            verifier_state.accumulator.clone()
        } else {
            panic!("Verifier state not initialized");
        }
    }

    /// Returns the polynomial commitments.
    pub fn get_commitments(&self) -> Rc<RefCell<Vec<PCS::Commitment>>> {
        self.commitments.clone()
    }

    /// Sets the polynomial commitments (used by verifier).
    pub fn set_commitments(&self, commitments: Vec<PCS::Commitment>) {
        *self.commitments.borrow_mut() = commitments;
    }

    /// Returns the final memory values after trace execution (prover only).
    pub fn get_val_final(&self) -> &[i64] {
        if let Some(ref prover_state) = self.prover_state {
            &prover_state.val_final
        } else {
            panic!("Prover state not initialized");
        }
    }

    /// Returns the maximum number of variables for multilinear polynomials (prover only).
    /// This determines the padding size for batch opening proofs.
    pub fn get_max_num_vars(&self) -> usize {
        if let Some(ref prover_state) = self.prover_state {
            prover_state.preprocessing.max_num_vars
        } else {
            panic!("Prover state not initialized");
        }
    }

    /// Gets the opening for a given virtual polynomial from whichever accumulator is available.
    pub fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        if let Some(ref prover_state) = self.prover_state {
            prover_state
                .accumulator
                .borrow()
                .get_virtual_polynomial_opening(polynomial, sumcheck)
        } else if let Some(ref verifier_state) = self.verifier_state {
            verifier_state
                .accumulator
                .borrow()
                .get_virtual_polynomial_opening(polynomial, sumcheck)
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    /// Gets the opening for a given committed polynomial from whichever accumulator is available.
    pub fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        if let Some(ref prover_state) = self.prover_state {
            prover_state
                .accumulator
                .borrow()
                .get_committed_polynomial_opening(polynomial, sumcheck)
        } else if let Some(ref verifier_state) = self.verifier_state {
            verifier_state
                .accumulator
                .borrow()
                .get_committed_polynomial_opening(polynomial, sumcheck)
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    /// Appends public data to the transcript for Fiat-Shamir transformation.
    ///
    /// Commits program I/O, memory size, and trace length to ensure
    /// verifier and prover derive identical challenges.
    pub fn fiat_shamir_preamble(&mut self) {
        let transcript = self.get_transcript();
        transcript.borrow_mut().append_bytes(&{
            let io: Vec<u64> = self.program_io.output.inner.iter().map(normalize).collect();
            io.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()
        });
        transcript.borrow_mut().append_bytes(&{
            let io: Vec<u64> = self.program_io.input.inner.iter().map(normalize).collect();
            io.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()
        });
        transcript.borrow_mut().append_u64(self.memory_K as u64);

        if let Some(ref verifier_state) = self.verifier_state {
            transcript
                .borrow_mut()
                .append_u64(verifier_state.trace_length as u64);
        } else if let Some(ref prover_state) = self.prover_state {
            transcript
                .borrow_mut()
                .append_u64(prover_state.trace.len() as u64);
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    /// Returns the memory size parameter K.
    pub fn get_memory_K(&self) -> usize {
        self.memory_K
    }

    /// Returns the precompile preprocessing data.
    pub fn get_precompile_preprocessing(&self) -> &PrecompilePreprocessing {
        if let Some(ref verifier_state) = self.verifier_state {
            &verifier_state.preprocessing.shared.precompiles
        } else if let Some(ref prover_state) = self.prover_state {
            &prover_state.preprocessing.shared.precompiles
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    /// Returns the bytecode preprocessing data.
    pub fn get_bytecode_pp(&self) -> &BytecodePreprocessing {
        if let Some(ref verifier_state) = self.verifier_state {
            &verifier_state.preprocessing.shared.bytecode
        } else if let Some(ref prover_state) = self.prover_state {
            &prover_state.preprocessing.shared.bytecode
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    /// Returns the fp lookup preprocessing data.
    pub fn get_fp_lookup_pp(&self) -> &crate::jolt::fp_lookups::FpLookupPreprocessing {
        if let Some(ref verifier_state) = self.verifier_state {
            &verifier_state.preprocessing.shared.fp_lookups
        } else if let Some(ref prover_state) = self.prover_state {
            &prover_state.preprocessing.shared.fp_lookups
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }
}
