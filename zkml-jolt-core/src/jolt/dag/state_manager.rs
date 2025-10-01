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
use onnx_tracer::{ProgramIO, trace_types::normalize};

use crate::jolt::{
    JoltProverPreprocessing, JoltVerifierPreprocessing,
    bytecode::JoltONNXBytecode,
    pcs::{ProverOpeningAccumulator, ReducedOpeningProof, SumcheckId, VerifierOpeningAccumulator},
    precompiles::PrecompileSNARK,
    trace::JoltONNXCycle,
    witness::{CommittedPolynomial, VirtualPolynomial},
};

#[derive(PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, FromPrimitive)]
#[repr(u8)]
pub enum ProofKeys {
    Stage1Sumcheck,
    Stage2Sumcheck,
    Stage3Sumcheck,
    Stage4Sumcheck,
    PrecompileProof,
    ReducedOpeningProof,
}

#[derive(Clone, Debug)]
pub enum ProofData<F: JoltField, PCS: CommitmentScheme<Field = F>, ProofTranscript: Transcript> {
    SumcheckProof(SumcheckInstanceProof<F, ProofTranscript>),
    ReducedOpeningProof(ReducedOpeningProof<F, PCS, ProofTranscript>),
    PrecompileProof(PrecompileSNARK<F, ProofTranscript>),
}

pub type Proofs<F, PCS, ProofTranscript> = BTreeMap<ProofKeys, ProofData<F, PCS, ProofTranscript>>;

pub struct ProverState<'a, F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: &'a JoltProverPreprocessing<F, PCS>,
    pub trace: Vec<JoltONNXCycle>,
    pub accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
    pub val_final: Vec<i64>,
}

pub struct VerifierState<'a, F: JoltField, PCS>
where
    PCS: CommitmentScheme<Field = F>,
{
    pub preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    pub trace_length: usize,
    pub accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
}

pub struct StateManager<
    'a,
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
> {
    pub transcript: Rc<RefCell<ProofTranscript>>,
    pub proofs: Rc<RefCell<Proofs<F, PCS, ProofTranscript>>>,
    pub commitments: Rc<RefCell<Vec<PCS::Commitment>>>,
    pub memory_K: usize,
    pub twist_sumcheck_switch_index: usize,
    pub program_io: ProgramIO,
    pub prover_state: Option<ProverState<'a, F, PCS>>,
    pub verifier_state: Option<VerifierState<'a, F, PCS>>,
}

impl<'a, F, ProofTranscript, PCS> StateManager<'a, F, ProofTranscript, PCS>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
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
                val_final[instr.td as usize] = cycle.td_write().1 as u32 as i32 as i64
            });
        Self {
            transcript,
            proofs,
            commitments,
            program_io,
            memory_K,
            twist_sumcheck_switch_index,
            prover_state: Some(ProverState {
                preprocessing,
                trace,
                accumulator: opening_accumulator,
                val_final,
            }),
            verifier_state: None,
        }
    }

    /// Only used in tests; in practice, the verifier state manager is
    /// constructed using `JoltProof::to_verifier_state_manager`
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
            prover_state: None,
            verifier_state: Some(VerifierState {
                preprocessing,
                trace_length,
                accumulator: opening_accumulator,
            }),
        }
    }

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

    pub fn get_bytecode(&self) -> &[JoltONNXBytecode] {
        if let Some(ref verifier_state) = self.verifier_state {
            &verifier_state.preprocessing.shared.bytecode.bytecode
        } else if let Some(ref prover_state) = self.prover_state {
            &prover_state.preprocessing.shared.bytecode.bytecode
        } else {
            panic!("Neither prover nor verifier state initialized");
        }
    }

    pub fn get_prover_accumulator(&self) -> Rc<RefCell<ProverOpeningAccumulator<F>>> {
        if let Some(ref prover_state) = self.prover_state {
            prover_state.accumulator.clone()
        } else {
            panic!("Prover state not initialized");
        }
    }

    pub fn get_transcript(&self) -> Rc<RefCell<ProofTranscript>> {
        self.transcript.clone()
    }

    pub fn get_verifier_accumulator(&self) -> Rc<RefCell<VerifierOpeningAccumulator<F>>> {
        if let Some(ref verifier_state) = self.verifier_state {
            verifier_state.accumulator.clone()
        } else {
            panic!("Verifier state not initialized");
        }
    }

    pub fn get_commitments(&self) -> Rc<RefCell<Vec<PCS::Commitment>>> {
        self.commitments.clone()
    }

    pub fn set_commitments(&self, commitments: Vec<PCS::Commitment>) {
        *self.commitments.borrow_mut() = commitments;
    }

    pub fn get_val_final(&self) -> &[i64] {
        if let Some(ref prover_state) = self.prover_state {
            &prover_state.val_final
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

    pub fn get_memory_K(&self) -> usize {
        self.memory_K
    }
}
