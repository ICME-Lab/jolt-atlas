use jolt_core::{
    field::JoltField, poly::commitment::commitment_scheme::CommitmentScheme,
    transcripts::Transcript,
};

use crate::jolt::{
    dag::{stage::SumcheckStages, state_manager::StateManager},
    memory::{read_write_checking::MemoryReadWriteChecking, val_evaluation::ValEvaluationSumcheck},
    sumcheck::SumcheckInstance,
};

pub mod read_write_checking;
pub mod val_evaluation;

#[derive(Default)]
pub struct MemoryDag {}

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStages<F, ProofTranscript, PCS> for MemoryDag
{
    fn stage4_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let read_write_checking = MemoryReadWriteChecking::new_prover(state_manager);
        vec![Box::new(read_write_checking)]
    }

    fn stage4_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let read_write_checking = MemoryReadWriteChecking::new_verifier(state_manager);
        vec![Box::new(read_write_checking)]
    }

    fn stage5_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let val_evaluation = ValEvaluationSumcheck::new_prover(state_manager);
        vec![Box::new(val_evaluation)]
    }

    fn stage5_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let val_evaluation = ValEvaluationSumcheck::new_verifier(state_manager);
        vec![Box::new(val_evaluation)]
    }
}
