use crate::jolt::{
    JoltSNARK, ProverDebugInfo,
    bytecode::BytecodeDag,
    dag::{
        stage::SumcheckStages,
        state_manager::{ProofData, ProofKeys, StateManager},
    },
    executor::LookupsDag,
    memory::MemoryDag,
    precompiles::PrecompileSNARK,
    spartan::SpartanDag,
    sumcheck::{BatchedSumcheck, SumcheckInstance},
    witness::{AllCommittedPolynomials, CommittedPolynomial},
};
#[cfg(test)]
use crate::utils::mcc::sanity_check_mcc;
use anyhow::Context;
#[cfg(not(target_arch = "wasm32"))]
use jolt_core::utils::profiling::print_current_memory_usage;
use jolt_core::{
    field::JoltField,
    poly::commitment::{commitment_scheme::CommitmentScheme, dory::DoryGlobals},
    transcripts::Transcript,
    utils::thread::drop_in_background_thread,
    zkvm::witness::DTH_ROOT_OF_K,
};
use rayon::prelude::*;
use std::collections::HashMap;

pub enum JoltDAG {}

impl JoltDAG {
    #[allow(clippy::type_complexity)]
    pub fn prove<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        mut state_manager: StateManager<'a, F, ProofTranscript, PCS>,
    ) -> Result<
        (
            JoltSNARK<F, PCS, ProofTranscript>,
            Option<ProverDebugInfo<F, ProofTranscript, PCS>>,
        ),
        anyhow::Error,
    > {
        state_manager.fiat_shamir_preamble();

        // Initialize DoryGlobals at the beginning to keep it alive for the entire proof
        let (preprocessing, trace, _) = state_manager.get_prover_data();
        let trace_length = trace.len();
        let padded_trace_length = trace_length.next_power_of_two();
        #[cfg(test)]
        sanity_check_mcc(preprocessing.bytecode(), trace, preprocessing.memory_K());

        println!("bytecode size: {}", preprocessing.shared.bytecode.code_size);
        println!("trace length: {trace_length}");

        let _memory_K = state_manager.memory_K;
        let _guard = (
            DoryGlobals::initialize(DTH_ROOT_OF_K, padded_trace_length),
            AllCommittedPolynomials::initialize(),
        );

        // Generate and commit to all witness polynomials
        let opening_proof_hints = Self::generate_and_commit_polynomials(&mut state_manager)?;

        // Append commitments to transcript
        let commitments = state_manager.get_commitments();
        let transcript = state_manager.get_transcript();
        for commitment in commitments.borrow().iter() {
            transcript.borrow_mut().append_serializable(commitment);
        }
        drop(commitments);

        // Stage 1:
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 1 baseline");
        let span = tracing::span!(tracing::Level::INFO, "Stage 1 sumchecks");
        let _guard = span.enter();

        let (pp, trace, _) = state_manager.get_prover_data();
        let padded_trace_length = trace.len().next_power_of_two();
        let mut spartan_dag = SpartanDag::<F>::new::<ProofTranscript>(padded_trace_length);
        let mut lookups_dag = LookupsDag::default();
        let mut memory_dag = MemoryDag::default();
        let mut bytecode_dag = BytecodeDag::default();
        spartan_dag
            .stage1_prove(&mut state_manager)
            .context("Stage 1")?;

        drop(_guard);
        drop(span);

        // Stage 2:
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 2 baseline");
        let span = tracing::span!(tracing::Level::INFO, "Stage 2 sumchecks");
        let _guard = span.enter();

        let mut stage2_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage2_prover_instances(&mut state_manager))
            .chain(memory_dag.stage2_prover_instances(&mut state_manager))
            .collect();

        let stage2_instances_mut: Vec<&mut dyn SumcheckInstance<F>> = stage2_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
            .collect();

        let transcript = state_manager.get_transcript();
        let accumulator = state_manager.get_prover_accumulator();
        let (stage2_proof, _r_stage2) = BatchedSumcheck::prove(
            stage2_instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );

        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage2Sumcheck,
            ProofData::SumcheckProof(stage2_proof),
        );

        drop_in_background_thread(stage2_instances);

        drop(_guard);
        drop(span);

        // Stage 3:
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 3 baseline");
        let span = tracing::span!(tracing::Level::INFO, "Stage 3 sumchecks");
        let _guard = span.enter();

        let mut stage3_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage3_prover_instances(&mut state_manager))
            .chain(memory_dag.stage3_prover_instances(&mut state_manager))
            .chain(lookups_dag.stage3_prover_instances(&mut state_manager))
            .collect();

        let stage3_instances_mut: Vec<&mut dyn SumcheckInstance<F>> = stage3_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
            .collect();

        let (stage3_proof, _r_stage3) = BatchedSumcheck::prove(
            stage3_instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );

        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage3Sumcheck,
            ProofData::SumcheckProof(stage3_proof),
        );
        drop_in_background_thread(stage3_instances);
        drop(_guard);
        drop(span);

        // Stage 4:
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 4 baseline");
        let span = tracing::span!(tracing::Level::INFO, "Stage 4 sumchecks");
        let _guard = span.enter();

        let mut stage4_instances: Vec<_> = std::iter::empty()
            .chain(bytecode_dag.stage4_prover_instances(&mut state_manager))
            .chain(lookups_dag.stage4_prover_instances(&mut state_manager))
            .collect();

        let stage4_instances_mut: Vec<&mut dyn SumcheckInstance<F>> = stage4_instances
            .iter_mut()
            .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
            .collect();

        let (stage4_proof, _r_stage4) = BatchedSumcheck::prove(
            stage4_instances_mut,
            Some(accumulator.clone()),
            &mut *transcript.borrow_mut(),
        );

        state_manager.proofs.borrow_mut().insert(
            ProofKeys::Stage4Sumcheck,
            ProofData::SumcheckProof(stage4_proof),
        );

        drop_in_background_thread(stage4_instances);
        drop(_guard);
        drop(span);

        if pp.is_precompiles_enabled() {
            let precompile_proof = PrecompileSNARK::prove(&mut state_manager);
            state_manager.proofs.borrow_mut().insert(
                ProofKeys::PrecompileProof,
                ProofData::PrecompileProof(precompile_proof),
            );
        }

        // Batch-prove all openings
        let (_, trace, _) = state_manager.get_prover_data();

        let all_polys: Vec<CommittedPolynomial> =
            AllCommittedPolynomials::iter().copied().collect();
        let polynomials_map = CommittedPolynomial::generate_witness_batch(&all_polys, trace);

        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage("Stage 5 baseline");

        let opening_proof = accumulator.borrow_mut().reduce_and_prove(
            polynomials_map,
            opening_proof_hints,
            &preprocessing.generators,
            &mut *transcript.borrow_mut(),
        );

        state_manager.proofs.borrow_mut().insert(
            ProofKeys::ReducedOpeningProof,
            ProofData::ReducedOpeningProof(opening_proof),
        );

        // TODO(Forpee) Precompile Ra poly evals
        // #[cfg(test)]
        // assert!(
        //     state_manager
        //         .get_prover_accumulator()
        //         .borrow()
        //         .appended_virtual_openings
        //         .borrow()
        //         .is_empty(),
        //     "Not all virtual openings have been proven, missing: {:?}",
        //     state_manager
        //         .get_prover_accumulator()
        //         .borrow()
        //         .appended_virtual_openings
        //         .borrow()
        // );

        #[cfg(test)]
        println!(
            "Not all virtual openings have been proven, missing: {:#?}",
            state_manager
                .get_prover_accumulator()
                .borrow()
                .appended_virtual_openings
                .borrow()
        );

        #[cfg(test)]
        let debug_info = {
            let transcript = state_manager.transcript.take();
            let opening_accumulator = state_manager.get_prover_accumulator().borrow().clone();
            Some(ProverDebugInfo {
                transcript,
                opening_accumulator,
                prover_setup: preprocessing.generators.clone(),
            })
        };
        #[cfg(not(test))]
        let debug_info = None;

        let proof = JoltSNARK::from_prover_state_manager(state_manager);

        Ok((proof, debug_info))
    }

    pub fn verify<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        mut state_manager: StateManager<'a, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        state_manager.fiat_shamir_preamble();

        let _memory_K = state_manager.memory_K;
        let _guard = AllCommittedPolynomials::initialize();

        // Append commitments to transcript
        let commitments = state_manager.get_commitments();
        let transcript = state_manager.get_transcript();
        for commitment in commitments.borrow().iter() {
            transcript.borrow_mut().append_serializable(commitment);
        }

        // Stage 1:
        let (preprocessing, _, trace_length) = state_manager.get_verifier_data();
        let padded_trace_length = trace_length.next_power_of_two();
        let mut spartan_dag = SpartanDag::<F>::new::<ProofTranscript>(padded_trace_length);
        let mut lookups_dag = LookupsDag::default();
        let mut memory_dag = MemoryDag::default();
        let mut bytecode_dag = BytecodeDag::default();
        spartan_dag
            .stage1_verify(&mut state_manager)
            .context("Stage 1")?;

        // Stage 2:
        let stage2_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage2_verifier_instances(&mut state_manager))
            .chain(memory_dag.stage2_verifier_instances(&mut state_manager))
            .collect();
        let stage2_instances_ref: Vec<&dyn SumcheckInstance<F>> = stage2_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F>)
            .collect();

        let proofs = state_manager.proofs.borrow();
        let stage2_proof_data = proofs
            .get(&ProofKeys::Stage2Sumcheck)
            .expect("Stage 2 sumcheck proof not found");
        let stage2_proof = match stage2_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 2"),
        };

        let transcript = state_manager.get_transcript();
        let opening_accumulator = state_manager.get_verifier_accumulator();
        let _r_stage2 = BatchedSumcheck::verify(
            stage2_proof,
            stage2_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 2")?;

        drop(proofs);

        // Stage 3:
        let stage3_instances: Vec<_> = std::iter::empty()
            .chain(spartan_dag.stage3_verifier_instances(&mut state_manager))
            .chain(memory_dag.stage3_verifier_instances(&mut state_manager))
            .chain(lookups_dag.stage3_verifier_instances(&mut state_manager))
            .collect();
        let stage3_instances_ref: Vec<&dyn SumcheckInstance<F>> = stage3_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F>)
            .collect();

        let proofs = state_manager.proofs.borrow();
        let stage3_proof_data = proofs
            .get(&ProofKeys::Stage3Sumcheck)
            .expect("Stage 3 sumcheck proof not found");
        let stage3_proof = match stage3_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 3"),
        };

        let _r_stage3 = BatchedSumcheck::verify(
            stage3_proof,
            stage3_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 3")?;

        drop(proofs);

        // Stage 4:
        let stage4_instances: Vec<_> = std::iter::empty()
            .chain(bytecode_dag.stage4_verifier_instances(&mut state_manager))
            .chain(lookups_dag.stage4_verifier_instances(&mut state_manager))
            .collect();
        let stage4_instances_ref: Vec<&dyn SumcheckInstance<F>> = stage4_instances
            .iter()
            .map(|instance| &**instance as &dyn SumcheckInstance<F>)
            .collect();

        let proofs = state_manager.proofs.borrow();
        let stage4_proof_data = proofs
            .get(&ProofKeys::Stage4Sumcheck)
            .expect("Stage 4 sumcheck proof not found");
        let stage4_proof = match stage4_proof_data {
            ProofData::SumcheckProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 4"),
        };

        let _r_stage4 = BatchedSumcheck::verify(
            stage4_proof,
            stage4_instances_ref,
            Some(opening_accumulator.clone()),
            &mut *transcript.borrow_mut(),
        )
        .context("Stage 4")?;
        drop(proofs);

        if preprocessing.is_precompiles_enabled() {
            // precompile proof
            // Extract and clone precompile proof
            let precompile_proof = {
                let proofs = state_manager.proofs.borrow();
                let precompile_proof_data = proofs
                    .get(&ProofKeys::PrecompileProof)
                    .expect("Precompile proof not found");
                match precompile_proof_data {
                    ProofData::PrecompileProof(proof) => proof.clone(), // Clone to avoid borrow issues
                    _ => panic!("Invalid proof type for precompile"),
                }
            };

            // Verify with mutable reference to state_manager
            precompile_proof
                .verify(&mut state_manager)
                .context("Precompile")?;
        }

        // Batch-prove all openings - get fresh borrow after verify
        let proofs = state_manager.proofs.borrow();
        let batched_opening_proof = proofs
            .get(&ProofKeys::ReducedOpeningProof)
            .expect("Reduced opening proof not found");
        let batched_opening_proof = match batched_opening_proof {
            ProofData::ReducedOpeningProof(proof) => proof,
            _ => panic!("Invalid proof type for stage 4"),
        };

        let mut commitments_map = HashMap::new();
        for polynomial in AllCommittedPolynomials::iter() {
            commitments_map.insert(
                *polynomial,
                commitments.borrow()[polynomial.to_index()].clone(),
            );
        }
        let accumulator = state_manager.get_verifier_accumulator();
        accumulator
            .borrow_mut()
            .reduce_and_verify(
                &preprocessing.generators,
                &mut commitments_map,
                batched_opening_proof,
                &mut *transcript.borrow_mut(),
            )
            .context("Stage 5")?;

        Ok(())
    }

    // Prover utility to commit to all the polynomials for the PCS
    #[tracing::instrument(skip_all)]
    fn generate_and_commit_polynomials<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<Field = F>,
    >(
        prover_state_manager: &mut StateManager<'a, F, ProofTranscript, PCS>,
    ) -> Result<HashMap<CommittedPolynomial, PCS::OpeningProofHint>, anyhow::Error> {
        let (preprocessing, trace, _program_io) = prover_state_manager.get_prover_data();

        let all_polys: Vec<CommittedPolynomial> =
            AllCommittedPolynomials::iter().copied().collect();
        let committed_polys: Vec<_> = AllCommittedPolynomials::iter()
            .filter_map(|poly| {
                CommittedPolynomial::generate_witness_batch(&all_polys, trace).remove(poly)
            })
            .collect();

        let (commitments, hints): (Vec<PCS::Commitment>, Vec<PCS::OpeningProofHint>) =
            committed_polys
                .par_iter()
                .map(|poly| PCS::commit(poly, &preprocessing.generators))
                .unzip();
        let mut hint_map = HashMap::with_capacity(committed_polys.len());
        for (poly, hint) in AllCommittedPolynomials::iter().zip(hints) {
            hint_map.insert(*poly, hint);
        }

        prover_state_manager.set_commitments(commitments);

        drop_in_background_thread(committed_polys);

        Ok(hint_map)
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::memory::{
        read_write_checking::test::test_read_write_sumcheck,
        val_evaluation::test::test_val_evaluation_sumcheck,
    };
    use onnx_tracer::{builder, graph::model::Model, tensor::Tensor};

    // TODO(AntoineF4C5): Complete with other sumchecks
    fn test_dag_sumchecks<ModelFunc>(model_fn: ModelFunc, input: &Tensor<i32>)
    where
        ModelFunc: Fn() -> Model + Copy,
    {
        // Stage 1

        // Stage 2
        test_read_write_sumcheck(model_fn, input);

        // Stage 3
        test_val_evaluation_sumcheck(model_fn, input);

        // Stage 4

        // Stage 5
    }

    #[test]
    fn test_trace_sumchecks() {
        let input = Tensor::new(Some(&[1, 2, 3, 4]), &[1, 4]).unwrap();
        let model_fn = builder::add_model;

        test_dag_sumchecks(model_fn, &input);
    }
}
