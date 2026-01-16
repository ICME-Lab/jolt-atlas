use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier};
use atlas_onnx_tracer::{node::ComputationNode, ops::ReLU};
use joltworks::{
    self, field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for ReLU {
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        use crate::onnx_proof::{
            lookup_tables::relu::ReluTable,
            op_lookups::{
                self,
                ra_virtual::{InstructionRaSumcheckParams, InstructionRaSumcheckProver},
                read_raf_checking::{ReadRafSumcheckParams, ReadRafSumcheckProver},
            },
        };
        use common::consts::XLEN;
        use joltworks::{
            config::OneHotParams,
            subprotocols::{
                sumcheck::{BatchedSumcheck, Sumcheck},
                sumcheck_prover::SumcheckInstanceProver,
            },
            utils::math::Math,
        };

        let mut results = Vec::new();

        // Execution proof
        let params = ReadRafSumcheckParams::<F, ReluTable<XLEN>>::new(
            node.clone(),
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut execution_sumcheck = ReadRafSumcheckProver::initialize(
            params,
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        let (execution_proof, _) = Sumcheck::prove(
            &mut execution_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), execution_proof));

        // RaOneHotChecks proof
        let log_T = node.num_output_elements().log_2();
        let one_hot_params = OneHotParams::new(log_T);
        let ra_params = InstructionRaSumcheckParams::new(
            node.clone(),
            &OneHotParams::new(log_T),
            &prover.accumulator,
        );
        let ra_prover_sumcheck = InstructionRaSumcheckProver::initialize(ra_params, &prover.trace);

        let lookups_hamming_weight_params = op_lookups::ra_hamming_weight_params(
            node,
            &one_hot_params,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let lookups_booleanity_params = op_lookups::ra_booleanity_params(
            node,
            &one_hot_params,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let (lookups_ra_booleanity, lookups_ra_hamming_weight) = op_lookups::gen_ra_one_hot_provers(
            lookups_hamming_weight_params,
            lookups_booleanity_params,
            &prover.trace,
            node,
            &one_hot_params,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(ra_prover_sumcheck),
            Box::new(lookups_ra_booleanity),
            Box::new(lookups_ra_hamming_weight),
        ];
        let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((
            ProofId(node.idx, ProofType::RaOneHotChecks),
            ra_one_hot_proof,
        ));

        results
    }

    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        use crate::onnx_proof::{
            lookup_tables::relu::ReluTable,
            op_lookups::{
                self, ra_virtual::RaSumcheckVerifier, read_raf_checking::ReadRafSumcheckVerifier,
            },
        };
        use common::consts::XLEN;
        use joltworks::{
            config::OneHotParams,
            subprotocols::sumcheck::{BatchedSumcheck, Sumcheck},
            utils::math::Math,
        };

        // Verify execution proof
        let verifier_sumcheck = ReadRafSumcheckVerifier::<F, ReluTable<XLEN>>::new(
            node.clone(),
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
        let execution_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        Sumcheck::verify(
            execution_proof,
            &verifier_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Verify RaOneHotChecks
        let log_T = node.num_output_elements().log_2();
        let one_hot_params = OneHotParams::new(log_T);
        let ra_verifier_sumcheck =
            RaSumcheckVerifier::new(node.clone(), &one_hot_params, &verifier.accumulator);
        let (lookups_ra_booleanity, lookups_rs_hamming_weight) =
            op_lookups::new_ra_one_hot_verifiers(
                node,
                &one_hot_params,
                &verifier.accumulator,
                &mut verifier.transcript,
            );
        let ra_one_hot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        BatchedSumcheck::verify(
            ra_one_hot_proof,
            vec![
                &ra_verifier_sumcheck,
                &lookups_ra_booleanity,
                &lookups_rs_hamming_weight,
            ],
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }
}
