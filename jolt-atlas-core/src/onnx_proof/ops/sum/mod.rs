use atlas_onnx_tracer::{node::ComputationNode, ops::Sum};
use joltworks::{
    self,
    field::JoltField,
    subprotocols::sumcheck::{Sumcheck, SumcheckInstanceProof},
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

use crate::{
    onnx_proof::{
        ops::{
            sum::axis::{SumAxisParams, SumAxisProver, SumAxisVerifier},
            OperatorProofTrait,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils,
};

pub mod axis;

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Sum {
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let sum_config = utils::dims::sum_config(node, &prover.preprocessing.model);
        let params = SumAxisParams::new(node.clone(), sum_config, &prover.accumulator);
        let mut prover_sumcheck =
            SumAxisProver::initialize(&prover.trace, params, &prover.accumulator);
        let (proof, _) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        vec![(ProofId(node.idx, ProofType::Execution), proof)]
    }

    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let sum_config = utils::dims::sum_config(node, &verifier.preprocessing.model);
        let verifier_sumcheck =
            SumAxisVerifier::new(node.clone(), sum_config, &verifier.accumulator);
        Sumcheck::verify(
            proof,
            &verifier_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        Ok(())
    }
}
