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

/// Axis-wise sum implementations for sumcheck protocol.
pub mod axis;

/// Create a Sum prover instance for the ZK pipeline.
pub fn create_sum_prover<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    model: &atlas_onnx_tracer::model::Model,
    trace: &atlas_onnx_tracer::model::trace::Trace,
    accumulator: &joltworks::poly::opening_proof::ProverOpeningAccumulator<F>,
) -> Box<dyn joltworks::subprotocols::sumcheck_prover::SumcheckInstanceProver<F, T>> {
    let sum_config = utils::dims::sum_config(node, model);
    let params = SumAxisParams::new(node.clone(), sum_config, accumulator);
    Box::new(SumAxisProver::initialize(trace, params, accumulator))
}

/// Create a Sum verifier instance for the ZK pipeline.
pub fn create_sum_verifier<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    model: &atlas_onnx_tracer::model::Model,
    accumulator: &joltworks::poly::opening_proof::VerifierOpeningAccumulator<F>,
) -> Box<dyn joltworks::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier<F, T>> {
    let sum_config = utils::dims::sum_config(node, model);
    Box::new(SumAxisVerifier::new(node.clone(), sum_config, accumulator))
}

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Sum {
    #[tracing::instrument(skip_all, name = "Sum::prove")]
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

    #[tracing::instrument(skip_all, name = "Sum::verify")]
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
