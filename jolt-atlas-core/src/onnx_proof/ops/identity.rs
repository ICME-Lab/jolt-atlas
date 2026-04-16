use crate::{
    onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier},
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{node::ComputationNode, ops::Identity};
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Identity {
    #[tracing::instrument(skip_all, name = "Identity::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let node_poly_opening =
            AccOpeningAccessor::new(&prover.accumulator, node).get_reduced_opening();
        let (opening_point, claim) = node_poly_opening;
        let mut provider = AccOpeningAccessor::new(&mut prover.accumulator, node)
            .to_provider(&mut prover.transcript, opening_point);
        provider.append_node_io(Target::Input(0), claim);
        vec![]
    }

    #[tracing::instrument(skip_all, name = "Identity::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let (opening_point, claim) =
            AccOpeningAccessor::new(&verifier.accumulator, node).get_reduced_opening();
        let mut provider = AccOpeningAccessor::new(&mut verifier.accumulator, node)
            .to_provider(&mut verifier.transcript, opening_point);
        provider.append_node_io(Target::Input(0));
        let (_, operand_claim) = provider.get_node_io(Target::Input(0));

        if operand_claim != claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Identity claim should match operand claim".to_string(),
            ));
        }
        Ok(())
    }
}
