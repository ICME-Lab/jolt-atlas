use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Sum,
};
use common::VirtualPolynomial;
use joltworks::{
    self,
    field::JoltField,
    poly::opening_proof::SumcheckId,
    subprotocols::sumcheck::{Sumcheck, SumcheckInstanceProof},
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

use crate::{
    onnx_proof::{
        ops::{
            sum::axis::{sum_axis_prover, sum_axis_verifier, SumAxisProvider},
            OperatorProofTrait,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::{
        self,
        dims::{SumAxis, SumConfig},
    },
};

/// Axis-wise sum implementations for sumcheck protocol.
pub mod axis;

// ---------------------------------------------------------------------------
// NodeSumProvider — adapts a standalone Sum node for the generic sum-axis
// sumcheck via SumAxisProvider.
// ---------------------------------------------------------------------------

/// Provider for a standalone `Sum` computation node, implementing
/// [`SumAxisProvider`] so the generic sum-axis prover/verifier can be reused.
pub struct NodeSumProvider {
    node_idx: usize,
    input_idx: usize,
    sum_config: SumConfig,
}

impl NodeSumProvider {
    /// Create a provider from a standalone sum computation node.
    pub fn new(node: &ComputationNode, sum_config: SumConfig) -> Self {
        Self {
            node_idx: node.idx,
            input_idx: node.inputs[0],
            sum_config,
        }
    }
}

impl SumAxisProvider for NodeSumProvider {
    fn axis(&self) -> SumAxis {
        self.sum_config.axis()
    }

    fn operand_dims(&self) -> [usize; 2] {
        let d = self.sum_config.operand_dims();
        [d[0], d[1]]
    }

    fn sum_output_source(&self) -> (VirtualPolynomial, SumcheckId) {
        // The sum node's own output; resolve_vp_opening routes NodeOutput
        // through get_node_output_opening automatically.
        (
            VirtualPolynomial::NodeOutput(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn operand_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::NodeOutput(self.input_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }
}

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Sum {
    #[tracing::instrument(skip_all, name = "Sum::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let sum_config = utils::dims::sum_config(node, &prover.preprocessing.model);
        let provider = NodeSumProvider::new(node, sum_config);

        // Extract the operand from the trace.
        let LayerData {
            operands,
            output: _,
        } = Trace::layer_data(&prover.trace, node);
        let [operand] = operands[..] else {
            panic!("Expected one operand for Sum operation")
        };

        let mut prover_sumcheck = sum_axis_prover(&provider, operand, &prover.accumulator);
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
        let provider = NodeSumProvider::new(node, sum_config);
        let verifier_sumcheck = sum_axis_verifier(&provider, &verifier.accumulator);
        Sumcheck::verify(
            proof,
            &verifier_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        Ok(())
    }
}
