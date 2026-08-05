use crate::onnx_proof::{
    ops::{
        activation_clamped::{
            clamped_activation_committed_polynomials, prove_clamped_activation,
            verify_clamped_activation, SmallActivationTable,
        },
        OperatorProofTrait,
    },
    ProofId, Prover, Verifier,
};
use atlas_onnx_tracer::{node::ComputationNode, ops::Sigmoid, tensor::ops::nonlinearities};
use common::{
    consts::{ACTIVATION_TABLE_BOUND, MODEL_SCALE},
    CommittedPoly,
};
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// Marker implementing [`SmallActivationTable`] for Sigmoid's small dense lookup table.
pub(crate) struct SigmoidTableMarker;

impl SmallActivationTable for SigmoidTableMarker {
    fn materialize() -> Vec<i32> {
        crate::onnx_proof::neural_teleport::utils::materialize_signed_activation_table(
            ACTIVATION_TABLE_BOUND,
            1,
            MODEL_SCALE as i32,
            nonlinearities::sigmoid,
        )
    }
}

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Sigmoid {
    #[tracing::instrument(skip_all, name = "Sigmoid::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        prove_clamped_activation::<F, T, SigmoidTableMarker>(node, prover)
    }

    #[tracing::instrument(skip_all, name = "Sigmoid::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_clamped_activation::<F, T, SigmoidTableMarker>(node, verifier)
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        clamped_activation_committed_polynomials(node)
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };
    use common::consts::ACTIVATION_TABLE_BOUND;
    use rand::{rngs::StdRng, SeedableRng};

    fn sigmoid_model(input_shape: &[usize]) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(input_shape.to_vec());
        let res = b.sigmoid(i);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_sigmoid() {
        let t = 1 << 14;
        const MIN_INPUT: i32 = -(1 << 16);
        const MAX_INPUT: i32 = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::random_range(&mut rng, &[t], MIN_INPUT..MAX_INPUT);
        let model = sigmoid_model(&[t]);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "TODO: non-power-of-two sigmoid path not fully validated yet"]
    fn test_sigmoid_non_power_of_two_input_len() {
        let t = 1000;
        const MIN_INPUT_VALUE: i32 = -(1 << (ACTIVATION_TABLE_BOUND - 1));
        const MAX_INPUT_VALUE: i32 = 1 << (ACTIVATION_TABLE_BOUND - 1);
        let mut rng = StdRng::seed_from_u64(0x88A);
        let input = Tensor::random_range(&mut rng, &[t], MIN_INPUT_VALUE..MAX_INPUT_VALUE);
        let model = sigmoid_model(&[t]);
        unit_test_op(model, &[input]);
    }
}
