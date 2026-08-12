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
use atlas_onnx_tracer::{node::ComputationNode, ops::Tanh, tensor::ops::nonlinearities};
use common::{
    consts::{ACTIVATION_TABLE_VARS, MODEL_SCALE},
    CommittedPoly,
};
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// Marker implementing [`SmallActivationTable`] for Tanh's small dense lookup table.
pub(crate) struct TanhTableMarker;

impl SmallActivationTable for TanhTableMarker {
    fn materialize() -> Vec<i32> {
        crate::onnx_proof::neural_teleport::utils::materialize_signed_activation_table(
            ACTIVATION_TABLE_VARS,
            1,
            MODEL_SCALE as i32,
            nonlinearities::tanh,
        )
    }
}

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Tanh {
    #[tracing::instrument(skip_all, name = "Tanh::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        prove_clamped_activation::<F, T, TanhTableMarker>(node, prover)
    }

    #[tracing::instrument(skip_all, name = "Tanh::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_clamped_activation::<F, T, TanhTableMarker>(node, verifier)
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
    use common::consts::ACTIVATION_BOUND;
    use rand::{rngs::StdRng, SeedableRng};

    fn tanh_model(input_shape: &[usize]) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(input_shape.to_vec());
        let res = b.tanh(i);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_tanh() {
        let t = 1 << 14;
        const MIN_INPUT: i32 = -(1 << 16);
        const MAX_INPUT: i32 = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::random_range(&mut rng, &[t], MIN_INPUT..MAX_INPUT);
        let model = tanh_model(&[t]);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "non-power-of-two path not fully supported yet"]
    fn test_tanh_non_power_of_two_input_len() {
        let t = 1000;
        const MIN_INPUT_VALUE: i32 = -(1 << ACTIVATION_BOUND);
        const MAX_INPUT_VALUE: i32 = 1 << ACTIVATION_BOUND;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::random_range(&mut rng, &[t], MIN_INPUT_VALUE..MAX_INPUT_VALUE);
        let model = tanh_model(&[t]);
        unit_test_op(model, &[input]);
    }
}
