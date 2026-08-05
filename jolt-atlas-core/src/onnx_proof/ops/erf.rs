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
use atlas_onnx_tracer::{node::ComputationNode, ops::Erf, tensor::ops::nonlinearities};
use common::{
    consts::{ACTIVATION_TABLE_VARS, MODEL_SCALE},
    CommittedPoly,
};
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// Marker implementing [`SmallActivationTable`] for Erf's small dense lookup table.
pub(crate) struct ErfTableMarker;

impl SmallActivationTable for ErfTableMarker {
    fn materialize() -> Vec<i32> {
        crate::onnx_proof::neural_teleport::utils::materialize_signed_activation_table(
            ACTIVATION_TABLE_VARS,
            1,
            MODEL_SCALE as i32,
            nonlinearities::erffunc,
        )
    }
}

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Erf {
    #[tracing::instrument(skip_all, name = "Erf::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        prove_clamped_activation::<F, T, ErfTableMarker>(node, prover)
    }

    #[tracing::instrument(skip_all, name = "Erf::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_clamped_activation::<F, T, ErfTableMarker>(node, verifier)
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

    fn erf_model(input_shape: &[usize]) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(input_shape.to_vec());
        let res = b.erf(i);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_erf() {
        let t = 1 << 14;
        const MIN_INPUT: i32 = -(1 << 16);
        const MAX_INPUT: i32 = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::random_range(&mut rng, &[t], MIN_INPUT..MAX_INPUT);
        let model = erf_model(&[t]);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "TODO: non-power-of-two erf path not fully validated yet"]
    fn test_erf_non_power_of_two_input_len() {
        let t = 1000;
        const MIN_INPUT_VALUE: i32 = -(1 << ACTIVATION_BOUND);
        const MAX_INPUT_VALUE: i32 = 1 << ACTIVATION_BOUND;
        let mut rng = StdRng::seed_from_u64(0x88A);
        let input = Tensor::random_range(&mut rng, &[t], MIN_INPUT_VALUE..MAX_INPUT_VALUE);
        let model = erf_model(&[t]);
        unit_test_op(model, &[input]);
    }
}
