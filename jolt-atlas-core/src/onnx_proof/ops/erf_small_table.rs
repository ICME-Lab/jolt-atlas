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
use atlas_onnx_tracer::{ops::ErfSmallTable, tensor::ops::nonlinearities};
use common::{consts::SCALE_12, CommittedPoly};
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// Marker implementing [`SmallActivationTable`] for the clamped Erf variant's small
/// dense lookup table.
pub(crate) struct ErfTableMarker;

impl SmallActivationTable for ErfTableMarker {
    fn materialize() -> Vec<i32> {
        crate::onnx_proof::neural_teleport::utils::materialize_signed_activation_table(
            common::consts::ACTIVATION_TABLE_BOUND,
            1,
            SCALE_12 as i32,
            nonlinearities::erffunc,
        )
    }
}

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for ErfSmallTable {
    #[tracing::instrument(skip_all, name = "ErfSmallTable::prove")]
    fn prove(
        &self,
        node: &atlas_onnx_tracer::node::ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        prove_clamped_activation::<F, T, ErfTableMarker>(node, prover)
    }

    #[tracing::instrument(skip_all, name = "ErfSmallTable::verify")]
    fn verify(
        &self,
        node: &atlas_onnx_tracer::node::ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_clamped_activation::<F, T, ErfTableMarker>(node, verifier)
    }

    fn get_committed_polynomials(
        &self,
        node: &atlas_onnx_tracer::node::ComputationNode,
    ) -> Vec<CommittedPoly> {
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
    use rand::{rngs::StdRng, SeedableRng};

    fn erf_small_table_model(t: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![t]);
        let res = b.erf_small_table(i);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_erf_small_table() {
        let t = 1 << 14;
        const MIN_INPUT: i32 = -(1 << 16);
        const MAX_INPUT: i32 = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x88D);
        let input = Tensor::random_range(&mut rng, &[t], MIN_INPUT..MAX_INPUT);
        let model = erf_small_table_model(t);
        unit_test_op(model, &[input]);
    }
}
