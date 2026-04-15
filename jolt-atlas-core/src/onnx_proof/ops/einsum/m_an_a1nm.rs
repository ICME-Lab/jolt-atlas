use atlas_onnx_tracer::{model::trace::Trace, node::ComputationNode};
use joltworks::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::math::Math,
};

use crate::utils::{
    dims::EinsumDims,
    opening_id_builder::{AccOpeningAccessor, Target},
};

/// Parameters for the shared `m,an->a1nm` canonical einsum family.
#[derive(Clone)]
pub struct MAnA1nmParams<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    einsum_dims: EinsumDims,
}

impl<F: JoltField> MAnA1nmParams<F> {
    /// Creates params for the shared `m,an->a1nm` canonical family.
    pub fn new(
        computation_node: ComputationNode,
        einsum_dims: EinsumDims,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let atlas_onnx_tracer::ops::Operator::Einsum(einsum) = &computation_node.operator else {
            panic!("Expected Einsum operator")
        };
        assert_eq!(
            einsum.equation, "m,an->abnm",
            "MAnA1nmParams only supports the m,an->abnm source equation"
        );
        assert!(
            computation_node.output_dims.len() == 4,
            "m,an->a1nm expects a rank-4 output"
        );
        assert!(
            computation_node.output_dims[1] == 1,
            "m,an->abnm is interpreted as m,an->a1nm in the current tracer"
        );
        assert_eq!(
            einsum_dims.left_operand(),
            &[computation_node.output_dims[3]],
            "m,an->a1nm requires the left operand to be the trailing m axis"
        );
        assert_eq!(
            einsum_dims.right_operand(),
            &[
                computation_node.output_dims[0],
                computation_node.output_dims[2]
            ],
            "m,an->a1nm requires the right operand to align with the output a and n axes"
        );
        let r_node_output = accessor.get_reduced_opening().0;
        Self {
            r_node_output,
            computation_node,
            einsum_dims,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for MAnA1nmParams<F> {
    fn degree(&self) -> usize {
        1
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        accessor.get_reduced_opening().1
    }

    fn normalize_opening_point(
        &self,
        challenges: &[F],
    ) -> joltworks::poly::opening_proof::OpeningPoint<
        { joltworks::poly::opening_proof::BIG_ENDIAN },
        F,
    > {
        OpeningPoint::new(challenges.to_vec())
    }

    fn num_rounds(&self) -> usize {
        0
    }
}

/// Prover for the shared `m,an->a1nm` canonical einsum family.
pub struct MAnA1nmProver<F: JoltField> {
    params: MAnA1nmParams<F>,
    left_claim: F,
    right_claim: F,
}

impl<F: JoltField> MAnA1nmProver<F> {
    /// Initializes the prover.
    pub fn initialize(trace: &Trace, params: MAnA1nmParams<F>) -> Self {
        let layer = Trace::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = layer.operands[..] else {
            panic!("Expected two operands for m,an->a1nm operation")
        };

        let log_m = params.einsum_dims.left_operand()[0].log_2();
        let log_n = params.einsum_dims.right_operand()[1].log_2();
        let log_a = params.einsum_dims.right_operand()[0].log_2();

        let split_at_m = params.r_node_output.r.len().saturating_sub(log_m);
        let (r_prefix, r_m) = params.r_node_output.split_at(split_at_m);
        let left_claim = MultilinearPolynomial::from(left_operand.clone()).evaluate(&r_m.r);

        let (r_a, r_bn) = r_prefix.split_at(log_a);
        let (_, r_n) = r_bn.split_at(r_bn.r.len().saturating_sub(log_n));
        let r_right = [r_a.r.as_slice(), r_n.r.as_slice()].concat();
        let right_claim = MultilinearPolynomial::from(right_operand.clone()).evaluate(&r_right);

        debug_assert_eq!(r_m.r.len(), log_m);

        Self {
            params,
            left_claim,
            right_claim,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for MAnA1nmProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(
        &mut self,
        _round: usize,
        _previous_claim: F,
    ) -> joltworks::poly::unipoly::UniPoly<F> {
        unreachable!("m,an->a1nm has no sumcheck rounds")
    }

    fn ingest_challenge(&mut self, _r_j: F::Challenge, _round: usize) {
        unreachable!("m,an->a1nm has no sumcheck rounds")
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        let log_m = self.params.einsum_dims.left_operand()[0].log_2();
        let log_n = self.params.einsum_dims.right_operand()[1].log_2();
        let log_a = self.params.einsum_dims.right_operand()[0].log_2();

        let split_at_m = self.params.r_node_output.r.len().saturating_sub(log_m);
        let (r_prefix, r_m) = self.params.r_node_output.split_at(split_at_m);
        let left_opening_point = self.params.normalize_opening_point(&r_m.r);
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .to_provider(transcript, left_opening_point);
        provider.append_node_io(Target::Input(0), self.left_claim);

        let (r_a, r_bn) = r_prefix.split_at(log_a);
        let (_, r_n) = r_bn.split_at(r_bn.r.len().saturating_sub(log_n));
        let r_right = [r_a.r.as_slice(), r_n.r.as_slice()].concat();
        let right_opening_point = self.params.normalize_opening_point(&r_right);
        provider.update_point(right_opening_point);
        provider.append_node_io(Target::Input(1), self.right_claim);
    }
}

/// Verifier for the shared `m,an->a1nm` canonical einsum family.
pub struct MAnA1nmVerifier<F: JoltField> {
    params: MAnA1nmParams<F>,
}

impl<F: JoltField> MAnA1nmVerifier<F> {
    /// Creates the verifier.
    pub fn new(
        computation_node: ComputationNode,
        einsum_dims: EinsumDims,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = MAnA1nmParams::new(computation_node, einsum_dims, accumulator);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for MAnA1nmVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);
        let left_claim = accessor.get_node_io(Target::Input(0)).1;
        let right_claim = accessor.get_node_io(Target::Input(1)).1;
        left_claim * right_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        let log_n = self.params.einsum_dims.right_operand()[1].log_2();
        let log_a = self.params.einsum_dims.right_operand()[0].log_2();
        let log_m = self.params.einsum_dims.left_operand()[0].log_2();

        let split_at_m = self.params.r_node_output.r.len().saturating_sub(log_m);
        let (r_prefix, r_m) = self.params.r_node_output.split_at(split_at_m);
        let left_opening_point = self.params.normalize_opening_point(&r_m.r);
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .to_provider(transcript, left_opening_point);
        provider.append_node_io(Target::Input(0));

        let (r_a, r_bn) = r_prefix.split_at(log_a);
        let (_, r_n) = r_bn.split_at(r_bn.r.len().saturating_sub(log_n));
        let r_right = [r_a.r.as_slice(), r_n.r.as_slice()].concat();
        let right_opening_point = self.params.normalize_opening_point(&r_right);
        provider.update_point(right_opening_point);
        provider.append_node_io(Target::Input(1));
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

    fn m_an_a1nm_model(rng: &mut StdRng, m: usize, a: usize, n: usize) -> Model {
        let mut builder = ModelBuilder::new();
        let left = builder.input(vec![m]);
        let right = builder.constant(Tensor::random_small(rng, &[a, n]));
        let out = builder.einsum("m,an->abnm", vec![left, right], vec![a, 1, n, m]);
        builder.mark_output(out);
        builder.build()
    }

    #[test]
    fn test_m_an_a1nm() {
        let mut rng = StdRng::seed_from_u64(0);
        let input = Tensor::<i32>::random_small(&mut rng, &[4]);
        let model = m_an_a1nm_model(&mut rng, 4, 2, 8);
        unit_test_op(model, &[input]);
    }
}
