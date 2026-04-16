use crate::utils::{
    dims::{coord_to_linear, linear_to_coord},
    opening_access::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, Trace},
        ComputationGraph,
    },
    node::ComputationNode,
    ops::{Operator, Slice},
};
use common::parallel::par_enabled;
use joltworks::{
    field::{IntoOpening, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        sumcheck::Sumcheck,
        sumcheck::SumcheckInstanceProof,
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};
use rayon::prelude::*;

use crate::onnx_proof::{ops::OperatorProofTrait, ProofId, ProofType, Prover, Verifier};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Slice {
    #[tracing::instrument(skip_all, name = "Slice::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let params = SliceSumcheckParams::new(
            node.clone(),
            &prover.accumulator,
            &prover.preprocessing.model.graph,
        );
        let mut prover_sumcheck = SliceSumcheckProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        vec![(ProofId(node.idx, ProofType::Execution), proof)]
    }

    #[tracing::instrument(skip_all, name = "Slice::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let verifier_sumcheck = SliceSumcheckVerifier::new(
            node.clone(),
            &verifier.accumulator,
            &verifier.preprocessing.model.graph,
        );
        Sumcheck::verify(
            proof,
            &verifier_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        Ok(())
    }
}

/// Parameters for the slice selector sumcheck.
#[derive(Clone)]
pub struct SliceSumcheckParams<F: JoltField> {
    /// Slice node being proven.
    pub computation_node: ComputationNode,
    /// Reduced opening point for the slice output.
    pub r_output: OpeningPoint<BIG_ENDIAN, F>,
    /// Raw input shape.
    pub input_raw_dims: Vec<usize>,
    /// Raw output shape.
    pub output_raw_dims: Vec<usize>,
    /// Slice axis.
    pub axis: usize,
    /// Slice start index on the axis.
    pub start: usize,
    /// Slice end index on the axis.
    pub end: usize,
}

impl<F: JoltField> SliceSumcheckParams<F> {
    /// Build slice sumcheck parameters from the graph metadata and reduced output opening.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
        graph: &ComputationGraph,
    ) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let Operator::Slice(slice_op) = &computation_node.operator else {
            panic!("Expected Slice operator")
        };
        let axis = slice_op.axis;
        let start = slice_op.start;
        let end = slice_op.end;
        let r_output = accessor.get_reduced_opening().0;
        let input_raw_dims = graph
            .nodes
            .get(&computation_node.inputs[0])
            .expect("Slice node should have one input")
            .output_dims
            .clone();
        let output_raw_dims = computation_node.output_dims.clone();
        validate_slice_shapes(&input_raw_dims, &output_raw_dims, axis, start, end);
        Self {
            computation_node,
            r_output,
            input_raw_dims,
            output_raw_dims,
            axis,
            start,
            end,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for SliceSumcheckParams<F> {
    fn degree(&self) -> usize {
        2
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        accessor.get_reduced_opening().1
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.input_raw_dims
            .iter()
            .map(|dim| dim.next_power_of_two())
            .product::<usize>()
            .log_2()
    }
}

/// Prover state for the slice selector sumcheck.
pub struct SliceSumcheckProver<F: JoltField> {
    /// Static slice parameters shared across all rounds.
    pub params: SliceSumcheckParams<F>,
    /// Input polynomial over the padded input domain.
    pub input_mle: MultilinearPolynomial<F>,
    /// Selector polynomial mapping input coordinates to the output point.
    pub selector_mle: MultilinearPolynomial<F>,
}

impl<F: JoltField> SliceSumcheckProver<F> {
    /// Initialize the slice sumcheck prover from trace data and prepared parameters.
    pub fn initialize(trace: &Trace, params: SliceSumcheckParams<F>) -> Self {
        let LayerData { operands, .. } = Trace::layer_data(trace, &params.computation_node);
        let [input] = operands[..] else {
            panic!("Slice expects exactly one operand")
        };

        let input_mle = MultilinearPolynomial::from(input.padded_next_power_of_two());
        let selector = build_slice_selector(
            &params.input_raw_dims,
            &params.output_raw_dims,
            params.axis,
            params.start,
            &params.r_output.r,
        );
        let selector_mle = MultilinearPolynomial::from(selector);
        assert_eq!(input_mle.len(), selector_mle.len());

        Self {
            params,
            input_mle,
            selector_mle,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for SliceSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE_BOUND: usize = 2;
        let half_poly_len = self.input_mle.len() / 2;
        let uni_poly_evals: [F; 2] = (0..half_poly_len)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let input_evals =
                    self.input_mle
                        .sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                let selector_evals =
                    self.selector_mle
                        .sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                [
                    input_evals[0] * selector_evals[0],
                    input_evals[1] * selector_evals[1],
                ]
            })
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
        UniPoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.input_mle.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.selector_mle
            .bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .to_provider(transcript, opening_point);
        provider.append_node_io(Target::Input(0), self.input_mle.final_claim());
    }
}

/// Verifier state for the slice selector sumcheck.
pub struct SliceSumcheckVerifier<F: JoltField> {
    /// Static slice parameters shared across all rounds.
    pub params: SliceSumcheckParams<F>,
}

impl<F: JoltField> SliceSumcheckVerifier<F> {
    /// Build the slice verifier from the reduced output opening and graph metadata.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
        graph: &ComputationGraph,
    ) -> Self {
        let params = SliceSumcheckParams::new(computation_node, accumulator, graph);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for SliceSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);

        let input_claim = accessor.get_node_io(Target::Input(0)).1;
        let selector = build_slice_selector(
            &self.params.input_raw_dims,
            &self.params.output_raw_dims,
            self.params.axis,
            self.params.start,
            &self.params.r_output.r,
        );
        let selector_claim = MultilinearPolynomial::from(selector).evaluate(
            &self
                .params
                .normalize_opening_point(&sumcheck_challenges.into_opening())
                .r,
        );
        input_claim * selector_claim
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .to_provider(transcript, opening_point);
        provider.append_node_io(Target::Input(0));
    }
}

fn build_slice_selector<F: JoltField>(
    input_raw_dims: &[usize],
    output_raw_dims: &[usize],
    axis: usize,
    start: usize,
    r_output: &[F],
) -> Vec<F> {
    let input_raw_len: usize = input_raw_dims.iter().product();
    let input_padded_dims: Vec<usize> = input_raw_dims
        .iter()
        .map(|dim| dim.next_power_of_two())
        .collect();
    let output_padded_dims: Vec<usize> = output_raw_dims
        .iter()
        .map(|dim| dim.next_power_of_two())
        .collect();
    let input_domain_len: usize = input_padded_dims.iter().product();
    let output_domain_len: usize = output_padded_dims.iter().product();
    let output_num_vars = output_domain_len.log_2();
    assert_eq!(
        output_num_vars,
        r_output.len(),
        "Slice selector expects output challenge length to match output padded domain"
    );

    let mut selector = vec![F::zero(); input_domain_len];
    let eq_evals = EqPolynomial::evals(r_output);
    for output_linear_idx in 0..output_raw_dims.iter().product() {
        let output_coord = linear_to_coord(output_linear_idx, output_raw_dims);
        let mut input_coord = output_coord.clone();
        input_coord[axis] += start;

        let input_index = coord_to_linear(&input_coord, &input_padded_dims);
        let output_index = coord_to_linear(&output_coord, &output_padded_dims);
        selector[input_index] = eq_evals[output_index];
    }

    assert!(
        input_raw_len >= output_raw_dims.iter().product(),
        "Slice output raw length exceeds input raw length"
    );

    selector
}

fn validate_slice_shapes(
    input_dims: &[usize],
    output_dims: &[usize],
    axis: usize,
    start: usize,
    end: usize,
) {
    assert_eq!(input_dims.len(), output_dims.len(), "Slice rank mismatch");
    assert!(axis < input_dims.len(), "Slice axis out of range");
    assert!(start <= end, "Slice start must be <= end");
    assert!(end <= input_dims[axis], "Slice end out of bounds");

    for dim in 0..input_dims.len() {
        let expected = if dim == axis {
            end - start
        } else {
            input_dims[dim]
        };
        assert_eq!(
            output_dims[dim], expected,
            "Slice output shape mismatch at axis {dim}"
        );
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

    fn slice_model(input_shape: Vec<usize>, axis: usize, start: usize, end: usize) -> Model {
        let mut b = ModelBuilder::new();
        let input = b.input(input_shape);
        let out = b.slice(input, axis, start, end);
        b.mark_output(out);
        b.build()
    }

    #[test]
    fn test_slice_arbitrary_shapes() {
        let cases = vec![
            (vec![2, 5], 1, 1, 4),
            (vec![3, 4], 0, 1, 3),
            (vec![2, 3, 5], 2, 0, 4),
            (vec![2, 4, 3], 1, 1, 4),
            (vec![4, 6], 1, 2, 5),
            (vec![5, 3], 0, 0, 2),
        ];

        for (idx, (shape, axis, start, end)) in cases.into_iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(0x511CE + idx as u64);
            let input = Tensor::<i32>::random_small(&mut rng, &shape);
            let model = slice_model(shape, axis, start, end);
            unit_test_op(model, &[input]);
        }
    }
}
