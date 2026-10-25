use crate::node_view::ComputationNode;
use crate::node_view::ModelView;
use crate::onnx_proof::trace_view::TraceView;
use crate::utils::{
    dims::{coord_to_linear, linear_to_coord},
    opening_access::{AccOpeningAccessor, Target},
};
use atlas_trace::{op::Concat, trace::LayerData, Model, Op, Trace};
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

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Concat {
    #[tracing::instrument(skip_all, name = "Concat::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        let params = ConcatSumcheckParams::new(
            node.clone(),
            &prover.accumulator,
            &prover.preprocessing.model,
        );
        let mut prover_sumcheck = ConcatSumcheckProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        vec![(ProofId(node.idx, ProofType::Execution), proof)]
    }

    #[tracing::instrument(skip_all, name = "Concat::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let verifier_sumcheck = ConcatSumcheckVerifier::new(
            node.clone(),
            &verifier.accumulator,
            &verifier.preprocessing.model,
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

/// Parameters for the concat selector sumcheck.
#[derive(Clone)]
pub struct ConcatSumcheckParams<F: JoltField> {
    /// Concat node being proven.
    pub computation_node: ComputationNode,
    /// Reduced opening point for the concat output.
    pub r_output: OpeningPoint<BIG_ENDIAN, F>,
    /// Raw input shapes for each concat operand.
    pub input_raw_dims: Vec<Vec<usize>>,
    /// Raw output shape for the concat result.
    pub output_raw_dims: Vec<usize>,
    /// Normalized concat axis.
    pub axis: usize,
    /// Maximum padded input domain size, in variables.
    pub max_input_num_vars: usize,
}

impl<F: JoltField> ConcatSumcheckParams<F> {
    /// Build concat sumcheck parameters from the model metadata and reduced output opening.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
        model: &Model,
    ) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let Op::Concat(concat_op) = &computation_node.operator else {
            panic!("Expected Concat operator")
        };
        let r_output = accessor.get_reduced_opening().0;
        let input_raw_dims = model
            .operand_views(&computation_node)
            .iter()
            .map(|input_node| input_node.output_dims.clone())
            .collect::<Vec<_>>();
        let output_raw_dims = computation_node.output_dims.clone();
        let axis = concat_op.axis; // already normalized to a non-negative index
        validate_concat_shapes(
            &input_raw_dims
                .iter()
                .map(|dims| dims.as_slice())
                .collect::<Vec<_>>(),
            &output_raw_dims,
            axis,
        );
        let max_input_num_vars = input_raw_dims
            .iter()
            .map(|dims| padded_domain_len(dims).log_2())
            .max()
            .unwrap_or(0);

        Self {
            computation_node,
            r_output,
            input_raw_dims,
            output_raw_dims,
            axis,
            max_input_num_vars,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ConcatSumcheckParams<F> {
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
        self.max_input_num_vars
    }

    // output = sum_i(selector_claim_i * input_claim_i)
}

struct ConcatInputTerm<F: JoltField> {
    input_num_vars: usize,
    input_mle: MultilinearPolynomial<F>,
    selector_mle: MultilinearPolynomial<F>,
}

/// Prover state for the concat selector sumcheck.
pub struct ConcatSumcheckProver<F: JoltField> {
    /// Static concat parameters shared across all rounds.
    pub params: ConcatSumcheckParams<F>,
    input_terms: Vec<ConcatInputTerm<F>>,
}

impl<F: JoltField> ConcatSumcheckProver<F> {
    /// Initialize prover state from trace tensors and prepared concat parameters.
    pub fn initialize(trace: &Trace, params: ConcatSumcheckParams<F>) -> Self {
        let LayerData { operands, output } = TraceView::layer_data(trace, &params.computation_node);
        assert!(!operands.is_empty(), "Concat expects at least one operand");
        assert_eq!(
            operands.len(),
            params.input_raw_dims.len(),
            "Concat input shape count mismatch"
        );

        let output_padded_len = output.padded_pow2().len();
        let max_input_domain_len = 1usize << params.max_input_num_vars;
        assert!(
            max_input_domain_len <= output_padded_len,
            "Concat max input domain cannot exceed output padded domain"
        );

        let input_terms = operands
            .iter()
            .enumerate()
            .map(|(input_idx, input_tensor)| {
                let input_padded = input_tensor.padded_pow2();
                let input_num_vars = input_padded.len().log_2();
                let extended_input = extend_input_to_max_domain(
                    input_padded.data(),
                    input_num_vars,
                    params.max_input_num_vars,
                );
                let selector = build_concat_selector(
                    &params.input_raw_dims,
                    &params.output_raw_dims,
                    params.axis,
                    input_idx,
                    &params.r_output.r,
                    params.max_input_num_vars,
                );
                ConcatInputTerm {
                    input_num_vars,
                    input_mle: MultilinearPolynomial::from(extended_input),
                    selector_mle: MultilinearPolynomial::from(selector),
                }
            })
            .collect();

        Self {
            params,
            input_terms,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ConcatSumcheckProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE_BOUND: usize = 2;
        let half_poly_len = self
            .input_terms
            .first()
            .map(|term| term.input_mle.len() / 2)
            .unwrap_or(0);
        let uni_poly_evals: [F; 2] = (0..half_poly_len)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let mut evals = [F::zero(); 2];
                for term in &self.input_terms {
                    let input_evals =
                        term.input_mle
                            .sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                    let selector_evals =
                        term.selector_mle
                            .sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                    evals[0] += input_evals[0] * selector_evals[0];
                    evals[1] += input_evals[1] * selector_evals[1];
                }
                evals
            })
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);

        UniPoly::from_evals_and_hint(previous_claim, &uni_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        for term in &mut self.input_terms {
            term.input_mle.bind_parallel(r_j, BindingOrder::LowToHigh);
            term.selector_mle
                .bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let full_opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, Default::default()); // We will update the opening point for each input term
        for (i, term) in self.input_terms.iter().enumerate() {
            let live_opening_point = full_opening_point.r[..term.input_num_vars].to_vec().into();
            provider.append_nodeio_at(
                Target::Input(i),
                live_opening_point,
                term.input_mle.final_claim(),
            );
        }
    }
}

/// Verifier state for the concat selector sumcheck.
pub struct ConcatSumcheckVerifier<F: JoltField> {
    /// Static concat parameters shared across all rounds.
    pub params: ConcatSumcheckParams<F>,
}

impl<F: JoltField> ConcatSumcheckVerifier<F> {
    /// Build the concat verifier from the reduced output opening and model metadata.
    pub fn new(
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
        model: &Model,
    ) -> Self {
        let params = ConcatSumcheckParams::new(computation_node, accumulator, model);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ConcatSumcheckVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);
        let full_opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());

        self.params.input_raw_dims.iter().enumerate().fold(
            F::zero(),
            |running, (input_idx, _input_raw_dims)| {
                let input_claim = accessor.get_nodeio(Target::Input(input_idx)).1;

                let selector = build_concat_selector(
                    &self.params.input_raw_dims,
                    &self.params.output_raw_dims,
                    self.params.axis,
                    input_idx,
                    &self.params.r_output.r,
                    self.params.max_input_num_vars,
                );
                let selector_claim =
                    MultilinearPolynomial::from(selector).evaluate(&full_opening_point.r);
                running + input_claim * selector_claim
            },
        )
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let full_opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .into_provider(transcript, Default::default()); // We will update the opening point for each input term
        for (input_idx, input_raw_dims) in self.params.input_raw_dims.iter().enumerate() {
            let input_num_vars = padded_domain_len(input_raw_dims).log_2();
            let live_opening_point = full_opening_point.r[..input_num_vars].to_vec().into();
            provider.append_nodeio_at(Target::Input(input_idx), live_opening_point);
        }
    }
}

fn padded_domain_len(dims: &[usize]) -> usize {
    dims.iter().map(|dim| dim.next_power_of_two()).product()
}

fn extend_input_to_max_domain<T: Copy>(
    input_padded: &[T],
    input_num_vars: usize,
    max_num_vars: usize,
) -> Vec<T> {
    let pad_repeat = 1usize << (max_num_vars - input_num_vars);
    let mut extended = Vec::with_capacity(input_padded.len() * pad_repeat);
    for value in input_padded {
        for _ in 0..pad_repeat {
            extended.push(*value);
        }
    }
    extended
}

fn build_concat_selector<F: JoltField>(
    raw_input_dims: &[Vec<usize>],
    output_raw_dims: &[usize],
    axis: usize,
    input_idx: usize,
    r_output: &[F],
    max_input_num_vars: usize,
) -> Vec<F> {
    let input_raw_dims = raw_input_dims
        .get(input_idx)
        .unwrap_or_else(|| panic!("Concat input_idx {input_idx} out of range"));
    let input_raw_len: usize = input_raw_dims.iter().product();
    let input_padded_dims: Vec<usize> = input_raw_dims
        .iter()
        .map(|dim| dim.next_power_of_two())
        .collect();
    let input_domain_len: usize = input_padded_dims.iter().product();
    let input_num_vars = input_domain_len.log_2();

    let output_padded_dims: Vec<usize> = output_raw_dims
        .iter()
        .map(|dim| dim.next_power_of_two())
        .collect();
    let output_domain_len: usize = output_padded_dims.iter().product();
    let output_num_vars = output_domain_len.log_2();
    assert_eq!(
        output_num_vars,
        r_output.len(),
        "Concat selector expects output challenge length to match output padded domain"
    );

    let max_domain_len = 1usize << max_input_num_vars;
    let axis_offset = axis_offset(raw_input_dims, input_idx, axis);
    let mut selector = vec![F::zero(); max_domain_len];
    let eq_evals = EqPolynomial::evals(r_output);

    for linear_idx in 0..input_raw_len {
        let input_coord = linear_to_coord(linear_idx, input_raw_dims);
        let mut output_coord = input_coord.clone();
        output_coord[axis] += axis_offset;

        let input_index = coord_to_linear(&input_coord, &input_padded_dims);
        let output_index = coord_to_linear(&output_coord, &output_padded_dims);
        let full_index = input_index << (max_input_num_vars - input_num_vars);
        selector[full_index] = eq_evals[output_index];
    }

    selector
}

/// Validates concat shape rules.
fn validate_concat_shapes(inputs_dims: &[&[usize]], out_dims: &[usize], axis: usize) {
    assert!(!inputs_dims.is_empty(), "Concat expects at least one input");
    let rank = out_dims.len();
    for dims in inputs_dims {
        assert_eq!(dims.len(), rank, "Concat input rank mismatch");
    }

    for dim in 0..out_dims.len() {
        if dim == axis {
            let sum: usize = inputs_dims.iter().map(|dims| dims[dim]).sum();
            assert_eq!(
                out_dims[dim], sum,
                "Concat output axis dimension must equal sum of input dimensions"
            );
        } else {
            let out_dim = out_dims[dim];
            for dims in inputs_dims {
                assert_eq!(
                    dims[dim], out_dim,
                    "Concat non-axis dimensions must match between inputs"
                );
            }
        }
    }
}

/// Returns the axis offset (in output coordinates) for the given input.
fn axis_offset(raw_input_dims: &[Vec<usize>], input_idx: usize, axis: usize) -> usize {
    raw_input_dims
        .iter()
        .take(input_idx)
        .map(|dims| dims[axis])
        .sum()
}
