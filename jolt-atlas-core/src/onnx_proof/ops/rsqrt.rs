use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    onnx_proof::deferred_lookups::{DeferredBatch, DeferredOneHot},
    onnx_proof::{
        ops::{eval_reduction::NodeEvalReduction, OperatorProofTrait, ReductionFlow},
        range_checking::{
            range_check_operands::{
                RangeCheckOperands, RiRangeCheckOperands, RsRangeCheckOperands,
            },
            RangeCheckProvider,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_trace::{op::Rsqrt, trace::LayerData, Op, Trace};
use common::{consts::XLEN, CommittedPoly, VirtualPoly};
use jolt_poly::{GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use joltworks::{
    field::{IntoOpening, JoltField},
    lookup_tables::unsigned_less_than::UnsignedLessThanTable,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::SplitEqUnreducedFold,
    },
    subprotocols::{
        evaluation_reduction::EvalReductionProof,
        shout::RaOneHotEncoding,
        sumcheck::{Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
        upstream_sumcheck::round_check_failed,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

/// Dividend `S³` of the fused rsqrt division relation, where `S = 2^scale` is the
/// input's fixed-point scale.
///
/// The fused kernel computes `output = ⌊√⌊S³ / x̂⌋⌋`, so `S³` is the dividend that
/// both the division and square-root relations are built around.
pub fn rsqrt_dividend(node: &ComputationNode) -> i64 {
    assert!(
        matches!(node.operator, Op::Rsqrt(_)),
        "rsqrt_dividend called on a non-Rsqrt node"
    );
    1i64 << (3 * node.scale)
}

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Rsqrt {
    fn reduction_flow(&self) -> ReductionFlow {
        ReductionFlow::Custom
    }

    #[tracing::instrument(skip_all, name = "Rsqrt::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        let mut results = Vec::new();
        // Execution proof
        let params = RsqrtParams::new(node.clone(), &mut prover.transcript);
        let mut prover_sumcheck = RsqrtProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), proof));

        results
    }

    fn prove_with_reduction(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> (
        EvalReductionProof<F>,
        Vec<(ProofId, SumcheckInstanceProof<F>)>,
    ) {
        let mut proofs = self.prove(node, prover);
        let eval_reduction_proof = NodeEvalReduction::prove(prover, node);
        proofs.extend(prove_range_and_onehot(node, prover));
        (eval_reduction_proof, proofs)
    }

    #[tracing::instrument(skip_all, name = "Rsqrt::prove")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Execution proof
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let verifier_sumcheck = RsqrtVerifier::new(node.clone(), &mut verifier.transcript);
        Sumcheck::verify(
            proof,
            &verifier_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }

    fn verify_with_reduction(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
        eval_reduction_proof: &EvalReductionProof<F>,
    ) -> Result<(), ProofVerifyError> {
        self.verify(node, verifier)?;
        NodeEvalReduction::verify(verifier, node, eval_reduction_proof)?;
        verify_range_and_onehot(node, verifier)
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let mut polys = vec![CommittedPoly::RsqrtQuotient(node.idx)];

        let rc_operands = RangeCheckOperands::<RiRangeCheckOperands>::new(node);
        let encoding = rc_operands.get_encoding(node);
        let d = encoding.one_hot_params().instruction_d;
        for i in 0..d {
            polys.push(CommittedPoly::SqrtDivRangeCheckRaD(node.idx, i));
            polys.push(CommittedPoly::SqrtRangeCheckRaD(node.idx, i));
        }
        polys
    }
}

// FUSED RECIPROCAL-SQUARE-ROOT PROOF
//
// For a quantized input x̂ with scale S = 2^scale, the reciprocal square root at the
// same scale is  S / √(x̂ / S) = √(S³ / x̂).  The tracer fuses the division and the
// square root into a single kernel:
//
//     output = ⌊√⌊S³ / x̂⌋⌋
//
// We witness the intermediate `quotient = ⌊S³ / x̂⌋` and two remainders, and prove
// two relations:
//
//   division:  x̂ · quotient + div_remainder = S³      with 0 ≤ div_remainder < x̂
//   sqrt:      output² + sqrt_remainder = quotient     with 0 ≤ sqrt_remainder ≤ 2·output
//
// The two relations are folded together with a challenge `gamma` into one sumcheck:
//
//     0 = (x̂·quotient + div_remainder − S³) + gamma·(output² + sqrt_remainder − quotient)
//
// The bounds `div_remainder < x̂` and `sqrt_remainder ≤ 2·output` are enforced by the
// range checks in `prove_range_and_onehot`.

const DEGREE_BOUND: usize = 3;

/// Parameters for proving reciprocal square root (1/√x) operations.
///
/// Rsqrt requires proving multiple relations including the reciprocal and square root properties.
/// Stores the opening point and computation node information needed for the sumcheck protocol.
#[derive(Clone)]
pub struct RsqrtParams<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    /// Folding challenge for batching the inverse and sqrt relations.
    gamma: F,
}

impl<F: JoltField> RsqrtParams<F> {
    /// Create new rsqrt parameters from a computation node and transcript.
    /// Squeezes both the opening point and the folding challenge gamma.
    pub fn new<T: Transcript>(computation_node: ComputationNode, transcript: &mut T) -> Self {
        let num_vars = computation_node.pow2_padded_num_output_elements().log_2();
        let r_node_output = transcript.challenge_vector_optimized::<F>(num_vars);
        let gamma = transcript.challenge_scalar();
        Self {
            r_node_output: r_node_output.into(),
            computation_node,
            gamma,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for RsqrtParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.computation_node
            .pow2_padded_num_output_elements()
            .log_2()
    }

    // output = eq * (input*quotient + div_rem - S³ + gamma*(output² + sqrt_rem - quotient))
}

/// Prover state for the fused reciprocal-square-root sumcheck.
///
/// Maintains the equality polynomial, the input operand, the division `quotient`,
/// the `output` (rsqrt result), and the two remainders (`div_remainder`,
/// `sqrt_remainder`) needed to prove the two folded relations.
pub struct RsqrtProver<F: JoltField> {
    params: RsqrtParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    input: MultilinearPolynomial<F>,
    quotient: MultilinearPolynomial<F>,
    output: MultilinearPolynomial<F>,
    div_remainder: MultilinearPolynomial<F>,
    sqrt_remainder: MultilinearPolynomial<F>,
    /// `S³` dividend as a field constant, cached for `compute_message`.
    s_cubed: F,
    // folding challenge
    gamma: F,
}

impl<F: JoltField> RsqrtProver<F> {
    /// Initialize the prover with trace data, transcript, and parameters.
    pub fn initialize(trace: &Trace, params: RsqrtParams<F>) -> Self {
        let eq_r_node_output = GruenSplitEqPolynomial::new(
            &params.r_node_output.r,
            jolt_poly::BindingOrder::LowToHigh,
        );
        let LayerData { operands, output } = TraceView::layer_data(trace, &params.computation_node);
        let [input] = operands[..] else {
            panic!("Expected one operand for Rsqrt operation")
        };
        let s_cubed = rsqrt_dividend(&params.computation_node);
        // `quotient = ⌊S³ / x̂⌋` and `out²` can exceed i32 at higher scales, so both
        // are computed in i64; `quotient` is committed as u64. `div_remainder < x̂`
        // and `sqrt_remainder < 2·out+1` are small and stay i32.
        let quotient_data: Vec<u64> = input.iter().map(|&x| (s_cubed / x as i64) as u64).collect();
        let div_remainder_data: Vec<i32> =
            input.iter().map(|&x| (s_cubed % x as i64) as i32).collect();
        let sqrt_remainder_data: Vec<i32> = quotient_data
            .iter()
            .zip(output.iter())
            .map(|(&quotient, &out)| (quotient as i64 - (out as i64) * (out as i64)) as i32)
            .collect();

        let input = MultilinearPolynomial::from(input.clone());
        let quotient = MultilinearPolynomial::from(quotient_data);
        let div_remainder = MultilinearPolynomial::from(div_remainder_data);
        let output = MultilinearPolynomial::from(output.clone());
        let sqrt_remainder = MultilinearPolynomial::from(sqrt_remainder_data);
        let s_cubed = F::from_i64(s_cubed);
        #[cfg(test)]
        {
            let div_claim = (0..input.len())
                .map(|i| {
                    let x: F = input.get_bound_coeff(i);
                    let quotient = quotient.get_bound_coeff(i);
                    let div_remainder: F = div_remainder.get_bound_coeff(i);
                    // range checking: div_remainder < x
                    crate::onnx_proof::testing::probe::probe_assert!(div_remainder
                        .to_u64()
                        .zip(x.to_u64())
                        .is_some_and(|(r, x)| r < x));

                    x * quotient + div_remainder - s_cubed
                })
                .sum();
            crate::onnx_proof::testing::probe::probe_assert_eq!(F::zero(), div_claim);

            let sqrt_claim = (0..input.len())
                .map(|i| {
                    let quotient = quotient.get_bound_coeff(i);
                    let out: F = output.get_bound_coeff(i);
                    let sqrt_remainder: F = sqrt_remainder.get_bound_coeff(i);
                    // range checking: sqrt_remainder <= 2*output
                    crate::onnx_proof::testing::probe::probe_assert!(sqrt_remainder
                        .to_u64()
                        .zip(out.to_u64())
                        .is_some_and(|(r, o)| r <= 2 * o));

                    out * out + sqrt_remainder - quotient
                })
                .sum();
            crate::onnx_proof::testing::probe::probe_assert_eq!(F::zero(), sqrt_claim)
        }

        let gamma = params.gamma;
        Self {
            params,
            eq_r_node_output,
            input,
            quotient,
            output,
            div_remainder,
            sqrt_remainder,
            s_cubed,
            gamma,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for RsqrtProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
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
            .into_provider(transcript, opening_point);

        provider.append_nodeio(Target::Input(0), self.input.final_claim());
        provider.append_advice(CommittedPoly::RsqrtQuotient, self.quotient.final_claim());
        provider.append_nodeio(Target::Current, self.output.final_claim());
        provider.append_advice(VirtualPoly::DivRemainder, self.div_remainder.final_claim());
        provider.append_advice(
            VirtualPoly::SqrtRemainder,
            self.sqrt_remainder.final_claim(),
        );
    }
}

impl<F: JoltField> RsqrtProver<F> {
    fn bind(&mut self, r: F) {
        let r_j = F::challenge_from_field(r);
        self.eq_r_node_output.bind(r);
        self.input.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.quotient.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.output.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.div_remainder
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.sqrt_remainder
            .bind_parallel(r_j, BindingOrder::LowToHigh);
    }
}

impl<F: JoltField> ProveRounds<F> for RsqrtProver<F> {
    fn num_rounds(&self) -> usize {
        self.params.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = bind {
            self.bind(r);
        }
        let Self {
            eq_r_node_output,
            input,
            quotient,
            output,
            div_remainder,
            sqrt_remainder,
            s_cubed,
            gamma,
            ..
        } = self;
        eq_r_node_output
            .gruen_round_deg_3(
                &|g| {
                    let x0 = input.get_bound_coeff(2 * g);
                    let x1 = input.get_bound_coeff(2 * g + 1);
                    let quotient0 = quotient.get_bound_coeff(2 * g);
                    let quotient1 = quotient.get_bound_coeff(2 * g + 1);
                    let div_remainder0 = div_remainder.get_bound_coeff(2 * g);

                    let output0 = output.get_bound_coeff(2 * g);
                    let output1 = output.get_bound_coeff(2 * g + 1);
                    let sqrt_remainder0 = sqrt_remainder.get_bound_coeff(2 * g);

                    // division relation: x·quotient + div_remainder − S³
                    let div0 = x0 * quotient0 + div_remainder0 - *s_cubed;
                    // sqrt relation: output² + sqrt_remainder − quotient
                    let sqrt0 = output0 * output0 + sqrt_remainder0 - quotient0;

                    let div_quad = (x1 - x0) * (quotient1 - quotient0);
                    let sqrt_quad = (output1 - output0) * (output1 - output0);
                    [div0 + *gamma * sqrt0, div_quad + *gamma * sqrt_quad]
                },
                &|g| {
                    let quotient1 = quotient.get_bound_coeff(2 * g + 1);
                    let output1 = output.get_bound_coeff(2 * g + 1);
                    let div1 = input.get_bound_coeff(2 * g + 1) * quotient1
                        + div_remainder.get_bound_coeff(2 * g + 1)
                        - *s_cubed;
                    let sqrt1 =
                        output1 * output1 + sqrt_remainder.get_bound_coeff(2 * g + 1) - quotient1;
                    div1 + *gamma * sqrt1
                },
                previous_claim,
            )
            .map_err(round_check_failed(round, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

/// Verifier for reciprocal square root sumcheck protocol.
///
/// Verifies that the prover's sumcheck messages are consistent with the claimed
/// rsqrt operation output and all required rsqrt relations.
pub struct RsqrtVerifier<F: JoltField> {
    params: RsqrtParams<F>,
    gamma: F,
}

impl<F: JoltField> RsqrtVerifier<F> {
    /// Create a new verifier for the rsqrt operation with folding challenge from transcript.
    pub fn new<T: Transcript>(computation_node: ComputationNode, transcript: &mut T) -> Self {
        let params = RsqrtParams::new(computation_node, transcript);
        let gamma = params.gamma;
        Self { params, gamma }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for RsqrtVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);

        let r_node_output = &self.params.r_node_output.r;
        let r_node_output_prime = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let eq_eval = EqPolynomial::mle(r_node_output, &r_node_output_prime);

        let quotient_claim = accessor.get_advice(CommittedPoly::RsqrtQuotient).1;
        let div_remainder_claim = accessor.get_advice(VirtualPoly::DivRemainder).1;
        let output_claim = accessor.get_nodeio(Target::Current).1;
        let sqrt_remainder_claim = accessor.get_advice(VirtualPoly::SqrtRemainder).1;
        let input_claim = accessor.get_nodeio(Target::Input(0)).1;
        let s_cubed = F::from_i64(rsqrt_dividend(&self.params.computation_node));

        eq_eval
            * (input_claim * quotient_claim + div_remainder_claim - s_cubed
                + self.gamma
                    * (output_claim * output_claim + sqrt_remainder_claim - quotient_claim))
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
            .into_provider(transcript, opening_point);

        provider.append_nodeio(Target::Input(0));
        provider.append_advice(CommittedPoly::RsqrtQuotient);
        provider.append_nodeio(Target::Current);
        provider.append_advice(VirtualPoly::DivRemainder);
        provider.append_advice(VirtualPoly::SqrtRemainder);
    }
}

fn prove_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
    // Both range checks are built now (challenges drawn, operand claims
    // appended) and proven after the node loop in the `RangeCheck` batch; their
    // one-hot checks follow that batch.
    let div_rangecheck_provider = RangeCheckProvider::<RiRangeCheckOperands>::new(node);
    let (div_rc_prover, div_lookup_indices) = div_rangecheck_provider
        .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    prover.defer(DeferredBatch::RangeCheck, Box::new(div_rc_prover));

    let sqrt_rangecheck_provider = RangeCheckProvider::<RsRangeCheckOperands>::new(node);
    let (sqrt_rc_prover, sqrt_lookup_indices) = sqrt_rangecheck_provider
        .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    prover.defer(DeferredBatch::RangeCheck, Box::new(sqrt_rc_prover));

    let div_operands = RangeCheckOperands::<RiRangeCheckOperands>::new(node);
    prover.deferred_onehots.push((
        DeferredOneHot::RangeCheck(div_operands.get_encoding(node)),
        div_lookup_indices,
    ));
    let sqrt_operands = RangeCheckOperands::<RsRangeCheckOperands>::new(node);
    prover.deferred_onehots.push((
        DeferredOneHot::RangeCheck(sqrt_operands.get_encoding(node)),
        sqrt_lookup_indices,
    ));

    Vec::new()
}

fn verify_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    let div_rangecheck_provider = RangeCheckProvider::<RiRangeCheckOperands>::new(node);
    let div_rc_verifier = div_rangecheck_provider
        .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
    verifier.defer(DeferredBatch::RangeCheck, Box::new(div_rc_verifier));

    let sqrt_rangecheck_provider = RangeCheckProvider::<RsRangeCheckOperands>::new(node);
    let sqrt_rc_verifier = sqrt_rangecheck_provider
        .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
    verifier.defer(DeferredBatch::RangeCheck, Box::new(sqrt_rc_verifier));

    let div_operands = RangeCheckOperands::<RiRangeCheckOperands>::new(node);
    verifier
        .deferred_onehots
        .push(DeferredOneHot::RangeCheck(div_operands.get_encoding(node)));
    let sqrt_operands = RangeCheckOperands::<RsRangeCheckOperands>::new(node);
    verifier
        .deferred_onehots
        .push(DeferredOneHot::RangeCheck(sqrt_operands.get_encoding(node)));

    Ok(())
}
