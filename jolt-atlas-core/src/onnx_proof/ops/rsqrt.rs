use crate::{
    onnx_proof::{
        ops::{eval_reduction::NodeEvalReduction, OperatorProofTrait, ReductionFlow},
        range_checking::{
            range_check_operands::{RiRangeCheckOperands, RsRangeCheckOperands},
            RangeCheckEncoding, RangeCheckProvider,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::{Operator, Rsqrt},
};
use common::{consts::XLEN, CommittedPoly, VirtualPoly};
#[cfg(feature = "zk")]
use joltworks::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};
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
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        evaluation_reduction::EvalReductionProof,
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
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
    let Operator::Rsqrt(op) = &node.operator else {
        panic!("rsqrt_dividend called on a non-Rsqrt node");
    };
    1i64 << (3 * op.scale)
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
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
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
        Vec<(ProofId, SumcheckInstanceProof<F, T>)>,
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
        let encoding = RangeCheckEncoding::<RiRangeCheckOperands>::new(node);
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

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }

    // output = eq * (input*quotient + div_rem - S³ + gamma*(output² + sqrt_rem - quotient))
    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        use crate::utils::opening_access::OpeningIdBuilder;
        let builder = OpeningIdBuilder::new(&self.computation_node);
        let input_id = builder.nodeio(Target::Input(0));
        let quotient_id = builder.advice(CommittedPoly::RsqrtQuotient);
        let output_id = builder.nodeio(Target::Current);
        let div_rem_id = builder.advice(VirtualPoly::DivRemainder);
        let sqrt_rem_id = builder.advice(VirtualPoly::SqrtRemainder);
        Some(OutputClaimConstraint::sum_of_products(vec![
            // eq * input * quotient
            ProductTerm::scaled(
                ValueSource::Challenge(0),
                vec![
                    ValueSource::Opening(input_id),
                    ValueSource::Opening(quotient_id),
                ],
            ),
            // eq * div_remainder
            ProductTerm::scaled(
                ValueSource::Challenge(1),
                vec![ValueSource::Opening(div_rem_id)],
            ),
            // -eq * S³  (constant term, no openings)
            ProductTerm::scaled(ValueSource::Challenge(2), vec![]),
            // eq * gamma * output²
            ProductTerm::scaled(
                ValueSource::Challenge(3),
                vec![
                    ValueSource::Opening(output_id),
                    ValueSource::Opening(output_id),
                ],
            ),
            // eq * gamma * sqrt_remainder
            ProductTerm::scaled(
                ValueSource::Challenge(4),
                vec![ValueSource::Opening(sqrt_rem_id)],
            ),
            // -eq * gamma * quotient
            ProductTerm::scaled(
                ValueSource::Challenge(5),
                vec![ValueSource::Opening(quotient_id)],
            ),
        ]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let r_prime: Vec<F> = self
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let eq_eval = EqPolynomial::mle(&self.r_node_output.r, &r_prime);
        let eq_gamma = eq_eval * self.gamma;
        let s_cubed = F::from_i64(rsqrt_dividend(&self.computation_node));
        vec![
            eq_eval,            // Ch(0): eq * input * quotient
            eq_eval,            // Ch(1): eq * div_remainder
            -eq_eval * s_cubed, // Ch(2): -eq * S³ (constant)
            eq_gamma,           // Ch(3): eq * gamma * output²
            eq_gamma,           // Ch(4): eq * gamma * sqrt_remainder
            -eq_gamma,          // Ch(5): -eq * gamma * quotient
        ]
    }
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
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
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
                    assert!(div_remainder.to_u64().unwrap() < x.to_u64().unwrap());

                    x * quotient + div_remainder - s_cubed
                })
                .sum();
            assert_eq!(F::zero(), div_claim);

            let sqrt_claim = (0..input.len())
                .map(|i| {
                    let quotient = quotient.get_bound_coeff(i);
                    let out: F = output.get_bound_coeff(i);
                    let sqrt_remainder: F = sqrt_remainder.get_bound_coeff(i);
                    // range checking: sqrt_remainder <= 2*output
                    assert!(sqrt_remainder.to_u64().unwrap() <= 2 * out.to_u64().unwrap());

                    out * out + sqrt_remainder - quotient
                })
                .sum();
            assert_eq!(F::zero(), sqrt_claim)
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

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_node_output,
            input,
            quotient,
            output,
            div_remainder,
            sqrt_remainder,
            ..
        } = self;
        let [q_constant, q_quadratic] = eq_r_node_output.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let x0 = input.get_bound_coeff(2 * g);
            let x1 = input.get_bound_coeff(2 * g + 1);
            let quotient0 = quotient.get_bound_coeff(2 * g);
            let quotient1 = quotient.get_bound_coeff(2 * g + 1);
            let div_remainder0 = div_remainder.get_bound_coeff(2 * g);

            let output0 = output.get_bound_coeff(2 * g);
            let output1 = output.get_bound_coeff(2 * g + 1);
            let sqrt_remainder0 = sqrt_remainder.get_bound_coeff(2 * g);

            // division relation: x·quotient + div_remainder − S³
            let div0 = x0 * quotient0 + div_remainder0 - self.s_cubed;
            // sqrt relation: output² + sqrt_remainder − quotient
            let sqrt0 = output0 * output0 + sqrt_remainder0 - quotient0;

            let div_quad = (x1 - x0) * (quotient1 - quotient0);
            let sqrt_quad = (output1 - output0) * (output1 - output0);
            [div0 + self.gamma * sqrt0, div_quad + self.gamma * sqrt_quad]
        });
        eq_r_node_output.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
        self.input.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.quotient.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.output.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.div_remainder
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.sqrt_remainder
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
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
    let mut proofs = Vec::new();

    let div_rangecheck_provider = RangeCheckProvider::<RiRangeCheckOperands>::new(node);
    let (div_rc_prover, div_lookup_indices) = div_rangecheck_provider
        .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );

    let sqrt_rangecheck_provider = RangeCheckProvider::<RsRangeCheckOperands>::new(node);
    let (sqrt_rc_prover, sqrt_lookup_indices) = sqrt_rangecheck_provider
        .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );

    let mut rc_instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
        vec![Box::new(div_rc_prover), Box::new(sqrt_rc_prover)];
    let (rangecheck_proof, _) = BatchedSumcheck::prove(
        rc_instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    proofs.push((ProofId(node.idx, ProofType::RangeCheck), rangecheck_proof));

    let div_encoding = RangeCheckEncoding::<RiRangeCheckOperands>::new(node);
    let [div_ra, div_hw, div_bool] = shout::ra_onehot_provers(
        &div_encoding,
        &div_lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );

    let sqrt_encoding = RangeCheckEncoding::<RsRangeCheckOperands>::new(node);
    let [sqrt_ra, sqrt_hw, sqrt_bool] = shout::ra_onehot_provers(
        &sqrt_encoding,
        &sqrt_lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );

    let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
        vec![div_ra, div_hw, div_bool, sqrt_ra, sqrt_hw, sqrt_bool];
    let (proof, _) = BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    proofs.push((ProofId(node.idx, ProofType::RaOneHotChecks), proof));

    proofs
}

fn verify_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    let rangecheck_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RangeCheck))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;

    let div_rangecheck_provider = RangeCheckProvider::<RiRangeCheckOperands>::new(node);
    let div_rc_verifier = div_rangecheck_provider
        .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );

    let sqrt_rangecheck_provider = RangeCheckProvider::<RsRangeCheckOperands>::new(node);
    let sqrt_rc_verifier = sqrt_rangecheck_provider
        .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
    let rc_instances: Vec<&dyn SumcheckInstanceVerifier<_, _>> =
        vec![&div_rc_verifier, &sqrt_rc_verifier];
    BatchedSumcheck::verify(
        rangecheck_proof,
        rc_instances,
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    let ra_one_hot_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;
    let div_encoding = RangeCheckEncoding::<RiRangeCheckOperands>::new(node);
    let [div_ra, div_hw, div_bool] = shout::ra_onehot_verifiers(
        &div_encoding,
        &verifier.accumulator,
        &mut verifier.transcript,
    );
    let sqrt_encoding = RangeCheckEncoding::<RsRangeCheckOperands>::new(node);
    let [sqrt_ra, sqrt_hw, sqrt_bool] = shout::ra_onehot_verifiers(
        &sqrt_encoding,
        &verifier.accumulator,
        &mut verifier.transcript,
    );

    let instances: Vec<&dyn SumcheckInstanceVerifier<_, _>> = vec![
        &*div_ra,
        &*div_hw,
        &*div_bool,
        &*sqrt_ra,
        &*sqrt_hw,
        &*sqrt_bool,
    ];
    BatchedSumcheck::verify(
        ra_one_hot_proof,
        instances,
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };
    use rand::{rngs::StdRng, SeedableRng};

    /// Upper bound on the random quantized inputs; positive so the division is well-defined.
    const INPUT_MAX: i32 = 1 << 16;

    fn rsqrt_model(T: usize, scale: u32) -> Model {
        let mut b = ModelBuilder::with_scale(scale);
        let input = b.input(vec![T]);
        let res = b.rsqrt(input);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_rsqrt() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_range(&mut rng, &[T], 1..INPUT_MAX);
        let model = rsqrt_model(T, 8);
        unit_test_op(model, &[input]);
    }

    /// At scale 12, `S³ = 2³⁶` and `quotient = ⌊S³/x̂⌋` exceeds i32 for small inputs
    /// (`x̂ < 32`), exercising the u64 quotient / i64 `out²` witness path.
    #[test]
    fn test_rsqrt_scale_12() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x88c);
        // Include small inputs so the quotient overflows i32.
        let input = Tensor::<i32>::random_range(&mut rng, &[T], 1..INPUT_MAX);
        let model = rsqrt_model(T, 12);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "non-power-of-two path not fully supported yet"]
    fn test_rsqrt_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::<i32>::random_range(&mut rng, &[t], 1..INPUT_MAX);
        let model = rsqrt_model(t, 8);
        unit_test_op(model, &[input]);
    }
}
