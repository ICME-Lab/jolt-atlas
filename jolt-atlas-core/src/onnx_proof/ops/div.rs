use crate::{
    onnx_proof::{
        ops::{eval_reduction::NodeEvalReduction, OperatorProofTrait, ReductionFlow},
        range_checking::{
            range_check_operands::DivRangeCheckOperands, RangeCheckEncoding, RangeCheckProvider,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Div,
    tensor::Tensor,
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

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Div {
    fn reduction_flow(&self) -> ReductionFlow {
        ReductionFlow::Custom
    }

    #[tracing::instrument(skip_all, name = "Div::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();

        // Execution proof
        let params = DivParams::new(node.clone(), &mut prover.transcript);
        let mut exec_sumcheck = DivProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut exec_sumcheck,
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

        // Reduce node-output openings first, then bind the reduced output claim
        // as the committed Div quotient opening used by PCS opening verification.
        let eval_reduction_proof = NodeEvalReduction::prove(prover, node);

        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
        let reduced = accessor.get_reduced_opening();
        let mut provider = accessor.into_provider(&mut prover.transcript, reduced.0.clone());

        provider.append_advice(CommittedPoly::DivNodeQuotient, reduced.1);

        if node.is_scalar() {
            return (eval_reduction_proof, proofs);
        }

        proofs.extend(prove_range_and_onehot(node, prover));
        (eval_reduction_proof, proofs)
    }

    #[tracing::instrument(skip_all, name = "Div::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Execution verification
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let exec_sumcheck = DivVerifier::new(node.clone(), &mut verifier.transcript);
        Sumcheck::verify(
            proof,
            &exec_sumcheck,
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
        let reduced = AccOpeningAccessor::new(&verifier.accumulator, node).get_reduced_opening();

        let mut provider = AccOpeningAccessor::new(&mut verifier.accumulator, node)
            .into_provider(&mut verifier.transcript, reduced.0.clone());
        provider.append_advice(CommittedPoly::DivNodeQuotient);

        let quotient_claim = provider.get_advice(CommittedPoly::DivNodeQuotient).1;

        if quotient_claim != reduced.1 {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Div quotient claim does not match reduced node-output claim".to_string(),
            ));
        }

        if node.is_scalar() {
            return Ok(());
        }

        verify_range_and_onehot(node, verifier)?;
        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let mut polys = vec![CommittedPoly::DivNodeQuotient(node.idx)];
        if node.is_scalar() {
            return polys;
        }
        let encoding = RangeCheckEncoding::<DivRangeCheckOperands>::new(node);
        let d = encoding.one_hot_params().instruction_d;
        polys.extend((0..d).map(|i| CommittedPoly::DivRangeCheckRaD(node.idx, i)));
        polys
    }
}

// TODO: Reduce two claims to 1 via 4.5.2 PAZK for Quotient polynomial openings

const DEGREE_BOUND: usize = 3;

/// Parameters for proving element-wise division operations.
///
/// Division requires proving a * q = b * R where a/b = q with remainder R.
/// Stores the opening point and computation node information needed for the sumcheck protocol.
#[derive(Clone)]
pub struct DivParams<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
}

impl<F: JoltField> DivParams<F> {
    /// Create new division parameters from a computation node and transcript.
    pub fn new<T: Transcript>(computation_node: ComputationNode, transcript: &mut T) -> Self {
        let num_vars = computation_node.pow2_padded_num_output_elements().log_2();
        let r_node_output = transcript.challenge_vector_optimized::<F>(num_vars);

        Self {
            r_node_output: r_node_output.into(),
            computation_node,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for DivParams<F> {
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

    // output = eq_eval * (right * q + R - left)
    //        = eq_eval * right * q + eq_eval * R + (-eq_eval) * left
    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        use crate::utils::opening_access::OpeningIdBuilder;
        let builder = OpeningIdBuilder::new(&self.computation_node);
        let left_id = builder.nodeio(Target::Input(0));
        let right_id = builder.nodeio(Target::Input(1));
        let q_id = builder.nodeio(Target::Current);
        let r_id = builder.advice(VirtualPoly::DivRemainder);
        Some(OutputClaimConstraint::sum_of_products(vec![
            ProductTerm::scaled(
                ValueSource::Challenge(0),
                vec![ValueSource::Opening(right_id), ValueSource::Opening(q_id)],
            ),
            ProductTerm::scaled(ValueSource::Challenge(1), vec![ValueSource::Opening(r_id)]),
            ProductTerm::scaled(
                ValueSource::Challenge(2),
                vec![ValueSource::Opening(left_id)],
            ),
        ]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let r_node_output_prime: Vec<F> = self
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let eq_eval = EqPolynomial::mle(&self.r_node_output.r, &r_node_output_prime);
        vec![eq_eval, eq_eval, -eq_eval]
    }
}

/// Prover state for element-wise division sumcheck protocol.
///
/// Maintains the equality polynomial, operand polynomials, quotient q, and remainder R
/// needed to prove the division relation: left = right * q + R.
pub struct DivProver<F: JoltField> {
    params: DivParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    left_operand: MultilinearPolynomial<F>,
    right_operand: MultilinearPolynomial<F>,
    q: MultilinearPolynomial<F>,
    R: MultilinearPolynomial<F>,
}

impl<F: JoltField> DivProver<F> {
    /// Initialize the prover with trace data and parameters.
    #[tracing::instrument(skip_all)]
    pub fn initialize(trace: &Trace, params: DivParams<F>) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand, right_operand] = operands[..] else {
            panic!("Expected two operands for Div operation")
        };
        let R_tensor = {
            let data: Vec<i32> = left_operand
                .iter()
                .zip(right_operand.iter())
                .map(|(&a, &b)| {
                    let mut R = a % b;
                    if (R < 0 && b > 0) || R > 0 && b < 0 {
                        R += b
                    }
                    R
                })
                .collect();
            Tensor::<i32>::construct(data, left_operand.dims().to_vec())
        };
        let left_operand = MultilinearPolynomial::from(left_operand.clone());
        let right_operand = MultilinearPolynomial::from(right_operand.clone());
        let q = MultilinearPolynomial::from(output.clone());
        let R = MultilinearPolynomial::from(R_tensor);
        #[cfg(test)]
        {
            let claim = (0..left_operand.len())
                .map(|i| {
                    let a = left_operand.get_bound_coeff(i);
                    let b = right_operand.get_bound_coeff(i);
                    let q = q.get_bound_coeff(i);
                    let R = R.get_bound_coeff(i);
                    b * q + R - a
                })
                .sum();
            assert_eq!(F::zero(), claim)
        }
        Self {
            params,
            eq_r_node_output,
            left_operand,
            right_operand,
            q,
            R,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for DivProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            eq_r_node_output,
            left_operand,
            right_operand,
            q,
            R,
            ..
        } = self;
        let [q_constant, q_quadratic] = eq_r_node_output.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let lo0 = left_operand.get_bound_coeff(2 * g);
            let ro0 = right_operand.get_bound_coeff(2 * g);
            let ro1 = right_operand.get_bound_coeff(2 * g + 1);
            let q0 = q.get_bound_coeff(2 * g);
            let q1 = q.get_bound_coeff(2 * g + 1);
            let R0 = R.get_bound_coeff(2 * g);
            let c0 = (ro0 * q0) + R0 - lo0;

            let e = (ro1 - ro0) * (q1 - q0);
            [c0, e]
        });
        eq_r_node_output.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
        self.left_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.right_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.q.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.R.bind_parallel(r_j, BindingOrder::LowToHigh);
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

        provider.append_nodeio(Target::Input(0), self.left_operand.final_claim());
        provider.append_nodeio(Target::Input(1), self.right_operand.final_claim());
        provider.append_nodeio(Target::Current, self.q.final_claim());

        provider.append_advice(VirtualPoly::DivRemainder, self.R.final_claim());
    }
}

/// Verifier for element-wise division sumcheck protocol.
///
/// Verifies that the prover's sumcheck messages are consistent with the claimed
/// division operation output and the division relation.
pub struct DivVerifier<F: JoltField> {
    params: DivParams<F>,
}

impl<F: JoltField> DivVerifier<F> {
    /// Create a new verifier for the division operation.
    pub fn new<T: Transcript>(computation_node: ComputationNode, transcript: &mut T) -> Self {
        let params = DivParams::new(computation_node, transcript);
        Self { params }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for DivVerifier<F> {
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

        let q_claim = accessor.get_nodeio(Target::Current).1;
        let R_claim = accessor.get_advice(VirtualPoly::DivRemainder).1;
        let left_operand_claim = accessor.get_nodeio(Target::Input(0)).1;
        let right_operand_claim = accessor.get_nodeio(Target::Input(1)).1;

        eq_eval * ((right_operand_claim * q_claim) + R_claim - left_operand_claim)
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
        provider.append_nodeio(Target::Input(1));
        provider.append_nodeio(Target::Current);
        provider.append_advice(VirtualPoly::DivRemainder);
    }
}

fn prove_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
    let mut results = Vec::new();

    let rangecheck_provider = RangeCheckProvider::<DivRangeCheckOperands>::new(node);
    let (mut rangecheck_sumcheck, lookup_indices) = rangecheck_provider
        .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    let (rangecheck_proof, _) = Sumcheck::prove(
        &mut rangecheck_sumcheck,
        &mut prover.accumulator,
        &mut prover.transcript,
    );

    results.push((ProofId(node.idx, ProofType::RangeCheck), rangecheck_proof));

    let encoding = RangeCheckEncoding::<DivRangeCheckOperands>::new(node);

    let [ra_sumcheck, hw_sumcheck, bool_sumcheck] = shout::ra_onehot_provers(
        &encoding,
        &lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );

    let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
        vec![ra_sumcheck, hw_sumcheck, bool_sumcheck];
    let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    results.push((
        ProofId(node.idx, ProofType::RaOneHotChecks),
        ra_one_hot_proof,
    ));

    results
}

fn verify_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    let rangecheck_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RangeCheck))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;

    let rangecheck_provider = RangeCheckProvider::<DivRangeCheckOperands>::new(node);
    let rangecheck_verifier = rangecheck_provider
        .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
    Sumcheck::verify(
        rangecheck_proof,
        &rangecheck_verifier,
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    let ra_one_hot_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;
    let encoding = RangeCheckEncoding::<DivRangeCheckOperands>::new(node);
    let [ra_sumcheck, hw_sumcheck, bool_sumcheck] =
        shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);
    BatchedSumcheck::verify(
        ra_one_hot_proof,
        vec![&*ra_sumcheck, &*hw_sumcheck, &*bool_sumcheck],
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

    const SCALE: i32 = 8;

    fn div_model(T: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let c = b.constant(Tensor::construct(vec![1 << SCALE; T], vec![T]));
        let res = b.div(i, c);
        b.mark_output(res);
        b.build()
    }

    fn recip_model(T: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let c = b.constant(Tensor::construct(vec![1 << (SCALE * 2); T], vec![T]));
        let res = b.div(c, i);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_div_by_const() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random(&mut rng, &[T]);
        let model = div_model(T);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_div_by_tensor() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let model = recip_model(T);
        let mut input = Tensor::<i32>::random_range(&mut rng, &[T], 1..SCALE * SCALE);
        input.iter_mut().for_each(|v| {
            if *v == 0 {
                *v = 1
            }
        });
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "non-power-of-two path not fully supported yet"]
    fn test_div_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0x889);
        let mut input = Tensor::<i32>::random_range(&mut rng, &[t], 1..SCALE * SCALE);
        input.iter_mut().for_each(|v| {
            if *v == 0 {
                *v = 1
            }
        });
        let model = div_model(t);
        unit_test_op(model, &[input]);
    }
}
