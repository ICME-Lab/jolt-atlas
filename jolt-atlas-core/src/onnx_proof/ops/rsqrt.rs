use crate::onnx_proof::{
    ops::{eval_reduction::NodeEvalReduction, OperatorProofTrait, ReductionFlow},
    range_checking::{
        range_check_operands::{RiRangeCheckOperands, RsRangeCheckOperands},
        RangeCheckEncoding, RangeCheckProvider,
    },
    ProofId, ProofType, Prover, Verifier,
};
use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Rsqrt,
};
use common::{consts::XLEN, CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::{IntoOpening, JoltField},
    lookup_tables::unsigned_less_than::UnsignedLessThanTable,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            CommittedOpeningId, OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator,
            SumcheckId, VerifierOpeningAccumulator, VirtualOpeningId, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

/// Fixed-point scaling factor for reciprocal square root calculations.
pub const Q: i32 = 256;
/// Square of the fixed-point scaling factor (Q * Q = 65536).
pub const Q_SQUARE: i32 = Q * Q;

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
        let mut prover_sumcheck =
            RsqrtProver::initialize(&prover.trace, &mut prover.transcript, params);
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
        joltworks::subprotocols::evaluation_reduction::EvalReductionProof<F>,
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
        eval_reduction_proof: &joltworks::subprotocols::evaluation_reduction::EvalReductionProof<F>,
    ) -> Result<(), ProofVerifyError> {
        self.verify(node, verifier)?;
        NodeEvalReduction::verify(verifier, node, eval_reduction_proof)?;
        verify_range_and_onehot(node, verifier)
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPolynomial> {
        let mut polys = vec![CommittedPolynomial::RsqrtNodeInv(node.idx)];
        let encoding = RangeCheckEncoding::<RiRangeCheckOperands>::new(node);
        let d = encoding.one_hot_params().instruction_d;
        for i in 0..d {
            polys.push(CommittedPolynomial::SqrtDivRangeCheckRaD(node.idx, i));
            polys.push(CommittedPolynomial::SqrtRangeCheckRaD(node.idx, i));
        }
        polys
    }
}

// Decomposes rsqrt into an inverse and a square root where the inverse `inv` of x is such that:
// 1 = x * inv + r_i  where 0 <= r_i < x
// and the square root `sqrt` of inv is such that:
// inv = sqrt * sqrt + r_s  where 0 <= r_s < 2 * sqrt + 1

// HANDLING SCALE:
// Any input to the model is quantized by a scale factor of S
// Therefore, for an input x, the quantized representation is x̂ = x * S
// The inverse inv of x is given by:
// inv = 1 / x = S / x̂
// Therefore, the quantized representation of inv is:
// in̂v = S * inv = S^2 / x̂
// Similarly, for the square root sqt of in̂v:
// sqt = sqrt(in̂v) = sqrt(S / x̂)
// Therefore, the quantized representation of sqrt is:
// sq̂t = S * sqt = S * sqrt(S / x̂) = sqrt(S^3 / x̂) = sqrt(S * in̂v)
// The two relations that we will batch together in a sumcheck instance are:
// - 0 = x̂ * in̂v + r_i - S^2
// - 0 = S * in̂v - sq̂t * sqt - r_s

// Possible optimization is to only commit to the result and a remainder,
// and find the associated range check for the unique remainder.

const DEGREE_BOUND: usize = 3;

/// Parameters for proving reciprocal square root (1/√x) operations.
///
/// Rsqrt requires proving multiple relations including the reciprocal and square root properties.
/// Stores the opening point and computation node information needed for the sumcheck protocol.
#[derive(Clone)]
pub struct RsqrtParams<F: JoltField> {
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
}

impl<F: JoltField> RsqrtParams<F> {
    /// Create new rsqrt parameters from a computation node and transcript.
    pub fn new<T: Transcript>(computation_node: ComputationNode, transcript: &mut T) -> Self {
        let num_vars = computation_node.pow2_padded_num_output_elements().log_2();
        let r_node_output = transcript.challenge_vector_optimized::<F>(num_vars);
        Self {
            r_node_output: r_node_output.into(),
            computation_node,
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
}

/// Prover state for reciprocal square root sumcheck protocol.
///
/// Maintains the equality polynomial, operand polynomial, inverse, rsqrt result,
/// and remainders (r_i, r_s) needed to prove the rsqrt relation along with a folding challenge.
pub struct RsqrtProver<F: JoltField> {
    params: RsqrtParams<F>,
    eq_r_node_output: GruenSplitEqPolynomial<F>,
    left_operand: MultilinearPolynomial<F>,
    inv: MultilinearPolynomial<F>,
    rsqrt: MultilinearPolynomial<F>,
    r_i: MultilinearPolynomial<F>,
    r_s: MultilinearPolynomial<F>,
    // folding challenge
    gamma: F,
}

impl<F: JoltField> RsqrtProver<F> {
    /// Initialize the prover with trace data, transcript, and parameters.
    pub fn initialize<T: Transcript>(
        trace: &Trace,
        transcript: &mut T,
        params: RsqrtParams<F>,
    ) -> Self {
        let eq_r_node_output =
            GruenSplitEqPolynomial::new(&params.r_node_output.r, BindingOrder::LowToHigh);
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
        let [left_operand] = operands[..] else {
            panic!("Expected one operand for Rsqrt operation")
        };
        let inv_data: Vec<i32> = left_operand.iter().map(|&x| Q_SQUARE / x).collect();
        let ri_data: Vec<i32> = left_operand.iter().map(|&x| Q_SQUARE % x).collect();
        let rs_data: Vec<i32> = inv_data
            .iter()
            .zip(output.iter())
            .map(|(&inv, &sqrt)| Q * inv - sqrt * sqrt)
            .collect();

        let left_operand = MultilinearPolynomial::from(left_operand.clone());
        let inv = MultilinearPolynomial::from(inv_data.clone());
        let r_i = MultilinearPolynomial::from(ri_data.clone());
        let rsqrt = MultilinearPolynomial::from(output.clone());
        let r_s = MultilinearPolynomial::from(rs_data.clone());
        #[cfg(test)]
        {
            let claim_inv = (0..left_operand.len())
                .map(|i| {
                    let a: F = left_operand.get_bound_coeff(i);
                    let inv = inv.get_bound_coeff(i);
                    let r_i: F = r_i.get_bound_coeff(i);
                    // range checking
                    assert!(r_i.to_u64().unwrap() < a.to_u64().unwrap());

                    a * inv + r_i - F::from_i32(Q_SQUARE)
                })
                .sum();
            assert_eq!(F::zero(), claim_inv);

            let claim_sqrt = (0..left_operand.len())
                .map(|i| {
                    let inv = inv.get_bound_coeff(i);
                    let sqrt: F = rsqrt.get_bound_coeff(i);
                    let r_s: F = r_s.get_bound_coeff(i);
                    // range checking
                    assert!(r_s.to_u64().unwrap() <= 2 * sqrt.to_u64().unwrap());

                    sqrt * sqrt + r_s - F::from_i32(Q) * inv
                })
                .sum();
            assert_eq!(F::zero(), claim_sqrt)
        }

        let gamma = transcript.challenge_scalar();
        Self {
            params,
            eq_r_node_output,
            left_operand,
            inv,
            r_i,
            rsqrt,
            r_s,
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
            left_operand,
            inv,
            rsqrt,
            r_i,
            r_s,
            ..
        } = self;
        let [q_constant, q_quadratic] = eq_r_node_output.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let lo0 = left_operand.get_bound_coeff(2 * g);
            let lo1 = left_operand.get_bound_coeff(2 * g + 1);
            let inv0 = inv.get_bound_coeff(2 * g);
            let inv1 = inv.get_bound_coeff(2 * g + 1);
            let r_i0 = r_i.get_bound_coeff(2 * g);

            let rsqrt0 = rsqrt.get_bound_coeff(2 * g);
            let rsqrt1 = rsqrt.get_bound_coeff(2 * g + 1);
            let r_s0 = r_s.get_bound_coeff(2 * g);

            let c0 = lo0 * inv0 + r_i0 - F::from_i32(Q_SQUARE);

            let c1 = rsqrt0 * rsqrt0 + r_s0 - F::from_i32(Q) * inv0;

            let e0 = (lo1 - lo0) * (inv1 - inv0);
            let e1 = (rsqrt1 - rsqrt0) * (rsqrt1 - rsqrt0);
            [c0 + self.gamma * c1, e0 + self.gamma * e1]
        });
        eq_r_node_output.gruen_poly_deg_3(q_constant, q_quadratic, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_node_output.bind(r_j);
        self.left_operand
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.inv.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.rsqrt.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.r_i.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.r_s.bind_parallel(r_j, BindingOrder::LowToHigh);
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
        let left_operand_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(
            transcript,
            left_operand_id,
            opening_point.clone(),
            self.left_operand.final_sumcheck_claim(),
        );
        let inv_id = CommittedOpeningId::new(
            CommittedPolynomial::RsqrtNodeInv(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_dense(
            transcript,
            inv_id,
            opening_point.r.clone(),
            self.inv.final_sumcheck_claim(),
        );
        let rsqrt_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(
            transcript,
            rsqrt_id,
            opening_point.clone(),
            self.rsqrt.final_sumcheck_claim(),
        );
        let r_i_id = VirtualOpeningId::new(
            VirtualPolynomial::DivRemainder(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(
            transcript,
            r_i_id,
            opening_point.clone(),
            self.r_i.final_sumcheck_claim(),
        );
        let r_s_id = VirtualOpeningId::new(
            VirtualPolynomial::SqrtRemainder(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(
            transcript,
            r_s_id,
            opening_point.clone(),
            self.r_s.final_sumcheck_claim(),
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
        let gamma = transcript.challenge_scalar();
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
        let r_node_output = self.params.r_node_output.r.clone();
        let r_node_output_prime = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening())
            .r;
        let eq_eval = EqPolynomial::mle(&r_node_output, &r_node_output_prime);
        let inv_id = CommittedOpeningId::new(
            CommittedPolynomial::RsqrtNodeInv(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        let inv_claim = accumulator.get_committed_polynomial_opening(inv_id).1;
        let r_i_id = VirtualOpeningId::new(
            VirtualPolynomial::DivRemainder(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        let r_i_claim = accumulator.get_virtual_polynomial_opening(r_i_id).1;
        let rsqrt_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        let rsqrt_claim = accumulator.get_virtual_polynomial_opening(rsqrt_id).1;
        let r_s_id = VirtualOpeningId::new(
            VirtualPolynomial::SqrtRemainder(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        let r_s_claim = accumulator.get_virtual_polynomial_opening(r_s_id).1;
        let left_operand_claim = accumulator.get_node_output_claim(
            self.params.computation_node.inputs[0],
            self.params.computation_node.idx,
        );
        eq_eval
            * (left_operand_claim * inv_claim + r_i_claim - F::from_i32(Q_SQUARE)
                + self.gamma * (rsqrt_claim * rsqrt_claim + r_s_claim - F::from_i32(Q) * inv_claim))
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
        let left_operand_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutput(self.params.computation_node.inputs[0]),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(transcript, left_operand_id, opening_point.clone());
        let inv_id = CommittedOpeningId::new(
            CommittedPolynomial::RsqrtNodeInv(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_dense(transcript, inv_id, opening_point.r.clone());
        let rsqrt_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutput(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(transcript, rsqrt_id, opening_point.clone());
        let r_i_id = VirtualOpeningId::new(
            VirtualPolynomial::DivRemainder(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(transcript, r_i_id, opening_point.clone());
        let r_s_id = VirtualOpeningId::new(
            VirtualPolynomial::SqrtRemainder(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(transcript, r_s_id, opening_point.clone());
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
    use super::Q_SQUARE;
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };
    use rand::{rngs::StdRng, SeedableRng};

    fn rsqrt_model(T: usize) -> Model {
        let mut b = ModelBuilder::new();
        let input = b.input(vec![T]);
        let res = b.rsqrt(input);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_rsqrt() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_range(&mut rng, &[T], 1..Q_SQUARE);
        let model = rsqrt_model(T);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "non-power-of-two path not fully supported yet"]
    fn test_rsqrt_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::<i32>::random_range(&mut rng, &[t], 1..Q_SQUARE);
        let model = rsqrt_model(t);
        unit_test_op(model, &[input]);
    }
}
