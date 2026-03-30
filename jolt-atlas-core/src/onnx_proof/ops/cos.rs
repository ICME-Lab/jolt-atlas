use crate::onnx_proof::neural_teleport::{
    cos::{CosTable, COS_LOG_TABLE_SIZE},
    division::{
        compute_division, TeleportDivisionParams, TeleportDivisionProver, TeleportDivisionVerifier,
    },
    utils::compute_ra_evals_direct,
};
use crate::onnx_proof::{
    ops::{eval_reduction::NodeEvalReduction, OperatorProofTrait, ReductionFlow},
    range_checking::{
        range_check_operands::TeleportRangeCheckOperands, RangeCheckEncoding, RangeCheckProvider,
    },
    ProofId, ProofType, Prover, Verifier,
};
use atlas_onnx_tracer::{
    model::{
        consts::EIGHT_PI_APPROX,
        trace::{LayerData, Trace},
        ComputationGraph,
    },
    node::ComputationNode,
    ops::Cos,
};
use common::{consts::XLEN, CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::{IntoOpening, JoltField},
    lookup_tables::unsigned_less_than::UnsignedLessThanTable,
    poly::{
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            CommittedOpeningId, OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator,
            SumcheckId, VerifierOpeningAccumulator, VirtualOpeningId, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Cos {
    fn reduction_flow(&self) -> ReductionFlow {
        ReductionFlow::Custom
    }

    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();

        // Stage 1a: Neural teleportation remainder proof
        let div_params = TeleportDivisionParams::new_from_transcript(
            node.clone(),
            &mut prover.transcript,
            EIGHT_PI_APPROX,
        );
        let mut div_sumcheck = TeleportDivisionProver::new(&prover.trace, div_params);

        // Prove the division sumcheck for the teleportation remainder
        let (div_proof, _) = Sumcheck::prove(
            &mut div_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::NeuralTeleport), div_proof));

        // Stage 1b: Execution proof for the cos layer
        let params = CosParams::new(
            node.clone(),
            &prover.preprocessing.model.graph,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut exec_sumcheck = CosProver::initialize(
            &prover.trace,
            params,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        let (exec_proof, _) = Sumcheck::prove(
            &mut exec_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), exec_proof));

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
        let mut execution_proofs = self.prove(node, prover);

        let quotient_vid = VirtualOpeningId::new(
            VirtualPolynomial::TeleportQuotient(node.idx),
            SumcheckId::NodeExecution(node.idx),
        );
        let teleport_q = prover
            .accumulator
            .get_virtual_polynomial_opening(quotient_vid);

        let quotient_cid = CommittedOpeningId::new(
            CommittedPolynomial::TeleportNodeQuotient(node.idx),
            SumcheckId::NodeExecution(node.idx),
        );
        prover.accumulator.append_dense(
            &mut prover.transcript,
            quotient_cid,
            teleport_q.0.r.clone(),
            teleport_q.1,
        );

        let eval_reduction_proof = NodeEvalReduction::prove(prover, node);
        execution_proofs.extend(prove_range_and_onehot(node, prover));
        (eval_reduction_proof, execution_proofs)
    }

    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Stage 1a: Neural teleportation remainder verification
        let div_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::NeuralTeleport))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;

        let div_verifier = TeleportDivisionVerifier::new_from_transcript(
            node.clone(),
            EIGHT_PI_APPROX,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            div_proof,
            &div_verifier,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 1b: Execution verification for the cos layer
        let cos_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;

        let exec_sumcheck = CosVerifier::new(
            node.clone(),
            &verifier.preprocessing.model.graph,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            cos_proof,
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
        eval_reduction_proof: &joltworks::subprotocols::evaluation_reduction::EvalReductionProof<F>,
    ) -> Result<(), ProofVerifyError> {
        self.verify(node, verifier)?;

        let quotient_vid = VirtualOpeningId::new(
            VirtualPolynomial::TeleportQuotient(node.idx),
            SumcheckId::NodeExecution(node.idx),
        );
        let teleport_q = verifier
            .accumulator
            .get_virtual_polynomial_opening(quotient_vid);

        let quotient_cid = CommittedOpeningId::new(
            CommittedPolynomial::TeleportNodeQuotient(node.idx),
            SumcheckId::NodeExecution(node.idx),
        );
        verifier.accumulator.append_dense(
            &mut verifier.transcript,
            quotient_cid,
            teleport_q.0.r.clone(),
        );
        let committed_q = verifier
            .accumulator
            .get_committed_polynomial_opening(quotient_cid)
            .1;
        if committed_q != teleport_q.1 {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Teleport quotient claim does not match committed quotient claim".to_string(),
            ));
        }

        NodeEvalReduction::verify(verifier, node, eval_reduction_proof)?;
        verify_range_and_onehot(node, verifier)
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPolynomial> {
        let cos_encoding = CosRaEncoding { node_idx: node.idx };
        let rc_encoding = RangeCheckEncoding::<TeleportRangeCheckOperands>::new(node);
        let cos_d = cos_encoding.one_hot_params().instruction_d;
        let rc_d = rc_encoding.one_hot_params().instruction_d;
        let mut polys = vec![CommittedPolynomial::TeleportNodeQuotient(node.idx)];
        polys.extend((0..cos_d).map(|i| CommittedPolynomial::CosRaD(node.idx, i)));
        polys.extend((0..rc_d).map(|i| CommittedPolynomial::TeleportRangeCheckRaD(node.idx, i)));
        polys
    }
}

fn prove_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
    let mut proofs = Vec::new();

    let LayerData { operands, .. } = Trace::layer_data(&prover.trace, node);
    let input = operands[0];
    let (_, remainder) = compute_division(input, EIGHT_PI_APPROX);
    let cos_lookup_indices = remainder
        .par_iter()
        .map(|&x| x as usize)
        .collect::<Vec<usize>>();

    let rangecheck_provider = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node);
    let (rangecheck_sumcheck, rc_lookup_indices) = rangecheck_provider
        .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );

    let cos_encoding = CosRaEncoding { node_idx: node.idx };
    let cos_ra_onehot_provers = shout::ra_onehot_provers(
        &cos_encoding,
        &cos_lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let mut range_and_cos_instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
        vec![Box::new(rangecheck_sumcheck)];
    range_and_cos_instances.extend(cos_ra_onehot_provers);
    let (cos_ra_one_hot_proof, _) = BatchedSumcheck::prove(
        range_and_cos_instances
            .iter_mut()
            .map(|v| &mut **v as _)
            .collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    proofs.push((
        ProofId(node.idx, ProofType::RaOneHotChecks),
        cos_ra_one_hot_proof,
    ));

    let rc_encoding = RangeCheckEncoding::<TeleportRangeCheckOperands>::new(node);
    let [rc_ra, rc_hw, rc_bool] = shout::ra_onehot_provers(
        &rc_encoding,
        &rc_lookup_indices,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![rc_ra, rc_hw, rc_bool];
    let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    proofs.push((
        ProofId(node.idx, ProofType::RaHammingWeight),
        ra_one_hot_proof,
    ));

    proofs
}

fn verify_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    let cos_ra_one_hot_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;

    let rangecheck_provider = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node);
    let rangecheck_verifier = rangecheck_provider
        .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
    let cos_encoding = CosRaEncoding { node_idx: node.idx };
    let cos_ra_onehot_verifier = shout::ra_onehot_verifiers(
        &cos_encoding,
        &verifier.accumulator,
        &mut verifier.transcript,
    );
    let cos_ra_onehot_verifier: Vec<&dyn SumcheckInstanceVerifier<F, T>> =
        cos_ra_onehot_verifier.iter().map(|v| &**v as _).collect();
    let mut range_and_cos_instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> =
        vec![&rangecheck_verifier];
    range_and_cos_instances.extend(cos_ra_onehot_verifier);
    BatchedSumcheck::verify(
        cos_ra_one_hot_proof,
        range_and_cos_instances,
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    let ra_one_hot_proof = verifier
        .proofs
        .get(&ProofId(node.idx, ProofType::RaHammingWeight))
        .ok_or(ProofVerifyError::MissingProof(node.idx))?;

    let rc_encoding = RangeCheckEncoding::<TeleportRangeCheckOperands>::new(node);
    let [rc_ra, rc_hw, rc_bool] = shout::ra_onehot_verifiers(
        &rc_encoding,
        &verifier.accumulator,
        &mut verifier.transcript,
    );
    BatchedSumcheck::verify(
        ra_one_hot_proof,
        vec![&*rc_ra, &*rc_hw, &*rc_bool],
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;

    Ok(())
}

const DEGREE_BOUND: usize = 2;

#[derive(Clone)]
/// Parameters for the cosine trigonometric function.
///
/// Cos uses a lookup table preceded by a teleportation step that
/// reduces the input domain.
pub struct CosParams<F: JoltField> {
    gamma: F,
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
}

impl<F: JoltField> CosParams<F> {
    /// Create a new CosParams instance for the given computation node.
    pub fn new(
        computation_node: ComputationNode,
        _graph: &ComputationGraph,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar();
        let remainder_id = VirtualOpeningId::new(
            VirtualPolynomial::TeleportRemainder(computation_node.idx),
            SumcheckId::NodeExecution(computation_node.idx),
        );
        let r_node_output = accumulator.get_virtual_polynomial_opening(remainder_id).0.r;

        Self {
            gamma,
            r_node_output: r_node_output.into(),
            computation_node,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for CosParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let output_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutput(self.computation_node.idx),
            SumcheckId::NodeExecution(self.computation_node.idx),
        );
        let rv_claim = accumulator.get_virtual_polynomial_opening(output_id).1;

        let remainder_id = VirtualOpeningId::new(
            VirtualPolynomial::TeleportRemainder(self.computation_node.idx),
            SumcheckId::NodeExecution(self.computation_node.idx),
        );
        let remainder_claim = accumulator.get_virtual_polynomial_opening(remainder_id).1;

        rv_claim + self.gamma * remainder_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        COS_LOG_TABLE_SIZE
    }
}

/// Prover state for cos execution sumcheck instance.
///
/// This implements a Read-Raf sumcheck for Cos lookup, asserting that each output[i] = CosTable[input[i]],
/// and that input[i] = ∑ Ra[k] * Int[k].
pub struct CosProver<F: JoltField> {
    params: CosParams<F>,
    cos_table: MultilinearPolynomial<F>,
    input_onehot: MultilinearPolynomial<F>,
    identity: IdentityPolynomial<F>,
}

impl<F: JoltField> CosProver<F> {
    /// Initialize the prover state.
    pub fn initialize(
        trace: &Trace,
        params: CosParams<F>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
        let input = operands[0];
        let (_quotient_tensor, remainder_tensor) = compute_division(input, EIGHT_PI_APPROX);

        assert!(remainder_tensor
            .iter()
            .all(|&x| (0..EIGHT_PI_APPROX).contains(&x)));

        let cos_table = MultilinearPolynomial::from(CosTable::materialize());
        let input_onehot: Vec<F> = compute_ra_evals_direct(
            &params.r_node_output.r,
            &remainder_tensor,
            1 << COS_LOG_TABLE_SIZE,
        );

        let output_claim =
            MultilinearPolynomial::from(output.clone()).evaluate(&params.r_node_output.r);
        // Special case where we add a new opening for the node output at node's own index,
        // This is due to the fact that `remainder` poly claim is used as input claim for cos lookup sumcheck, together with a node output claim.
        // Both claims are derived from a different opening point, so we need to derive a new claim for one of (`output`, `remainder`).
        // We chose `output` to prevent us from having to use n-to-1 reductions on the `remainder` and rather only implement it on NodeOutput.
        // Making further work for Issue#138 easier.
        let output_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutput(params.computation_node.idx),
            SumcheckId::NodeExecution(params.computation_node.idx),
        );
        accumulator.append_virtual(
            transcript,
            output_id,
            params.r_node_output.clone(),
            output_claim,
        );

        let input_onehot = MultilinearPolynomial::from(input_onehot);
        assert_eq!(input_onehot.len(), cos_table.len());
        let identity = IdentityPolynomial::new(COS_LOG_TABLE_SIZE);

        #[cfg(test)]
        {
            let remainder_id = VirtualOpeningId::new(
                VirtualPolynomial::TeleportRemainder(params.computation_node.idx),
                SumcheckId::NodeExecution(params.computation_node.idx),
            );
            let output_id = VirtualOpeningId::new(
                VirtualPolynomial::NodeOutput(params.computation_node.idx),
                SumcheckId::NodeExecution(params.computation_node.idx),
            );
            let remainder_claim = accumulator.get_virtual_polynomial_opening(remainder_id).1;
            let rv_claim = accumulator.get_virtual_polynomial_opening(output_id).1;
            let claim = (0..input_onehot.len())
                .map(|i| {
                    let a = input_onehot.get_bound_coeff(i);
                    let b = cos_table.get_bound_coeff(i);
                    let int = F::from_u32(i as u32);
                    a * (b + params.gamma * int)
                })
                .sum();
            assert_eq!(rv_claim + params.gamma * remainder_claim, claim)
        }

        Self {
            params,
            cos_table,
            input_onehot,
            identity,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for CosProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            input_onehot,
            cos_table,
            identity,
            ..
        } = self;

        let univariate_poly_evals: [F; 2] = (0..input_onehot.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals =
                    input_onehot.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                let table_evals =
                    cos_table.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                let id_evals = identity.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);

                [
                    ra_evals[0] * (table_evals[0] + id_evals[0] * self.params.gamma),
                    ra_evals[1] * (table_evals[1] + id_evals[1] * self.params.gamma),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        UniPoly::from_evals_and_hint(previous_claim, &univariate_poly_evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.input_onehot
            .bind_parallel(r_j, BindingOrder::LowToHigh);
        self.cos_table.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.identity.bind_parallel(r_j, BindingOrder::LowToHigh);
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
        let r = [
            opening_point.r.as_slice(),
            self.params.r_node_output.r.as_slice(),
        ]
        .concat();
        let cos_ra_id = VirtualOpeningId::new(
            VirtualPolynomial::CosRa(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(
            transcript,
            cos_ra_id,
            OpeningPoint::new(r),
            self.input_onehot.final_sumcheck_claim(),
        );
    }
}

/// Verifier state for cos execution sumcheck instance.
///
/// Verifies that the prover correctly evaluated the Cos lookup table at the teleported remainders,
/// And that the one-hot encoding of the input is correct.
pub struct CosVerifier<F: JoltField> {
    params: CosParams<F>,
    cos_table: MultilinearPolynomial<F>,
}

impl<F: JoltField> CosVerifier<F> {
    /// Initialize the verifier state.
    pub fn new(
        computation_node: ComputationNode,
        graph: &ComputationGraph,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = CosParams::new(computation_node, graph, accumulator, transcript);

        let output_id = VirtualOpeningId::new(
            VirtualPolynomial::NodeOutput(params.computation_node.idx),
            SumcheckId::NodeExecution(params.computation_node.idx),
        );
        accumulator.append_virtual(transcript, output_id, params.r_node_output.clone());

        let cos_table = MultilinearPolynomial::from(CosTable::materialize());
        Self { params, cos_table }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for CosVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());

        let cos_ra_id = VirtualOpeningId::new(
            VirtualPolynomial::CosRa(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        let ra_claim = accumulator.get_virtual_polynomial_opening(cos_ra_id).1;
        let table_claim = self.cos_table.evaluate(&opening_point.r);
        let int_eval = IdentityPolynomial::new(COS_LOG_TABLE_SIZE).evaluate(&opening_point.r);

        ra_claim * (table_claim + self.params.gamma * int_eval)
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
        let r = [
            opening_point.r.as_slice(),
            self.params.r_node_output.r.as_slice(),
        ]
        .concat();
        let cos_ra_id = VirtualOpeningId::new(
            VirtualPolynomial::CosRa(self.params.computation_node.idx),
            SumcheckId::NodeExecution(self.params.computation_node.idx),
        );
        accumulator.append_virtual(transcript, cos_ra_id, OpeningPoint::new(r));
    }
}

/// Encoding impl for Cos one-hot read-address checks.
///
/// This encodes the relation between the teleportation remainder and the cos lookup table index,
pub struct CosRaEncoding {
    /// Index of the computation node in the trace.
    pub node_idx: usize,
}

impl RaOneHotEncoding for CosRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPolynomial {
        CommittedPolynomial::CosRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> VirtualOpeningId {
        VirtualOpeningId::new(
            VirtualPolynomial::TeleportRemainder(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> VirtualOpeningId {
        VirtualOpeningId::new(
            VirtualPolynomial::CosRa(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        COS_LOG_TABLE_SIZE
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), COS_LOG_TABLE_SIZE)
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

    use super::EIGHT_PI_APPROX;

    fn cos_model(input_shape: &[usize]) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(input_shape.to_vec());
        let res = b.cos(i);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_cos_random_inputs() {
        let t = 1 << 13;
        let mut rng = StdRng::seed_from_u64(0xC05);
        let input = Tensor::random_range(&mut rng, &[t], -50000..50000);
        let model = cos_model(&[t]);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_cos_periodic_boundary_inputs() {
        let input = Tensor::new(
            Some(&[
                -EIGHT_PI_APPROX - 1,
                -EIGHT_PI_APPROX,
                -1,
                0,
                1,
                EIGHT_PI_APPROX - 1,
                EIGHT_PI_APPROX,
                EIGHT_PI_APPROX + 1,
            ]),
            &[8],
        )
        .unwrap();
        let model = cos_model(&[8]);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "TODO: non-power-of-two cos path not fully validated yet"]
    fn test_cos_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0xC06);
        let input = Tensor::random_range(&mut rng, &[t], -50000..50000);
        let model = cos_model(&[t]);
        unit_test_op(model, &[input]);
    }
}
