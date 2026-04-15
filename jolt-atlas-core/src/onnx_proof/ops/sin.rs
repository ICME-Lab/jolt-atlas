use crate::onnx_proof::neural_teleport::{
    division::{
        compute_division, TeleportDivisionParams, TeleportDivisionProver, TeleportDivisionVerifier,
    },
    range_and_onehot::{
        prove_range_and_onehot, verify_range_and_onehot, NeuralTeleportRangeOneHot,
    },
    sin::{SinTable, SIN_LOG_TABLE_SIZE},
    utils::compute_ra_evals_direct,
};
use crate::onnx_proof::{
    ops::{eval_reduction::NodeEvalReduction, OperatorProofTrait, ReductionFlow},
    range_checking::{range_check_operands::TeleportRangeCheckOperands, RangeCheckEncoding},
    ProofId, ProofType, Prover, Verifier,
};
use crate::utils::opening_id_builder::{AccOpeningAccessor, Target};
use atlas_onnx_tracer::{
    model::{
        consts::FOUR_PI_APPROX,
        trace::{LayerData, Trace},
        ComputationGraph,
    },
    node::ComputationNode,
    ops::Sin,
};
use common::parallel::par_enabled;
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::{IntoOpening, JoltField},
    poly::{
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        shout::RaOneHotEncoding,
        sumcheck::{Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Sin {
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
            FOUR_PI_APPROX,
        );
        let mut div_sumcheck = TeleportDivisionProver::new(&prover.trace, div_params);

        // Prove the division sumcheck for the teleportation remainder
        let (div_proof, _) = Sumcheck::prove(
            &mut div_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::NeuralTeleport), div_proof));

        // Stage 1b: Execution proof for the sin layer
        let params = SinParams::new(
            node.clone(),
            &prover.preprocessing.model.graph,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut exec_sumcheck = SinProver::initialize(
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

        // teleportation quotient is never virtualized, so we need to commit to it.
        let accessor = AccOpeningAccessor::new(&mut prover.accumulator, node);
        let teleport_q = accessor.get_advice(VirtualPoly::TeleportQuotient);
        let mut provider = accessor.to_provider(&mut prover.transcript, teleport_q.0.clone());

        provider.append_advice(CommittedPoly::TeleportNodeQuotient, teleport_q.1);

        let eval_reduction_proof = NodeEvalReduction::prove(prover, node);
        execution_proofs.extend(prove_range_and_onehot(node, prover, self));
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
            FOUR_PI_APPROX,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            div_proof,
            &div_verifier,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 1b: Execution verification for the sin layer
        let sin_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;

        let exec_sumcheck = SinVerifier::new(
            node.clone(),
            &verifier.preprocessing.model.graph,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            sin_proof,
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

        let accessor = AccOpeningAccessor::new(&mut verifier.accumulator, node);
        let teleport_q = accessor.get_advice(VirtualPoly::TeleportQuotient);
        let mut provider = accessor.to_provider(&mut verifier.transcript, teleport_q.0.clone());

        provider.append_advice(CommittedPoly::TeleportNodeQuotient);
        let committed_q = provider.get_advice(CommittedPoly::TeleportNodeQuotient).1;
        if committed_q != teleport_q.1 {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Teleport quotient claim does not match committed quotient claim".to_string(),
            ));
        }

        NodeEvalReduction::verify(verifier, node, eval_reduction_proof)?;
        verify_range_and_onehot(node, verifier, self)
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let sin_encoding = SinRaEncoding { node_idx: node.idx };
        let rc_encoding = RangeCheckEncoding::<TeleportRangeCheckOperands>::new(node);
        let sin_d = sin_encoding.one_hot_params().instruction_d;
        let rc_d = rc_encoding.one_hot_params().instruction_d;
        let mut polys = vec![CommittedPoly::TeleportNodeQuotient(node.idx)];
        polys.extend((0..sin_d).map(|i| CommittedPoly::SinRaD(node.idx, i)));
        polys.extend((0..rc_d).map(|i| CommittedPoly::TeleportRangeCheckRaD(node.idx, i)));
        polys
    }
}

const DEGREE_BOUND: usize = 2;

#[derive(Clone)]
/// Parameters for the sine trigonometric function.
///
/// Sin uses a lookup table preceded by a teleportation step that
/// reduces the input domain.
pub struct SinParams<F: JoltField> {
    gamma: F,
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
}

impl<F: JoltField> SinParams<F> {
    /// Create a new SinParams instance for the given computation node.
    pub fn new(
        computation_node: ComputationNode,
        _graph: &ComputationGraph,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let gamma = transcript.challenge_scalar();
        let r_node_output = accessor.get_advice(VirtualPoly::TeleportRemainder).0;

        Self {
            gamma,
            r_node_output,
            computation_node,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for SinParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let rv_claim = accessor.get_node_io(Target::Current).1;
        let remainder_claim = accessor.get_advice(VirtualPoly::TeleportRemainder).1;

        rv_claim + self.gamma * remainder_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        SIN_LOG_TABLE_SIZE
    }
}

/// Prover state for sin execution sumcheck instance.
///
/// This implements a Read-Raf sumcheck for Sin lookup, asserting that each output[i] = SinTable[input[i]],
/// and that input[i] = ∑ Ra[k] * Int[k].
pub struct SinProver<F: JoltField> {
    params: SinParams<F>,
    sin_table: MultilinearPolynomial<F>,
    input_onehot: MultilinearPolynomial<F>,
    identity: IdentityPolynomial<F>,
}

impl<F: JoltField> SinProver<F> {
    /// Initialize the prover state.
    pub fn initialize(
        trace: &Trace,
        params: SinParams<F>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let LayerData { operands, output } = Trace::layer_data(trace, &params.computation_node);
        let input = operands[0];
        let (_quotient_tensor, remainder_tensor) = compute_division(input, FOUR_PI_APPROX);

        assert!(remainder_tensor
            .iter()
            .all(|&x| (0..FOUR_PI_APPROX).contains(&x)));

        let sin_table = MultilinearPolynomial::from(SinTable::materialize());
        let input_onehot: Vec<F> = compute_ra_evals_direct(
            &params.r_node_output.r,
            &remainder_tensor,
            1 << SIN_LOG_TABLE_SIZE,
        );

        let output_claim =
            MultilinearPolynomial::from(output.clone()).evaluate(&params.r_node_output.r);
        // Special case where we add a new opening for the node output at node's own index,
        // This is due to the fact that `remainder` poly claim is used as input claim for sin lookup sumcheck, together with a node output claim.
        // Both claims are derived from a different opening point, so we need to derive a new claim for one of (`output`, `remainder`).
        // We chose `output` to prevent us from having to use n-to-1 reductions on the `remainder` and rather only implement it on NodeOutput.
        // Making further work for Issue#138 easier.
        let mut provider = AccOpeningAccessor::new(accumulator, &params.computation_node)
            .to_provider(transcript, params.r_node_output.clone());
        provider.append_node_io(Target::Current, output_claim);

        let input_onehot = MultilinearPolynomial::from(input_onehot);
        assert_eq!(input_onehot.len(), sin_table.len());
        let identity = IdentityPolynomial::new(SIN_LOG_TABLE_SIZE);

        #[cfg(test)]
        {
            let remainder_claim = provider.get_advice(VirtualPoly::TeleportRemainder).1;
            let rv_claim = provider.get_node_io(Target::Current).1;
            let claim = (0..input_onehot.len())
                .map(|i| {
                    let a = input_onehot.get_bound_coeff(i);
                    let b = sin_table.get_bound_coeff(i);
                    let int = F::from_u32(i as u32);
                    a * (b + params.gamma * int)
                })
                .sum();
            assert_eq!(rv_claim + params.gamma * remainder_claim, claim)
        }

        Self {
            params,
            sin_table,
            input_onehot,
            identity,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for SinProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            input_onehot,
            sin_table,
            identity,
            ..
        } = self;

        let univariate_poly_evals: [F; 2] = (0..input_onehot.len() / 2)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|i| {
                let ra_evals =
                    input_onehot.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                let table_evals =
                    sin_table.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
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
        self.sin_table.bind_parallel(r_j, BindingOrder::LowToHigh);
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
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .to_provider(transcript, OpeningPoint::new(r));
        provider.append_advice(VirtualPoly::SinRa, self.input_onehot.final_claim());
    }
}

/// Verifier state for sin execution sumcheck instance.
///
/// Verifies that the prover correctly evaluated the Sin lookup table at the teleported remainders,
/// And that the one-hot encoding of the input is correct.
pub struct SinVerifier<F: JoltField> {
    params: SinParams<F>,
    sin_table: MultilinearPolynomial<F>,
}

impl<F: JoltField> SinVerifier<F> {
    /// Initialize the verifier state.
    pub fn new(
        computation_node: ComputationNode,
        graph: &ComputationGraph,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = SinParams::new(computation_node, graph, accumulator, transcript);
        let mut provider = AccOpeningAccessor::new(accumulator, &params.computation_node)
            .to_provider(transcript, params.r_node_output.clone());
        provider.append_node_io(Target::Current);

        let sin_table = MultilinearPolynomial::from(SinTable::materialize());
        Self { params, sin_table }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for SinVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.params.computation_node);
        let opening_point = self
            .params
            .normalize_opening_point(&sumcheck_challenges.into_opening());

        let ra_claim = accessor.get_advice(VirtualPoly::SinRa).1;
        let table_claim = self.sin_table.evaluate(&opening_point.r);
        let int_eval = IdentityPolynomial::new(SIN_LOG_TABLE_SIZE).evaluate(&opening_point.r);

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
        let mut provider = AccOpeningAccessor::new(accumulator, &self.params.computation_node)
            .to_provider(transcript, OpeningPoint::new(r));
        provider.append_advice(VirtualPoly::SinRa);
    }
}

/// Encoding impl for Sin one-hot read-address checks.
///
/// This encodes the relation between the teleportation remainder and the sin lookup table index,
pub struct SinRaEncoding {
    /// Index of the computation node in the trace.
    pub node_idx: usize,
}

impl RaOneHotEncoding for SinRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::SinRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::TeleportRemainder(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::SinRa(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        SIN_LOG_TABLE_SIZE
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), SIN_LOG_TABLE_SIZE)
    }
}

impl<F: JoltField, T: Transcript> NeuralTeleportRangeOneHot<F, T> for Sin {
    type RaEncoding = SinRaEncoding;

    fn lookup_indices(&self, node: &ComputationNode, trace: &Trace) -> Vec<usize> {
        let LayerData { operands, .. } = Trace::layer_data(trace, node);
        let input = operands[0];
        let (_, remainder) = compute_division(input, FOUR_PI_APPROX);
        remainder.par_iter().map(|&x| x as usize).collect()
    }

    fn ra_encoding(&self, node: &ComputationNode) -> Self::RaEncoding {
        SinRaEncoding { node_idx: node.idx }
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

    use super::FOUR_PI_APPROX;

    fn sin_model(input_shape: &[usize]) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(input_shape.to_vec());
        let res = b.sin(i);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_sin_random_inputs() {
        let t = 1 << 13;
        let mut rng = StdRng::seed_from_u64(0xC05);
        let input = Tensor::random_range(&mut rng, &[t], -50000..50000);
        let model = sin_model(&[t]);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_sin_periodic_boundary_inputs() {
        let input = Tensor::new(
            Some(&[
                -FOUR_PI_APPROX - 1,
                -FOUR_PI_APPROX,
                -1,
                0,
                1,
                FOUR_PI_APPROX - 1,
                FOUR_PI_APPROX,
                FOUR_PI_APPROX + 1,
            ]),
            &[8],
        )
        .unwrap();
        let model = sin_model(&[8]);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "TODO: non-power-of-two sin path not fully validated yet"]
    fn test_sin_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0xC06);
        let input = Tensor::random_range(&mut rng, &[t], -50000..50000);
        let model = sin_model(&[t]);
        unit_test_op(model, &[input]);
    }
}
