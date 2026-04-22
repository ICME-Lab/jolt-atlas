use crate::{
    onnx_proof::{
        neural_teleport::{
            division::{
                compute_division, TeleportDivisionParams, TeleportDivisionProver,
                TeleportDivisionVerifier,
            },
            n_bits_to_usize,
            range_and_onehot::{
                prove_range_and_onehot, verify_range_and_onehot, NeuralTeleportRangeOneHot,
            },
            utils::compute_ra_evals_nbits_2comp,
            TanhTable,
        },
        ops::OperatorProofTrait,
        range_checking::{range_check_operands::TeleportRangeCheckOperands, RangeCheckEncoding},
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::AccOpeningAccessor,
};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, Trace},
        ComputationGraph,
    },
    node::{handlers::activation::NEURAL_TELEPORT_LOG_TABLE_SIZE, ComputationNode},
    ops::Tanh,
};
use common::parallel::par_enabled;
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::{IntoOpening, JoltField},
    poly::{
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        teleport_id_poly::TeleportIdPolynomial,
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

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Tanh {
    #[tracing::instrument(skip_all, name = "Tanh::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();

        // Stage 1a: Neural teleportation division proof
        let div_params = TeleportDivisionParams::new(node.clone(), &prover.accumulator, self.tau);
        let mut div_sumcheck = TeleportDivisionProver::new(&prover.trace, div_params);

        // Run division sumcheck first (output claim will be cached)
        let (div_proof, _) = Sumcheck::prove(
            &mut div_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::NeuralTeleport), div_proof));

        // Stage 1b: Tanh lookup proof (uses quotient from division)
        // This must be done AFTER division sumcheck completes
        // so that the quotient opening is cached in the accumulator
        let params = TanhParams::new(
            node.clone(),
            &prover.preprocessing.model.graph,
            &prover.accumulator,
            &mut prover.transcript,
            self.clone(),
        );
        let mut exec_sumcheck = TanhProver::initialize(
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

        results.extend(prove_range_and_onehot(node, prover, self));

        results
    }

    #[tracing::instrument(skip_all, name = "Tanh::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Stage 1a: Division verification
        let div_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::NeuralTeleport))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;

        let div_verifier =
            TeleportDivisionVerifier::new(node.clone(), &verifier.accumulator, self.tau);
        Sumcheck::verify(
            div_proof,
            &div_verifier,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 1b: Tanh verification
        let tanh_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;

        let exec_sumcheck = TanhVerifier::new(
            node.clone(),
            &verifier.preprocessing.model.graph,
            &mut verifier.accumulator,
            &mut verifier.transcript,
            self.clone(),
        );
        Sumcheck::verify(
            tanh_proof,
            &exec_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        verify_range_and_onehot(node, verifier, self)?;

        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let tanh_encoding = TanhRaEncoding {
            node_idx: node.idx,
            log_table: NEURAL_TELEPORT_LOG_TABLE_SIZE,
        };
        let rc_encoding = RangeCheckEncoding::<TeleportRangeCheckOperands>::new(node);
        let tanh_d = tanh_encoding.one_hot_params().instruction_d;
        let rc_d = rc_encoding.one_hot_params().instruction_d;
        let mut polys = vec![];
        polys.extend((0..tanh_d).map(|i| CommittedPoly::TanhRaD(node.idx, i)));
        polys.extend((0..rc_d).map(|i| CommittedPoly::TeleportRangeCheckRaD(node.idx, i)));
        polys
    }
}

const DEGREE_BOUND: usize = 2;

/// Parameters for proving hyperbolic tangent (tanh) activation operations.
///
/// Tanh uses a lookup table approach with range checking. The folding challenge gamma
/// is used to combine multiple checks.
#[derive(Clone)]
pub struct TanhParams<F: JoltField> {
    gamma: F,
    r_node_output: OpeningPoint<BIG_ENDIAN, F>,
    computation_node: ComputationNode,
    op: Tanh,
}

impl<F: JoltField> TanhParams<F> {
    /// Create new tanh parameters from a computation node, graph, accumulator, transcript, and operation.
    pub fn new(
        computation_node: ComputationNode,
        _graph: &ComputationGraph,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        op: Tanh,
    ) -> Self {
        let accessor = AccOpeningAccessor::new(accumulator, &computation_node);
        let gamma = transcript.challenge_scalar();
        let r_node_output = accessor.get_reduced_opening().0;

        Self {
            gamma,
            r_node_output,
            computation_node,
            op,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for TanhParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let accessor = AccOpeningAccessor::new(accumulator, &self.computation_node);
        let rv_claim = accessor.get_reduced_opening().1;

        // Use quotient claim instead of input claim (neural teleportation)
        let quotient_id = OpeningId::new(
            VirtualPoly::TeleportQuotient(self.computation_node.idx),
            SumcheckId::Raf,
        );
        let quotient_claim = accessor.get_custom(quotient_id).1;

        rv_claim + self.gamma * quotient_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.op.log_table
    }
}

/// Prover state for tanh activation sumcheck protocol.
///
/// This implements a Read-Raf sumcheck for Tanh lookup, asserting that each output[i] = TanhTable[input[i]]
/// where input[i] = Ra[k] * Int[k] with Ra as one-hot encoding and Int as custom identity polynomial.
pub struct TanhProver<F: JoltField> {
    params: TanhParams<F>,
    tanh_table: MultilinearPolynomial<F>,
    input_onehot: MultilinearPolynomial<F>,
    identity: TeleportIdPolynomial<F>,
}

impl<F: JoltField> TanhProver<F> {
    /// Initialize the prover with trace data, parameters, accumulator, and transcript.
    pub fn initialize(
        trace: &Trace,
        params: TanhParams<F>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let LayerData { operands, .. } = Trace::layer_data(trace, &params.computation_node);
        let input = operands[0];

        // Compute quotient from division (neural teleportation)
        let (quotient_tensor, _remainder) = compute_division(input, params.op.tau);

        // Ensure input is within expected range for table size: 2^(log_table_size - 1) <= input < 2^(log_table_size - 1)
        // Inputs outside this range will error
        // TODO: Pass these input in a clamping lookup table, since anyway tanh(±∞) = ±1, so we only need to handle a limited input range.
        assert!(quotient_tensor.iter().all(|&x| {
            let lower_bound = -(1 << (params.op.log_table - 1));
            let upper_bound = (1 << (params.op.log_table - 1)) - 1;
            x >= lower_bound && x <= upper_bound
        }));

        // Create and materialize the tanh lookup table (reduced size)
        let tanh_table = TanhTable::new(params.op.log_table, params.op.tau);
        let tanh_table = MultilinearPolynomial::from(tanh_table.materialize());

        // Compute one-hot encoding of QUOTIENT values (not input)
        let input_onehot: Vec<F> = compute_ra_evals_nbits_2comp(
            &params.r_node_output.r,
            &quotient_tensor,
            params.op.log_table,
        );

        // Cache quotient claim (used in tanh lookup)
        // We do not reuse the claim from the division sumcheck, because the opening point is different
        // TODO(AntoineF4C5): Reuse the quotient claim from proving division.
        // TODO(ClankPan): erf.rs has a similar implementation
        // REQUIRED:
        // - Computing an opening for output at same opening point than quotient tensor (and later perfom n-to-1 opening reduction).
        // - Handling the difference between polynomials built from u32 and i32 tensors,
        //   Namely we currently always use polynomials built from i32 tensors, except for raf-checking.
        let quotient_claim = MultilinearPolynomial::from(quotient_tensor.into_container_data()) // TODO: unify tensor representations (always i32 or always u32)
            .evaluate(&params.r_node_output.r);
        let mut provider = AccOpeningAccessor::new(accumulator, &params.computation_node)
            .into_provider(transcript, params.r_node_output.clone());
        // Edge case where we need to insert for SumcheckId::Raf sumcheck
        // TODO(AntoineF4C5): Clean once #208 is dealt with
        let raf_opening_id = OpeningId::new(
            VirtualPoly::TeleportQuotient(params.computation_node.idx),
            SumcheckId::Raf,
        );
        provider.append_custom(raf_opening_id, quotient_claim);

        let input_onehot = MultilinearPolynomial::from(input_onehot);
        assert_eq!(input_onehot.len(), tanh_table.len());
        let identity = TeleportIdPolynomial::new(params.op.log_table);

        #[cfg(test)]
        {
            let quotient_claim = MultilinearPolynomial::from(quotient_tensor.into_container_data())
                .evaluate(&params.r_node_output.r);
            let rv_claim = provider.get_reduced_opening().1;
            let claim = (0..input_onehot.len())
                .map(|i| {
                    use crate::onnx_proof::neural_teleport::usize_to_n_bits;

                    let a = input_onehot.get_bound_coeff(i);
                    let b = tanh_table.get_bound_coeff(i);
                    let int = F::from_u32(usize_to_n_bits(i, params.op.log_table) as u32);
                    a * (b + params.gamma * int)
                })
                .sum();
            assert_eq!(rv_claim + params.gamma * quotient_claim, claim)
        }

        Self {
            params,
            tanh_table,
            input_onehot,
            identity,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for TanhProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            input_onehot,
            tanh_table,
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
                    tanh_table.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
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
        self.tanh_table.bind_parallel(r_j, BindingOrder::LowToHigh);
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
            .into_provider(transcript, OpeningPoint::new(r));
        provider.append_advice(VirtualPoly::TanhRa, self.input_onehot.final_claim());
    }
}

/// Verifier for tanh activation sumcheck protocol.
///
/// Verifies that the prover's sumcheck messages are consistent with the claimed
/// tanh activation lookup table output.
pub struct TanhVerifier<F: JoltField> {
    params: TanhParams<F>,
    tanh_table: MultilinearPolynomial<F>,
}

impl<F: JoltField> TanhVerifier<F> {
    /// Create a new verifier for the tanh operation.
    pub fn new(
        computation_node: ComputationNode,
        graph: &ComputationGraph,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        op: Tanh,
    ) -> Self {
        let params = TanhParams::new(computation_node, graph, accumulator, transcript, op);

        // Cache quotient polynomial opening
        let mut provider = AccOpeningAccessor::new(accumulator, &params.computation_node)
            .into_provider(transcript, params.r_node_output.clone());
        // Edge case where we need to insert for SumcheckId::Raf sumcheck
        // TODO(AntoineF4C5): Clean once #208 is dealt with
        let raf_opening_id = OpeningId::new(
            VirtualPoly::TeleportQuotient(params.computation_node.idx),
            SumcheckId::Raf,
        );
        provider.append_custom(raf_opening_id);

        // Materialize the tanh table for verification
        let tanh_table = TanhTable::new(params.op.log_table, params.op.tau);
        let tanh_table = MultilinearPolynomial::from(tanh_table.materialize());

        Self { params, tanh_table }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for TanhVerifier<F> {
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

        let ra_claim = accessor.get_advice(VirtualPoly::TanhRa).1;

        // Evaluate tanh table at the opening point
        let table_claim = self.tanh_table.evaluate(&opening_point.r);

        let int_eval =
            TeleportIdPolynomial::new(self.params.op.log_table).evaluate(&opening_point.r);

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
            .into_provider(transcript, OpeningPoint::new(r));
        provider.append_advice(VirtualPoly::TanhRa);
    }
}

// ---------------------------------------------------------------------------
// TanhRaEncoding — implements RaOneHotEncoding for Tanh's stage-2 one-hot checks
// ---------------------------------------------------------------------------

/// Encoding impl for tanh one-hot checking.
///
/// Used in the stage-2 one-hot checks for tanh lookup table accesses.
pub struct TanhRaEncoding {
    /// Index of the computation node this encoding belongs to.
    pub node_idx: usize,
    /// Log2 of the lookup table size.
    pub log_table: usize,
}

impl RaOneHotEncoding for TanhRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::TanhRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::TeleportQuotient(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::TanhRa(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        self.log_table
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.log_table)
    }
}

impl<F: JoltField, T: Transcript> NeuralTeleportRangeOneHot<F, T> for Tanh {
    type RaEncoding = TanhRaEncoding;

    fn lookup_indices(&self, node: &ComputationNode, trace: &Trace) -> Vec<usize> {
        let LayerData { operands, .. } = Trace::layer_data(trace, node);
        let input = operands[0];
        let (quotient, _remainder) = compute_division(input, self.tau);
        quotient
            .par_iter()
            .map(|&x| n_bits_to_usize(x, self.log_table))
            .collect()
    }

    fn ra_encoding(&self, node: &ComputationNode) -> Self::RaEncoding {
        TanhRaEncoding {
            node_idx: node.idx,
            log_table: self.log_table,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        node::handlers::activation::NEURAL_TELEPORT_LOG_TABLE_SIZE,
        tensor::Tensor,
    };
    use rand::{rngs::StdRng, SeedableRng};

    fn tanh_model(input_shape: &[usize]) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(input_shape.to_vec());
        let res = b.tanh(i);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_tanh() {
        let T = 1 << 14;
        const MIN_INPUT_VALUE: i32 = -(1 << (NEURAL_TELEPORT_LOG_TABLE_SIZE - 1));
        const MAX_INPUT_VALUE: i32 = 1 << (NEURAL_TELEPORT_LOG_TABLE_SIZE - 1);
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::random_range(&mut rng, &[T], MIN_INPUT_VALUE..MAX_INPUT_VALUE);
        let model = tanh_model(&[T]);
        unit_test_op(model, &[input]);
    }

    #[test]
    #[ignore = "non-power-of-two path not fully supported yet"]
    fn test_tanh_non_power_of_two_input_len() {
        let t = 1000;
        const MIN_INPUT_VALUE: i32 = -(1 << (NEURAL_TELEPORT_LOG_TABLE_SIZE - 1));
        const MAX_INPUT_VALUE: i32 = 1 << (NEURAL_TELEPORT_LOG_TABLE_SIZE - 1);
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::random_range(&mut rng, &[t], MIN_INPUT_VALUE..MAX_INPUT_VALUE);
        let model = tanh_model(&[t]);
        unit_test_op(model, &[input]);
    }
}
