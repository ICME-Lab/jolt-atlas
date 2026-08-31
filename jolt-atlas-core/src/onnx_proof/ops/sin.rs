use crate::onnx_proof::{
    neural_teleport::{
        division::{
            cache_teleport_input_claim_prove, cache_teleport_quotient_prove,
            cache_teleport_quotient_verify, compute_division, verify_teleport_input_claim,
        },
        sin::{SinTable, SIN_TABLE_VARS},
        trig_downscale::{cache_downscaled_prove, cache_downscaled_verify, TrigDownscaleOperands},
        utils::compute_ra_evals_from_usize_indices,
    },
    op_lookups::{OpLookupEncoding, OpLookupProvider},
    ops::OperatorProofTrait,
    range_checking::{range_check_operands::TeleportRangeCheckOperands, RangeCheckProvider},
    ProofId, ProofType, Prover, Verifier,
};
use crate::utils::opening_access::AccOpeningAccessor;
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, Trace},
        ComputationGraph,
    },
    node::ComputationNode,
    ops::Sin,
    tensor::Tensor,
};
use common::consts::{MODEL_SCALE, TRIG_PERIOD_MODULUS, XLEN};
use common::parallel::par_enabled;
use common::{CommittedPoly, VirtualPoly};
#[cfg(feature = "zk")]
use joltworks::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::{IntoOpening, JoltField},
    lookup_tables::{right_shift::RightShiftTable, unsigned_less_than::UnsignedLessThanTable},
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
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
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
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        assert_eq!(
            self.scale, MODEL_SCALE as i32,
            "Sin teleportation proving is only calibrated for MODEL_SCALE={MODEL_SCALE} \
             (got {})",
            self.scale
        );
        let mut results = Vec::new();

        // Plumbing: cache the quotient's claim.
        cache_teleport_quotient_prove(node, prover, TRIG_PERIOD_MODULUS as i32);

        // Plumbing: downscale.
        let LayerData { operands, .. } = Trace::layer_data(&prover.trace, node);
        let input = operands[0];
        let (_quotient, remainder) = compute_division(input, TRIG_PERIOD_MODULUS as i32);
        let downscaled = cache_downscaled_prove(node, prover, &remainder);

        // Stage 1: batch every read-raf "value" sumcheck together — downscale, the Sin
        // table lookup, and the remainder range-check.
        let dsc_provider: OpLookupProvider<TrigDownscaleOperands> =
            OpLookupProvider::new(node.clone());
        let (dsc_exec, dsc_lookup_indices) = dsc_provider
            .read_raf_prove::<F, T, RightShiftTable<XLEN>, XLEN>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );

        let params = SinParams::new(
            node.clone(),
            &prover.preprocessing.model.graph,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let (sin_exec, sin_lookup_indices) =
            SinProver::initialize(&downscaled, params, &mut prover.accumulator);

        let rc_provider = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node);
        let (rangecheck_exec, rc_lookup_indices) = rc_provider
            .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );

        let mut exec_instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = vec![
            Box::new(dsc_exec),
            Box::new(sin_exec),
            Box::new(rangecheck_exec),
        ];
        let (exec_proof, _) = BatchedSumcheck::prove(
            exec_instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), exec_proof));

        // Stage 2: batch every one-hot correctness triple together — downscale, Sin,
        // and the range-check.
        let dsc_encoding = dsc_provider.encoding();
        let sin_encoding = SinRaEncoding::new(node);
        let rc_encoding = rc_provider.encoding();

        let mut onehot_instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = Vec::new();
        onehot_instances.extend(shout::ra_onehot_provers(
            &dsc_encoding,
            &dsc_lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        ));
        onehot_instances.extend(shout::ra_onehot_provers(
            &sin_encoding,
            &sin_lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        ));
        onehot_instances.extend(shout::ra_onehot_provers(
            &rc_encoding,
            &rc_lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        ));
        let (onehot_proof, _) = BatchedSumcheck::prove(
            onehot_instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::RaOneHotChecks), onehot_proof));

        // Plumbing: derive `a`'s input claim now that `q` and `r` both hold claims at the
        // node's reduced output-opening point.
        cache_teleport_input_claim_prove(node, prover, TRIG_PERIOD_MODULUS as i32);

        results
    }

    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        if self.scale != MODEL_SCALE as i32 {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "Sin teleportation verifying is only calibrated for MODEL_SCALE={MODEL_SCALE} \
                     (got {})",
                self.scale
            )));
        }
        // Plumbing: cache the quotient's claim.
        cache_teleport_quotient_verify(node, verifier);

        // Plumbing: downscale.
        cache_downscaled_verify(node, verifier);

        // Stage 1: batch every read-raf "value" sumcheck together — downscale, the Sin
        // table lookup, and the remainder range-check.
        let dsc_provider: OpLookupProvider<TrigDownscaleOperands> =
            OpLookupProvider::new(node.clone());
        let dsc_verifier = dsc_provider.read_raf_verify::<F, T, RightShiftTable<XLEN>, XLEN>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );

        let exec_sumcheck = SinVerifier::new(
            node.clone(),
            &verifier.preprocessing.model.graph,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );

        let rc_provider = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node);
        let rangecheck_verifier = rc_provider.read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );

        let sin_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        BatchedSumcheck::verify(
            sin_proof,
            vec![&dsc_verifier, &exec_sumcheck, &rangecheck_verifier],
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 2: batch every one-hot correctness triple together — downscale, Sin,
        // and the range-check.
        let dsc_encoding = dsc_provider.encoding();
        let sin_encoding = SinRaEncoding::new(node);
        let rc_encoding = rc_provider.encoding();

        let onehot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let mut onehot_verifiers: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> = Vec::new();
        onehot_verifiers.extend(shout::ra_onehot_verifiers(
            &dsc_encoding,
            &verifier.accumulator,
            &mut verifier.transcript,
        ));
        onehot_verifiers.extend(shout::ra_onehot_verifiers(
            &sin_encoding,
            &verifier.accumulator,
            &mut verifier.transcript,
        ));
        onehot_verifiers.extend(shout::ra_onehot_verifiers(
            &rc_encoding,
            &verifier.accumulator,
            &mut verifier.transcript,
        ));
        BatchedSumcheck::verify(
            onehot_proof,
            onehot_verifiers.iter().map(|v| &**v as _).collect(),
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Plumbing: derive `a`'s input claim now that `q` and `r` both hold claims at the
        // node's reduced output-opening point.
        verify_teleport_input_claim(node, verifier, TRIG_PERIOD_MODULUS as i32)?;

        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let sin_encoding = SinRaEncoding::new(node);
        let rc_encoding = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node).encoding();
        let dsc_encoding: OpLookupEncoding<TrigDownscaleOperands> =
            OpLookupProvider::<TrigDownscaleOperands>::new(node.clone()).encoding();
        let sin_d = sin_encoding.one_hot_params().instruction_d;
        let rc_d = rc_encoding.one_hot_params().instruction_d;
        let dsc_d = dsc_encoding.one_hot_params().instruction_d;
        let mut polys = vec![CommittedPoly::TeleportNodeQuotient(node.idx)];
        polys.extend((0..dsc_d).map(|i| CommittedPoly::TrigDownscaleRaD(node.idx, i)));
        polys.extend((0..sin_d).map(|i| CommittedPoly::SinRaD(node.idx, i)));
        polys.extend((0..rc_d).map(|i| CommittedPoly::TeleportRangeCheckRaD(node.idx, i)));
        polys
    }
}

const DEGREE_BOUND: usize = 2;

#[derive(Clone)]
/// Parameters for the sine trigonometric function.
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
        let (r_node_output, _) = accessor.get_reduced_opening();

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
        let rv_claim = accessor.get_reduced_opening().1;
        let downscaled_claim = accessor.get_advice(VirtualPoly::TrigDownscaled).1;

        rv_claim + self.gamma * downscaled_claim
    }

    fn normalize_opening_point(&self, challenges: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        SIN_TABLE_VARS
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

    // output = ra_claim * (table_claim + gamma * int_eval)
    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        use crate::utils::opening_access::OpeningIdBuilder;
        let builder = OpeningIdBuilder::new(&self.computation_node);
        let ra_id = builder.advice(VirtualPoly::SinRa);
        Some(OutputClaimConstraint::sum_of_products(vec![
            ProductTerm::scaled(ValueSource::Challenge(0), vec![ValueSource::Opening(ra_id)]),
        ]))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let opening_point = self.normalize_opening_point(&sumcheck_challenges.into_opening());
        let sin_table = MultilinearPolynomial::from(SinTable::materialize());
        let table_claim = sin_table.evaluate(&opening_point.r);
        let int_eval = IdentityPolynomial::new(SIN_TABLE_VARS).evaluate(&opening_point.r);
        vec![table_claim + self.gamma * int_eval]
    }
}

/// Read-Raf sumcheck prover for the Sin table lookup: `output[i] = SinTable[downscaled[i]]`.
pub struct SinProver<F: JoltField> {
    params: SinParams<F>,
    sin_table: MultilinearPolynomial<F>,
    input_onehot: MultilinearPolynomial<F>,
    identity: IdentityPolynomial<F>,
}

impl<F: JoltField> SinProver<F> {
    /// Initialize the prover state.
    pub fn initialize(
        downscaled: &Tensor<i32>,
        params: SinParams<F>,
        _accumulator: &mut ProverOpeningAccumulator<F>,
    ) -> (Self, Vec<usize>) {
        assert!(downscaled
            .iter()
            .all(|&x| (0..SinTable::table_size() as i32).contains(&x)));

        let sin_table = MultilinearPolynomial::from(SinTable::materialize());
        let lookup_indices: Vec<usize> = downscaled
            .par_iter()
            .with_min_len(par_enabled())
            .map(|&x| x as usize)
            .collect();
        let input_onehot: Vec<F> = compute_ra_evals_from_usize_indices(
            &params.r_node_output.r,
            &lookup_indices,
            1 << SIN_TABLE_VARS,
        );

        let input_onehot = MultilinearPolynomial::from(input_onehot);
        assert_eq!(input_onehot.len(), sin_table.len());
        let identity = IdentityPolynomial::new(SIN_TABLE_VARS);

        #[cfg(test)]
        {
            let accessor = AccOpeningAccessor::new(_accumulator, &params.computation_node);
            let downscaled_claim = accessor.get_advice(VirtualPoly::TrigDownscaled).1;
            let rv_claim = accessor.get_reduced_opening().1;
            let claim = (0..input_onehot.len())
                .map(|i| {
                    let a = input_onehot.get_bound_coeff(i);
                    let b = sin_table.get_bound_coeff(i);
                    let int = F::from_u32(i as u32);
                    a * (b + params.gamma * int)
                })
                .sum();
            assert_eq!(rv_claim + params.gamma * downscaled_claim, claim)
        }

        (
            Self {
                params,
                sin_table,
                input_onehot,
                identity,
            },
            lookup_indices,
        )
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
            .into_provider(transcript, OpeningPoint::new(r));
        provider.append_advice(VirtualPoly::SinRa, self.input_onehot.final_claim());
    }
}

/// Read-Raf sumcheck verifier for the Sin table lookup.
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
        let int_eval = IdentityPolynomial::new(SIN_TABLE_VARS).evaluate(&opening_point.r);

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
        provider.append_advice(VirtualPoly::SinRa);
    }
}

/// One-hot read-address encoding for the Sin table lookup.
pub struct SinRaEncoding {
    /// Index of the computation node in the trace.
    pub node_idx: usize,
}

impl SinRaEncoding {
    /// Create a new Sin table one-hot encoding for the given computation node.
    pub fn new(node: &ComputationNode) -> Self {
        Self { node_idx: node.idx }
    }
}

impl RaOneHotEncoding for SinRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::SinRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::TrigDownscaled(self.node_idx),
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
        SIN_TABLE_VARS
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), SIN_TABLE_VARS)
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

    use common::consts::TRIG_PERIOD_MODULUS;

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
        let m = TRIG_PERIOD_MODULUS as i32;
        let input = Tensor::new(Some(&[-m - 1, -m, -1, 0, 1, m - 1, m, m + 1]), &[8]).unwrap();
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
