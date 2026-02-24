use crate::onnx_proof::neural_teleport::{
    division::{
        compute_division, TeleportDivisionParams, TeleportDivisionProver, TeleportDivisionVerifier,
    },
    erf::ErfTable,
    n_bits_to_usize,
};
use crate::onnx_proof::{
    ops::{tanh::compute_ra_evals, OperatorProofTrait},
    range_checking::{
        range_check_operands::TeleportRangeCheckOperands, RangeCheckEncoding, RangeCheckProvider,
    },
    ProofId, ProofType, Prover, Verifier,
};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, Trace},
        ComputationGraph,
    },
    node::{handlers::activation::NEURAL_TELEPORT_LOG_TABLE_SIZE, ComputationNode},
    ops::Erf,
};
use common::{consts::XLEN, CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::JoltField,
    lookup_tables::unsigned_less_than::UnsignedLessThanTable,
    poly::{
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        teleport_id_poly::TeleportIdPolynomial,
        unipoly::UniPoly,
    },
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::BatchedSumcheck,
        sumcheck::{Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Erf {
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();

        // Stage 1a: Neural teleportation division proof
        let div_params = TeleportDivisionParams::new(node.clone(), &prover.accumulator, self.tau);
        let mut div_sumcheck = TeleportDivisionProver::new(&prover.trace, div_params);
        let (div_proof, _) = Sumcheck::prove(
            &mut div_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::NeuralTeleport), div_proof));

        // Stage 1b: Erf lookup proof
        let params = ErfParams::new(
            node.clone(),
            &prover.preprocessing.model.graph,
            &prover.accumulator,
            &mut prover.transcript,
            self.clone(),
        );
        let mut exec_sumcheck = ErfProver::initialize(
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

        let LayerData { operands, .. } = Trace::layer_data(&prover.trace, node);
        let input = operands[0];

        // Compute quotient for neural teleportation
        let (quotient, _remainder) = compute_division(input, self.tau);
        let lookup_indices = quotient
            .par_iter()
            .map(|&x| n_bits_to_usize(x, self.log_table))
            .collect::<Vec<usize>>();

        // Stage 2: Range check proof for division and first One-Hot checks for ErfRa
        let rangecheck_provider = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node);
        let (rangecheck_sumcheck, rc_lookup_indices) = rangecheck_provider
            .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );
        let erf_encoding = ErfRaEncoding {
            node_idx: node.idx,
            log_table: self.log_table,
        };
        let ra_onehot_provers = shout::ra_onehot_provers(
            &erf_encoding,
            &lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![Box::new(rangecheck_sumcheck)];
        instances.extend(ra_onehot_provers);

        let (erf_ra_one_hot_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((
            ProofId(node.idx, ProofType::RaOneHotChecks),
            erf_ra_one_hot_proof,
        ));

        // Stage 3: one-hot checks for division
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
        results.push((
            ProofId(node.idx, ProofType::RaHammingWeight),
            ra_one_hot_proof,
        ));

        results
    }

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

        // Stage 1b: Erf verification
        let erf_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;

        let exec_sumcheck = ErfVerifier::new(
            node.clone(),
            &verifier.preprocessing.model.graph,
            &mut verifier.accumulator,
            &mut verifier.transcript,
            self.clone(),
        );
        Sumcheck::verify(
            erf_proof,
            &exec_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 2: Range check verification for division and first One-Hot checks for ErfRa
        let erf_ra_one_hot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;

        let rangecheck_provider = RangeCheckProvider::<TeleportRangeCheckOperands>::new(node);
        let rangecheck_verifier = rangecheck_provider
            .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
                &mut verifier.accumulator,
                &mut verifier.transcript,
            );
        let erf_encoding = ErfRaEncoding {
            node_idx: node.idx,
            log_table: self.log_table,
        };
        let ra_onehot_verifier = shout::ra_onehot_verifiers(
            &erf_encoding,
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        let ra_onehot_verifier: Vec<&dyn SumcheckInstanceVerifier<F, T>> =
            ra_onehot_verifier.iter().map(|v| &**v as _).collect();
        let mut instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> = vec![&rangecheck_verifier];
        instances.extend(ra_onehot_verifier);
        BatchedSumcheck::verify(
            erf_ra_one_hot_proof,
            instances,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 3: one-hot check verification for division
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

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPolynomial> {
        let erf_encoding = ErfRaEncoding {
            node_idx: node.idx,
            log_table: NEURAL_TELEPORT_LOG_TABLE_SIZE,
        };
        let rc_encoding = RangeCheckEncoding::<TeleportRangeCheckOperands>::new(node);
        let erf_d = erf_encoding.one_hot_params().instruction_d;
        let rc_d = rc_encoding.one_hot_params().instruction_d;
        let mut polys = vec![];
        polys.extend((0..erf_d).map(|i| CommittedPolynomial::ErfRaD(node.idx, i)));
        polys.extend((0..rc_d).map(|i| CommittedPolynomial::TeleportRangeCheckRaD(node.idx, i)));
        polys
    }
}

const DEGREE_BOUND: usize = 2; // TODO

/// Parameters for proving error function (erf) activation operations.
///
/// Mirrors the parameter layout used by `TanhParams` so both lookup-style
/// activations follow the same transcript/challenge flow.
#[derive(Clone)]
pub struct ErfParams<F: JoltField> {
    /// Folding challenge used to combine multiple checks into one claim.
    pub gamma: F,
    /// Opening point sampled from the output claim of this node.
    pub r_node_output: Vec<F::Challenge>,
    /// Computation node currently being proven.
    pub computation_node: ComputationNode,
    /// Operator parameters (e.g. fixed-point scale for erf).
    pub op: Erf,
    /// Phantom marker for the field type.
    pub _marker: core::marker::PhantomData<F>,
}

impl<F: JoltField> ErfParams<F> {
    /// Build erf parameters from the current accumulator/transcript state.
    ///
    /// This samples a fresh folding challenge and reuses the node-output opening
    /// point already present in the accumulator, matching the tanh flow.
    pub fn new(
        computation_node: ComputationNode,
        _graph: &ComputationGraph,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        op: Erf,
    ) -> Self {
        let gamma = transcript.challenge_scalar();
        let r_node_output = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(computation_node.idx),
                SumcheckId::Execution,
            )
            .0
            .r;

        Self {
            gamma,
            r_node_output,
            computation_node,
            op,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ErfParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        let rv_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutput(self.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

        let quotient_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::TeleportQuotient(self.computation_node.idx),
                SumcheckId::Raf,
            )
            .1;

        rv_claim + self.gamma * quotient_claim
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    fn num_rounds(&self) -> usize {
        self.op.log_table
    }
}

/// Prover state for erf activation sumcheck protocol.
///
/// This implements a Read-Raf sumcheck for Erf lookup, asserting that each output[i] = ErfTable[input[i]]
/// where input[i] = Ra[k] * Int[k] with Ra as one-hot encoding and Int as custom identity polynomial.
pub struct ErfProver<F: JoltField> {
    pub params: ErfParams<F>,
    pub erf_table: MultilinearPolynomial<F>,
    pub input_onehot: MultilinearPolynomial<F>,
    pub identity: TeleportIdPolynomial<F>,
}

impl<F: JoltField> ErfProver<F> {
    /// Initialize the prover with trace data, parameters, accumulator, and transcript.
    pub fn initialize(
        trace: &Trace,
        params: ErfParams<F>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let LayerData { operands, .. } = Trace::layer_data(trace, &params.computation_node);
        let input = operands[0];

        let (quotient_tensor, _remainder) = compute_division(input, params.op.tau);

        // Ensure input is within expected range for table size: 2^(log_table_size - 1) <= input < 2^(log_table_size - 1)
        // Inputs outside this range will error
        // TODO: Pass these input in a clamping lookup table, since anyway tanh(±∞) = ±1, so we only need to handle a limited input range.
        assert!(quotient_tensor.iter().all(|&x| {
            let lower_bound = -(1 << (params.op.log_table - 1));
            let upper_bound = (1 << (params.op.log_table - 1)) - 1;
            x >= lower_bound && x <= upper_bound
        }));

        // Create and materialize the erf lookup table (reduced size)
        let erf_table = ErfTable::new(params.op.log_table);
        let erf_table = MultilinearPolynomial::from(erf_table.materialize());

        // Use the compute_ra_evals in tanh.rs
        let input_onehot: Vec<F> =
            compute_ra_evals(&params.r_node_output, &quotient_tensor, params.op.log_table);

        // TODO(ClankPan): Follow up on the TODOs in tanh.rs.
        let quotient_claim = MultilinearPolynomial::from(quotient_tensor.into_container_data())
            .evaluate(&params.r_node_output);
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::TeleportQuotient(params.computation_node.idx),
            SumcheckId::Raf,
            params.r_node_output.clone().into(),
            quotient_claim,
        );

        let input_onehot = MultilinearPolynomial::from(input_onehot);
        assert_eq!(input_onehot.len(), erf_table.len());
        let identity = TeleportIdPolynomial::new(params.op.log_table);

        Self {
            params,
            erf_table,
            input_onehot,
            identity,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for ErfProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let Self {
            input_onehot,
            erf_table,
            identity,
            ..
        } = self;

        let univariate_poly_evals: [F; 2] = (0..input_onehot.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals =
                    input_onehot.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
                let table_evals =
                    erf_table.sumcheck_evals(i, DEGREE_BOUND, BindingOrder::LowToHigh);
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
        self.erf_table.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.identity.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let r = [opening_point.r.as_slice(), &self.params.r_node_output].concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::ErfRa(self.params.computation_node.idx),
            SumcheckId::Execution,
            r.into(),
            self.input_onehot.final_sumcheck_claim(),
        );
    }
}

pub struct ErfVerifier<F: JoltField> {
    pub params: ErfParams<F>,
    pub erf_table: MultilinearPolynomial<F>,
}

impl<F: JoltField> ErfVerifier<F> {
    pub fn new(
        computation_node: ComputationNode,
        graph: &ComputationGraph,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
        op: Erf,
    ) -> Self {
        let params = ErfParams::new(computation_node, graph, accumulator, transcript, op);

        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::TeleportQuotient(params.computation_node.idx),
            SumcheckId::Raf,
            params.r_node_output.clone().into(),
        );

        let erf_table = ErfTable::new(params.op.log_table);
        let erf_table = MultilinearPolynomial::from(erf_table.materialize());

        Self { params, erf_table }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T> for ErfVerifier<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

        let ra_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::ErfRa(self.params.computation_node.idx),
                SumcheckId::Execution,
            )
            .1;

        // Evaluate erf table at the opening point
        let table_claim = self.erf_table.evaluate(&opening_point.r);

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
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        let r = [opening_point.r.as_slice(), &self.params.r_node_output].concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::ErfRa(self.params.computation_node.idx),
            SumcheckId::Execution,
            r.into(),
        );
    }
}

// ---------------------------------------------------------------------------
// ErfRaEncoding — implements RaOneHotEncoding for Erf's stage-2 one-hot checks
// ---------------------------------------------------------------------------

/// Encoding impl for erf one-hot checking.
///
/// Used in the stage-2 one-hot checks for erf lookup table accesses.
pub struct ErfRaEncoding {
    /// Index of the computation node this encoding belongs to.
    pub node_idx: usize,
    /// Log2 of the lookup table size.
    pub log_table: usize,
}

impl RaOneHotEncoding for ErfRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPolynomial {
        CommittedPolynomial::ErfRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::TeleportQuotient(self.node_idx),
            SumcheckId::Execution,
        )
    }

    fn ra_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::ErfRa(self.node_idx),
            SumcheckId::Execution,
        )
    }

    fn log_k(&self) -> usize {
        self.log_table
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.log_table)
    }
}
