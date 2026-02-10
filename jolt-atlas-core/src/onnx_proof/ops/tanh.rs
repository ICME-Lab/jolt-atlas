use crate::onnx_proof::{
    neural_teleport::{
        division::{
            compute_division, TeleportDivisionParams, TeleportDivisionProver,
            TeleportDivisionVerifier,
        },
        n_bits_to_usize,
        tanh::TanhTable,
    },
    ops::OperatorProofTrait,
    range_checking::{
        self,
        read_raf_checking::{RangecheckRafSumcheckProver, RangecheckRafSumcheckVerifier},
        sumcheck_instance::TeleportRangeCheckOperands,
    },
    ProofId, ProofType, Prover, Verifier,
};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, Trace},
        ComputationGraph,
    },
    node::ComputationNode,
    ops::Tanh,
    tensor::Tensor,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::config::OneHotParams;
use joltworks::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
            VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
        },
        unipoly::UniPoly,
    },
    subprotocols::{
        booleanity::{
            BooleanitySumcheckParams, BooleanitySumcheckVerifier, SmallBooleanitySumcheckProver,
        },
        hamming_booleanity::{
            HammingBooleanitySumcheckParams, HammingBooleanitySumcheckProver,
            HammingBooleanitySumcheckVerifier,
        },
        hamming_weight::{
            HammingWeightSumcheckParams, HammingWeightSumcheckProver, HammingWeightSumcheckVerifier,
        },
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math, thread::unsafe_allocate_zero_vec},
};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    slice::ParallelSlice,
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Tanh {
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();

        // Stage 1a: Neural teleportation division proof
        let div_params = TeleportDivisionParams::new(node.clone(), &prover.accumulator);
        let mut div_sumcheck = TeleportDivisionProver::new(&prover.trace, div_params);

        // Run division sumcheck first (output claim will be cached)
        let (div_proof, _) = Sumcheck::prove(
            &mut div_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), div_proof));

        // Stage 1b: Tanh lookup proof (uses quotient from division)
        // This must be done AFTER division sumcheck completes
        // so that the quotient opening is cached in the accumulator
        let params = TanhParams::new(
            node.clone(),
            &prover.preprocessing.model.graph,
            &prover.accumulator,
            &mut prover.transcript,
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

        results.push((ProofId(node.idx, ProofType::TanhExecution), exec_proof));

        // Stage 1b: Range check proof for division
        let mut rangecheck_sumcheck =
            RangecheckRafSumcheckProver::<_, TeleportRangeCheckOperands>::new_from_prover(
                node, prover,
            );
        let (rangecheck_proof, _) = Sumcheck::prove(
            &mut rangecheck_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::RangeCheck), rangecheck_proof));

        // Stage 2: RaOneHotChecks proof for division range checking
        let log_T = node.num_output_elements().log_2();
        let one_hot_params = OneHotParams::new(log_T);
        let (ra_sumcheck, hw_sumcheck, bool_sumcheck) =
            range_checking::new_ra_one_hot_sumcheck_provers::<F, TeleportRangeCheckOperands>(
                node.clone(),
                &one_hot_params,
                prover,
            );
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> = vec![
            Box::new(ra_sumcheck),
            Box::new(bool_sumcheck),
            Box::new(hw_sumcheck),
        ];
        let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((
            ProofId(node.idx, ProofType::RaOneHotChecks),
            ra_one_hot_proof,
        ));

        // Stage 3: TanhRa one-hot checks (booleanity + hamming booleanity)
        let (tanh_hb_prover, tanh_bool_prover) = build_stage3_provers::<F>(node, prover);
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![Box::new(tanh_hb_prover), Box::new(tanh_bool_prover)];
        let (tanh_ra_one_hot_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((
            ProofId(node.idx, ProofType::TanhRaOneHotChecks),
            tanh_ra_one_hot_proof,
        ));

        // Stage 4: TanhRa hamming weight
        let mut tanh_hw_sumcheck = build_stage4_prover::<F>(node, prover);
        let (tanh_hw_proof, _) = Sumcheck::prove(
            &mut tanh_hw_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((
            ProofId(node.idx, ProofType::TanhRaHammingWeight),
            tanh_hw_proof,
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
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;

        let div_verifier = TeleportDivisionVerifier::new(node.clone(), &verifier.accumulator);
        Sumcheck::verify(
            div_proof,
            &div_verifier,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 1b: Tanh verification (after division is complete)
        let tanh_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::TanhExecution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;

        let exec_sumcheck = TanhVerifier::new(
            node.clone(),
            &verifier.preprocessing.model.graph,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            tanh_proof,
            &exec_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 1b: Range check verification for division
        let rangecheck_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RangeCheck))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let rangecheck_verifier =
            RangecheckRafSumcheckVerifier::<_, TeleportRangeCheckOperands>::new_from_verifier(
                node, verifier,
            );
        Sumcheck::verify(
            rangecheck_proof,
            &rangecheck_verifier,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 2: RaOneHotChecks verification for division range checking
        let ra_one_hot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let log_T = node.num_output_elements().log_2();
        let one_hot_params = OneHotParams::new(log_T);
        let (ra_sumcheck, hw_sumcheck, bool_sumcheck) =
            range_checking::new_ra_one_hot_sumcheck_verifiers::<F, TeleportRangeCheckOperands>(
                node.clone(),
                &one_hot_params,
                verifier,
            );
        BatchedSumcheck::verify(
            ra_one_hot_proof,
            vec![&ra_sumcheck, &bool_sumcheck, &hw_sumcheck],
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 3: TanhRa one-hot checks verification (booleanity + hamming booleanity)
        let tanh_ra_one_hot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::TanhRaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let (tanh_hb_verifier, tanh_bool_verifier) = build_stage3_verifiers::<F>(node, verifier);
        let instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> =
            vec![&tanh_hb_verifier, &tanh_bool_verifier];
        BatchedSumcheck::verify(
            tanh_ra_one_hot_proof,
            instances,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Stage 4: TanhRa hamming weight verification
        let tanh_hw_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::TanhRaHammingWeight))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let tanh_hw_verifier = build_stage4_verifier::<F>(node, verifier);
        Sumcheck::verify(
            tanh_hw_proof,
            &tanh_hw_verifier,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPolynomial> {
        vec![CommittedPolynomial::TanhRa(node.idx)]
    }
}

const DEGREE_BOUND: usize = 2;

#[derive(Clone)]
pub struct TanhParams<F: JoltField> {
    gamma: F,
    r_node_output: Vec<F::Challenge>,
    computation_node: ComputationNode,
    table_size: usize,
    log_table_size: usize,
}

impl<F: JoltField> TanhParams<F> {
    pub fn new(
        computation_node: ComputationNode,
        _graph: &ComputationGraph,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let gamma = transcript.challenge_scalar();

        // With neural teleportation, we reduce the domain by dividing by NEURAL_TELEPORT_DIVISOR
        let log_table_size = 16; // Reduced by division

        let table_size = 1 << log_table_size;

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
            table_size,
            log_table_size,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for TanhParams<F> {
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

        // Use quotient claim instead of input claim (neural teleportation)
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
        self.log_table_size
    }
}

// This is a Read-Raf sumcheck for Tanh lookup,
// Where we assert that each output[i] = TanhTable[input[i]]
// and input[i] = Ra[k] * Int[k] where Ra is one-hot encoding and Int is identity
pub struct TanhProver<F: JoltField> {
    params: TanhParams<F>,
    tanh_table: MultilinearPolynomial<F>,
    input_onehot: MultilinearPolynomial<F>,
    identity: IdentityPolynomial<F>,
}

impl<F: JoltField> TanhProver<F> {
    pub fn initialize(
        trace: &Trace,
        params: TanhParams<F>,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let LayerData { operands, .. } = Trace::layer_data(trace, &params.computation_node);
        let input = operands[0];

        // Compute quotient from division (neural teleportation)
        let (quotient_tensor, _remainder) = compute_division(input);

        // Create and materialize the tanh lookup table (reduced size)
        let tanh_table_impl = TanhTable::new(params.log_table_size);
        let tanh_table_values = tanh_table_impl.materialize();
        let tanh_table: Vec<F> = tanh_table_values.iter().map(|&v| F::from_i32(v)).collect();
        let tanh_table = MultilinearPolynomial::from(tanh_table);

        // Compute one-hot encoding of QUOTIENT values (not input)
        println!("1");
        let input_onehot: Vec<F> =
            compute_ra_evals(&params.r_node_output, &quotient_tensor, params.table_size);

        // Cache quotient claim (used in tanh lookup)
        // We do not reuse the claim from the division sumcheck, because the opening point is different
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
        assert_eq!(input_onehot.len(), tanh_table.len());
        let identity = IdentityPolynomial::new(params.log_table_size);

        #[cfg(test)]
        {
            let quotient_claim = MultilinearPolynomial::from(quotient_tensor.into_container_data())
                .evaluate(&params.r_node_output);
            let rv_claim = accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::NodeOutput(params.computation_node.idx),
                    SumcheckId::Execution,
                )
                .1;
            let claim = (0..input_onehot.len())
                .map(|i| {
                    use crate::onnx_proof::neural_teleport::usize_to_n_bits;

                    let a = input_onehot.get_bound_coeff(i);
                    let b = tanh_table.get_bound_coeff(i);
                    let int = F::from_u32(usize_to_n_bits(i, 16) as u32);
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
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

        let r_input_onehot = [self.params.r_node_output.as_slice(), &opening_point.r].concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutputRa(self.params.computation_node.idx),
            SumcheckId::Raf,
            r_input_onehot.into(),
            self.input_onehot.final_sumcheck_claim(),
        );
    }
}

pub struct TanhVerifier<F: JoltField> {
    params: TanhParams<F>,
    tanh_table: MultilinearPolynomial<F>,
}

impl<F: JoltField> TanhVerifier<F> {
    pub fn new(
        computation_node: ComputationNode,
        graph: &ComputationGraph,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl Transcript,
    ) -> Self {
        let params = TanhParams::new(computation_node, graph, accumulator, transcript);

        // Cache quotient polynomial opening (neural teleportation)
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::TeleportQuotient(params.computation_node.idx),
            SumcheckId::Raf,
            params.r_node_output.clone().into(),
        );

        // Materialize the tanh table for verification
        let tanh_table_impl = TanhTable::new(params.log_table_size);
        let tanh_table_values = tanh_table_impl.materialize();
        let tanh_table: Vec<F> = tanh_table_values.iter().map(|&v| F::from_i32(v)).collect();
        let tanh_table = MultilinearPolynomial::from(tanh_table);

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
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

        let ra_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::NodeOutputRa(self.params.computation_node.idx),
                SumcheckId::Raf,
            )
            .1;

        // Evaluate tanh table at the opening point
        let table_claim = self.tanh_table.evaluate(&opening_point.r);

        let int_eval =
            IdentityPolynomial::new(self.params.log_table_size).evaluate(&opening_point.r);

        ra_claim * (table_claim + self.params.gamma * int_eval)
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);

        let r_input_onehot = [self.params.r_node_output.as_slice(), &opening_point.r].concat();
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::NodeOutputRa(self.params.computation_node.idx),
            SumcheckId::Raf,
            r_input_onehot.into(),
        );
    }
}

// From the input values, computes the one-hot read address vector
fn compute_ra_evals<F>(r: &[F::Challenge], input: &Tensor<i32>, table_size: usize) -> Vec<F>
where
    F: JoltField,
{
    let E = EqPolynomial::evals(r);
    let num_threads = rayon::current_num_threads();
    let chunk_size = input.len().div_ceil(num_threads);

    // TODO: Handle table_size correctly
    let input_usize = input
        .par_iter()
        .map(|&x| n_bits_to_usize(x, 16))
        .collect::<Vec<usize>>();

    let partial_results: Vec<Vec<F>> = input_usize
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let mut local_ra = unsafe_allocate_zero_vec::<F>(table_size);
            let base_idx = chunk_idx * chunk_size;
            chunk.iter().enumerate().for_each(|(local_j, &k)| {
                let global_j = base_idx + local_j;
                local_ra[k] += E[global_j];
            });
            local_ra
        })
        .collect();

    let mut ra = unsafe_allocate_zero_vec::<F>(table_size);
    for partial in partial_results {
        ra.par_iter_mut()
            .zip(partial.par_iter())
            .for_each(|(dest, &src)| *dest += src);
    }
    ra
}

// Stage 3: TanhRa one-hot checks (hamming booleanity + booleanity)
fn build_stage3_provers<F: JoltField>(
    computation_node: &ComputationNode,
    prover: &mut Prover<F, impl Transcript>,
) -> (
    HammingBooleanitySumcheckProver<F>,
    SmallBooleanitySumcheckProver<F>,
) {
    let LayerData { operands, .. } = Trace::layer_data(&prover.trace, computation_node);
    let input = operands[0];

    // Compute quotient for neural teleportation
    let (quotient, _remainder) = compute_division(input);

    let log_table_size = 16;
    let table_size = 1 << log_table_size;
    let num_lookups = quotient.len();

    let hb_params = ra_hamming_bool_params::<F>(
        computation_node,
        num_lookups,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let bool_params = ra_booleanity_params::<F>(
        computation_node,
        table_size,
        num_lookups,
        &prover.accumulator,
        &mut prover.transcript,
    );

    let hw = {
        let mut lookup_vec = vec![F::one(); quotient.len()];
        lookup_vec.resize(quotient.len().next_power_of_two(), F::zero());
        lookup_vec
    };
    println!("2");
    let ra_evals = compute_ra_evals(&bool_params.r_cycle, &quotient, table_size);

    let quotient_u = quotient.iter().map(|&x| Some(x as u16)).collect();

    let hb_sumcheck = HammingBooleanitySumcheckProver::gen(hb_params, vec![hw]);
    let bool_sumcheck =
        SmallBooleanitySumcheckProver::gen(bool_params, vec![ra_evals], vec![quotient_u]);

    (hb_sumcheck, bool_sumcheck)
}

fn build_stage3_verifiers<F: JoltField>(
    computation_node: &ComputationNode,
    verifier: &mut Verifier<'_, F, impl Transcript>,
) -> (
    HammingBooleanitySumcheckVerifier<F>,
    BooleanitySumcheckVerifier<F>,
) {
    let graph = &verifier.preprocessing.model.graph;
    let input_node = graph.nodes.get(&computation_node.inputs[0]).unwrap();

    let log_table_size = 16;
    let table_size = 1 << log_table_size;
    let num_lookups = input_node.num_output_elements();

    let hb_params = ra_hamming_bool_params::<F>(
        computation_node,
        num_lookups,
        &verifier.accumulator,
        &mut verifier.transcript,
    );
    let bool_params = ra_booleanity_params::<F>(
        computation_node,
        table_size,
        num_lookups,
        &verifier.accumulator,
        &mut verifier.transcript,
    );

    let hb_sumcheck = HammingBooleanitySumcheckVerifier::new(hb_params);
    let bool_sumcheck = BooleanitySumcheckVerifier::new(bool_params);

    (hb_sumcheck, bool_sumcheck)
}

// Stage 4: TanhRa hamming weight
fn build_stage4_prover<F: JoltField>(
    computation_node: &ComputationNode,
    prover: &mut Prover<F, impl Transcript>,
) -> HammingWeightSumcheckProver<F> {
    let LayerData { operands, .. } = Trace::layer_data(&prover.trace, computation_node);
    let input = operands[0];

    // Compute quotient for neural teleportation
    let (quotient, _remainder) = compute_division(input);

    let log_table_size = 16;
    let table_size = 1 << log_table_size;

    let hw_params = ra_hamming_weight_params::<F>(
        computation_node,
        table_size,
        &prover.accumulator,
        &mut prover.transcript,
    );

    println!("3");
    let ra_evals = compute_ra_evals(&hw_params.r_cycle, &quotient, table_size);

    HammingWeightSumcheckProver::gen(hw_params, vec![ra_evals])
}

fn build_stage4_verifier<F: JoltField>(
    computation_node: &ComputationNode,
    verifier: &mut Verifier<'_, F, impl Transcript>,
) -> HammingWeightSumcheckVerifier<F> {
    let log_table_size = 16;
    let table_size = 1 << log_table_size;

    let hw_params = ra_hamming_weight_params::<F>(
        computation_node,
        table_size,
        &verifier.accumulator,
        &mut verifier.transcript,
    );

    HammingWeightSumcheckVerifier::new(hw_params)
}

fn ra_hamming_weight_params<F: JoltField>(
    computation_node: &ComputationNode,
    table_size: usize,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    _transcript: &mut impl Transcript,
) -> HammingWeightSumcheckParams<F> {
    let polynomial_types = vec![CommittedPolynomial::TanhRa(computation_node.idx)];

    // Use quotient polynomial for r_lookup (neural teleportation)
    let r_lookup = opening_accumulator
        .get_virtual_polynomial_opening(
            VirtualPolynomial::TeleportQuotient(computation_node.idx),
            SumcheckId::Execution,
        )
        .0
        .r;

    HammingWeightSumcheckParams {
        d: 1,
        num_rounds: table_size.log_2(),
        gamma_powers: vec![F::one()],
        polynomial_types,
        sumcheck_id: SumcheckId::HammingWeight,
        r_cycle: r_lookup,
    }
}

fn ra_hamming_bool_params<F: JoltField>(
    computation_node: &ComputationNode,
    num_lookups: usize,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    _transcript: &mut impl Transcript,
) -> HammingBooleanitySumcheckParams<F> {
    let polynomial_types = vec![VirtualPolynomial::HammingWeight];

    // Use quotient polynomial for r_lookup (neural teleportation)
    let r_lookup = opening_accumulator
        .get_virtual_polynomial_opening(
            VirtualPolynomial::TeleportQuotient(computation_node.idx),
            SumcheckId::Execution,
        )
        .0
        .r;

    HammingBooleanitySumcheckParams {
        d: 1,
        num_rounds: num_lookups.log_2(),
        gamma_powers: vec![F::one()],
        polynomial_types,
        r_cycle: r_lookup,
        sumcheck_id: SumcheckId::RamHammingBooleanity,
    }
}

fn ra_booleanity_params<F: JoltField>(
    computation_node: &ComputationNode,
    table_size: usize,
    num_lookups: usize,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> BooleanitySumcheckParams<F> {
    let polynomial_type = CommittedPolynomial::TanhRa(computation_node.idx);

    // Use quotient polynomial for r_lookup (neural teleportation)
    let r_lookup = opening_accumulator
        .get_virtual_polynomial_opening(
            VirtualPolynomial::TeleportQuotient(computation_node.idx),
            SumcheckId::Execution,
        )
        .0
        .r;
    let r_address = transcript.challenge_vector_optimized::<F>(table_size.log_2());

    BooleanitySumcheckParams {
        d: 1,
        log_k_chunk: table_size.log_2(),
        log_t: num_lookups.log_2(),
        r_cycle: r_lookup,
        r_address,
        polynomial_types: vec![polynomial_type],
        gammas: vec![F::Challenge::from(1)],
        sumcheck_id: SumcheckId::Booleanity,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use ark_bn254::Fr;
    use atlas_onnx_tracer::{model, utils::f32::F32};
    use joltworks::transcripts::Blake2bTranscript;
    use rand::{rngs::StdRng, SeedableRng};

    use crate::onnx_proof::AtlasSharedPreprocessing;

    use super::*;

    #[test]
    fn test_tanh() {
        let log_T = 10;
        let T = 1 << log_T;

        const MAX_INPUT_VALUE: i32 = 1 << 16;

        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::random_range(&mut rng, &[T], 0..MAX_INPUT_VALUE);
        let model = model::test::tanh_model(&[T]);
        let trace = model.trace(&[input]);

        let prover_transcript = Blake2bTranscript::new(&[]);
        let preprocessing: AtlasSharedPreprocessing =
            AtlasSharedPreprocessing::preprocess(model.clone());
        let prover_opening_accumulator = ProverOpeningAccumulator::new();
        let mut prover = Prover {
            trace: trace.clone(),
            preprocessing,
            accumulator: prover_opening_accumulator,
            transcript: prover_transcript,
        };

        let output_index = model.outputs()[0];
        let computation_node = &model[output_index];
        let LayerData {
            operands: _,
            output,
        } = Trace::layer_data(&trace, computation_node);

        let r_node_output: Vec<<Fr as JoltField>::Challenge> = prover
            .transcript
            .challenge_vector_optimized::<Fr>(computation_node.num_output_elements().log_2());

        let tanh_claim = MultilinearPolynomial::from(output.clone()).evaluate(&r_node_output);
        prover.accumulator.append_virtual(
            &mut prover.transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.clone().into(),
            tanh_claim,
        );

        let proofs = Tanh {
            scale: F32(0.0),
            tau: 0,
        }
        .prove(computation_node, &mut prover);
        let proofs = BTreeMap::from_iter(proofs);

        let io = Trace::io(&trace, &model);

        let verifier_transcript = Blake2bTranscript::new(&[]);
        let verifier_opening_accumulator = VerifierOpeningAccumulator::new();
        let mut verifier = Verifier {
            preprocessing: &prover.preprocessing.clone(),
            accumulator: verifier_opening_accumulator,
            transcript: verifier_transcript,
            io: &io,
            proofs: &proofs,
        };

        let _r_node_output = verifier
            .transcript
            .challenge_vector_optimized::<Fr>(computation_node.num_output_elements().log_2());
        assert_eq!(r_node_output, _r_node_output);

        for (key, (_, value)) in &prover.accumulator.openings {
            let empty_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![]);
            verifier
                .accumulator
                .openings
                .insert(*key, (empty_point, *value));
        }

        verifier.accumulator.append_virtual(
            &mut verifier.transcript,
            VirtualPolynomial::NodeOutput(output_index),
            SumcheckId::Execution,
            r_node_output.into(),
        );

        let res = Tanh {
            scale: F32(0.0),
            tau: 0,
        }
        .verify(computation_node, &mut verifier);

        let r_prover: Fr = prover.transcript.challenge_scalar();
        let r_verifier: Fr = verifier.transcript.challenge_scalar();
        assert_eq!(r_prover, r_verifier);

        verifier.transcript.compare_to(prover.transcript);
        res.unwrap();
    }
}
