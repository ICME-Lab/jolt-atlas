use atlas_onnx_tracer::{model::trace::Trace, node::ComputationNode, ops::Operator};
use common::{consts::XLEN, CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    config::OneHotParams,
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        opening_proof::{OpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    },
    subprotocols::{
        booleanity::{
            BooleanitySumcheckParams, BooleanitySumcheckProver, BooleanitySumcheckVerifier,
        },
        hamming_weight::{
            HammingWeightSumcheckParams, HammingWeightSumcheckProver, HammingWeightSumcheckVerifier,
        },
    },
    transcripts::Transcript,
    utils::{lookup_bits::LookupBits, math::Math, thread::unsafe_allocate_zero_vec},
};
use rayon::prelude::*;

use crate::onnx_proof::{
    range_checking::{
        ra_virtual::{InstructionRaSumcheckProver, InstructionRaSumcheckVerifier},
        sumcheck_instance::ReadRafSumcheckHelper,
    },
    Prover, Verifier,
};

pub mod ra_virtual;
pub mod read_raf_checking;
pub mod sumcheck_instance;

pub const LOG_K: usize = XLEN * 2;

pub fn ra_hamming_weight_params<F: JoltField, Helper: ReadRafSumcheckHelper>(
    computation_node: &ComputationNode,
    one_hot_params: &OneHotParams,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> HammingWeightSumcheckParams<F> {
    let helper = Helper::new(computation_node);

    let gamma_powers = transcript.challenge_scalar_powers(one_hot_params.instruction_d);

    let polynomial_types: Vec<CommittedPolynomial> = (0..one_hot_params.instruction_d)
        .map(|i| helper.rad_poly(i))
        .collect();

    let r_cycle = opening_accumulator
        .get_virtual_polynomial_opening(
            VirtualPolynomial::NodeOutput(computation_node.idx),
            SumcheckId::Execution,
        )
        .0
        .r;

    HammingWeightSumcheckParams {
        d: one_hot_params.instruction_d,
        num_rounds: one_hot_params.log_k_chunk,
        gamma_powers,
        polynomial_types,
        sumcheck_id: SumcheckId::HammingWeight,
        r_cycle,
    }
}

pub fn ra_booleanity_params<F: JoltField, Helper: ReadRafSumcheckHelper>(
    computation_node: &ComputationNode,
    one_hot_params: &OneHotParams,
    opening_accumulator: &dyn OpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> BooleanitySumcheckParams<F> {
    let helper = Helper::new(computation_node);

    let polynomial_types: Vec<CommittedPolynomial> = (0..one_hot_params.instruction_d)
        .map(|i| helper.rad_poly(i))
        .collect();

    let (r_cycle, _) = opening_accumulator.get_virtual_polynomial_opening(
        VirtualPolynomial::NodeOutput(computation_node.idx),
        SumcheckId::Execution,
    );

    let gammas = transcript.challenge_vector_optimized::<F>(one_hot_params.instruction_d);
    let r_address = transcript.challenge_vector_optimized::<F>(one_hot_params.log_k_chunk);

    BooleanitySumcheckParams {
        d: one_hot_params.instruction_d,
        log_k_chunk: one_hot_params.log_k_chunk,
        log_t: computation_node.num_output_elements().log_2(),
        r_cycle: r_cycle.r.clone(),
        r_address,
        gammas,
        polynomial_types,
        sumcheck_id: SumcheckId::Booleanity,
    }
}

pub fn new_ra_one_hot_sumcheck_provers<F: JoltField, Helper: ReadRafSumcheckHelper>(
    computation_node: ComputationNode,
    one_hot_params: &OneHotParams,
    prover: &mut Prover<F, impl Transcript>,
) -> (
    InstructionRaSumcheckProver<F, Helper>,
    HammingWeightSumcheckProver<F>,
    BooleanitySumcheckProver<F>,
) {
    let rad_prover = InstructionRaSumcheckProver::new_from_prover(
        computation_node.clone(),
        one_hot_params,
        prover,
    );

    let hamming_weight_params = ra_hamming_weight_params::<F, Helper>(
        &computation_node,
        one_hot_params,
        &prover.accumulator,
        &mut prover.transcript,
    );
    let booleanity_params = ra_booleanity_params::<F, Helper>(
        &computation_node,
        one_hot_params,
        &prover.accumulator,
        &mut prover.transcript,
    );

    let (hamming_weight_prover, booleanity_prover) = gen_ra_one_hot_provers::<F, Helper>(
        hamming_weight_params,
        booleanity_params,
        &prover.trace,
        &computation_node,
        one_hot_params,
    );

    (rad_prover, hamming_weight_prover, booleanity_prover)
}

pub fn gen_ra_one_hot_provers<F: JoltField, Helper: ReadRafSumcheckHelper>(
    hamming_weight_params: HammingWeightSumcheckParams<F>,
    booleanity_params: BooleanitySumcheckParams<F>,
    trace: &Trace,
    computation_node: &ComputationNode,
    one_hot_params: &OneHotParams,
) -> (HammingWeightSumcheckProver<F>, BooleanitySumcheckProver<F>) {
    let (left_operand, right_operand) = Helper::get_operands_tensors(trace, computation_node);
    let lookup_indices = Helper::compute_lookup_indices(&left_operand, &right_operand);

    let ra_evals = compute_ra_evals(&lookup_indices, one_hot_params, &booleanity_params.r_cycle);
    let H_indices = compute_instruction_h_indices(&lookup_indices, one_hot_params);

    (
        HammingWeightSumcheckProver::gen(hamming_weight_params, ra_evals.clone()),
        BooleanitySumcheckProver::gen(booleanity_params, ra_evals, H_indices),
    )
}

pub fn new_ra_one_hot_sumcheck_verifiers<F: JoltField, Helper: ReadRafSumcheckHelper>(
    computation_node: ComputationNode,
    one_hot_params: &OneHotParams,
    verifier: &mut Verifier<F, impl Transcript>,
) -> (
    InstructionRaSumcheckVerifier<F, Helper>,
    HammingWeightSumcheckVerifier<F>,
    BooleanitySumcheckVerifier<F>,
) {
    let rad_verifier = InstructionRaSumcheckVerifier::new_from_verifier(
        computation_node.clone(),
        one_hot_params,
        verifier,
    );
    let (hamming_weight_verifier, booleanity_verifier) = new_ra_one_hot_verifiers::<F, Helper>(
        &computation_node,
        one_hot_params,
        &verifier.accumulator,
        &mut verifier.transcript,
    );

    (rad_verifier, hamming_weight_verifier, booleanity_verifier)
}

pub fn new_ra_one_hot_verifiers<F: JoltField, Helper: ReadRafSumcheckHelper>(
    computation_node: &ComputationNode,
    one_hot_params: &OneHotParams,
    opening_accumulator: &VerifierOpeningAccumulator<F>,
    transcript: &mut impl Transcript,
) -> (
    HammingWeightSumcheckVerifier<F>,
    BooleanitySumcheckVerifier<F>,
) {
    let hamming_weight_params = ra_hamming_weight_params::<_, Helper>(
        computation_node,
        one_hot_params,
        opening_accumulator,
        transcript,
    );
    let booleanity_params = ra_booleanity_params::<_, Helper>(
        computation_node,
        one_hot_params,
        opening_accumulator,
        transcript,
    );
    (
        HammingWeightSumcheckVerifier::new(hamming_weight_params),
        BooleanitySumcheckVerifier::new(booleanity_params),
    )
}

#[tracing::instrument(skip_all, name = "instruction_lookups::compute_instruction_h_indices")]
fn compute_instruction_h_indices(
    trace: &[LookupBits],
    one_hot_params: &OneHotParams,
) -> Vec<Vec<Option<u8>>> {
    (0..one_hot_params.instruction_d)
        .map(|i| {
            trace
                .par_iter()
                .map(|lookup_index| Some(one_hot_params.lookup_index_chunk(lookup_index.into(), i)))
                .collect()
        })
        .collect()
}

#[tracing::instrument(skip_all, name = "instruction_lookups::compute_ra_evals")]
fn compute_ra_evals<F: JoltField>(
    trace: &[LookupBits],
    one_hot_params: &OneHotParams,
    r_cycle: &[F::Challenge],
) -> Vec<Vec<F>> {
    let eq_r_cycle = EqPolynomial::evals(r_cycle);

    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
            let mut result: Vec<Vec<F>> = (0..one_hot_params.instruction_d)
                .map(|_| unsafe_allocate_zero_vec(one_hot_params.k_chunk))
                .collect();
            let mut j = chunk_index * chunk_size;
            for lookup_index in trace_chunk {
                for i in 0..one_hot_params.instruction_d {
                    let k = one_hot_params.lookup_index_chunk(lookup_index.into(), i);
                    result[i][k as usize] += eq_r_cycle[j];
                }
                j += 1;
            }
            result
        })
        .reduce(
            || {
                (0..one_hot_params.instruction_d)
                    .map(|_| unsafe_allocate_zero_vec(one_hot_params.k_chunk))
                    .collect()
            },
            |mut running, new| {
                running.iter_mut().zip(new.into_iter()).for_each(|(x, y)| {
                    x.par_iter_mut()
                        .zip(y.into_par_iter())
                        .for_each(|(x, y)| *x += y)
                });
                running
            },
        )
}

pub trait InterleavedBitsMarker {
    fn is_interleaved_operands(&self) -> bool;
}

impl InterleavedBitsMarker for ComputationNode {
    fn is_interleaved_operands(&self) -> bool {
        matches!(self.operator, Operator::And2(_))
    }
}

pub trait CommitToOneHotEncodingsMarker {
    fn commit_to_one_encodings(&self) -> bool;
}

impl CommitToOneHotEncodingsMarker for ComputationNode {
    fn commit_to_one_encodings(&self) -> bool {
        matches!(self.operator, Operator::And2(_) | Operator::ReLU(_))
    }
}
