//! # Witness Generation Module
//!
//! This module provides functionality for generating witness polynomials from neural network
//! model traces. The witness generation process is a critical component of the proof system:
//!
//! 1. **Trace Execution**: The model is executed with specific inputs, generating a trace of all
//!    intermediate computations.
//! 2. **Witness Generation**: For each committed polynomial in the protocol, this module generates
//!    the corresponding witness data from the trace.
//! 3. **Polynomial Commitment**: The generated polynomials are committed to using the commitment
//!    scheme (e.g., multilinear polynomial commitments or one-hot polynomial commitments).
//! 4. **Verifier Interaction**: The commitments are sent to the verifier as part of the proof.
//!
//! The [`WitnessGenerator`] trait defines the interface for generating witness polynomials from
//! different types of committed polynomials (e.g., node outputs, division quotients, range check
//! addresses, etc.).

use crate::{
    onnx_proof::{
        clamp_lookups::{clamp_intermediate, clamp_lookup_bits, CLAMP_LOG_K},
        neural_teleport::{division::compute_division, n_bits_to_usize},
        ops::{rsqrt::rsqrt_dividend, softmax_last_axis::rc::sat_diff_rc_bits},
        range_checking::range_check_operands::{
            DivRangeCheckOperands, MeanOfSquaresRangeCheckOperands, RangeCheckOperands,
            RangeCheckingOperandsTrait, RiRangeCheckOperands, RsRangeCheckOperands,
            TeleportRangeCheckOperands,
        },
    },
    utils::{adjusted_remainder, compute_lookup_indices_from_operands},
};
use atlas_onnx_tracer::{
    model::{consts::FOUR_PI_APPROX, trace::Trace, Model},
    node::ComputationNode,
    ops::{
        softmax::{generate_exp_lut_decomposed, softmax_last_axis_decomposed},
        Operator, SoftmaxLastAxis,
    },
    tensor::Tensor,
    utils::quantize::scale_to_multiplier,
};
use common::{
    consts::{LOG_K, XLEN},
    parallel::par_enabled,
    CommittedPoly,
};
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
    subprotocols,
    utils::{lookup_bits::LookupBits, math::Math},
};
use rayon::prelude::*;

/// Builds a one-hot RaD witness for any of the range-checking operand types.
///
/// Every range-check variant (`DivRangeCheckRaD`, `SqrtRangeCheckRaD`, etc.) follows the
/// same three steps: get operand tensors via the trait, compute lookup indices, then call
/// `build_one_hot_rad_witness`. This helper collapses all four arms to a single call site.
fn build_range_check_rad_witness<F: JoltField, H: RangeCheckingOperandsTrait>(
    model: &Model,
    trace: &Trace,
    node_idx: usize,
    d: usize,
) -> MultilinearPolynomial<F> {
    let computation_node = &model.graph.nodes[&node_idx];
    let inputs = trace.padded_operand_tensors(computation_node);
    let rc_operands = RangeCheckOperands::<H>::new(computation_node);

    let [left_operand, right_operand] = rc_operands
        .build_lookup_operands(&inputs)
        .try_into()
        .expect("Expected two operands for range-checking");

    let lookup_indices = rc_operands.compute_lookup_indices(&left_operand, &right_operand);
    build_one_hot_rad_witness(&lookup_indices, d, LOG_K)
}

/// Builds a one-hot polynomial witness for the `d`-th dimension of a read-after-decompose (RaD)
/// address polynomial.
///
/// This pattern appears in every operand type that decomposes lookup indices across multiple
/// dimensions (NodeOutputRaD, DivRangeCheckRaD, SqrtRangeCheckRaD, etc.).
fn build_one_hot_rad_witness<F: JoltField>(
    lookup_indices: &[LookupBits],
    d: usize,
    log_k: usize,
) -> MultilinearPolynomial<F> {
    let one_hot_params = OneHotParams::new(lookup_indices.len().log_2(), log_k);
    let addresses: Vec<_> = lookup_indices
        .par_iter()
        .with_min_len(par_enabled())
        .map(|lookup_index| Some(one_hot_params.lookup_index_chunk(lookup_index.into(), d) as u16))
        .collect();
    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
        addresses,
        one_hot_params.k_chunk,
    ))
}

/// Builds a one-hot RaD witness for neural-teleport activation lookups (tanh/erf).
///
/// Both activations share the exact same witness construction pipeline:
/// compute quotient via teleport-division, map quotient to lookup indices, then
/// construct the `d`-th one-hot address chunk.
fn build_teleport_activation_rad_witness<F: JoltField>(
    input: &Tensor<i32>,
    tau: i32,
    log_table: usize,
    d: usize,
) -> MultilinearPolynomial<F> {
    let (quotient, _remainder) = compute_division(input, tau);
    let lookup_indices: Vec<usize> = quotient
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&x| n_bits_to_usize(x, log_table))
        .collect();
    let one_hot_params = OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_table);
    let h_indices =
        subprotocols::shout::compute_instruction_h_indices(&lookup_indices, &one_hot_params);
    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
        h_indices[d]
            .par_iter()
            .with_min_len(par_enabled())
            .map(|&h| h.map(|h| h as u16))
            .collect(),
        one_hot_params.k_chunk,
    ))
}

// TODO: RM once clamp is implemented for tanh
fn build_teleport_activation_rad_witness_tanh<F: JoltField>(
    input: &Tensor<i32>,
    tau: i32,
    log_table: usize,
    d: usize,
) -> MultilinearPolynomial<F> {
    let (quotient, _remainder) = compute_division(input, tau);
    let clamped_tensor = atlas_onnx_tracer::ops::tanh::clamp_tensor(
        &quotient,
        -(1 << (log_table - 1)),
        (1 << (log_table - 1)) - 1,
    );
    let lookup_indices: Vec<usize> = clamped_tensor
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&x| n_bits_to_usize(x, log_table))
        .collect();
    let one_hot_params = OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_table);
    let h_indices =
        subprotocols::shout::compute_instruction_h_indices(&lookup_indices, &one_hot_params);
    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
        h_indices[d]
            .par_iter()
            .with_min_len(par_enabled())
            .map(|&h| h.map(|h| h as u16))
            .collect(),
        one_hot_params.k_chunk,
    ))
}

// TODO: group for all chunked committed polynomials (CommittedPoly::*RaD)
/// Generate the witnesses for **all** of a node's committed polynomials, computing
/// each expensive per-node re-execution **once** and slicing every one-hot chunk
/// from it.
///
/// A fused node's saturating-clamp and rescaling-remainder one-hots decompose into
/// many `LOG_K_CHUNK`-wide chunks (16 for the 64-bit clamp, plus the remainder /
/// range-check chunks). Generating each chunk via [`WitnessGenerator::generate_witness`]
/// independently re-runs the (expensive, e.g. `einsum_acc_i64` over a full
/// contraction) accumulation once *per chunk* — ~16-36× per node in practice. This
/// groups a node's polynomials so the accumulation runs once, then reuses the
/// cached lookup tensor for every chunk. Non-fused polynomials fall back to
/// [`WitnessGenerator::generate_witness`].
#[tracing::instrument(skip_all, name = "generate_node_witnesses", fields(node = node.idx))]
pub(crate) fn generate_node_witnesses<F: JoltField, T: joltworks::transcripts::Transcript>(
    node: &ComputationNode,
    model: &Model,
    trace: &Trace,
) -> Vec<(CommittedPoly, MultilinearPolynomial<F>)> {
    use crate::onnx_proof::{
        fused_rebase::{rebase_bits, try_rebase_intermediates},
        ops::NodeCommittedPolynomials,
    };

    // Per-node lookup tensors, each computed at most once (lazily). For a
    // fused-rescale node the clamp quotient and rescale remainder derive from
    // one shared accumulation pass (`try_rebase_intermediates`), which fills
    // both caches at once — computing them independently would re-run the
    // accumulation twice.
    let mut clamp_bits: Option<Vec<LookupBits>> = None;
    let mut remainder_indices: Option<Vec<usize>> = None;
    let mut mos_range_bits: Option<Vec<LookupBits>> = None;

    NodeCommittedPolynomials::get_committed_polynomials::<F, T>(node)
        .into_iter()
        .map(|poly| {
            let witness = match &poly {
                CommittedPoly::ClampRaD(_, d) => {
                    if clamp_bits.is_none() {
                        match try_rebase_intermediates(node, trace) {
                            Some(ints) => {
                                clamp_bits = Some(clamp_lookup_bits(&ints.quotient));
                                remainder_indices = Some(
                                    ints.remainder.data().iter().map(|&v| v as usize).collect(),
                                );
                            }
                            // Add/Sub/Sum: clamp only, no fused rescale.
                            None => {
                                clamp_bits =
                                    Some(clamp_lookup_bits(&clamp_intermediate(node, trace)));
                            }
                        }
                    }
                    build_one_hot_rad_witness(clamp_bits.as_ref().unwrap(), *d, CLAMP_LOG_K)
                }
                CommittedPoly::RescaleRemainderRaD(_, d) => {
                    let bits = rebase_bits(&node.operator)
                        .expect("RescaleRemainderRaD requested for a non-rescaling operator");
                    if remainder_indices.is_none() {
                        let ints = try_rebase_intermediates(node, trace)
                            .expect("RescaleRemainderRaD requested for a non-rescaling operator");
                        remainder_indices =
                            Some(ints.remainder.data().iter().map(|&v| v as usize).collect());
                        clamp_bits.get_or_insert_with(|| clamp_lookup_bits(&ints.quotient));
                    }
                    build_onehot_witness(remainder_indices.as_ref().unwrap(), bits as usize, *d)
                }
                CommittedPoly::MeanOfSquaresRangeCheckRaD(_, d) => {
                    let bits = mos_range_bits.get_or_insert_with(|| {
                        let padded_operands = trace.padded_operand_tensors(node);
                        let rc_operands =
                            RangeCheckOperands::<MeanOfSquaresRangeCheckOperands>::new(node);
                        let [left, right] = rc_operands
                            .build_lookup_operands(&padded_operands)
                            .try_into()
                            .unwrap();

                        rc_operands.compute_lookup_indices(&left, &right)
                    });
                    build_one_hot_rad_witness(bits, *d, LOG_K)
                }
                other => other.generate_witness(model, trace),
            };
            (poly, witness)
        })
        .collect()
}

/// Trait for generating witness polynomials from model execution traces.
///
/// This trait defines the interface for converting committed polynomial specifications
/// into concrete witness data. The witness generation process takes a neural network
/// model and its execution trace, then produces the multilinear polynomial that the
/// prover commits to and sends to the verifier.
///
/// # Type Parameters
///
/// * `F` - The finite field over which polynomials are defined
///
/// # Example Flow
///
/// ```text
/// Model + Trace → WitnessGenerator → MultilinearPolynomial → Commitment → Verifier
/// ```
pub trait WitnessGenerator<F: JoltField> {
    /// Generates a witness polynomial from a model and its execution trace.
    ///
    /// This method creates the concrete polynomial data that will be committed to by the prover.
    /// Different types of committed polynomials (node outputs, range check addresses, division
    /// remainders, etc.) require different witness generation logic.
    ///
    /// # Arguments
    ///
    /// * `model` - The neural network model containing the computation graph
    /// * `trace` - The execution trace containing intermediate values and operands
    ///
    /// # Returns
    ///
    /// A multilinear polynomial representing the witness data. This can be:
    /// - A regular `MultilinearPolynomial` for dense data
    /// - A `OneHotPolynomial` for sparse one-hot encoded data (e.g., lookup addresses)
    fn generate_witness(&self, model: &Model, trace: &Trace) -> MultilinearPolynomial<F>;
}

/// Implementation of witness generation for different types of committed polynomials.
///
/// This implementation handles witness generation for all committed polynomial types used in
/// the proof system, including:
/// - Node output read-after-decompose (RaD) addresses
/// - Division quotients and remainders
/// - Range check addresses for various operations
/// - Softmax intermediate values
/// - Gather operation addresses
/// - Tanh lookup table addresses
impl<F: JoltField> WitnessGenerator<F> for CommittedPoly {
    #[tracing::instrument(skip_all)]
    fn generate_witness(&self, model: &Model, trace: &Trace) -> MultilinearPolynomial<F> {
        match self {
            CommittedPoly::NodeOutputRaD(node_idx, d) => {
                let computation_node = &model.graph.nodes[node_idx];
                let layer_data = Trace::layer_data(trace, computation_node);
                let padded_operands: Vec<_> = layer_data
                    .operands
                    .iter()
                    .map(|tensor| tensor.padded_next_power_of_two())
                    .collect();
                let operand_refs: Vec<_> = padded_operands.iter().collect();
                let lookup_indices = compute_lookup_indices_from_operands(&operand_refs, false);
                build_one_hot_rad_witness(&lookup_indices, *d, XLEN)
            }
            CommittedPoly::ClampRaD(node_idx, d) => {
                // Saturating Add/Sub: the lookup index is the pre-clamp i64
                // accumulation, recovered by re-executing the binop.
                let computation_node = &model.graph.nodes[node_idx];
                let intermediate = clamp_intermediate(computation_node, trace);
                let lookup_bits = clamp_lookup_bits(&intermediate);
                build_one_hot_rad_witness(&lookup_bits, *d, CLAMP_LOG_K)
            }
            CommittedPoly::DivNodeQuotient(node_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(computation_node.operator, Operator::Div(_)));
                let layer_data = Trace::layer_data(trace, computation_node);
                let q = layer_data.output;
                MultilinearPolynomial::from(q.clone())
            }
            CommittedPoly::TeleportNodeQuotient(node_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                // For nodes that use the quotient for a lookup (erf/sigmoid ...),
                // the quotient is virtualized from a one-hot encoding.
                assert!(matches!(
                    computation_node.operator,
                    Operator::Cos(_) | Operator::Sin(_)
                ));
                let tau = match &computation_node.operator {
                    Operator::Cos(_) | Operator::Sin(_) => FOUR_PI_APPROX,
                    _ => unreachable!(
                        "teleport quotient witness requested for non-teleport operator"
                    ),
                };
                let layer_data = Trace::layer_data(trace, computation_node);
                let input = &layer_data.operands[0];
                let (quotient, _remainder) = compute_division(input, tau);
                MultilinearPolynomial::from(quotient)
            }
            CommittedPoly::ScalarConstDivNodeRemainder(node_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                let Operator::ScalarConstDiv(op) = &computation_node.operator else {
                    panic!("Expected ScalarConstDiv operator at node {node_idx}");
                };
                let b = op.divisor;
                let layer_data = Trace::layer_data(trace, computation_node);
                let [left_operand] = layer_data.operands[..] else {
                    panic!("Expected one operand for ScalarConstDiv operation")
                };
                let remainder_data: Vec<i32> = left_operand
                    .iter()
                    .map(|&a| adjusted_remainder(a, b))
                    .collect();
                MultilinearPolynomial::from(Tensor::<i32>::construct(
                    remainder_data,
                    left_operand.dims().to_vec(),
                ))
            }
            CommittedPoly::RsqrtQuotient(node_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(computation_node.operator, Operator::Rsqrt(_)));
                let layer_data = Trace::layer_data(trace, computation_node);
                let [input] = layer_data.operands[..] else {
                    panic!("Expected one operand for Rsqrt operation")
                };
                let s_cubed = rsqrt_dividend(computation_node);
                // `quotient = ⌊S³ / x̂⌋` can exceed i32 at higher scales (e.g. up to
                // `S³` when `x̂ = 1`), so it is committed as u64.
                let quotient_data: Vec<u64> =
                    input.iter().map(|&x| (s_cubed / x as i64) as u64).collect();
                MultilinearPolynomial::from(quotient_data)
            }
            CommittedPoly::DivRangeCheckRaD(node_idx, d) => {
                build_range_check_rad_witness::<F, DivRangeCheckOperands>(
                    model, trace, *node_idx, *d,
                )
            }
            CommittedPoly::MeanOfSquaresRangeCheckRaD(node_idx, d) => {
                build_range_check_rad_witness::<F, MeanOfSquaresRangeCheckOperands>(
                    model, trace, *node_idx, *d,
                )
            }
            CommittedPoly::SqrtRangeCheckRaD(node_idx, d) => {
                build_range_check_rad_witness::<F, RsRangeCheckOperands>(
                    model, trace, *node_idx, *d,
                )
            }
            CommittedPoly::SqrtDivRangeCheckRaD(node_idx, d) => {
                build_range_check_rad_witness::<F, RiRangeCheckOperands>(
                    model, trace, *node_idx, *d,
                )
            }
            CommittedPoly::TeleportRangeCheckRaD(node_idx, d) => {
                build_range_check_rad_witness::<F, TeleportRangeCheckOperands>(
                    model, trace, *node_idx, *d,
                )
            }
            CommittedPoly::GatherRa(node_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(
                    computation_node.operator,
                    Operator::GatherSmall(_)
                ));
                let layer_data = Trace::layer_data(trace, computation_node);
                let indexes = &layer_data.operands[1];
                let non_zero_addresses: Vec<_> = indexes
                    .data()
                    .par_iter()
                    .with_min_len(par_enabled())
                    .map(|&index| Some(index as u16))
                    .collect();
                let input_dict = &model.graph.nodes.get(&computation_node.inputs[0]).unwrap();
                let num_words = input_dict.output_dims[0];
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    non_zero_addresses,
                    num_words,
                ))
            }
            CommittedPoly::GatherRaD(node_idx, d_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                let Operator::GatherLarge(gather_op) = &computation_node.operator else {
                    panic!("Expected GatherLarge operator for GatherRaD committed polynomial");
                };
                let layer_data = Trace::layer_data(trace, computation_node);
                let indexes = layer_data.operands[1].padded_next_power_of_two();
                let lookup_indices: Vec<usize> = indexes
                    .data()
                    .par_iter()
                    .with_min_len(par_enabled())
                    .map(|&x| x as usize)
                    .collect();
                let num_words = gather_op.dict_len.next_power_of_two();
                let one_hot_params = OneHotParams::from_config_and_log_K(
                    &OneHotConfig::default(),
                    num_words.log_2(),
                );
                let h_indices = subprotocols::shout::compute_instruction_h_indices(
                    &lookup_indices,
                    &one_hot_params,
                );
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    h_indices[*d_idx]
                        .par_iter()
                        .with_min_len(par_enabled())
                        .map(|&h| h.map(|h| h as u16))
                        .collect(),
                    one_hot_params.k_chunk,
                ))
            }

            CommittedPoly::TanhRaD(node_idx, d_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                let Operator::Tanh(inner) = &computation_node.operator else {
                    panic!("Expected Tanh operator for TanhRa committed polynomial");
                };
                let layer_data = Trace::layer_data(trace, computation_node);
                let input = &layer_data.operands[0];
                build_teleport_activation_rad_witness_tanh(
                    input,
                    inner.tau,
                    inner.log_table,
                    *d_idx,
                )
            }

            CommittedPoly::ErfRaD(node_idx, d_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                let Operator::Erf(inner) = &computation_node.operator else {
                    panic!("Expected Erf operator for ErfRa committed polynomial");
                };
                let layer_data = Trace::layer_data(trace, computation_node);
                let input = &layer_data.operands[0];
                build_teleport_activation_rad_witness(input, inner.tau, inner.log_table, *d_idx)
            }

            CommittedPoly::SigmoidRaD(node_idx, d_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                let Operator::Sigmoid(inner) = &computation_node.operator else {
                    panic!("Expected Sigmoid operator for SigmoidRa committed polynomial");
                };
                let layer_data = Trace::layer_data(trace, computation_node);
                let input = &layer_data.operands[0];
                build_teleport_activation_rad_witness(input, inner.tau, inner.log_table, *d_idx)
            }

            CommittedPoly::CosRaD(node_idx, d_idx) | CommittedPoly::SinRaD(node_idx, d_idx) => {
                const COS_LOG_TABLE_SIZE: usize =
                    (FOUR_PI_APPROX as usize).next_power_of_two().ilog2() as usize;

                let computation_node = &model.graph.nodes[node_idx];
                assert!(
                    matches!(
                        computation_node.operator,
                        Operator::Cos(_) | Operator::Sin(_)
                    ),
                    "Expected Cos or Sin operator for CosRaD/SinRaD committed polynomial"
                );
                let layer_data = Trace::layer_data(trace, computation_node);
                let input = &layer_data.operands[0];

                let (_quotient, remainder) = compute_division(input, FOUR_PI_APPROX);
                let lookup_indices: Vec<usize> = remainder
                    .par_iter()
                    .with_min_len(par_enabled())
                    .map(|&x| x as usize)
                    .collect();
                let one_hot_params = OneHotParams::from_config_and_log_K(
                    &OneHotConfig::default(),
                    COS_LOG_TABLE_SIZE,
                );
                let h_indices = subprotocols::shout::compute_instruction_h_indices(
                    &lookup_indices,
                    &one_hot_params,
                );
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    h_indices[*d_idx]
                        .par_iter()
                        .with_min_len(par_enabled())
                        .map(|&h| h.map(|h| h as u16))
                        .collect(),
                    one_hot_params.k_chunk,
                ))
            }
            CommittedPoly::SoftmaxRemainderRaD(node_idx, d) => {
                let node = &model.graph.nodes[node_idx];
                let Operator::SoftmaxLastAxis(SoftmaxLastAxis { scale }) = &node.operator else {
                    panic!("Expected SoftmaxLastAxis at node {node_idx}");
                };
                let log_scale = *scale as usize;
                let st = softmax_last_axis_full_trace(node, trace);
                let lookup_indices: Vec<usize> = st.R.iter().map(|&v| v as usize).collect();
                build_onehot_witness(&lookup_indices, log_scale, *d)
            }
            CommittedPoly::SoftmaxExpRemainderRaD(node_idx, d) => {
                let node = &model.graph.nodes[node_idx];
                let Operator::SoftmaxLastAxis(SoftmaxLastAxis { scale }) = &node.operator else {
                    panic!("Expected SoftmaxLastAxis at node {node_idx}");
                };
                let log_scale = *scale as usize;
                let st = softmax_last_axis_full_trace(node, trace);
                let lookup_indices: Vec<usize> = st
                    .decomposed_exp
                    .r_exp
                    .iter()
                    .map(|&v| v as usize)
                    .collect();
                build_onehot_witness(&lookup_indices, log_scale, *d)
            }
            CommittedPoly::SoftmaxSatDiffRaD(node_idx, d) => {
                let node = &model.graph.nodes[node_idx];
                let Operator::SoftmaxLastAxis(SoftmaxLastAxis { scale }) = &node.operator else {
                    panic!("Expected SoftmaxLastAxis at node {node_idx}");
                };
                let log_scale = *scale as usize;
                let st = softmax_last_axis_full_trace(node, trace);
                let lookup_indices: Vec<usize> = st
                    .decomposed_exp
                    .sat_diff
                    .iter()
                    .map(|&v| v as usize)
                    .collect();
                // Completeness gate: every honest sat_diff must fit the range-check
                // width. `build_onehot_witness` addresses a 2^bits table, so an
                // over-budget (or negative-as-usize) value would silently wrap to its
                // low bits. Fire loudly in debug/CI instead — this is the invariant the
                // narrowed width relies on (see rc::sat_diff_rc_bits).
                debug_assert!(
                    lookup_indices
                        .iter()
                        .all(|&v| v < (1usize << sat_diff_rc_bits(log_scale))),
                    "softmax sat_diff exceeds range-check budget 2^{} at log_scale {log_scale}",
                    sat_diff_rc_bits(log_scale),
                );
                build_onehot_witness(&lookup_indices, sat_diff_rc_bits(log_scale), *d)
            }
            CommittedPoly::SoftmaxZHiRaD(node_idx, d) => {
                let node = &model.graph.nodes[node_idx];
                let st = softmax_last_axis_full_trace(node, trace);
                let decomp = generate_exp_lut_decomposed(st.scale as i32);
                let log_hi = decomp.lut_hi.len().next_power_of_two().log_2();
                let lookup_indices: Vec<usize> =
                    st.decomposed_exp.z_hi.iter().map(|&v| v as usize).collect();
                build_onehot_witness(&lookup_indices, log_hi, *d)
            }
            CommittedPoly::SoftmaxZLoRaD(node_idx, d) => {
                let node = &model.graph.nodes[node_idx];
                let st = softmax_last_axis_full_trace(node, trace);
                let decomp = generate_exp_lut_decomposed(st.scale as i32);
                let log_lo = decomp.lut_lo.len().next_power_of_two().log_2();
                let lookup_indices: Vec<usize> =
                    st.decomposed_exp.z_lo.iter().map(|&v| v as usize).collect();
                build_onehot_witness(&lookup_indices, log_lo, *d)
            }
            CommittedPoly::RescaleRemainderRaD(node_idx, d) => {
                // Fused rescaling remainder `R = acc mod 2^S ∈ [0, 2^S)`, padded
                // to the node-output cycle domain . Shared by einsum, Mul,
                // Square, Cube via the per-operator re-execution dispatch.
                let node = &model.graph.nodes[node_idx];
                let bits = crate::onnx_proof::fused_rebase::rebase_bits(&node.operator)
                    .expect("RescaleRemainderRaD requested for a non-rescaling operator");
                let r = crate::onnx_proof::fused_rebase::rebase_remainder(node, trace);
                let lookup_indices: Vec<usize> = r.data().iter().map(|&v| v as usize).collect();
                build_onehot_witness(&lookup_indices, bits as usize, *d)
            }
        }
    }
}

/// Re-runs the decomposed softmax trace for a `SoftmaxLastAxis` node.
///
/// Returns the full [`SoftmaxLastAxisTrace`] from which the individual lookup
/// indices (R, r_exp, sat_diff, z_hi, z_lo) can be extracted.
fn softmax_last_axis_full_trace(
    node: &ComputationNode,
    trace: &Trace,
) -> atlas_onnx_tracer::ops::softmax::SoftmaxLastAxisTrace {
    let Operator::SoftmaxLastAxis(SoftmaxLastAxis { scale }) = &node.operator else {
        panic!("Expected SoftmaxLastAxis operator at node {}", node.idx);
    };
    let layer_data = Trace::layer_data(trace, node);
    let input = &layer_data.operands[0];
    softmax_last_axis_decomposed(input, scale_to_multiplier(*scale) as i32).1
}

/// Build a one-hot RaD witness for an identity range-check or Shout ra
/// polynomial, given raw lookup indices and the log2 of the table /
/// range-check size.
fn build_onehot_witness<F: JoltField>(
    lookup_indices: &[usize],
    log_k: usize,
    d: usize,
) -> MultilinearPolynomial<F> {
    let one_hot_params = OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k);
    let h_indices =
        subprotocols::shout::compute_instruction_h_indices(lookup_indices, &one_hot_params);
    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
        h_indices[d]
            .par_iter()
            .with_min_len(par_enabled())
            .map(|&h| h.map(|h| h as u16))
            .collect(),
        one_hot_params.k_chunk,
    ))
}
