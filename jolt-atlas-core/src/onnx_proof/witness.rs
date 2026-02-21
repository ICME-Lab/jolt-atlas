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
        neural_teleport::{division::compute_division, n_bits_to_usize},
        op_lookups::InterleavedBitsMarker,
        ops::{rsqrt::Q_SQUARE, softmax_axes::softmax::scalar_div::S},
        range_checking::range_check_operands::{
            DivRangeCheckOperands, RangeCheckingOperandsTrait, RiRangeCheckOperands,
            RsRangeCheckOperands, TeleportRangeCheckOperands,
        },
    },
    utils::{adjusted_remainder, compute_lookup_indices_from_operands},
};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, Trace},
        Model,
    },
    node::ComputationNode,
    ops::Operator,
    tensor::{
        ops::nonlinearities::{softmax_fixed_128, SoftmaxTrace, LOG_EXP_LUT_SIZE},
        Tensor,
    },
};
use common::CommittedPolynomial;
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
    subprotocols,
    utils::{lookup_bits::LookupBits, math::Math},
};
use rayon::prelude::*;

/// Computes the softmax intermediate trace for one feature-group chunk of a `SoftmaxAxes` node.
///
/// Both `SoftmaxRemainder` and `SoftmaxExponentiationRaD` need identical slicing and
/// `softmax_fixed_128` logic; this helper eliminates the duplication.
fn softmax_chunk_trace(node: &ComputationNode, trace: &Trace, feature_idx: usize) -> SoftmaxTrace {
    let features = node.output_dims[2];
    let LayerData { operands, .. } = Trace::layer_data(trace, node);
    let operand = operands[0];
    let chunk = operand.data()[feature_idx * features..(feature_idx + 1) * features].to_vec();
    let (_, softmax_trace) = softmax_fixed_128::<true>(&Tensor::construct(chunk, vec![features]));
    softmax_trace.expect("softmax trace is always Some when TRACE = true")
}

/// Builds a one-hot RaD witness for any of the range-checking operand types.
///
/// Every range-check variant (`DivRangeCheckRaD`, `SqrtRangeCheckRaD`, etc.) follows the
/// same three steps: get operand tensors via the trait, compute lookup indices, then call
/// `build_one_hot_rad_witness`. This helper collapses all four arms to a single call site.
fn build_range_check_rad_witness<F: JoltField, R: RangeCheckingOperandsTrait>(
    model: &Model,
    trace: &Trace,
    node_idx: usize,
    d: usize,
) -> MultilinearPolynomial<F> {
    let computation_node = &model.graph.nodes[&node_idx];
    let (left_operand, right_operand) = R::get_operands_tensors(trace, computation_node);
    let lookup_indices = R::compute_lookup_indices(&left_operand, &right_operand);
    build_one_hot_rad_witness(&lookup_indices, d)
}

/// Builds a one-hot polynomial witness for the `d`-th dimension of a read-after-decompose (RaD)
/// address polynomial.
///
/// This pattern appears in every operand type that decomposes lookup indices across multiple
/// dimensions (NodeOutputRaD, DivRangeCheckRaD, SqrtRangeCheckRaD, etc.).
fn build_one_hot_rad_witness<F: JoltField>(
    lookup_indices: &[LookupBits],
    d: usize,
) -> MultilinearPolynomial<F> {
    let one_hot_params = OneHotParams::new(lookup_indices.len().log_2());
    let addresses: Vec<_> = lookup_indices
        .par_iter()
        .map(|lookup_index| Some(one_hot_params.lookup_index_chunk(lookup_index.into(), d) as u16))
        .collect();
    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
        addresses,
        one_hot_params.k_chunk,
    ))
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
impl<F: JoltField> WitnessGenerator<F> for CommittedPolynomial {
    #[tracing::instrument(skip_all)]
    fn generate_witness(&self, model: &Model, trace: &Trace) -> MultilinearPolynomial<F> {
        match self {
            CommittedPolynomial::NodeOutputRaD(node_idx, d) => {
                let computation_node = &model.graph.nodes[node_idx];
                let layer_data = Trace::layer_data(trace, computation_node);
                let is_interleaved_operands = computation_node.is_interleaved_operands();
                let lookup_indices = compute_lookup_indices_from_operands(
                    &layer_data.operands,
                    is_interleaved_operands,
                );
                build_one_hot_rad_witness(&lookup_indices, *d)
            }
            CommittedPolynomial::DivNodeQuotient(node_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(computation_node.operator, Operator::Div(_)));
                let layer_data = Trace::layer_data(trace, computation_node);
                let q = layer_data.output;
                MultilinearPolynomial::from(q.clone())
            }
            CommittedPolynomial::ScalarConstDivNodeRemainder(node_idx) => {
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
            CommittedPolynomial::RsqrtNodeRsqrt(node_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(computation_node.operator, Operator::Rsqrt(_)));
                let layer_data = Trace::layer_data(trace, computation_node);
                let rsqrt = layer_data.output;
                MultilinearPolynomial::from(rsqrt.clone())
            }
            CommittedPolynomial::RsqrtNodeInv(node_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(computation_node.operator, Operator::Rsqrt(_)));
                let layer_data = Trace::layer_data(trace, computation_node);
                let [left_operand] = layer_data.operands[..] else {
                    panic!("Expected one operand for Rsqrt operation")
                };
                let inv_data: Vec<i32> = left_operand.iter().map(|&x| Q_SQUARE / x).collect();
                MultilinearPolynomial::from(inv_data)
            }
            CommittedPolynomial::DivRangeCheckRaD(node_idx, d) => {
                build_range_check_rad_witness::<F, DivRangeCheckOperands>(
                    model, trace, *node_idx, *d,
                )
            }
            CommittedPolynomial::SqrtRangeCheckRaD(node_idx, d) => {
                build_range_check_rad_witness::<F, RsRangeCheckOperands>(
                    model, trace, *node_idx, *d,
                )
            }
            CommittedPolynomial::SqrtDivRangeCheckRaD(node_idx, d) => {
                build_range_check_rad_witness::<F, RiRangeCheckOperands>(
                    model, trace, *node_idx, *d,
                )
            }
            CommittedPolynomial::TeleportRangeCheckRaD(node_idx, d) => {
                build_range_check_rad_witness::<F, TeleportRangeCheckOperands>(
                    model, trace, *node_idx, *d,
                )
            }
            // TODO: Generate batch witness for Softmax committed polynomials
            CommittedPolynomial::SoftmaxRemainder(node_idx, feature_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(
                    computation_node.operator,
                    Operator::SoftmaxAxes(_)
                ));
                let st = softmax_chunk_trace(computation_node, trace, *feature_idx);
                let remainder_data: Vec<i32> = st
                    .exp_q_values
                    .iter()
                    .map(|&a| adjusted_remainder(a * S as i32, st.exp_sum_q))
                    .collect();
                MultilinearPolynomial::from(Tensor::<i32>::construct(
                    remainder_data,
                    st.exp_q_values.dims().to_vec(),
                ))
            }
            CommittedPolynomial::SoftmaxExponentiationRaD(node_idx, feature_idx, d_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(
                    computation_node.operator,
                    Operator::SoftmaxAxes(_)
                ));
                let st = softmax_chunk_trace(computation_node, trace, *feature_idx);
                let lookup_indices: Vec<usize> = st
                    .abs_centered_logits
                    .par_iter()
                    .map(|&x| x as usize)
                    .collect();
                let one_hot_params =
                    OneHotParams::from_config_and_log_K(&OneHotConfig::default(), LOG_EXP_LUT_SIZE);
                let h_indices = subprotocols::shout::compute_instruction_h_indices(
                    &lookup_indices,
                    &one_hot_params,
                );
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    h_indices[*d_idx]
                        .par_iter()
                        .map(|&h| h.map(|h| h as u16))
                        .collect(),
                    one_hot_params.k_chunk,
                ))
            }

            CommittedPolynomial::GatherRa(node_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(computation_node.operator, Operator::Gather(_)));
                let layer_data = Trace::layer_data(trace, computation_node);
                let indexes = &layer_data.operands[1];
                let non_zero_addresses: Vec<_> = indexes
                    .data()
                    .par_iter()
                    .map(|&index| Some(index as u16))
                    .collect();
                let input_dict = &model.graph.nodes.get(&computation_node.inputs[0]).unwrap();
                let num_words = input_dict.output_dims[0];
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    non_zero_addresses,
                    num_words,
                ))
            }

            CommittedPolynomial::TanhRaD(node_idx, d_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                let Operator::Tanh(inner) = &computation_node.operator else {
                    panic!("Expected Tanh operator for TanhRa committed polynomial");
                };
                let layer_data = Trace::layer_data(trace, computation_node);
                let input = &layer_data.operands[0];
                let (quotient, _remainder) = compute_division(input, inner.tau);
                let lookup_indices: Vec<usize> = quotient
                    .par_iter()
                    .map(|&x| n_bits_to_usize(x, inner.log_table))
                    .collect();
                let one_hot_params =
                    OneHotParams::from_config_and_log_K(&OneHotConfig::default(), inner.log_table);
                let h_indices = subprotocols::shout::compute_instruction_h_indices(
                    &lookup_indices,
                    &one_hot_params,
                );
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    h_indices[*d_idx]
                        .par_iter()
                        .map(|&h| h.map(|h| h as u16))
                        .collect(),
                    one_hot_params.k_chunk,
                ))
            }
        }
    }
}
