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

use atlas_trace::{Model, Op, Tensor, Trace};

use crate::node_view::ComputationNode;
use crate::node_view::ModelView;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    onnx_proof::ops::softmax_last_axis::{addresses, table_addresses},
    utils::{compute_lookup_indices, n_bits_to_usize},
};
use atlas_algebra::{
    field::JoltField,
    poly::{multilinear::MultilinearPolynomial, one_hot::OneHotPolynomial},
    utils::{lookup_bits::LookupBits, math::Math},
};
use atlas_sumcheck::config::{OneHotConfig, OneHotParams};
use common::{
    consts::{SOFTMAX_CLAMP_BOUND, XLEN},
    parallel::par_enabled,
    CommittedPoly,
};
use rayon::prelude::*;

/// Builds a one-hot polynomial witness for the `d`-th dimension of a read-after-decompose (RaD)
/// address polynomial.
///
/// This pattern appears in every operand type that decomposes lookup indices across multiple
/// dimensions (NodeOutputRaD, SymmetricClampRaD, etc.).
pub(crate) fn build_one_hot_rad_witness<F: JoltField>(
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

/// Builds a one-hot RaD witness for the clamped Erf/Sigmoid/Tanh variants' small
/// dense activation-table execution stage: clamps the raw input, maps to lookup
/// indices, then constructs the `d`-th one-hot address chunk.
fn build_activation_small_rad_witness<F: JoltField>(
    input: &Tensor<i32>,
    vars: usize,
    d: usize,
) -> MultilinearPolynomial<F> {
    let bound = 1i32 << (vars - 1);
    let clamped = input.map(|x| x.clamp(-bound, bound - 1));
    let lookup_indices: Vec<usize> = clamped
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&x| n_bits_to_usize(x, vars))
        .collect();
    let one_hot_params = OneHotParams::from_config_and_log_K(&OneHotConfig::default(), vars);
    let h_indices =
        atlas_sumcheck::shout::compute_instruction_h_indices(&lookup_indices, &one_hot_params);
    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
        h_indices[d]
            .par_iter()
            .with_min_len(par_enabled())
            .map(|&h| h.map(|h| h as u16))
            .collect(),
        one_hot_params.k_chunk,
    ))
}

/// Generate the witnesses for **all** of a node's committed polynomials, each through
/// [`WitnessGenerator::generate_witness`].
#[tracing::instrument(skip_all, name = "generate_node_witnesses", fields(node = node.idx))]
pub(crate) fn generate_node_witnesses<F: JoltField>(
    node: &ComputationNode,
    model: &Model,
    trace: &Trace,
) -> Vec<(CommittedPoly, MultilinearPolynomial<F>)> {
    use crate::onnx_proof::ops::NodeCommittedPolynomials;

    NodeCommittedPolynomials::get_committed_polynomials(node)
        .expect("the model's nodes have the shapes their ops prove")
        .into_iter()
        .map(|poly| {
            let witness = poly.generate_witness(model, trace);
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
                let computation_node = &model.view(*node_idx);
                let layer_data = TraceView::layer_data(trace, computation_node);
                let padded_operands: Vec<_> = layer_data
                    .operands
                    .iter()
                    .map(|tensor| tensor.padded_pow2())
                    .collect();
                let operand_refs: Vec<_> = padded_operands.iter().collect();
                let lookup_indices = compute_lookup_indices(operand_refs[0]);
                build_one_hot_rad_witness(&lookup_indices, *d, XLEN)
            }
            CommittedPoly::SymmetricClampRaD(node_idx, d) => {
                let computation_node = &model.view(*node_idx);
                let layer_data = TraceView::layer_data(trace, computation_node);
                let padded_operands: Vec<_> = layer_data
                    .operands
                    .iter()
                    .map(|tensor| tensor.padded_pow2())
                    .collect();
                let operand_refs: Vec<_> = padded_operands.iter().collect();
                let lookup_indices = compute_lookup_indices(operand_refs[0]);
                build_one_hot_rad_witness(&lookup_indices, *d, XLEN)
            }
            CommittedPoly::ActivationClampRaD(node_idx, d) => {
                let computation_node = &model.view(*node_idx);
                let layer_data = TraceView::layer_data(trace, computation_node);
                let padded_operands: Vec<_> = layer_data
                    .operands
                    .iter()
                    .map(|tensor| tensor.padded_pow2())
                    .collect();
                let operand_refs: Vec<_> = padded_operands.iter().collect();
                let lookup_indices = compute_lookup_indices(operand_refs[0]);
                build_one_hot_rad_witness(&lookup_indices, *d, XLEN)
            }
            CommittedPoly::ActivationSmallRaD(node_idx, d) => {
                let computation_node = &model.view(*node_idx);
                let layer_data = TraceView::layer_data(trace, computation_node);
                let input = &layer_data.operands[0];
                let vars = crate::onnx_proof::lookup::SmallTableDef::for_node(computation_node)
                    .expect("the node reads a small table")
                    .vars;
                build_activation_small_rad_witness(input, vars, *d)
            }
            CommittedPoly::GlobalRemainderDigit(..) => {
                unreachable!("range bucket witnesses are built at the model level")
            }
            CommittedPoly::DivNodeQuotient(node_idx) => {
                let computation_node = &model.view(*node_idx);
                assert!(matches!(computation_node.operator, Op::Div(_)));
                let layer_data = TraceView::layer_data(trace, computation_node);
                let q = layer_data.output;
                MultilinearPolynomial::from(q.clone())
            }
            CommittedPoly::TeleportNodeQuotient(node_idx) => {
                let computation_node = &model.view(*node_idx);
                // For nodes that use the quotient for a lookup (erf/sigmoid ...),
                // the quotient is virtualized from a one-hot encoding.
                assert!(matches!(computation_node.operator, Op::Cos(_) | Op::Sin(_)));
                MultilinearPolynomial::from(
                    TraceView::teleport_advice(trace, computation_node)
                        .quotient
                        .clone(),
                )
            }
            CommittedPoly::RsqrtQuotient(node_idx) => {
                let computation_node = &model.view(*node_idx);
                assert!(matches!(computation_node.operator, Op::Rsqrt(_)));
                // `quotient = ⌊S³ / x̂⌋` can exceed i32 at higher scales (e.g. up to
                // `S³` when `x̂ = 1`), so it is committed as u64.
                let quotient_data: Vec<u64> = TraceView::rsqrt_advice(trace, computation_node)
                    .quotient
                    .iter()
                    .map(|&q| q as u64)
                    .collect();
                MultilinearPolynomial::from(quotient_data)
            }
            CommittedPoly::GatherRa(node_idx) => {
                let computation_node = &model.view(*node_idx);
                assert!(matches!(computation_node.operator, Op::GatherSmall(_)));
                let layer_data = TraceView::layer_data(trace, computation_node);
                let indexes = &layer_data.operands[1];
                let non_zero_addresses: Vec<_> = indexes
                    .data()
                    .par_iter()
                    .with_min_len(par_enabled())
                    .map(|&index| Some(index as u16))
                    .collect();
                let input_dict = &model.view(computation_node.inputs[0]);
                let num_words = input_dict.output_dims[0];
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    non_zero_addresses,
                    num_words,
                ))
            }
            CommittedPoly::GatherRaD(node_idx, d_idx) => {
                let computation_node = &model.view(*node_idx);
                let Op::GatherLarge(gather_op) = &computation_node.operator else {
                    panic!("Expected GatherLarge operator for GatherRaD committed polynomial");
                };
                let layer_data = TraceView::layer_data(trace, computation_node);
                let indexes = layer_data.operands[1].padded_pow2();
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
                let h_indices = atlas_sumcheck::shout::compute_instruction_h_indices(
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

            CommittedPoly::CosRaD(node_idx, d_idx) | CommittedPoly::SinRaD(node_idx, d_idx) => {
                use crate::onnx_proof::ops::trig::TRIG_TABLE_VARS;
                use common::consts::TRIG_DOWNSCALE_BITS;

                let computation_node = &model.view(*node_idx);
                assert!(
                    matches!(computation_node.operator, Op::Cos(_) | Op::Sin(_)),
                    "Expected Cos or Sin operator for CosRaD/SinRaD committed polynomial"
                );
                let remainder = &TraceView::teleport_advice(trace, computation_node).remainder;
                let lookup_indices: Vec<usize> = remainder
                    .par_iter()
                    .with_min_len(par_enabled())
                    .map(|&x| (x >> TRIG_DOWNSCALE_BITS) as usize)
                    .collect();
                let one_hot_params =
                    OneHotParams::from_config_and_log_K(&OneHotConfig::default(), TRIG_TABLE_VARS);
                let h_indices = atlas_sumcheck::shout::compute_instruction_h_indices(
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
            CommittedPoly::TrigDownscaleRaD(node_idx, d_idx) => {
                let computation_node = &model.view(*node_idx);
                assert!(
                    matches!(computation_node.operator, Op::Cos(_) | Op::Sin(_)),
                    "Expected Cos or Sin operator for TrigDownscaleRaD committed polynomial"
                );
                let remainder = &TraceView::teleport_advice(trace, computation_node).remainder;
                let lookup_indices = compute_lookup_indices(remainder);
                build_one_hot_rad_witness(&lookup_indices, *d_idx, XLEN)
            }
            CommittedPoly::SoftmaxShift(node_idx) => {
                let node = &model.view(*node_idx);
                let st = softmax_last_axis_full_trace(node, trace);
                let mut shift = st.shift.clone();
                shift.resize(shift.len().next_power_of_two(), 0);
                MultilinearPolynomial::from(shift)
            }
            CommittedPoly::SoftmaxAddrRaD(node_idx, d) => {
                let node = &model.view(*node_idx);
                let st = softmax_last_axis_full_trace(node, trace);
                build_onehot_witness(&table_addresses(node, st), SOFTMAX_CLAMP_BOUND, *d)
            }
            CommittedPoly::SoftmaxClampRaD(node_idx, d) => {
                // The lookup index is the signed address `u = c − x`.
                let node = &model.view(*node_idx);
                let st = softmax_last_axis_full_trace(node, trace);
                let lookup_indices = compute_lookup_indices(&addresses(node, st));
                build_one_hot_rad_witness(&lookup_indices, *d, XLEN)
            }
        }
    }
}

/// The softmax node's full witness, captured in the trace at execution time
/// (the old pipeline re-ran the decomposed softmax here).
fn softmax_last_axis_full_trace<'a>(
    node: &ComputationNode,
    trace: &'a Trace,
) -> &'a atlas_trace::SoftmaxWitness {
    trace
        .softmax_witness(node.idx)
        .unwrap_or_else(|| panic!("softmax node {} has no trace witness", node.idx))
}

/// Build a one-hot RaD witness for a Shout ra
/// polynomial, given raw lookup indices and the log2 of the table /
/// range-check size.
fn build_onehot_witness<F: JoltField>(
    lookup_indices: &[usize],
    log_k: usize,
    d: usize,
) -> MultilinearPolynomial<F> {
    let one_hot_params = OneHotParams::from_config_and_log_K(&OneHotConfig::default(), log_k);
    let h_indices =
        atlas_sumcheck::shout::compute_instruction_h_indices(lookup_indices, &one_hot_params);
    MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
        h_indices[d]
            .par_iter()
            .with_min_len(par_enabled())
            .map(|&h| h.map(|h| h as u16))
            .collect(),
        one_hot_params.k_chunk,
    ))
}
