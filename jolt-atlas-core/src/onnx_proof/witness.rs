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
    onnx_proof::{
        neural_teleport::{division::compute_division, n_bits_to_usize},
        ops::rsqrt::rsqrt_dividend,
        ops::softmax_last_axis::{generate_exp_lut_decomposed, softmax_z},
        range_checking::range_check_operands::{
            range_check_lookup_indices, DivRangeCheckOperands, MeanOfSquaresRangeCheckOperands,
            RangeCheckWitness, RiRangeCheckOperands, RsRangeCheckOperands,
            TeleportRangeCheckOperands,
        },
    },
    utils::{adjusted_remainder, compute_lookup_indices_from_operands},
};
use atlas_algebra::{
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
    utils::{lookup_bits::LookupBits, math::Math},
};
use atlas_sumcheck::config::{OneHotConfig, OneHotParams};
use common::{
    consts::{ACTIVATION_BOUND, ACTIVATION_TABLE_VARS, LOG_K, TRIG_PERIOD_MODULUS, XLEN},
    parallel::par_enabled,
    CommittedPoly,
};
use rayon::prelude::*;

/// Builds a one-hot RaD witness for any of the range-checking operand types.
///
/// Every range-check variant (`DivRangeCheckRaD`, `SqrtRangeCheckRaD`, etc.) follows the
/// same three steps: get operand tensors via the trait, compute lookup indices, then call
/// `build_one_hot_rad_witness`. This helper collapses all four arms to a single call site.
fn build_range_check_rad_witness<F: JoltField, H: RangeCheckWitness>(
    model: &Model,
    trace: &Trace,
    node_idx: usize,
    d: usize,
) -> MultilinearPolynomial<F> {
    let computation_node = &model.view(node_idx);
    let inputs = TraceView::padded_operand_tensors(trace, computation_node);
    let [left_operand, right_operand] = H::build_lookup_operands(computation_node, trace, &inputs)
        .try_into()
        .expect("Expected two operands for range-checking");

    let lookup_indices = range_check_lookup_indices(&left_operand, &right_operand);
    build_one_hot_rad_witness(&lookup_indices, d, LOG_K)
}

/// Builds a one-hot polynomial witness for the `d`-th dimension of a read-after-decompose (RaD)
/// address polynomial.
///
/// This pattern appears in every operand type that decomposes lookup indices across multiple
/// dimensions (NodeOutputRaD, DivRangeCheckRaD, SqrtRangeCheckRaD, etc.).
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
    d: usize,
) -> MultilinearPolynomial<F> {
    let bound = 1i32 << ACTIVATION_BOUND;
    let clamped = input.map(|x| x.clamp(-bound, bound - 1));
    let lookup_indices: Vec<usize> = clamped
        .par_iter()
        .with_min_len(par_enabled())
        .map(|&x| n_bits_to_usize(x, ACTIVATION_TABLE_VARS))
        .collect();
    let one_hot_params =
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), ACTIVATION_TABLE_VARS);
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

/// Generate the witnesses for **all** of a node's committed polynomials. A MeanOfSquares node's
/// range-check lookup operands are computed once and every one-hot chunk is sliced from them;
/// every other polynomial goes through [`WitnessGenerator::generate_witness`].
#[tracing::instrument(skip_all, name = "generate_node_witnesses", fields(node = node.idx))]
pub(crate) fn generate_node_witnesses<F: JoltField>(
    node: &ComputationNode,
    model: &Model,
    trace: &Trace,
) -> Vec<(CommittedPoly, MultilinearPolynomial<F>)> {
    use crate::onnx_proof::ops::NodeCommittedPolynomials;

    let mut mos_range_bits: Option<Vec<LookupBits>> = None;

    NodeCommittedPolynomials::get_committed_polynomials(node)
        .expect("the model's nodes have the shapes their ops prove")
        .into_iter()
        .map(|poly| {
            let witness = match &poly {
                CommittedPoly::MeanOfSquaresRangeCheckRaD(_, d) => {
                    let bits = mos_range_bits.get_or_insert_with(|| {
                        let padded_operands = TraceView::padded_operand_tensors(trace, node);
                        let [left, right] = MeanOfSquaresRangeCheckOperands::build_lookup_operands(
                            node,
                            trace,
                            &padded_operands,
                        )
                        .try_into()
                        .unwrap();

                        range_check_lookup_indices(&left, &right)
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
                let computation_node = &model.view(*node_idx);
                let layer_data = TraceView::layer_data(trace, computation_node);
                let padded_operands: Vec<_> = layer_data
                    .operands
                    .iter()
                    .map(|tensor| tensor.padded_pow2())
                    .collect();
                let operand_refs: Vec<_> = padded_operands.iter().collect();
                let lookup_indices = compute_lookup_indices_from_operands(&operand_refs, false);
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
                let lookup_indices = compute_lookup_indices_from_operands(&operand_refs, false);
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
                let lookup_indices = compute_lookup_indices_from_operands(&operand_refs, false);
                build_one_hot_rad_witness(&lookup_indices, *d, XLEN)
            }
            CommittedPoly::ActivationSmallRaD(node_idx, d) => {
                let computation_node = &model.view(*node_idx);
                let layer_data = TraceView::layer_data(trace, computation_node);
                let input = &layer_data.operands[0];
                build_activation_small_rad_witness(input, *d)
            }
            CommittedPoly::GlobalRemainderDigit(..)
            | CommittedPoly::GlobalClampOutDigit(..)
            | CommittedPoly::GlobalClampSlackRaD(..) => {
                unreachable!("packed clamp bucket witnesses are built at the model level")
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
                let tau = match &computation_node.operator {
                    Op::Cos(_) | Op::Sin(_) => TRIG_PERIOD_MODULUS as i32,
                    _ => unreachable!(
                        "teleport quotient witness requested for non-teleport operator"
                    ),
                };
                let layer_data = TraceView::layer_data(trace, computation_node);
                let input = &layer_data.operands[0];
                let (quotient, _remainder) = compute_division(input, tau);
                MultilinearPolynomial::from(quotient)
            }
            CommittedPoly::ScalarConstDivNodeRemainder(node_idx) => {
                let computation_node = &model.view(*node_idx);
                let Op::ScalarConstDiv(op) = &computation_node.operator else {
                    panic!("Expected ScalarConstDiv operator at node {node_idx}");
                };
                let b = op.divisor;
                let layer_data = TraceView::layer_data(trace, computation_node);
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
                let computation_node = &model.view(*node_idx);
                assert!(matches!(computation_node.operator, Op::Rsqrt(_)));
                let layer_data = TraceView::layer_data(trace, computation_node);
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
                use crate::onnx_proof::neural_teleport::cos::COS_TABLE_VARS;
                use common::consts::TRIG_DOWNSCALE_BITS;

                let computation_node = &model.view(*node_idx);
                assert!(
                    matches!(computation_node.operator, Op::Cos(_) | Op::Sin(_)),
                    "Expected Cos or Sin operator for CosRaD/SinRaD committed polynomial"
                );
                let layer_data = TraceView::layer_data(trace, computation_node);
                let input = &layer_data.operands[0];

                let (_quotient, remainder) = compute_division(input, TRIG_PERIOD_MODULUS as i32);
                let lookup_indices: Vec<usize> = remainder
                    .par_iter()
                    .with_min_len(par_enabled())
                    .map(|&x| (x >> TRIG_DOWNSCALE_BITS) as usize)
                    .collect();
                let one_hot_params =
                    OneHotParams::from_config_and_log_K(&OneHotConfig::default(), COS_TABLE_VARS);
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
                let layer_data = TraceView::layer_data(trace, computation_node);
                let input = &layer_data.operands[0];
                let (_quotient, remainder) = compute_division(input, TRIG_PERIOD_MODULUS as i32);
                let lookup_indices = compute_lookup_indices_from_operands(&[&remainder], false);
                build_one_hot_rad_witness(&lookup_indices, *d_idx, XLEN)
            }
            CommittedPoly::SoftmaxRemainderRaD(node_idx, d) => {
                let node = &model.view(*node_idx);
                let log_scale = softmax_op_scale(node) as usize;
                let st = softmax_last_axis_full_trace(node, trace);
                let lookup_indices: Vec<usize> = st.r.iter().map(|&v| v as usize).collect();
                build_onehot_witness(&lookup_indices, log_scale, *d)
            }
            CommittedPoly::SoftmaxExpRemainderRaD(node_idx, d) => {
                let node = &model.view(*node_idx);
                let log_scale = softmax_op_scale(node) as usize;
                let st = softmax_last_axis_full_trace(node, trace);
                let lookup_indices: Vec<usize> = st.r_exp.iter().map(|&v| v as usize).collect();
                build_onehot_witness(&lookup_indices, log_scale, *d)
            }
            CommittedPoly::SoftmaxZHiRaD(node_idx, d) => {
                let node = &model.view(*node_idx);
                let st = softmax_last_axis_full_trace(node, trace);
                let decomp = generate_exp_lut_decomposed(st.scale as i32)
                    .expect("the softmax multiplier is a scale the semantics define");
                let log_hi = decomp.lut_hi.len().next_power_of_two().log_2();
                let lookup_indices: Vec<usize> = st.z_hi.iter().map(|&v| v as usize).collect();
                build_onehot_witness(&lookup_indices, log_hi, *d)
            }
            CommittedPoly::SoftmaxZLoRaD(node_idx, d) => {
                let node = &model.view(*node_idx);
                let st = softmax_last_axis_full_trace(node, trace);
                let decomp = generate_exp_lut_decomposed(st.scale as i32)
                    .expect("the softmax multiplier is a scale the semantics define");
                let log_lo = decomp.lut_lo.len().next_power_of_two().log_2();
                let lookup_indices: Vec<usize> = st.z_lo.iter().map(|&v| v as usize).collect();
                build_onehot_witness(&lookup_indices, log_lo, *d)
            }
            CommittedPoly::SoftmaxClampRaD(node_idx, d) => {
                // Softmax's saturating-clamp lookup: the lookup index is the raw
                // `z = max_k - x` value (no offset, `z >= 0` always).
                let node = &model.view(*node_idx);
                let &last_dim = node
                    .output_dims
                    .last()
                    .expect("softmax node must have at least one output dimension");
                let st = softmax_last_axis_full_trace(node, trace);
                let z = softmax_z(&st.x, &st.max_k, last_dim);
                let operand = Tensor::from_slice(&z, &[z.len()])
                    .expect("softmax_z tensor construction")
                    .padded_pow2();
                let lookup_indices = compute_lookup_indices_from_operands(&[&operand], false);
                build_one_hot_rad_witness(&lookup_indices, *d, XLEN)
            }
        }
    }
}

/// The softmax node's log2 scale (the model's global scale).
fn softmax_op_scale(node: &ComputationNode) -> i32 {
    assert!(
        matches!(node.operator, Op::SoftmaxLastAxis(_)),
        "Expected SoftmaxLastAxis at node {}",
        node.idx
    );
    node.scale as i32
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
