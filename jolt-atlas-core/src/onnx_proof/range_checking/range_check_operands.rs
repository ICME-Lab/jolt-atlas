//! Range-check operand helpers, with the prover's lookup-operand builders.

pub use jolt_atlas_verifier::onnx_proof::range_checking::range_check_operands::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::{ops::mean_of_squares::mos_divisor, trace_view::TraceView};
use crate::utils::compute_lookup_indices_from_operands;
use atlas_algebra::utils::lookup_bits::LookupBits;
use atlas_trace::{NodeId, Op, Tensor, Trace};
use common::consts::TRIG_PERIOD_MODULUS;

/// The prover's side of a [`RangeCheckingOperandsTrait`] helper: the lookup operands
/// (remainder and bound) it range-checks.
pub trait RangeCheckWitness: RangeCheckingOperandsTrait {
    // NOTE: In the future, range-checks with a constant value such as "tau" (neural teleportation) may be proven using a single-operand lookup table (#253).
    // Hence, "build_lookup_operands" returns a vector of tensors, to account for both cases.
    /// Builds the encoding operands for the lookup table from `node`'s operand tensors.
    /// `trace` is available so implementations can reuse trace-captured intermediates instead
    /// of re-executing kernels.
    fn build_lookup_operands(
        node: &ComputationNode,
        trace: &Trace,
        operand_tensors: &[Tensor<i32>],
    ) -> Vec<Tensor<i32>>;
}

/// The interleaved lookup indices of a range check's `(remainder, bound)` operands.
pub fn range_check_lookup_indices(
    left_operand: &Tensor<i32>,
    right_operand: &Tensor<i32>,
) -> Vec<LookupBits> {
    compute_lookup_indices_from_operands(&[left_operand, right_operand], true)
}

impl RangeCheckWitness for DivRangeCheckOperands {
    fn build_lookup_operands(
        node: &ComputationNode,
        trace: &Trace,
        operand_tensors: &[Tensor<i32>],
    ) -> Vec<Tensor<i32>> {
        assert_eq!(
            operand_tensors.len(),
            2,
            "Expected exactly two operand tensors"
        );
        let divisor = &operand_tensors[1];
        vec![
            TraceView::division_remainder(trace, node).clone(),
            divisor.clone(),
        ]
    }
}

impl RangeCheckWitness for RiRangeCheckOperands {
    fn build_lookup_operands(
        node: &ComputationNode,
        trace: &Trace,
        operand_tensors: &[Tensor<i32>],
    ) -> Vec<Tensor<i32>> {
        assert_eq!(
            operand_tensors.len(),
            1,
            "Expected exactly one operand tensor"
        );
        let input_tensor = &operand_tensors[0];
        vec![
            TraceView::rsqrt_advice(trace, node).div_remainder.clone(),
            input_tensor.clone(),
        ]
    }
}

impl RangeCheckWitness for RsRangeCheckOperands {
    fn build_lookup_operands(
        node: &ComputationNode,
        trace: &Trace,
        operand_tensors: &[Tensor<i32>],
    ) -> Vec<Tensor<i32>> {
        assert_eq!(
            operand_tensors.len(),
            1,
            "Expected exactly one operand tensor"
        );
        // Quantized sqrt(q) outputs v such that v² ≤ q < (v+1)², hence the remainder
        // r = q − v² satisfies 0 ≤ r < 2v + 1.
        let bound = trace.node_outputs[node.idx].map(|v| 2 * v + 1);
        vec![
            TraceView::rsqrt_advice(trace, node).sqrt_remainder.clone(),
            bound,
        ]
    }
}

impl RangeCheckWitness for MeanOfSquaresRangeCheckOperands {
    fn build_lookup_operands(
        node: &ComputationNode,
        trace: &Trace,
        _operand_tensors: &[Tensor<i32>],
    ) -> Vec<Tensor<i32>> {
        // The executor always captures the fused intermediates; read the
        // rescaling remainder from the trace instead of re-executing the
        // whole O(count·elements) mean-of-squares kernel.
        let remainder = trace
            .fused(NodeId(node.idx as u32))
            .unwrap_or_else(|| {
                panic!(
                    "mean-of-squares node {} has no fused intermediates in the trace",
                    node.idx
                )
            })
            .remainder
            .clone();
        let Op::MeanOfSquares(op) = &node.operator else {
            panic!(
                "MeanOfSquares range check: unexpected operator {:?}",
                node.operator
            )
        };
        let divisor_i32 = i32::try_from(mos_divisor(op, node.scale))
            .expect("MeanOfSquares divisor N·2^S must fit i32 for the range check");
        let bound = Tensor::construct(vec![divisor_i32], vec![1])
            .expand(remainder.dims())
            .unwrap();
        vec![remainder, bound]
    }
}

impl RangeCheckWitness for TeleportRangeCheckOperands {
    fn build_lookup_operands(
        node: &ComputationNode,
        trace: &Trace,
        operand_tensors: &[Tensor<i32>],
    ) -> Vec<Tensor<i32>> {
        assert_eq!(
            operand_tensors.len(),
            1,
            "Expected exactly one operand tensor for neural teleportation"
        );
        let input_tensor = &operand_tensors[0];

        let tau = TRIG_PERIOD_MODULUS as i32;
        let divisor_tensor = Tensor::construct(vec![tau], vec![1])
            .expand(input_tensor.dims())
            .unwrap();

        vec![
            TraceView::teleport_advice(trace, node).remainder.clone(),
            divisor_tensor,
        ]
    }
}
