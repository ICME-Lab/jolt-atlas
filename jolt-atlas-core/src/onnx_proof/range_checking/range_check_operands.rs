use atlas_onnx_tracer::{
    model::{
        consts::FOUR_PI_APPROX,
        trace::{LayerData, Trace},
    },
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    utils::lookup_bits::LookupBits,
};

use crate::{
    onnx_proof::{
        neural_teleport::division::compute_division,
        ops::rsqrt::rsqrt_dividend,
    },
    utils::{adjusted_remainder, compute_lookup_indices_from_operands},
};
use atlas_onnx_tracer::ops::mean_of_squares::{mos_divisor, mos_remainder};

/// Common fields shared by all range-checking operand types.
///
/// Every operation that needs range-checking (Div, Rsqrt, Tanh) stores the same
/// triple of values: a node index, input operands, and a virtual read-address polynomial.
/// This struct eliminates the field duplication across the individual operand types.
pub struct RangeCheckOperandsBase {
    /// Index of the computation node in the graph.
    pub node_idx: usize,
    /// Polynomial to be range-checked
    pub remainder: VirtualPoly,
    /// Bound polynomial, or `None` when the bound is a field constant (e.g. τ for neural-teleport).
    pub bound: Option<VirtualPoly>,
    /// Virtual polynomial representing the range-check read-address (Ra).
    pub virtual_ra: VirtualPoly,
    /// The operator this range-check is associated with, used for operator-specific logic in operand extraction and claim transformation.
    pub operator: Operator,
}

/// Trait defining the interface for operations that require range-checking.
///
/// This trait provides all the information needed from operations (Div, Rsqrt, Tanh) that require
/// range-checking to verify that remainder values are within valid bounds. It abstracts the common
/// patterns for range-checking across different operations.
///
/// Implementors only need to provide [`Self::new`], [`Self::base`],
/// [`Self::get_operands_tensors`], and [`Self::rad_poly`].
/// Common accessors (`node_idx`, `get_input_operands`, `get_output_operand`,
/// `operand_claims`) are provided as defaults via [`Self::base`].
pub trait RangeCheckingOperandsTrait {
    /// Create a new operands instance from a computation node.
    fn new(node: &ComputationNode) -> Self;

    /// Returns a reference to the common base fields.
    fn base(&self) -> &RangeCheckOperandsBase;

    /// Get the index of the computation node these operands are associated with.
    fn node_idx(&self) -> usize {
        self.base().node_idx
    }

    /// Get the virtual polynomials representing the input operands for range-checking.
    fn get_input_operands(&self) -> Vec<VirtualPoly> {
        let mut ops = vec![self.base().remainder];
        if let Some(bound) = self.base().bound {
            ops.push(bound);
        }
        ops
    }

    /// Get the virtual polynomial representing the range-check read-address (Ra) output.
    fn get_output_operand(&self) -> VirtualPoly {
        self.base().virtual_ra
    }

    /// Optional transformation applied to the right operand claim before returning.
    ///
    /// Override this for operations where the range-check bound is a function of the
    /// right operand (e.g., sqrt range check uses `2·v̂ + 1` instead of `v̂`).
    fn transform_right_claim<F: JoltField>(&self, claim: F) -> F {
        claim
    }

    /// Extract the operand claims from the accumulator for the left and right operands.
    fn operand_claims<F: JoltField>(&self, accumulator: &dyn OpeningAccumulator<F>) -> (F, F) {
        let operand_claims = self
            .get_input_operands()
            .iter()
            .map(|operand| {
                let operand_id =
                    OpeningId::new(*operand, SumcheckId::NodeExecution(self.node_idx()));
                let (_, claim) = accumulator.get_virtual_polynomial_opening(operand_id);
                claim
            })
            .collect::<Vec<_>>();
        (
            operand_claims[0],
            self.transform_right_claim(operand_claims[1]),
        )
    }

    /// Extract or compute the operand tensors (remainder and bound) from the trace.
    ///
    /// Returns a tuple of (left_operand, right_operand) tensors used for range-checking.
    fn get_operands_tensors(trace: &Trace, node: &ComputationNode) -> (Tensor<i32>, Tensor<i32>);

    /// Compute lookup indices for the range-checking table from the operand tensors.
    ///
    /// Each lookup index verifies that `left_operand < right_operand` using the
    /// unsigned less-than lookup table.
    fn compute_lookup_indices(
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
    ) -> Vec<LookupBits> {
        compute_lookup_indices_from_operands(&[left_operand, right_operand], true)
    }

    /// Get the committed polynomial for the d-th dimension of the range-check read-address.
    fn rad_poly(&self, d: usize) -> CommittedPoly;
}

/// Operands for division range-checking.
///
/// For integer division `a / b = q`, there exists a remainder `r` such that `a = q·b + r`.
/// This struct holds the operands needed to verify `0 ≤ r < b`.
pub struct DivRangeCheckOperands {
    base: RangeCheckOperandsBase,
}

/// Operands for the fused rsqrt division range-check.
///
/// For the quotient `q = ⌊S³ / x̂⌋`, verifies the division remainder
/// `r = S³ mod x̂` satisfies `0 ≤ r < x̂`.
pub struct RiRangeCheckOperands {
    base: RangeCheckOperandsBase,
}

/// Operands for the fused rsqrt square-root range-check.
///
/// For the output `v̂ = ⌊√q⌋` of the quotient `q = ⌊S³ / x̂⌋`, verifies that the
/// remainder `r = q - v̂²` satisfies `0 ≤ r ≤ 2·v̂`.
pub struct RsRangeCheckOperands {
    base: RangeCheckOperandsBase,
}

impl RangeCheckingOperandsTrait for DivRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        Self {
            base: RangeCheckOperandsBase {
                node_idx: node.idx,
                remainder: VirtualPoly::DivRemainder(node.idx),
                bound: Some(VirtualPoly::NodeOutput(node.inputs[1])),
                virtual_ra: VirtualPoly::DivRangeCheckRa(node.idx),
                operator: node.operator.clone(),
            },
        }
    }

    fn base(&self) -> &RangeCheckOperandsBase {
        &self.base
    }

    fn get_operands_tensors(trace: &Trace, node: &ComputationNode) -> (Tensor<i32>, Tensor<i32>) {
        let LayerData {
            output: _,
            operands,
        } = Trace::layer_data(trace, node);

        let [dividend, divisor] = operands[..] else {
            panic!("Expected exactly two input tensors");
        };

        let remainder_data: Vec<i32> = dividend
            .iter()
            .zip(divisor.iter())
            .map(|(&a, &b)| adjusted_remainder(a, b))
            .collect();
        let remainder = Tensor::<i32>::construct(remainder_data, dividend.dims().to_vec());

        (remainder, divisor.clone())
    }

    fn rad_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::DivRangeCheckRaD(self.base.node_idx, d)
    }
}

impl RangeCheckingOperandsTrait for RiRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        Self {
            base: RangeCheckOperandsBase {
                node_idx: node.idx,
                remainder: VirtualPoly::DivRemainder(node.idx),
                bound: Some(VirtualPoly::NodeOutput(node.inputs[0])),
                virtual_ra: VirtualPoly::DivRangeCheckRa(node.idx),
                operator: node.operator.clone(),
            },
        }
    }

    fn base(&self) -> &RangeCheckOperandsBase {
        &self.base
    }

    fn get_operands_tensors(trace: &Trace, node: &ComputationNode) -> (Tensor<i32>, Tensor<i32>) {
        let LayerData {
            output: _,
            operands,
        } = Trace::layer_data(trace, node);

        let [input_tensor] = operands[..] else {
            panic!("Expected exactly one input tensor");
        };

        let s_cubed = rsqrt_dividend(node);
        let remainder = {
            let data: Vec<i32> = input_tensor
                .iter()
                .map(|&x| (s_cubed % x as i64) as i32)
                .collect();
            Tensor::<i32>::construct(data, input_tensor.dims().to_vec())
        };

        (remainder, input_tensor.clone())
    }

    fn rad_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::SqrtDivRangeCheckRaD(self.base.node_idx, d)
    }
}

impl RangeCheckingOperandsTrait for RsRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        Self {
            base: RangeCheckOperandsBase {
                node_idx: node.idx,
                remainder: VirtualPoly::SqrtRemainder(node.idx),
                bound: Some(VirtualPoly::NodeOutput(node.idx)),
                virtual_ra: VirtualPoly::SqrtRangeCheckRa(node.idx),
                operator: node.operator.clone(),
            },
        }
    }

    fn base(&self) -> &RangeCheckOperandsBase {
        &self.base
    }

    /// For sqrt range check: the bound is `2·v̂ + 1`, not just `v̂`.
    fn transform_right_claim<F: JoltField>(&self, claim: F) -> F {
        claim * F::from_i32(2) + F::one()
    }

    fn get_operands_tensors(trace: &Trace, node: &ComputationNode) -> (Tensor<i32>, Tensor<i32>) {
        let LayerData { output, operands } = Trace::layer_data(trace, node);

        let [input_tensor] = operands[..] else {
            panic!("Expected exactly one input tensor");
        };

        let s_cubed = rsqrt_dividend(node);
        // `quotient = ⌊S³ / x̂⌋` and `out²` can exceed i32 at higher scales, so both
        // are computed in i64. The remainder `quotient − out²` is small (< 2·out+1).
        let remainder = {
            let data: Vec<i32> = input_tensor
                .iter()
                .zip(output.iter())
                .map(|(&x, &out)| {
                    let quotient = s_cubed / x as i64;
                    (quotient - (out as i64) * (out as i64)) as i32
                })
                .collect();
            Tensor::<i32>::construct(data, input_tensor.dims().to_vec())
        };

        (remainder, output.clone())
    }

    // Override to implement the specific lookup indices computation for sqrt range check
    //
    // The fused rsqrt output is v̂ = ⌊√q⌋ for the quotient q = ⌊S³ / x̂⌋, so there is
    // a remainder r with:
    //     q = v̂ · v̂ + r
    //
    // Bounding r: let v = √q (not floored), so v = v̂ + e with 0 ≤ e < 1. Then
    //     q = v · v = (v̂ + e)² = v̂ · v̂ + 2·e·v̂ + e²
    // and since e < 1:
    //     0 ≤ r < 2 · v̂ + 1
    fn compute_lookup_indices(
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
    ) -> Vec<LookupBits> {
        let upper_bound = {
            let data = right_operand.iter().map(|v| 2 * v + 1).collect();
            Tensor::<i32>::construct(data, right_operand.dims().to_vec())
        };

        compute_lookup_indices_from_operands(&[left_operand, &upper_bound], true)
    }

    fn rad_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::SqrtRangeCheckRaD(self.base.node_idx, d)
    }
}

/// Operands for the fused mean-of-squares rescaling-remainder range check.
///
/// MoS computes `out = SatClamp((Σx²) / D)` with `D = N·2^S` (a per-node
/// **constant**); the reduction binds `Σx² = rescaled·D + R` and this check
/// proves `0 ≤ R < D`. Because the bound is constant it mirrors the
/// [`TeleportRangeCheckOperands`] pattern (`bound: None`) rather than Div's
/// per-element divisor .
pub struct MeanOfSquaresRangeCheckOperands {
    base: RangeCheckOperandsBase,
    /// The constant divisor `D = N·2^S`, recovered from the node's input dims.
    divisor: i64,
}

impl RangeCheckingOperandsTrait for MeanOfSquaresRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        Self {
            base: RangeCheckOperandsBase {
                node_idx: node.idx,
                remainder: VirtualPoly::RescaleRemainder(node.idx),
                // The bound is the constant `D = N·2^S`, not a committed polynomial.
                bound: None,
                virtual_ra: VirtualPoly::MeanOfSquaresRangeCheckRa(node.idx),
                operator: node.operator.clone(),
            },
            divisor: {
                let Operator::MeanOfSquares(op) = &node.operator else {
                    panic!("MeanOfSquares range check: expected MeanOfSquares operator");
                };
                mos_divisor(op)
            },
        }
    }

    fn base(&self) -> &RangeCheckOperandsBase {
        &self.base
    }

    fn get_operands_tensors(trace: &Trace, node: &ComputationNode) -> (Tensor<i32>, Tensor<i32>) {
        let LayerData { operands, .. } = Trace::layer_data(trace, node);
        let op = match &node.operator {
            Operator::MeanOfSquares(op) => op,
            other => panic!("MeanOfSquares range check: unexpected operator {other:?}"),
        };
        let remainder = mos_remainder(op, &operands);
        let divisor_i32 = mean_of_squares_divisor_i32(op);
        let bound = Tensor::construct(vec![divisor_i32], vec![1])
            .expand(remainder.dims())
            .unwrap();
        (remainder, bound)
    }

    /// The right claim is the constant `D`; only the remainder claim is fetched
    /// from the accumulator.
    fn operand_claims<F: JoltField>(&self, accumulator: &dyn OpeningAccumulator<F>) -> (F, F) {
        let remainder_id = OpeningId::new(
            self.base.remainder,
            SumcheckId::NodeExecution(self.base.node_idx),
        );
        let (_, remainder_claim) = accumulator.get_virtual_polynomial_opening(remainder_id);
        (remainder_claim, F::from_i32(i32_divisor(self.divisor)))
    }

    fn rad_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::MeanOfSquaresRangeCheckRaD(self.base.node_idx, d)
    }
}

/// The fused mean-of-squares divisor `D = N·2^S` as an `i32` (the range-check
/// value domain), recovered from the operator's stored count + scale.
fn mean_of_squares_divisor_i32(op: &atlas_onnx_tracer::ops::MeanOfSquares) -> i32 {
    i32_divisor(mos_divisor(op))
}

fn i32_divisor(divisor: i64) -> i32 {
    i32::try_from(divisor).expect("MeanOfSquares divisor N·2^S must fit i32 for the range check")
}

/// Operands for neural teleportation (tanh) range-checking.
///
/// For tanh computation involving division by τ, verifies that the remainder satisfies the bounds.
pub struct TeleportRangeCheckOperands {
    base: RangeCheckOperandsBase,
}

impl RangeCheckingOperandsTrait for TeleportRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        Self {
            base: RangeCheckOperandsBase {
                node_idx: node.idx,
                remainder: VirtualPoly::TeleportRemainder(node.idx),
                // The bound is the constant τ, not a committed polynomial.
                bound: None,
                virtual_ra: VirtualPoly::TeleportRangeCheckRa(node.idx),
                operator: node.operator.clone(),
            },
        }
    }

    fn base(&self) -> &RangeCheckOperandsBase {
        &self.base
    }

    fn get_operands_tensors(trace: &Trace, node: &ComputationNode) -> (Tensor<i32>, Tensor<i32>) {
        let LayerData {
            output: _,
            operands,
        } = Trace::layer_data(trace, node);

        let [input_tensor] = operands[..] else {
            panic!("Expected exactly one input tensor for neural teleportation");
        };

        let tau = match &node.operator {
            Operator::Tanh(inner) => inner.tau,
            Operator::Erf(inner) => inner.tau,
            Operator::Sigmoid(inner) => inner.tau,
            Operator::Cos(_) | Operator::Sin(_) => FOUR_PI_APPROX,
            _ => {
                panic!(
                    "Expected Tanh, Erf, Sigmoid, Cos, or Sin operator for neural teleportation division"
                )
            }
        };

        let (_, remainder) = compute_division(input_tensor, tau);
        let divisor_tensor = Tensor::construct(vec![tau], vec![1])
            .expand(input_tensor.dims())
            .unwrap();

        (remainder, divisor_tensor)
    }

    /// The right claim is the constant τ; only the remainder claim is fetched from the accumulator.
    fn operand_claims<F: JoltField>(&self, accumulator: &dyn OpeningAccumulator<F>) -> (F, F) {
        let remainder_id = OpeningId::new(
            self.base.remainder,
            SumcheckId::NodeExecution(self.base.node_idx),
        );
        let (_, remainder_claim) = accumulator.get_virtual_polynomial_opening(remainder_id);
        let tau = match &self.base().operator {
            Operator::Tanh(inner) => inner.tau,
            Operator::Erf(inner) => inner.tau,
            Operator::Sigmoid(inner) => inner.tau,
            Operator::Cos(_) | Operator::Sin(_) => FOUR_PI_APPROX,
            _ => {
                panic!(
                    "Expected Tanh, Erf, Sigmoid, Cos, or Sin operator for neural teleportation division"
                )
            }
        };
        (remainder_claim, F::from_i32(tau))
    }

    fn rad_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::TeleportRangeCheckRaD(self.base.node_idx, d)
    }
}
