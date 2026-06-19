use atlas_onnx_tracer::{
    model::consts::FOUR_PI_APPROX,
    node::ComputationNode,
    ops::{MeanOfSquares, Operator},
    tensor::Tensor,
};
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    utils::lookup_bits::LookupBits,
};
use num::integer::Roots;

use crate::{
    onnx_proof::{neural_teleport::division::compute_division, range_checking::RangeCheckEncoding},
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
/// Implementors only need to provide [`Self::new_params`],
/// [`Self::get_operands_tensors`], and [`Self::rad_poly`].
///
/// Common accessors (`transform_operand_claims`,
/// `compute_lookup_indices`) are provided as defaults.
pub trait RangeCheckingOperandsTrait {
    /// Create a new instance of the implementing type from a computation node.
    fn new(node: &ComputationNode) -> Self;

    /// Create a new base instance from a computation node.
    fn new_params(node: &ComputationNode) -> RangeCheckOperandsBase;

    // NOTE: In the future, range-checks with a constant value such as "tau" (neural teleportation) may be proven using a single-operand lookup table (#253).
    // Hence, "build_lookup_operands" returns a vector of tensors, to account for both cases.
    /// Builds the encoding operands for the lookup table from the model's operand tensors.
    fn build_lookup_operands(&self, operand_tensors: &[Tensor<i32>]) -> Vec<Tensor<i32>>;

    /// Get the committed polynomial for the d-th dimension of the range-check read-address.
    fn rad_poly(index: usize, d: usize) -> CommittedPoly;

    /// Optional transformation applied to the operand claims before returning.
    ///
    /// Override this for operations where the range-check bound is not directly
    /// the right operand (e.g., for sqrt range-check, the bound is `2·v̂ + 1`).
    fn transform_operand_claims<F: JoltField>(&self, claims: Vec<F>) -> (F, F) {
        (claims[0], claims[1])
    }

    /// Optional transformation applied to the claim of the range-check output before returning.
    ///
    /// Override this for operations where the range-check output claim is not directly the Ra polynomial.
    fn transform_output_claim<F: JoltField>(claim: F) -> F {
        claim
    }
}

/// Operands for division range-checking.
///
/// For integer division `a / b = q`, there exists a remainder `r` such that `a = q·b + r`.
/// This struct holds the operands needed to verify `0 ≤ r < b`.
pub struct DivRangeCheckOperands;

/// Operands for the fused rsqrt division range-check.
///
/// For the quotient `q = ⌊S³ / x̂⌋`, verifies the division remainder
/// `r = S³ mod x̂` satisfies `0 ≤ r < x̂`.
pub struct RiRangeCheckOperands {
    scale: i32,
}

/// Operands for the fused rsqrt square-root range-check.
///
/// For the output `v̂ = ⌊√q⌋` of the quotient `q = ⌊S³ / x̂⌋`, verifies that the
/// remainder `r = q - v̂²` satisfies `0 ≤ r ≤ 2·v̂`.
pub struct RsRangeCheckOperands {
    scale: i32,
}

impl RangeCheckingOperandsTrait for DivRangeCheckOperands {
    fn new(_node: &ComputationNode) -> Self {
        Self
    }

    fn new_params(node: &ComputationNode) -> RangeCheckOperandsBase {
        RangeCheckOperandsBase {
            node_idx: node.idx,
            remainder: VirtualPoly::DivRemainder(node.idx),
            bound: Some(VirtualPoly::NodeOutput(node.inputs[1])),
            virtual_ra: VirtualPoly::DivRangeCheckRa(node.idx),
            operator: node.operator.clone(),
        }
    }

    fn build_lookup_operands(&self, operand_tensors: &[Tensor<i32>]) -> Vec<Tensor<i32>> {
        assert_eq!(
            operand_tensors.len(),
            2,
            "Expected exactly two operand tensors"
        );
        let dividend = &operand_tensors[0];
        let divisor = &operand_tensors[1];

        let remainder_data: Vec<i32> = dividend
            .iter()
            .zip(divisor.iter())
            .map(|(&a, &b)| adjusted_remainder(a, b))
            .collect();
        let remainder = Tensor::<i32>::construct(remainder_data, dividend.dims().to_vec());

        vec![remainder, divisor.clone()]
    }

    fn rad_poly(index: usize, d: usize) -> CommittedPoly {
        CommittedPoly::DivRangeCheckRaD(index, d)
    }
}

impl RangeCheckingOperandsTrait for RiRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        let Operator::Rsqrt(inner) = &node.operator else {
            panic!("Expected Rsqrt operator");
        };
        Self { scale: inner.scale }
    }

    fn new_params(node: &ComputationNode) -> RangeCheckOperandsBase {
        RangeCheckOperandsBase {
            node_idx: node.idx,
            remainder: VirtualPoly::DivRemainder(node.idx),
            bound: Some(VirtualPoly::NodeOutput(node.inputs[0])),
            virtual_ra: VirtualPoly::DivRangeCheckRa(node.idx),
            operator: node.operator.clone(),
        }
    }

    fn build_lookup_operands(&self, operand_tensors: &[Tensor<i32>]) -> Vec<Tensor<i32>> {
        assert_eq!(
            operand_tensors.len(),
            1,
            "Expected exactly one operand tensor"
        );
        let input_tensor = &operand_tensors[0];

        let s_cubed = 1i64 << (3 * self.scale);
        let remainder = {
            let data: Vec<i32> = input_tensor
                .iter()
                .map(|&x| (s_cubed % x as i64) as i32)
                .collect();
            Tensor::<i32>::construct(data, input_tensor.dims().to_vec())
        };

        vec![remainder, input_tensor.clone()]
    }

    fn rad_poly(index: usize, d: usize) -> CommittedPoly {
        CommittedPoly::SqrtDivRangeCheckRaD(index, d)
    }
}

impl RangeCheckingOperandsTrait for RsRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        let Operator::Rsqrt(inner) = &node.operator else {
            panic!("Expected Sqrt operator");
        };
        Self { scale: inner.scale }
    }

    fn new_params(node: &ComputationNode) -> RangeCheckOperandsBase {
        RangeCheckOperandsBase {
            node_idx: node.idx,
            remainder: VirtualPoly::SqrtRemainder(node.idx),
            bound: Some(VirtualPoly::NodeOutput(node.idx)),
            virtual_ra: VirtualPoly::SqrtRangeCheckRa(node.idx),
            operator: node.operator.clone(),
        }
    }

    /// For sqrt range check: the bound is `2·v̂ + 1`, not just `v̂`.
    fn transform_operand_claims<F: JoltField>(&self, claims: Vec<F>) -> (F, F) {
        let left_claim = claims[0];
        let right_claim = claims[1];
        (left_claim, right_claim + right_claim + F::one())
    }

    fn build_lookup_operands(&self, operand_tensors: &[Tensor<i32>]) -> Vec<Tensor<i32>> {
        assert_eq!(
            operand_tensors.len(),
            1,
            "Expected exactly one operand tensor"
        );
        let input_tensor = &operand_tensors[0];

        let s_cubed = 1i64 << (3 * self.scale);
        // `quotient = ⌊S³ / x̂⌋` and `out²` can exceed i32 at higher scales, so both
        // are computed in i64. The remainder `quotient − out²` is small (< 2·out+1).

        let (remainder, bound) = {
            let (rem_data, bound_data): (Vec<i32>, Vec<i32>) = input_tensor
                .iter()
                .map(|&inv| {
                    let inv_qsq = s_cubed / inv as i64;
                    let v = inv_qsq.sqrt();

                    // Quantized sqrt(x) outputs v such that v² ≤ x < (v+1)²,
                    // hence the remainder r = x - v² satisfies 0 ≤ r < 2v + 1.
                    let upper_bound = (2 * v + 1) as i32;

                    ((inv_qsq - v * v) as i32, upper_bound)
                })
                .unzip();
            (
                Tensor::<i32>::construct(rem_data, input_tensor.dims().to_vec()),
                Tensor::<i32>::construct(bound_data, input_tensor.dims().to_vec()),
            )
        };

        vec![remainder, bound]
    }

    fn rad_poly(index: usize, d: usize) -> CommittedPoly {
        CommittedPoly::SqrtRangeCheckRaD(index, d)
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
    /// The mean of squares operator
    op: MeanOfSquares,
}

impl RangeCheckingOperandsTrait for MeanOfSquaresRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        let op = match &node.operator {
            Operator::MeanOfSquares(op) => op.clone(),
            other => panic!("MeanOfSquares range check: unexpected operator {other:?}"),
        };
        Self { op }
    }

    fn new_params(node: &ComputationNode) -> RangeCheckOperandsBase {
        RangeCheckOperandsBase {
            node_idx: node.idx,
            remainder: VirtualPoly::RescaleRemainder(node.idx),
            bound: None,
            virtual_ra: VirtualPoly::MeanOfSquaresRangeCheckRa(node.idx),
            operator: node.operator.clone(),
        }
    }

    fn build_lookup_operands(&self, operand_tensors: &[Tensor<i32>]) -> Vec<Tensor<i32>> {
        let operand_refs: Vec<&Tensor<i32>> = operand_tensors.iter().collect();
        let remainder = mos_remainder(&self.op, &operand_refs);
        let divisor_i32 = mean_of_squares_divisor_i32(&self.op);
        let bound = Tensor::construct(vec![divisor_i32], vec![1])
            .expand(remainder.dims())
            .unwrap();
        vec![remainder, bound]
    }

    fn transform_operand_claims<F: JoltField>(&self, claims: Vec<F>) -> (F, F) {
        let left_claim = claims[0];

        (left_claim, F::from_i32(i32_divisor(mos_divisor(&self.op))))
    }

    fn rad_poly(index: usize, d: usize) -> CommittedPoly {
        CommittedPoly::MeanOfSquaresRangeCheckRaD(index, d)
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
    tau: i32,
}

impl RangeCheckingOperandsTrait for TeleportRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
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
        Self { tau }
    }

    fn new_params(node: &ComputationNode) -> RangeCheckOperandsBase {
        RangeCheckOperandsBase {
            node_idx: node.idx,
            remainder: VirtualPoly::TeleportRemainder(node.idx),
            // The bound is the constant τ, not a committed polynomial.
            bound: None,
            virtual_ra: VirtualPoly::TeleportRangeCheckRa(node.idx),
            operator: node.operator.clone(),
        }
    }

    fn build_lookup_operands(&self, operand_tensors: &[Tensor<i32>]) -> Vec<Tensor<i32>> {
        assert_eq!(
            operand_tensors.len(),
            1,
            "Expected exactly one operand tensor for neural teleportation"
        );
        let input_tensor = &operand_tensors[0];

        let (_, remainder) = compute_division(input_tensor, self.tau);
        let divisor_tensor = Tensor::construct(vec![self.tau], vec![1])
            .expand(input_tensor.dims())
            .unwrap();

        vec![remainder, divisor_tensor]
    }

    fn transform_operand_claims<F: JoltField>(&self, claims: Vec<F>) -> (F, F) {
        let left_claim = claims[0];

        (left_claim, F::from_i32(self.tau))
    }

    fn rad_poly(index: usize, d: usize) -> CommittedPoly {
        CommittedPoly::TeleportRangeCheckRaD(index, d)
    }
}

/// A wrapper struct that holds the range-checking information
pub struct RangeCheckOperands<Helper: RangeCheckingOperandsTrait> {
    base: RangeCheckOperandsBase,
    helper: Helper,
}

impl<H: RangeCheckingOperandsTrait> RangeCheckOperands<H> {
    /// Create a new range-checking operands instance from a computation node.
    pub fn new(node: &ComputationNode) -> Self {
        let base = H::new_params(node);
        Self {
            base,
            helper: H::new(node),
        }
    }

    /// Get the index of the computation node associated with this range-checking operands.
    pub fn node_idx(&self) -> usize {
        self.base.node_idx
    }

    /// Get the virtual polynomials representing the input operands for range-checking.
    pub fn get_input_operands_id(&self) -> Vec<VirtualPoly> {
        let mut ops = vec![self.base.remainder];
        if let Some(bound) = self.base.bound {
            ops.push(bound);
        }
        ops
    }

    /// Get the virtual polynomial representing the range-check read-address (Ra) output.
    pub fn get_output_operand_id(&self) -> VirtualPoly {
        self.base.virtual_ra
    }

    /// Extract the operand claims from the accumulator for the left and right operands.
    pub fn operand_claims<F: JoltField>(&self, accumulator: &dyn OpeningAccumulator<F>) -> (F, F) {
        let operand_claims = self
            .get_input_operands_id()
            .iter()
            .map(|operand| {
                let operand_id =
                    OpeningId::new(*operand, SumcheckId::NodeExecution(self.base.node_idx));
                let (_, claim) = accumulator.get_virtual_polynomial_opening(operand_id);
                claim
            })
            .collect::<Vec<_>>();

        self.helper.transform_operand_claims(operand_claims)
    }

    /// Computes the lookup operand tensors from the node input operands.
    pub fn build_lookup_operands(&self, operand_tensors: &[Tensor<i32>]) -> Vec<Tensor<i32>> {
        self.helper.build_lookup_operands(operand_tensors)
    }

    /// Computes the interleaved one-hot encoding required to perform the range-checking lookup.
    pub fn compute_lookup_indices(
        &self,
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
    ) -> Vec<LookupBits> {
        compute_lookup_indices_from_operands(&[left_operand, right_operand], true)
    }

    /// Gets the RA one hot encoding for one-hotness checks
    pub fn get_encoding(&self, node: &ComputationNode) -> RangeCheckEncoding {
        RangeCheckEncoding::new(node, self)
    }
}
