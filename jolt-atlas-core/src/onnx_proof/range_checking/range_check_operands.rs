use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, SumcheckId},
    utils::lookup_bits::LookupBits,
};

use crate::{
    onnx_proof::{
        neural_teleport::division::compute_division,
        ops::rsqrt::{Q, Q_SQUARE},
    },
    utils::compute_lookup_indices_from_operands,
};

/// Common fields shared by all range-checking operand types.
///
/// Every operation that needs range-checking (Div, Rsqrt, Tanh) stores the same
/// triple of values: a node index, input operands, and a virtual read-address polynomial.
/// This struct eliminates the field duplication across the individual operand types.
pub struct RangeCheckOperandsBase {
    /// Index of the computation node in the graph.
    pub node_idx: usize,
    /// Input operand virtual polynomials: [left_operand, right_operand].
    pub input_operands: Vec<VirtualPolynomial>,
    /// Virtual polynomial representing the range-check read-address (Ra).
    pub virtual_ra: VirtualPolynomial,
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
    /// Create a new helper from a computation node.
    fn new(node: &ComputationNode) -> Self;

    /// Returns a reference to the common base fields.
    fn base(&self) -> &RangeCheckOperandsBase;

    /// Get the index of the computation node this helper is associated with.
    fn node_idx(&self) -> usize {
        self.base().node_idx
    }

    /// Get the virtual polynomials representing the input operands for range-checking.
    fn get_input_operands(&self) -> Vec<VirtualPolynomial> {
        self.base().input_operands.to_vec()
    }

    /// Get the virtual polynomial representing the range-check read-address (Ra) output.
    fn get_output_operand(&self) -> VirtualPolynomial {
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
                let (_, claim) =
                    accumulator.get_virtual_polynomial_opening(*operand, SumcheckId::Raf);
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
    fn rad_poly(&self, d: usize) -> CommittedPolynomial;
}

/// Operands for division range-checking.
///
/// For integer division `a / b = q`, there exists a remainder `r` such that `a = q·b + r`.
/// This struct holds the operands needed to verify `0 ≤ r < b`.
pub struct DivRangeCheckOperands {
    base: RangeCheckOperandsBase,
}

/// Operands for reciprocal square root (rsqrt) division range-checking.
///
/// For rsqrt computation involving `Q²/x`, this verifies the intermediate division remainder.
pub struct RiRangeCheckOperands {
    base: RangeCheckOperandsBase,
}

/// Operands for reciprocal square root (rsqrt) final range-checking.
///
/// For square root `v = √(S·x̂)`, verifies that the remainder `r = S·x̂ - v̂²` satisfies `0 ≤ r < 2v̂ + 1`.
pub struct RsRangeCheckOperands {
    base: RangeCheckOperandsBase,
}

impl RangeCheckingOperandsTrait for DivRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        Self {
            base: RangeCheckOperandsBase {
                node_idx: node.idx,
                input_operands: vec![
                    VirtualPolynomial::DivRemainder(node.idx),
                    VirtualPolynomial::NodeOutput(node.inputs[1]),
                ],
                virtual_ra: VirtualPolynomial::DivRangeCheckRa(node.idx),
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

        let remainder = {
            let data: Vec<i32> = dividend
                .iter()
                .zip(divisor.iter())
                .map(|(&a, &b)| {
                    let mut R = a % b;
                    if (R < 0 && b > 0) || R > 0 && b < 0 {
                        R += b
                    }
                    R
                })
                .collect();
            Tensor::<i32>::construct(data, dividend.dims().to_vec())
        };

        (remainder, divisor.clone())
    }

    fn rad_poly(&self, d: usize) -> CommittedPolynomial {
        CommittedPolynomial::DivRangeCheckRaD(self.base.node_idx, d)
    }
}

impl RangeCheckingOperandsTrait for RiRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        Self {
            base: RangeCheckOperandsBase {
                node_idx: node.idx,
                input_operands: vec![
                    VirtualPolynomial::DivRemainder(node.idx),
                    VirtualPolynomial::NodeOutput(node.inputs[0]),
                ],
                virtual_ra: VirtualPolynomial::DivRangeCheckRa(node.idx),
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

        let remainder = {
            let data: Vec<i32> = input_tensor.iter().map(|&x| Q_SQUARE % x).collect();
            Tensor::<i32>::construct(data, input_tensor.dims().to_vec())
        };

        (remainder, input_tensor.clone())
    }

    fn rad_poly(&self, d: usize) -> CommittedPolynomial {
        CommittedPolynomial::SqrtDivRangeCheckRaD(self.base.node_idx, d)
    }
}

impl RangeCheckingOperandsTrait for RsRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        Self {
            base: RangeCheckOperandsBase {
                node_idx: node.idx,
                input_operands: vec![
                    VirtualPolynomial::SqrtRemainder(node.idx),
                    VirtualPolynomial::NodeOutput(node.idx),
                ],
                virtual_ra: VirtualPolynomial::SqrtRangeCheckRa(node.idx),
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

        let inv = {
            let data: Vec<i32> = input_tensor.iter().map(|&x| Q_SQUARE / x).collect();
            Tensor::<i32>::construct(data, input_tensor.dims().to_vec())
        };
        let remainder = {
            let data: Vec<i32> = inv
                .iter()
                .zip(output.iter())
                .map(|(&inv, &sqrt)| Q * inv - sqrt * sqrt)
                .collect();
            Tensor::<i32>::construct(data, input_tensor.dims().to_vec())
        };

        (remainder, output.clone())
    }

    // Override to implement the specific lookup indices computation for sqrt range check
    //
    // let x̂ the (quantized) input
    // â = ⌊a * S⌋          | Recall S is the scaling factor, quantization precision equals 1/S
    // let v the square root of x̂, i.e., v = sqrt(S · x̂) | We multiply by S so that v has the same scale as x̂
    // Hence v̂ = ⌊v⌋        | Here v is already scaled by S
    // Due to quantization, there exists a non-zero remainder r such that:
    //     S · x̂ = v̂ · v̂ + r
    //
    // Defining the bounds for r:
    // We have S · x̂ = v · v (notice v is not quantized)
    // Where S · v = v̂ + e | e is the quantization error for v, with 0 ≤ e < 1
    // Therefore:
    //     S · x̂ = (v̂ + e) · (v̂ + e) = v̂ · v̂ + 2 · e · v̂ + e · e
    // Since e < 1, this gives us the bounds for r:
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

    fn rad_poly(&self, d: usize) -> CommittedPolynomial {
        CommittedPolynomial::SqrtRangeCheckRaD(self.base.node_idx, d)
    }
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
                input_operands: vec![
                    VirtualPolynomial::TeleportRemainder(node.idx),
                    VirtualPolynomial::NodeOutput(node.inputs[0]),
                ],
                virtual_ra: VirtualPolynomial::TeleportRangeCheckRa(node.idx),
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

        let tau = if let Operator::Tanh(inner) = &node.operator {
            inner.tau
        } else {
            panic!("Expected Tanh operator for neural teleportation division");
        };

        let (_, remainder) = compute_division(input_tensor, tau);
        let divisor_tensor = Tensor::construct(vec![tau], vec![1])
            .expand(input_tensor.dims())
            .unwrap();

        (remainder, divisor_tensor)
    }

    fn rad_poly(&self, d: usize) -> CommittedPolynomial {
        CommittedPolynomial::TeleportRangeCheckRaD(self.base.node_idx, d)
    }
}
