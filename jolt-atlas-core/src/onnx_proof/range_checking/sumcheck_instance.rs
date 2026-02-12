use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use common::{CommittedPolynomial, VirtualPolynomial};
use joltworks::{field::JoltField, utils::lookup_bits::LookupBits};

use crate::onnx_proof::{
    neural_teleport::division::compute_division,
    ops::rsqrt::{Q, Q_SQUARE},
    range_checking::read_raf_checking::compute_lookup_indices_from_operands,
};

pub trait ReadRafSumcheckHelper {
    fn new(node: &ComputationNode) -> Self;

    fn compute_input_claim<F: JoltField>(input_claims: &[F], gamma: F, gamma_sqr: F) -> F {
        let [left_claim, right_claim] = input_claims else {
            panic!("Expected exactly two input operands for division range check");
        };

        F::one() + gamma * left_claim + gamma_sqr * right_claim
    }

    fn get_input_operands(&self) -> Vec<VirtualPolynomial>;

    fn get_output_operand(&self) -> VirtualPolynomial;

    fn get_operands_tensors(trace: &Trace, node: &ComputationNode) -> (Tensor<i32>, Tensor<i32>);

    fn compute_lookup_indices(
        left_operand: &Tensor<i32>,
        right_operand: &Tensor<i32>,
    ) -> Vec<LookupBits> {
        compute_lookup_indices_from_operands(&[left_operand, right_operand])
    }

    fn rad_poly(&self, d: usize) -> CommittedPolynomial;
}

// Generic struct that holds input and output operands for sumcheck instances
pub struct DivRangeCheckOperands {
    pub node_idx: usize,
    pub input_operands: Vec<VirtualPolynomial>,
    pub virtual_ra: VirtualPolynomial,
}

pub struct RiRangeCheckOperands {
    pub node_idx: usize,
    pub input_operands: Vec<VirtualPolynomial>,
    pub virtual_ra: VirtualPolynomial,
}

pub struct RsRangeCheckOperands {
    pub node_idx: usize,
    pub input_operands: Vec<VirtualPolynomial>,
    pub virtual_ra: VirtualPolynomial,
}

impl ReadRafSumcheckHelper for DivRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        let input_operands = vec![
            VirtualPolynomial::DivRemainder(node.idx),
            VirtualPolynomial::NodeOutput(node.inputs[1]),
        ];
        let virtual_ra = VirtualPolynomial::DivRangeCheckRa(node.idx);

        Self {
            node_idx: node.idx,
            input_operands,
            virtual_ra,
        }
    }

    fn get_input_operands(&self) -> Vec<VirtualPolynomial> {
        self.input_operands.to_vec()
    }

    fn get_output_operand(&self) -> VirtualPolynomial {
        self.virtual_ra
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
        CommittedPolynomial::DivRangeCheckRaD(self.node_idx, d)
    }
}

impl ReadRafSumcheckHelper for RiRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        let input_operands = vec![
            VirtualPolynomial::DivRemainder(node.idx),
            VirtualPolynomial::NodeOutput(node.inputs[0]),
        ];
        let virtual_ra = VirtualPolynomial::DivRangeCheckRa(node.idx);

        Self {
            node_idx: node.idx,
            input_operands,
            virtual_ra,
        }
    }

    fn get_input_operands(&self) -> Vec<VirtualPolynomial> {
        self.input_operands.to_vec()
    }

    fn get_output_operand(&self) -> VirtualPolynomial {
        self.virtual_ra
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
        CommittedPolynomial::SqrtDivRangeCheckRaD(self.node_idx, d)
    }
}

impl ReadRafSumcheckHelper for RsRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        let input_operands = vec![
            VirtualPolynomial::SqrtRemainder(node.idx),
            VirtualPolynomial::NodeOutput(node.idx),
        ];
        let virtual_ra = VirtualPolynomial::SqrtRangeCheckRa(node.idx);

        Self {
            node_idx: node.idx,
            input_operands,
            virtual_ra,
        }
    }

    // Override to implement the specific input claim computation for sqrt range check
    // the right claim is encoded as 2 * right_claim + 1 in the lookup table
    // So that we the lookup output corresponds to the check r < 2 * v̂ + 1
    fn compute_input_claim<F: JoltField>(input_claims: &[F], gamma: F, gamma_sqr: F) -> F {
        let [left_claim, right_claim] = input_claims else {
            panic!("Expected exactly two input operands for square root range check");
        };

        F::one() + gamma * left_claim + gamma_sqr * (F::from_i32(2) * right_claim + F::one())
    }

    fn get_input_operands(&self) -> Vec<VirtualPolynomial> {
        self.input_operands.to_vec()
    }

    fn get_output_operand(&self) -> VirtualPolynomial {
        self.virtual_ra
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

        compute_lookup_indices_from_operands(&[left_operand, &upper_bound])
    }

    fn rad_poly(&self, d: usize) -> CommittedPolynomial {
        CommittedPolynomial::SqrtRangeCheckRaD(self.node_idx, d)
    }
}

pub struct TeleportRangeCheckOperands {
    pub node_idx: usize,
    pub input_operands: Vec<VirtualPolynomial>,
    pub virtual_ra: VirtualPolynomial,
}

impl ReadRafSumcheckHelper for TeleportRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        let input_operands = vec![
            VirtualPolynomial::TeleportRemainder(node.idx),
            VirtualPolynomial::NodeOutput(node.inputs[0]),
        ];
        let virtual_ra = VirtualPolynomial::TeleportRangeCheckRa(node.idx);

        Self {
            node_idx: node.idx,
            input_operands,
            virtual_ra,
        }
    }

    fn get_input_operands(&self) -> Vec<VirtualPolynomial> {
        self.input_operands.to_vec()
    }

    fn get_output_operand(&self) -> VirtualPolynomial {
        self.virtual_ra
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
        CommittedPolynomial::TeleportRangeCheckRaD(self.node_idx, d)
    }
}
