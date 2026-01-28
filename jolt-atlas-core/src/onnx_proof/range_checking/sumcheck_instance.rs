use atlas_onnx_tracer::{
    model::trace::{LayerData, Trace},
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use common::VirtualPolynomial;
use joltworks::field::JoltField;

use crate::onnx_proof::ops::rsqrt::{Q, Q_SQUARE};

pub trait ReadRafSumcheckHelper {
    fn new(node: &ComputationNode) -> Self;

    fn compute_input_claim<F: JoltField>(input_claims: &[F], gamma: F, gamma_sqr: F) -> F;

    fn get_input_operands(&self) -> Vec<VirtualPolynomial>;

    fn get_output_operand(&self) -> VirtualPolynomial;

    fn get_operands_tensors(trace: &Trace, node: &ComputationNode) -> (Tensor<i32>, Tensor<i32>);
}

// Generic struct that holds input and output operands for sumcheck instances
pub struct DivRangeCheckOperands {
    pub input_operands: Vec<VirtualPolynomial>,
    pub output_operands: VirtualPolynomial,
}

pub struct SqrtRangeCheckOperands {
    pub input_operands: Vec<VirtualPolynomial>,
    pub output_operands: VirtualPolynomial,
}

impl ReadRafSumcheckHelper for DivRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        let input_operands = match node.operator {
            Operator::Div(_) => vec![
                VirtualPolynomial::DivNodeRemainder(node.idx),
                VirtualPolynomial::NodeOutput(node.inputs[1]),
            ],
            Operator::Rsqrt(_) => vec![
                VirtualPolynomial::DivNodeRemainder(node.idx),
                VirtualPolynomial::NodeOutput(node.inputs[0]),
            ],
            _ => panic!("Unsupported operator for DivRangeCheckOperands"),
        };
        Self {
            input_operands,
            output_operands: VirtualPolynomial::DivRangeCheckRa(node.idx),
        }
    }

    fn compute_input_claim<F: JoltField>(input_claims: &[F], gamma: F, gamma_sqr: F) -> F {
        let [left_claim, right_claim] = input_claims else {
            panic!("Expected exactly two input operands for division range check");
        };

        F::one() + gamma * left_claim + gamma_sqr * right_claim
    }

    fn get_input_operands(&self) -> Vec<VirtualPolynomial> {
        self.input_operands.to_vec()
    }

    fn get_output_operand(&self) -> VirtualPolynomial {
        self.output_operands
    }

    fn get_operands_tensors(trace: &Trace, node: &ComputationNode) -> (Tensor<i32>, Tensor<i32>) {
        let LayerData {
            output: _,
            operands,
        } = Trace::layer_data(trace, node);

        match node.operator {
            Operator::Div(_) => {
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
            Operator::Rsqrt(_) => {
                let [input_tensor] = operands[..] else {
                    panic!("Expected exactly one input tensor");
                };
                let remainder = {
                    let data: Vec<i32> = input_tensor.iter().map(|&x| Q_SQUARE % x).collect();
                    Tensor::<i32>::construct(data, input_tensor.dims().to_vec())
                };

                #[cfg(test)]
                {
                    assert!(remainder
                        .iter()
                        .zip(input_tensor.iter())
                        .all(|(&r, &x)| r >= 0 && r < x));
                }
                (remainder, input_tensor.clone())
            }
            _ => panic!("Unsupported operator for DivRangeCheckOperands"),
        }
    }
}

impl ReadRafSumcheckHelper for SqrtRangeCheckOperands {
    fn new(node: &ComputationNode) -> Self {
        let input_operands = vec![
            VirtualPolynomial::RsqrtNodeRs(node.idx),
            VirtualPolynomial::RsqrtNodeSqrt(node.idx),
        ];
        Self {
            input_operands,
            output_operands: VirtualPolynomial::RsqrtRsRangeCheckRa(node.idx),
        }
    }

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
        self.output_operands
    }

    fn get_operands_tensors(trace: &Trace, node: &ComputationNode) -> (Tensor<i32>, Tensor<i32>) {
        assert!(matches!(node.operator, Operator::Rsqrt(_)));
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
        let upper_bound = {
            let data = output.iter().map(|&x| 2 * x + 1).collect();
            Tensor::<i32>::construct(data, input_tensor.dims().to_vec())
        };

        (remainder, upper_bound)
    }
}
