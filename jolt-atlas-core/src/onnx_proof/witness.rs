use crate::onnx_proof::{
    op_lookups::{read_raf_checking::compute_lookup_indices_from_operands, InterleavedBitsMarker},
    ops::{rsqrt::Q_SQUARE, softmax_axes::softmax::scalar_div::S},
    range_checking::sumcheck_instance::{
        DivRangeCheckOperands, ReadRafSumcheckHelper, RiRangeCheckOperands, RsRangeCheckOperands,
    },
};
use atlas_onnx_tracer::{
    model::{
        trace::{LayerData, Trace},
        Model,
    },
    ops::Operator,
    tensor::{
        ops::nonlinearities::{softmax_fixed_128, EXP_LUT_SIZE},
        Tensor,
    },
};
use common::CommittedPolynomial;
use joltworks::{
    config::OneHotParams,
    field::JoltField,
    poly::{multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial},
    utils::math::Math,
};
use rayon::prelude::*;

pub trait WitnessGenerator<F: JoltField> {
    fn generate_witness(&self, model: &Model, trace: &Trace) -> MultilinearPolynomial<F>;
}

impl<F: JoltField> WitnessGenerator<F> for CommittedPolynomial {
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
                let one_hot_params = OneHotParams::new(lookup_indices.len().log_2());
                let addresses: Vec<_> = lookup_indices
                    .par_iter()
                    .map(|lookup_index| {
                        Some(one_hot_params.lookup_index_chunk(lookup_index.into(), *d) as u16)
                    })
                    .collect();
                MultilinearPolynomial::OneHot(
                    joltworks::poly::one_hot_polynomial::OneHotPolynomial::from_indices(
                        addresses,
                        one_hot_params.k_chunk,
                    ),
                )
            }
            CommittedPolynomial::DivNodeQuotient(node_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(computation_node.operator, Operator::Div(_)));
                let layer_data = Trace::layer_data(trace, computation_node);
                let q = layer_data.output;
                MultilinearPolynomial::from(q.clone())
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
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(computation_node.operator, Operator::Div(_)));
                let (left_operand, right_operand) =
                    DivRangeCheckOperands::get_operands_tensors(trace, computation_node);
                let lookup_indices =
                    DivRangeCheckOperands::compute_lookup_indices(&left_operand, &right_operand);
                let one_hot_params = OneHotParams::new(lookup_indices.len().log_2());
                let addresses: Vec<_> = lookup_indices
                    .par_iter()
                    .map(|lookup_index| {
                        Some(one_hot_params.lookup_index_chunk(lookup_index.into(), *d) as u16)
                    })
                    .collect();
                MultilinearPolynomial::OneHot(
                    joltworks::poly::one_hot_polynomial::OneHotPolynomial::from_indices(
                        addresses,
                        one_hot_params.k_chunk,
                    ),
                )
            }
            CommittedPolynomial::SqrtRangeCheckRaD(node_idx, d) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(computation_node.operator, Operator::Rsqrt(_)));
                let (left_operand, right_operand) =
                    RsRangeCheckOperands::get_operands_tensors(trace, computation_node);
                let lookup_indices =
                    RsRangeCheckOperands::compute_lookup_indices(&left_operand, &right_operand);
                let one_hot_params = OneHotParams::new(lookup_indices.len().log_2());
                let addresses: Vec<_> = lookup_indices
                    .par_iter()
                    .map(|lookup_index| {
                        Some(one_hot_params.lookup_index_chunk(lookup_index.into(), *d) as u16)
                    })
                    .collect();
                MultilinearPolynomial::OneHot(
                    joltworks::poly::one_hot_polynomial::OneHotPolynomial::from_indices(
                        addresses,
                        one_hot_params.k_chunk,
                    ),
                )
            }
            CommittedPolynomial::SqrtDivRangeCheckRaD(node_idx, d) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(computation_node.operator, Operator::Rsqrt(_)));
                let (left_operand, right_operand) =
                    RiRangeCheckOperands::get_operands_tensors(trace, computation_node);
                let lookup_indices =
                    RiRangeCheckOperands::compute_lookup_indices(&left_operand, &right_operand);
                let one_hot_params = OneHotParams::new(lookup_indices.len().log_2());
                let addresses: Vec<_> = lookup_indices
                    .par_iter()
                    .map(|lookup_index| {
                        Some(one_hot_params.lookup_index_chunk(lookup_index.into(), *d) as u16)
                    })
                    .collect();
                MultilinearPolynomial::OneHot(
                    joltworks::poly::one_hot_polynomial::OneHotPolynomial::from_indices(
                        addresses,
                        one_hot_params.k_chunk,
                    ),
                )
            }

            // TODO: Generate batch witness for Sofmax committed polynomials
            CommittedPolynomial::SoftmaxRemainder(node_idx, feature_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(
                    computation_node.operator,
                    Operator::SoftmaxAxes(_)
                ));
                let features = computation_node.output_dims[2];
                let LayerData {
                    output: _,
                    operands,
                } = Trace::layer_data(trace, computation_node);
                let operand = operands[0];
                let operand_chunk =
                    operand.data()[feature_idx * features..(feature_idx + 1) * features].to_vec();
                let (_, softmax_trace) = softmax_fixed_128::<true>(&Tensor::construct(
                    operand_chunk.to_vec(),
                    vec![features],
                ));
                let softmax_trace = softmax_trace.unwrap();
                let left_operand = &softmax_trace.exp_q_values;
                let right_operand = softmax_trace.exp_sum_q;
                let R_tensor = {
                    let data: Vec<i32> = left_operand
                        .iter()
                        .map(|&a| {
                            let mut R = (a * S as i32) % right_operand;
                            if (R < 0 && right_operand > 0) || R > 0 && right_operand < 0 {
                                R += right_operand
                            }
                            R
                        })
                        .collect();
                    Tensor::<i32>::construct(data, left_operand.dims().to_vec())
                };
                MultilinearPolynomial::from(R_tensor)
            }
            CommittedPolynomial::SoftmaxExponentiationRa(node_idx, feature_idx) => {
                let computation_node = &model.graph.nodes[node_idx];
                assert!(matches!(
                    computation_node.operator,
                    Operator::SoftmaxAxes(_)
                ));
                let features = computation_node.output_dims[2];
                let LayerData {
                    output: _,
                    operands,
                } = Trace::layer_data(trace, computation_node);
                let operand = operands[0];
                let operand_chunk =
                    operand.data()[feature_idx * features..(feature_idx + 1) * features].to_vec();
                let (_, softmax_trace) = softmax_fixed_128::<true>(&Tensor::construct(
                    operand_chunk.to_vec(),
                    vec![features],
                ));
                let softmax_trace = softmax_trace.unwrap();
                let lookup_indices = &softmax_trace.abs_centered_logits;

                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(
                    lookup_indices
                        .par_iter()
                        .map(|&lookup_index| Some(lookup_index as usize as u16))
                        .collect::<Vec<_>>(),
                    EXP_LUT_SIZE,
                ))
            }
        }
    }
}
