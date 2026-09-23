//! Preserve logical element order when a reshape changes the positions of padding.
use crate::{
    node::ComputationNode,
    ops::{Broadcast, Constant, GatherLarge, GatherSmall, Mul, Operator, Reshape},
    tensor::Tensor,
};
use std::collections::BTreeMap;

pub(crate) struct Repack {
    input_rows: usize,
    block: usize,
    indices: Vec<i32>,
    valid: Vec<i32>,
}

fn padded_size(dims: &[usize]) -> usize {
    dims.iter().map(|d| d.next_power_of_two()).product()
}

/// Keep a common suffix together so an attention reshape gathers whole heads,
/// rather than individual scalars. Every index refers to the padded input.
pub(crate) fn plan(input: &[usize], output: &[usize]) -> Option<Repack> {
    assert!(
        input.iter().chain(output).all(|&d| d > 0),
        "empty reshape dimensions are unsupported"
    );
    assert_eq!(
        input.iter().product::<usize>(),
        output.iter().product::<usize>()
    );
    let suffix = input
        .iter()
        .rev()
        .zip(output.iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let block = padded_size(&input[input.len() - suffix..]);
    let input = &input[..input.len() - suffix];
    let output = &output[..output.len() - suffix];
    let input_rows = padded_size(input);
    let output_rows = padded_size(output);
    assert!(
        input_rows <= i32::MAX as usize,
        "reshape gather indices exceed i32"
    );
    let mut indices = Vec::with_capacity(output_rows);
    let mut valid = Vec::with_capacity(output_rows);
    let mut identity = input_rows == output_rows;
    for physical in 0..output_rows {
        let mut remainder = physical;
        let mut logical = 0;
        let mut stride = 1;
        let mut real = true;
        for &d in output.iter().rev() {
            let coord = remainder % d.next_power_of_two();
            remainder /= d.next_power_of_two();
            real &= coord < d;
            logical += coord * stride;
            stride *= d;
        }
        let mut source = 0;
        let mut stride = 1;
        if real {
            for &d in input.iter().rev() {
                source += (logical % d) * stride;
                logical /= d;
                stride *= d.next_power_of_two();
            }
            identity &= source == physical;
        }
        indices.push(source as i32);
        valid.push(i32::from(real));
    }
    (!identity).then_some(Repack {
        input_rows,
        block,
        indices,
        valid,
    })
}

pub(crate) fn plans(nodes: &BTreeMap<usize, ComputationNode>) -> BTreeMap<usize, Repack> {
    nodes
        .iter()
        .filter_map(|(&idx, node)| {
            if let Operator::Reshape(_) = node.operator {
                let input = nodes[&node.inputs[0]].raw_or_padded_output_dims();
                plan(&input, &node.raw_or_padded_output_dims()).map(|p| (idx, p))
            } else {
                None
            }
        })
        .collect()
}

pub(crate) fn remapping(
    nodes: &BTreeMap<usize, ComputationNode>,
    plans: &BTreeMap<usize, Repack>,
) -> BTreeMap<usize, usize> {
    let mut next = 0;
    nodes
        .keys()
        .map(|&idx| {
            next += if plans.contains_key(&idx) { 7 } else { 1 };
            (idx, next - 1)
        })
        .collect()
}

/// Input nodes have already been padded. Lower only reshapes whose real
/// elements move, using existing proved operators and authenticated constants.
pub(crate) fn lower(
    nodes: &mut BTreeMap<usize, ComputationNode>,
    plans: BTreeMap<usize, Repack>,
    mapping: &BTreeMap<usize, usize>,
) {
    let old = std::mem::take(nodes);
    for (idx, mut node) in old {
        node.idx = mapping[&idx];
        node.inputs.iter_mut().for_each(|i| *i = mapping[i]);
        if let Some(p) = plans.get(&idx) {
            let first = node.idx - 6;
            let rows = p.indices.len();
            let dims = vec![rows, p.block];
            let gather = if p.input_rows <= 65536 {
                Operator::GatherSmall(GatherSmall {
                    axis: 0,
                    dict_len: p.input_rows,
                })
            } else {
                Operator::GatherLarge(GatherLarge {
                    axis: 0,
                    dict_len: p.input_rows,
                })
            };
            let generated = [
                ComputationNode::new(
                    first,
                    Operator::Reshape(Reshape {
                        shape: vec![p.input_rows, p.block],
                    }),
                    node.inputs.clone(),
                    vec![p.input_rows, p.block],
                ),
                ComputationNode::new(
                    first + 1,
                    Operator::Constant(Constant(Tensor::new(Some(&p.indices), &[rows]).unwrap())),
                    vec![],
                    vec![rows],
                ),
                ComputationNode::new(first + 2, gather, vec![first, first + 1], dims.clone()),
                ComputationNode::new(
                    first + 3,
                    Operator::Constant(Constant(Tensor::new(Some(&p.valid), &[rows, 1]).unwrap())),
                    vec![],
                    vec![rows, 1],
                ),
                ComputationNode::new(
                    first + 4,
                    Operator::Broadcast(Broadcast {
                        shape: dims.clone(),
                    }),
                    vec![first + 3],
                    dims.clone(),
                ),
                ComputationNode::new(
                    first + 5,
                    Operator::Mul(Mul { scale: 0 }),
                    vec![first + 2, first + 4],
                    dims,
                ),
            ];
            for generated in generated {
                nodes.insert(generated.idx, generated);
            }
            node.inputs = vec![first + 5];
        }
        nodes.insert(node.idx, node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Op;

    fn check(input_dims: &[usize], output_dims: &[usize]) {
        let values: Vec<i32> = (0..input_dims.iter().product::<usize>())
            .map(|v| v as i32 + 1)
            .collect();
        let mut input = Tensor::new(Some(&values), input_dims).unwrap();
        input.pad_next_power_of_two();
        let mut expected = Tensor::new(Some(&values), output_dims).unwrap();
        expected.pad_next_power_of_two();
        let mut nodes = BTreeMap::from([
            (
                0,
                ComputationNode::new(
                    0,
                    Operator::Input(Default::default()),
                    vec![],
                    input_dims.to_vec(),
                ),
            ),
            (
                1,
                ComputationNode::new(
                    1,
                    Operator::Reshape(Reshape {
                        shape: output_dims.to_vec(),
                    }),
                    vec![0],
                    output_dims.to_vec(),
                ),
            ),
        ]);
        let plans = plans(&nodes);
        let mapping = remapping(&nodes, &plans);
        for n in nodes.values_mut() {
            n.pad_output_dims_to_power_of_2();
            let dims = n.raw_or_padded_output_dims();
            if let Operator::Reshape(r) = &mut n.operator {
                r.shape = dims;
            }
        }
        lower(&mut nodes, plans, &mapping);
        let mut values = BTreeMap::from([(mapping[&0], input)]);
        for (idx, n) in &nodes {
            if values.contains_key(idx) {
                continue;
            }
            let inputs = n.inputs.iter().map(|i| &values[i]).collect();
            values.insert(*idx, n.operator.f(inputs));
        }
        assert_eq!(values[&mapping[&1]], expected);
    }

    #[test]
    fn reshape_moves_padding_between_groups() {
        check(&[2, 3, 4], &[6, 4]);
        check(&[6, 4], &[2, 3, 4]);
        check(&[2, 3, 5], &[6, 5]);
        check(&[6, 5], &[2, 3, 5]);
        check(&[1, 2, 7, 32, 64], &[1, 14, 32, 64]);
        check(&[1, 14, 32, 32], &[1, 2, 7, 32, 32]);
        check(&[3, 3], &[9]);
        check(&[9], &[3, 3]);
    }

    #[test]
    fn compatible_padding_keeps_zero_copy_reshape() {
        assert!(plan(&[1, 32, 896], &[1, 32, 14, 64]).is_none());
        assert!(plan(&[2, 4, 8], &[8, 8]).is_none());
    }
    #[test]
    fn imported_reshape_preserves_forward_and_all_shadows() {
        use crate::model::{Model, RunArgs};
        for (name, dims) in [
            ("merge", vec![2, 3, 4]),
            ("split", vec![6, 4]),
            ("unequal", vec![3, 3]),
        ] {
            let path = format!(
                "{}/tests/fixtures/reshape-{name}.onnx",
                env!("CARGO_MANIFEST_DIR")
            );
            let args = RunArgs::default();
            let model = Model::load(&path, &args);
            let values: Vec<i32> = (0..dims.iter().product::<usize>())
                .map(|i| i as i32 - 9)
                .collect();
            let input = Tensor::new(Some(&values), &dims).unwrap();
            assert_eq!(model.forward(&[input])[0].inner, values);
            let originals = model.load_original_f64_constants(&path, &args);
            let floats: Vec<f64> = values.iter().map(|&v| v as f64).collect();
            let shadow = model.trace_with_true_f64_shadow(
                &[Tensor::new(Some(&floats), &dims).unwrap()],
                &originals,
                args.scale,
            );
            let output = model.graph.outputs[0];
            let mut expected =
                Tensor::new(Some(&floats), &model.graph.original_output_dims[&output]).unwrap();
            expected.pad_next_power_of_two();
            assert_eq!(shadow.f64_outputs[&output], expected);
            // The regular shadows receive the same integer inputs expressed
            // in real units, unlike the original-weight shadow above.
            let factor = 2_f64.powi(args.scale);
            let dequantized: Vec<f64> = floats.iter().map(|v| v / factor).collect();
            let integer_input = Tensor::new(Some(&values), &dims).unwrap();
            let real_input = Tensor::new(Some(&dequantized), &dims).unwrap();
            let expected_real = expected.map(|v| v / factor);
            for regular in [
                model.trace_with_shadow(
                    std::slice::from_ref(&integer_input),
                    std::slice::from_ref(&real_input),
                    args.scale,
                ),
                model.trace_with_shadow_isolated(&[integer_input], &[real_input], args.scale),
            ] {
                assert_eq!(regular.f64_outputs[&output], expected_real);
                assert_eq!(
                    regular.i32_outputs[&output],
                    shadow.f64_outputs[&output].map(|v| v as i32)
                );
            }
        }
    }
}
