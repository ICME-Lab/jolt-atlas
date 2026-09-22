//! Exact integer masks and selection lowered to existing native relations.
//! These helpers append a block and return its output tensor. The complete
//! graph still requires normal validation, authenticated registration and proof.
use super::native_graph::{NativeGraph, NativeGraphNode};
use crate::utils::errors::ProofVerifyError;
fn next(graph: &NativeGraph, inputs: &[usize]) -> Result<usize, ProofVerifyError> {
    let first = graph
        .num_inputs()
        .checked_add(graph.nodes.len())
        .ok_or_else(|| {
            ProofVerifyError::InvalidOpeningProof("Native graph index overflow".into())
        })?;
    if first.checked_add(5).is_none() || inputs.iter().any(|id| *id >= first) {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "Integer block input must precede its output".into(),
        ));
    }
    Ok(first)
}
/// Boolean AND. Both original inputs must be zero or one.
pub fn append_and(
    graph: &mut NativeGraph,
    left: usize,
    right: usize,
) -> Result<usize, ProofVerifyError> {
    let first = next(graph, &[left, right])?;
    graph.nodes.extend([
        NativeGraphNode::lookup(left, vec![0, 2], 1),
        NativeGraphNode::lookup(right, vec![0, 1], 1),
        NativeGraphNode::mul(first, first + 1, 1),
    ]);
    Ok(first + 2)
}
/// Choose the true or false tensor using a Boolean mask. Every selected
/// signed32 value is unchanged, including both signed endpoints.
pub fn append_select(
    graph: &mut NativeGraph,
    mask: usize,
    when_true: usize,
    when_false: usize,
) -> Result<usize, ProofVerifyError> {
    let first = next(graph, &[mask, when_true, when_false])?;
    graph.nodes.extend([
        NativeGraphNode::lookup(mask, vec![0, 2], 1),
        NativeGraphNode::lookup(mask, vec![2, 0], 1),
        NativeGraphNode::mul(when_true, first, 1),
        NativeGraphNode::mul(when_false, first + 1, 1),
        NativeGraphNode::add(first + 2, first + 3),
    ]);
    Ok(first + 4)
}
/// Exact negation on the domain where the signed32 result exists.
/// The minimum signed integer is rejected, rather than assigned a clamp or
/// wraparound result. Model integration must retain this proved domain.
pub fn append_checked_neg(
    graph: &mut NativeGraph,
    input: usize,
) -> Result<usize, ProofVerifyError> {
    let first = next(graph, &[input])?;
    graph.nodes.extend([
        NativeGraphNode::sub(input, input),
        NativeGraphNode::sub(first, input),
        NativeGraphNode::add(input, first + 1),
        // For ordinary inputs the checked sum is zero. Negating MIN through
        // the clamp would give MAX, leaving -1, which this lookup rejects.
        NativeGraphNode::lookup(first + 2, vec![0, 0], 1),
    ]);
    Ok(first + 1)
}
/// Integer IsNan with an output shape equal to the input shape.
/// All integer values are finite. The original tensor still has signed ranges.
pub fn append_is_nan(graph: &mut NativeGraph, input: usize) -> Result<usize, ProofVerifyError> {
    let first = next(graph, &[input])?;
    graph.nodes.push(NativeGraphNode::sub(input, input));
    Ok(first)
}

/// Prove the first index attaining each last-axis maximum. The last-axis
/// width is a positive power of two, at most 2^30. This includes every input
/// position; a model with padded vocabulary must first prove its padding mask.
/// Outputs retain the input shape with last dimension one.
///
/// A proved maximum and clamped subtraction give a zero exactly at each
/// maximum. Prefix AND of the nonzero flags is one strictly before the first
/// maximum and zero thereafter. Its sum is exactly the first maximum index.
/// Every prefix stage opens its actual predecessor through the graph edges.
pub fn first_argmax(context: Vec<u8>, shape: Vec<usize>) -> Result<NativeGraph, ProofVerifyError> {
    let logical = *shape
        .last()
        .ok_or_else(|| ProofVerifyError::InvalidOpeningProof("Unsupported argmax shape".into()))?;
    first_argmax_with_count(context, shape, logical)
}

/// Prove first-maximum selection within a registered logical vocabulary.
/// Values beyond `logical` are replaced by MIN through proved slices and
/// constant lookups. Even when all valid logits equal MIN, the first maximum
/// lies in the nonempty logical prefix. No public mask witness is trusted.
pub fn first_argmax_with_count(
    context: Vec<u8>,
    shape: Vec<usize>,
    logical: usize,
) -> Result<NativeGraph, ProofVerifyError> {
    let invalid = || ProofVerifyError::InvalidOpeningProof("Unsupported argmax shape".into());
    let width = *shape.last().ok_or_else(invalid)?;
    if logical == 0 || logical > width {
        return Err(invalid());
    }
    if shape.iter().any(|n| !n.is_power_of_two()) || width > 1 << 30 {
        return Err(invalid());
    }
    let total = shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(invalid)?;
    let rows = total / width;
    let mut graph = NativeGraph {
        context,
        input_shapes: vec![shape.clone()],
        nodes: vec![],
        outputs: vec![],
    };
    fn push(g: &mut NativeGraph, node: NativeGraphNode) -> usize {
        let id = g.num_inputs() + g.nodes.len();
        g.nodes.push(node);
        id
    }
    fn mask_suffix(g: &mut NativeGraph, input: usize, shape: &[usize], logical: usize) -> usize {
        let axis = shape.len() - 1;
        let width = shape[axis];
        if logical == width {
            return input;
        }
        let half = width / 2;
        let mut left = push(g, NativeGraphNode::slice(input, axis, 0, half));
        let mut right = push(g, NativeGraphNode::slice(input, axis, half, half));
        let mut child = shape.to_vec();
        child[axis] = half;
        if logical <= half {
            left = mask_suffix(g, left, &child, logical);
            let zero = push(g, NativeGraphNode::sub(right, right));
            right = push(g, NativeGraphNode::lookup(zero, vec![i32::MIN; 2], 1));
        } else {
            right = mask_suffix(g, right, &child, logical - half);
        }
        push(g, NativeGraphNode::concat(left, right, axis))
    }
    let valid = mask_suffix(&mut graph, 0, &shape, logical);
    let maximum = push(&mut graph, NativeGraphNode::max_last_axis(valid, 32));
    let broadcast = push(
        &mut graph,
        NativeGraphNode::broadcast(maximum, shape.clone()),
    );
    // Saturation preserves zero versus positive even for MAX - MIN.
    let difference = push(&mut graph, NativeGraphNode::sub(broadcast, valid));
    let flags = push(
        &mut graph,
        NativeGraphNode::clamped_lookup(difference, vec![0, 1], 0, 1),
    );
    let mut prefix = flags;
    let mut stride = 1;
    while stride < width {
        let groups = total / (2 * stride);
        let block = push(
            &mut graph,
            NativeGraphNode::reshape(prefix, vec![groups, 2, stride]),
        );
        let left = push(&mut graph, NativeGraphNode::slice(block, 1, 0, 1));
        let right = push(&mut graph, NativeGraphNode::slice(block, 1, 1, 1));
        let last = push(&mut graph, NativeGraphNode::slice(left, 2, stride - 1, 1));
        let carry = push(
            &mut graph,
            NativeGraphNode::broadcast(last, vec![groups, 1, stride]),
        );
        let carried = append_and(&mut graph, carry, right)?;
        prefix = push(&mut graph, NativeGraphNode::concat(left, carried, 1));
        stride *= 2;
    }
    let flat = push(
        &mut graph,
        NativeGraphNode::reshape(prefix, vec![rows, width]),
    );
    let index = push(&mut graph, NativeGraphNode::sum(flat, vec![1]));
    let mut output_shape = shape;
    *output_shape.last_mut().unwrap() = 1;
    let output = push(&mut graph, NativeGraphNode::reshape(index, output_shape));
    graph.outputs = vec![output];
    graph.tensor_shapes()?;
    Ok(graph)
}

#[cfg(test)]
mod argmax_tests {
    use super::super::{
        native_graph::{NativeGraphProof, NativeGraphWitness},
        DoryScheme,
    };
    use super::*;
    use crate::poly::commitment::commitment_scheme::CommitmentScheme;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    #[test]
    fn first_argmax_proves_first_ties_and_signed_endpoints() {
        let pp = DoryScheme::setup_prover(12);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        for (shape, values, expected) in [
            (vec![1], vec![i32::MIN], vec![0]),
            (
                vec![2, 4],
                vec![3, 9, 9, 0, i32::MIN, i32::MAX, i32::MIN, i32::MAX],
                vec![1, 1],
            ),
            (vec![1, 8], vec![i32::MIN; 8], vec![0]),
            (
                vec![1, 8],
                vec![i32::MIN, 0, 1, 2, 3, 4, 5, i32::MAX],
                vec![7],
            ),
        ] {
            let graph = first_argmax(b"first maximum index".to_vec(), shape).unwrap();
            let (statement, witness) =
                NativeGraphWitness::commit(graph, vec![values], &pp).unwrap();
            assert_eq!(witness.outputs(), [expected]);
            let proof = NativeGraphProof::prove(&statement, witness, &pp, &gens).unwrap();
            let mut bytes = vec![];
            proof.serialize_compressed(&mut bytes).unwrap();
            NativeGraphProof::deserialize_compressed(bytes.as_slice())
                .unwrap()
                .verify(&statement, &vp, &gens)
                .unwrap();
            let mut wrong = statement;
            wrong.graph.nodes[3].lookup.as_mut().unwrap().table = vec![1, 0];
            assert!(proof.verify(&wrong, &vp, &gens).is_err());
        }
    }
    #[test]
    fn first_argmax_compares_all_small_integer_rows() {
        for width in [1, 2, 4, 8] {
            let graph = first_argmax(b"exhaustive maximum index".to_vec(), vec![width]).unwrap();
            let cases = 3usize.pow(width as u32);
            for mut code in 0..cases {
                let values = (0..width)
                    .map(|_| {
                        let x = [-1, 0, 1][code % 3];
                        code /= 3;
                        x
                    })
                    .collect::<Vec<_>>();
                let max = *values.iter().max().unwrap();
                let expected = values.iter().position(|x| *x == max).unwrap() as i32;
                let witness = NativeGraphWitness::uncommitted(&graph, vec![values]).unwrap();
                assert_eq!(witness.outputs(), [vec![expected]]);
            }
        }
    }
    #[test]
    fn first_argmax_proves_logical_vocabulary_and_padding() {
        let pp = DoryScheme::setup_prover(12);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 16);
        for (logical, values, expected) in [
            (
                1,
                vec![
                    i32::MIN,
                    i32::MAX,
                    i32::MAX,
                    i32::MAX,
                    i32::MAX,
                    i32::MAX,
                    i32::MAX,
                    i32::MAX,
                ],
                0,
            ),
            (3, vec![1, 9, 9, i32::MAX, 0, 1, 2, 3], 1),
            (5, vec![0, 1, 2, 3, 4, i32::MAX, 4, 4], 4),
            (7, vec![i32::MIN; 8], 0),
        ] {
            let graph =
                first_argmax_with_count(b"logical first maximum".to_vec(), vec![1, 8], logical)
                    .unwrap();
            let (statement, witness) =
                NativeGraphWitness::commit(graph, vec![values], &pp).unwrap();
            assert_eq!(witness.outputs(), [vec![expected]]);
            let proof = NativeGraphProof::prove(&statement, witness, &pp, &gens).unwrap();
            let mut encoded = vec![];
            proof.serialize_compressed(&mut encoded).unwrap();
            NativeGraphProof::deserialize_compressed(encoded.as_slice())
                .unwrap()
                .verify(&statement, &vp, &gens)
                .unwrap();
        }
        for logical in [0, 9] {
            assert!(first_argmax_with_count(
                b"invalid logical vocabulary".to_vec(),
                vec![8],
                logical
            )
            .is_err());
        }
    }
    #[test]
    fn first_argmax_logical_prefix_matches_small_rows() {
        for logical in 1..=4 {
            let graph =
                first_argmax_with_count(b"small logical vocabulary".to_vec(), vec![4], logical)
                    .unwrap();
            for mut code in 0..81 {
                let values = (0..4)
                    .map(|_| {
                        let x = [-1, 0, 1][code % 3];
                        code /= 3;
                        x
                    })
                    .collect::<Vec<_>>();
                let max = values[..logical].iter().max().unwrap();
                let expected = values[..logical].iter().position(|x| x == max).unwrap() as i32;
                let witness = NativeGraphWitness::uncommitted(&graph, vec![values]).unwrap();
                assert_eq!(witness.outputs(), [vec![expected]]);
            }
        }
    }
    #[test]
    fn first_argmax_rejects_invalid_shapes() {
        for shape in [vec![], vec![0], vec![3], vec![2, 3], vec![1usize << 31]] {
            assert!(first_argmax(vec![], shape).is_err());
        }
    }
}
