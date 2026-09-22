//! Greedy selection and stopping over the actual token and score tensors.
//! The caller must separately prove causal model execution and canonical text
//! conversion. Independent scores do not establish model provenance.
use super::{
    native_graph::{NativeGraph, NativeGraphNode},
    native_logic::first_argmax_with_count,
};
use crate::utils::errors::ProofVerifyError;

/// Public generation rule. Lengths count tokens and exclude zero padding.
#[derive(Clone, Copy)]
pub struct GreedySequenceRule {
    pub prompt_length: usize,
    /// Original sequence position represented by the first score row.
    pub score_start_position: usize,
    pub response_length: usize,
    pub maximum_new_tokens: usize,
    pub logical_vocabulary: usize,
    pub end_token: i32,
}
fn invalid() -> ProofVerifyError {
    ProofVerifyError::InvalidOpeningProof("Invalid greedy sequence rule or tensor shape".into())
}
fn push(g: &mut NativeGraph, n: NativeGraphNode) -> usize {
    let id = g.num_inputs() + g.nodes.len();
    g.nodes.push(n);
    id
}
fn require_zero(g: &mut NativeGraph, input: usize) {
    // Both lookups are required even when their outputs are not graph outputs.
    // The first allows only 0 or 1; the second excludes 1 via the negative address.
    let guard = push(g, NativeGraphNode::lookup(input, vec![0, -1], 1));
    push(g, NativeGraphNode::lookup(guard, vec![0, 0], 1));
}
fn require_equal(g: &mut NativeGraph, left: usize, right: usize) {
    // Clamped subtraction is zero iff the original signed integers are equal.
    let difference = push(g, NativeGraphNode::sub(left, right));
    require_zero(g, difference);
}
fn rebase(node: &mut NativeGraphNode, map: impl Fn(usize) -> usize) {
    node.input = map(node.input);
    if let Some(x) = &mut node.mul {
        x.right = map(x.right);
    }
    if let Some(x) = &mut node.add {
        x.right = map(x.right);
    }
    if let Some(x) = &mut node.einsum {
        x.right = map(x.right);
    }
    if let Some(x) = &mut node.concat {
        x.right = map(x.right);
    }
    if let Some(x) = &mut node.lookup {
        x.table_input = x.table_input.map(map);
    }
}
/// Append a required shifted selection relation using original graph tensor IDs.
/// Tokens have shape [sequence], scores [positions, padded vocabulary]. The
/// registered offset identifies the original position of the first score row.
/// The returned tensor contains only response selection indices, padded with
/// zeros to a power of two. Remaining token positions must be zero.
///
/// An end token is forbidden before the last response token. The last token
/// must be the end token unless the registered generation limit was reached.
/// The prompt may contain end tokens. Model causality and text conversion are
/// outside this component and are required before claiming full generation.
pub fn append_greedy_sequence(
    graph: &mut NativeGraph,
    tokens: usize,
    scores: usize,
    rule: GreedySequenceRule,
) -> Result<usize, ProofVerifyError> {
    if graph.context.is_empty() || graph.input_shapes.is_empty() {
        return Err(invalid());
    }
    let shapes = if graph.nodes.is_empty() {
        for shape in &graph.input_shapes {
            super::native_reduce::shape_bits(shape)?;
        }
        graph.input_shapes.clone()
    } else {
        graph.tensor_shapes()?
    };
    let token_shape = shapes.get(tokens).ok_or_else(invalid)?;
    let score_shape = shapes.get(scores).ok_or_else(invalid)?;
    let used = rule
        .prompt_length
        .checked_add(rule.response_length)
        .ok_or_else(invalid)?;
    if token_shape.len() != 1
        || score_shape.len() != 2
        || rule.prompt_length == 0
        || rule.score_start_position >= rule.prompt_length
        || rule
            .score_start_position
            .checked_add(score_shape[0])
            .is_none_or(|end| used.saturating_sub(1) > end)
        || rule.response_length == 0
        || rule.response_length > rule.maximum_new_tokens
        || used > token_shape[0]
        || rule.logical_vocabulary == 0
        || rule.logical_vocabulary > score_shape[1]
        || rule.end_token < 0
        || rule.end_token as usize >= rule.logical_vocabulary
    {
        return Err(invalid());
    }
    let mut block = NativeGraph {
        context: b"greedy sequence relation".to_vec(),
        input_shapes: vec![token_shape.clone(), score_shape.clone()],
        nodes: vec![],
        outputs: vec![0],
    };
    // Select only rows that predict response tokens before the expensive
    // maximum and prefix product relation. All slices open original scores.
    let count = rule.response_length.next_power_of_two();
    let start = rule.prompt_length - 1 - rule.score_start_position;
    let mut rows = (0..rule.response_length)
        .map(|i| push(&mut block, NativeGraphNode::slice(1, 0, start + i, 1)))
        .collect::<Vec<_>>();
    if rows.len() < count {
        let zero = push(&mut block, NativeGraphNode::sub(rows[0], rows[0]));
        rows.resize(count, zero);
    }
    while rows.len() > 1 {
        rows = rows
            .chunks_exact(2)
            .map(|pair| push(&mut block, NativeGraphNode::concat(pair[0], pair[1], 0)))
            .collect();
    }
    let selected_scores = rows[0];
    let selection = first_argmax_with_count(
        b"greedy score selection".to_vec(),
        vec![count, score_shape[1]],
        rule.logical_vocabulary,
    )?;
    let first = block.num_inputs() + block.nodes.len();
    let map = |id| {
        if id == 0 {
            selected_scores
        } else {
            first + id - 1
        }
    };
    let selected = map(selection.outputs[0]);
    for mut node in selection.nodes {
        rebase(&mut node, map);
        block.nodes.push(node);
    }
    let indices = push(&mut block, NativeGraphNode::reshape(selected, vec![count]));
    for position in rule.prompt_length..used {
        let token = push(&mut block, NativeGraphNode::slice(0, 0, position, 1));
        let chosen = push(
            &mut block,
            NativeGraphNode::slice(indices, 0, position - rule.prompt_length, 1),
        );
        require_equal(&mut block, token, chosen);
        let zero = push(&mut block, NativeGraphNode::sub(token, token));
        let end = push(
            &mut block,
            NativeGraphNode::lookup(zero, vec![rule.end_token; 2], 1),
        );
        if position + 1 == used {
            if rule.response_length < rule.maximum_new_tokens {
                require_equal(&mut block, token, end);
            }
        } else {
            let positive = push(&mut block, NativeGraphNode::sub(token, end));
            let negative = push(&mut block, NativeGraphNode::sub(end, token));
            let positive = push(
                &mut block,
                NativeGraphNode::clamped_lookup(positive, vec![0, 1], 0, 1),
            );
            let negative = push(
                &mut block,
                NativeGraphNode::clamped_lookup(negative, vec![0, 1], 0, 1),
            );
            let differs = push(&mut block, NativeGraphNode::add(positive, negative));
            let missing = push(&mut block, NativeGraphNode::lookup(differs, vec![-1, 0], 1));
            push(&mut block, NativeGraphNode::lookup(missing, vec![0, 0], 1));
        }
    }
    for position in used..token_shape[0] {
        let padding = push(&mut block, NativeGraphNode::slice(0, 0, position, 1));
        require_zero(&mut block, padding);
    }
    block.outputs = vec![indices];
    block.tensor_shapes()?;
    let first = graph
        .num_inputs()
        .checked_add(graph.nodes.len())
        .ok_or_else(invalid)?;
    first.checked_add(block.nodes.len()).ok_or_else(invalid)?;
    let map = |id| match id {
        0 => tokens,
        1 => scores,
        _ => first + id - 2,
    };
    let result = map(indices);
    for mut node in block.nodes {
        rebase(&mut node, map);
        graph.nodes.push(node);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::super::{
        native_graph::{NativeGraphProof, NativeGraphWitness},
        DoryScheme,
    };
    use super::*;
    use crate::poly::commitment::commitment_scheme::CommitmentScheme;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    fn rule(response_length: usize, maximum_new_tokens: usize) -> GreedySequenceRule {
        GreedySequenceRule {
            prompt_length: 1,
            score_start_position: 0,
            response_length,
            maximum_new_tokens,
            logical_vocabulary: 3,
            end_token: 2,
        }
    }
    fn graph(r: GreedySequenceRule) -> NativeGraph {
        let mut g = NativeGraph {
            context: b"registered greedy component".to_vec(),
            input_shapes: vec![vec![4], vec![4, 4]],
            nodes: vec![],
            outputs: vec![0],
        };
        let out = append_greedy_sequence(&mut g, 0, 1, r).unwrap();
        g.outputs = vec![out];
        g
    }
    // The first two rows select 1 then EOS=2. Padded score 3 must never win.
    fn scores() -> Vec<i32> {
        vec![0, 9, 8, i32::MAX, 0, 1, 9, i32::MAX, 3, 2, 1, 9, 0, 0, 0, 9]
    }
    #[test]
    fn greedy_sequence_proves_shift_ties_padding_and_stopping() {
        let pp = DoryScheme::setup_prover(12);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 64);
        for (r, tokens) in [
            (rule(2, 3), vec![2, 1, 2, 0]),
            (rule(1, 1), vec![0, 1, 0, 0]),
        ] {
            let (statement, witness) =
                NativeGraphWitness::commit(graph(r), vec![tokens, scores()], &pp).unwrap();
            let proof = NativeGraphProof::prove(&statement, witness, &pp, &gens).unwrap();
            let mut bytes = vec![];
            proof.serialize_compressed(&mut bytes).unwrap();
            let proof = NativeGraphProof::deserialize_compressed(bytes.as_slice()).unwrap();
            proof.verify(&statement, &vp, &gens).unwrap();
            let mut wrong = statement.clone();
            wrong.graph.context.push(0);
            assert!(proof.verify(&wrong, &vp, &gens).is_err());
        }
        let mut tied = scores();
        tied[..4].copy_from_slice(&[0, 9, 9, i32::MAX]);
        assert!(NativeGraphWitness::uncommitted(
            &graph(rule(2, 3)),
            vec![vec![0, 1, 2, 0], tied.clone()]
        )
        .is_ok());
        assert!(
            NativeGraphWitness::uncommitted(&graph(rule(2, 3)), vec![vec![0, 2, 2, 0], tied])
                .is_err()
        );
    }
    #[test]
    fn greedy_sequence_rejects_wrong_tokens_early_end_and_unfinished_response() {
        for tokens in [
            vec![0, 0, 2, 0],
            vec![0, 2, 2, 0],
            vec![0, 1, 1, 0],
            vec![0, 1, 2, 1],
        ] {
            assert!(
                NativeGraphWitness::uncommitted(&graph(rule(2, 3)), vec![tokens, scores()])
                    .is_err()
            );
        }
        assert!(NativeGraphWitness::uncommitted(
            &graph(rule(1, 3)),
            vec![vec![0, 1, 0, 0], scores()]
        )
        .is_err());
        let mut early = scores();
        early[..4].copy_from_slice(&[0, 1, 9, 9]);
        assert!(
            NativeGraphWitness::uncommitted(&graph(rule(2, 3)), vec![vec![0, 2, 2, 0], early])
                .is_err()
        );
    }
    #[test]
    fn greedy_sequence_uses_original_hidden_producer() {
        let pp = DoryScheme::setup_prover(12);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 64);
        let mut g = NativeGraph {
            context: b"original generated sequence".to_vec(),
            input_shapes: vec![vec![4], vec![4, 4]],
            nodes: vec![
                NativeGraphNode::lookup(0, vec![0, 1, 2, 3], 2),
                NativeGraphNode::reshape(1, vec![4, 4]),
            ],
            outputs: vec![2],
        };
        let out = append_greedy_sequence(&mut g, 2, 3, rule(2, 3)).unwrap();
        g.outputs = vec![out];
        let (statement, witness) =
            NativeGraphWitness::commit(g.clone(), vec![vec![0, 1, 2, 0], scores()], &pp).unwrap();
        let proof = NativeGraphProof::prove(&statement, witness, &pp, &gens).unwrap();
        proof.verify(&statement, &vp, &gens).unwrap();
        let (different, other) =
            NativeGraphWitness::commit(g, vec![vec![1, 1, 2, 0], scores()], &pp).unwrap();
        NativeGraphProof::prove(&different, other, &pp, &gens)
            .unwrap()
            .verify(&different, &vp, &gens)
            .unwrap();
        let mut mixed = statement;
        let id = common::CommittedPoly::DivNodeQuotient(2);
        mixed.commitments.insert(id, different.commitments[&id]);
        assert!(proof.verify(&mixed, &vp, &gens).is_err());
    }
    #[test]
    fn greedy_sequence_projects_only_required_score_rows() {
        let pp = DoryScheme::setup_prover(12);
        let vp = DoryScheme::setup_verifier(&pp);
        let gens = DoryScheme::pedersen_generators(&pp, 64);
        for (positions, offset, scores) in [
            (
                4,
                0,
                vec![
                    9,
                    0,
                    0,
                    i32::MAX,
                    0,
                    9,
                    8,
                    i32::MAX,
                    0,
                    1,
                    9,
                    i32::MAX,
                    0,
                    0,
                    0,
                    i32::MAX,
                ],
            ),
            (2, 1, vec![0, 9, 8, i32::MAX, 0, 1, 9, i32::MAX]),
        ] {
            let mut graph = NativeGraph {
                context: b"registered projected sequence".to_vec(),
                input_shapes: vec![vec![4], vec![positions, 4]],
                nodes: vec![],
                outputs: vec![],
            };
            let selected = append_greedy_sequence(
                &mut graph,
                0,
                1,
                GreedySequenceRule {
                    prompt_length: 2,
                    score_start_position: offset,
                    ..rule(2, 3)
                },
            )
            .unwrap();
            graph.outputs = vec![selected];
            let (statement, witness) =
                NativeGraphWitness::commit(graph, vec![vec![2, 0, 1, 2], scores], &pp).unwrap();
            assert_eq!(witness.outputs(), [vec![1, 2]]);
            NativeGraphProof::prove(&statement, witness, &pp, &gens)
                .unwrap()
                .verify(&statement, &vp, &gens)
                .unwrap();
        }
    }
    #[test]
    fn greedy_sequence_rejects_bad_registration_without_mutating_graph() {
        for r in [
            rule(0, 2),
            rule(3, 2),
            rule(4, 4),
            GreedySequenceRule {
                prompt_length: 0,
                ..rule(2, 3)
            },
            GreedySequenceRule {
                end_token: 3,
                ..rule(2, 3)
            },
            GreedySequenceRule {
                logical_vocabulary: 5,
                ..rule(2, 3)
            },
        ] {
            let mut g = NativeGraph {
                context: b"invalid".to_vec(),
                input_shapes: vec![vec![4], vec![4, 4]],
                nodes: vec![],
                outputs: vec![0],
            };
            assert!(append_greedy_sequence(&mut g, 0, 1, r).is_err());
            assert!(g.nodes.is_empty());
        }
    }
}
