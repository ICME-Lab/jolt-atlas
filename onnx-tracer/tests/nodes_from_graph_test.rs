//! Tests for nodes_from_graph functionality and its refactored components
//!
//! This module contains tests for the complex node creation logic in model.rs,
//! particularly testing the components that will be extracted during refactoring.

#[cfg(test)]
mod nodes_from_graph_tests {
    use onnx_tracer::{
        graph::{
            model::NodeType,
            node::{Node, RebaseScale, SupportedOp},
        },
        ops::{lookup::LookupOp, poly::PolyOp},
        utils::f32::F32,
    };
    use std::collections::{BTreeMap, HashMap};

    /// Test the RebaseScale expansion logic that splits a RebaseScale node into inner + div nodes
    #[test]
    fn test_rebase_scale_expansion_logic() {
        // Create a RebaseScale node that should be expanded
        let inner_op = SupportedOp::Linear(PolyOp::Add);
        let rebase_scale = RebaseScale {
            inner: Box::new(inner_op),
            multiplier: 2.0,
            target_scale: 6,
            original_scale: 7,
        };

        let original_node = Node {
            idx: 1,
            opkind: SupportedOp::RebaseScale(rebase_scale),
            inputs: vec![(0, 0)],
            out_dims: vec![2, 2],
            out_scale: 6,
            num_uses: 1,
        };

        // Simulate the expansion logic from nodes_from_graph
        if let SupportedOp::RebaseScale(rebase_scale) = &original_node.opkind {
            // This simulates the expansion logic that should be extracted
            let _ = BTreeMap::<usize, NodeType>::new();

            // Create first node: the inner operation
            let inner_node_idx = 1;
            let inner_node = Node {
                idx: inner_node_idx,
                opkind: (*rebase_scale.inner).clone(),
                inputs: original_node.inputs.clone(),
                out_dims: original_node.out_dims.clone(),
                out_scale: rebase_scale.original_scale,
                num_uses: 1, // Will be used by the div node
            };

            // Create second node: the division operation
            let div_node_idx = 2;
            let div_node = Node {
                idx: div_node_idx,
                opkind: SupportedOp::Nonlinear(LookupOp::Div {
                    denom: F32(rebase_scale.multiplier as f32),
                }),
                inputs: vec![(inner_node_idx, 0)], // Takes output from inner node
                out_dims: original_node.out_dims.clone(),
                out_scale: rebase_scale.target_scale,
                num_uses: original_node.num_uses, // Inherits the usage count from original node
            };

            // Verify the expanded nodes
            assert!(matches!(
                inner_node.opkind,
                SupportedOp::Linear(PolyOp::Add)
            ));
            assert_eq!(inner_node.out_scale, 7); // original_scale
            assert_eq!(inner_node.num_uses, 1);

            if let SupportedOp::Nonlinear(LookupOp::Div { denom }) = &div_node.opkind {
                assert_eq!(denom.0, 2.0); // multiplier
            }
            assert_eq!(div_node.out_scale, 6); // target_scale
            assert_eq!(div_node.inputs, vec![(1, 0)]);
        }
    }

    /// Test scale override functionality for input nodes
    #[test]
    fn test_input_scale_override() {
        // This tests the logic that overrides input scales based on override_input_scales
        use onnx_tracer::ops::{Input, InputType};

        let mut original_node = Node {
            idx: 0,
            opkind: SupportedOp::Input(Input {
                scale: 7, // original scale
                datum_type: InputType::F32,
            }),
            inputs: vec![],
            out_dims: vec![1, 4],
            out_scale: 7,
            num_uses: 1,
        };

        let override_input_scales = Some(vec![10]); // Override to scale 10
        let input_idx = 0;

        // Simulate the input scale override logic
        if let Some(ref scales) = override_input_scales {
            if let Some(inp) = original_node.opkind.get_input() {
                let scale = scales[input_idx];
                original_node.opkind = SupportedOp::Input(Input {
                    scale,
                    datum_type: inp.datum_type,
                });
                original_node.out_scale = scale;
            }
        }

        // Verify the override worked
        if let Some(input_op) = original_node.opkind.get_input() {
            assert_eq!(input_op.scale, 10);
        }
        assert_eq!(original_node.out_scale, 10);
    }

    /// Test output scale override functionality
    #[test]
    fn test_output_scale_override() {
        // This tests the logic that overrides output scales and creates RebaseScale nodes
        let mut original_node = Node {
            idx: 1,
            opkind: SupportedOp::Linear(PolyOp::Add),
            inputs: vec![(0, 0)],
            out_dims: vec![2, 2],
            out_scale: 7,
            num_uses: 1,
        };

        let mut override_output_scales = HashMap::new();
        override_output_scales.insert(1, 6); // Override node 1 to scale 6

        let node_index = 1;

        // Simulate the output scale override logic
        if let Some(ref scales) = Some(override_output_scales) {
            if scales.contains_key(&node_index) {
                let scale_diff = original_node.out_scale - scales[&node_index];
                original_node.opkind = if scale_diff > 0 {
                    RebaseScale::rebase(
                        original_node.opkind,
                        scales[&node_index],
                        original_node.out_scale,
                        1,
                    )
                } else {
                    RebaseScale::rebase_up(
                        original_node.opkind,
                        scales[&node_index],
                        original_node.out_scale,
                    )
                };
                original_node.out_scale = scales[&node_index];
            }
        }

        // Verify that a RebaseScale node was created
        assert!(matches!(original_node.opkind, SupportedOp::RebaseScale(_)));
        assert_eq!(original_node.out_scale, 6); // Should be updated to target scale
    }

    /// Test node indexing and remapping logic
    #[test]
    fn test_node_remapping() {
        // This tests the remapping logic that tracks node indices as they're added
        let mut remappings = BTreeMap::<usize, usize>::new();
        let mut nodes = BTreeMap::<usize, NodeType>::new();

        // Simulate adding nodes with remapping
        let node1 = Node {
            idx: 0,
            opkind: SupportedOp::Linear(PolyOp::Add),
            inputs: vec![],
            out_dims: vec![1, 4],
            out_scale: 7,
            num_uses: 1,
        };

        let original_idx = 0;
        let actual_idx = node1.idx;

        // Add to remapping and nodes collection
        remappings.insert(original_idx, actual_idx);
        nodes.insert(actual_idx, NodeType::Node(node1));

        // Verify remapping works
        assert_eq!(remappings[&0], 0);
        assert!(nodes.contains_key(&0));
    }

    /// Test consecutive indices enforcement
    #[test]
    fn test_consecutive_indices() {
        // This tests the ensure_consecutive_indices logic
        let mut nodes = BTreeMap::<usize, NodeType>::new();

        // Create nodes with non-consecutive indices (simulating after RebaseScale expansion)
        let node0 = Node {
            idx: 0,
            opkind: SupportedOp::Linear(PolyOp::Add),
            inputs: vec![],
            out_dims: vec![1, 4],
            out_scale: 7,
            num_uses: 1,
        };

        let node3 = Node {
            idx: 3, // Non-consecutive
            opkind: SupportedOp::Linear(PolyOp::Add),
            inputs: vec![(0, 0)],
            out_dims: vec![1, 4],
            out_scale: 7,
            num_uses: 1,
        };

        let node7 = Node {
            idx: 7, // Non-consecutive
            opkind: SupportedOp::Linear(PolyOp::Add),
            inputs: vec![(3, 0)],
            out_dims: vec![1, 4],
            out_scale: 7,
            num_uses: 1,
        };

        nodes.insert(0, NodeType::Node(node0));
        nodes.insert(3, NodeType::Node(node3));
        nodes.insert(7, NodeType::Node(node7));

        // Before consecutive enforcement, indices are [0, 3, 7]
        let original_keys: Vec<usize> = nodes.keys().cloned().collect();
        assert_eq!(original_keys, vec![0, 3, 7]);

        // This would be the consecutive enforcement logic (simplified)
        // In the actual refactor, this should be extracted to a separate function
        let node_list: Vec<_> = nodes.into_iter().collect();
        let mut consecutive_nodes = BTreeMap::new();

        for (new_idx, (_, mut node_type)) in node_list.into_iter().enumerate() {
            if let NodeType::Node(ref mut node) = node_type {
                node.idx = new_idx;
                // Update input references (simplified - in real code this needs to handle all nodes)
                for input in &mut node.inputs {
                    // This is simplified - the real logic needs to track the mapping
                    if input.0 == 3 {
                        input.0 = 1;
                    }
                    if input.0 == 7 {
                        input.0 = 2;
                    }
                }
            }
            consecutive_nodes.insert(new_idx, node_type);
        }

        // After consecutive enforcement, indices should be [0, 1, 2]
        let new_keys: Vec<usize> = consecutive_nodes.keys().cloned().collect();
        assert_eq!(new_keys, vec![0, 1, 2]);
    }
}
