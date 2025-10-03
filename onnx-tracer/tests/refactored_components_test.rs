//! Additional tests for refactored components
//!
//! Tests the individual extracted methods from the refactoring

#[cfg(test)]
mod refactored_components_tests {
    use onnx_tracer::{
        graph::{
            node::Node,
            utilities::{create_const_node, create_input_node, create_polyop_node},
        },
        ops::poly::PolyOp,
        tensor::Tensor,
    };

    /// Test the refactored extract_input_operand method
    #[test]
    fn test_extract_input_operand() {
        let node = create_polyop_node(
            PolyOp::Add,
            7,
            vec![(0, 0), (1, 0), (2, 0)], // Three inputs
            vec![1, 4],
            2,
            1,
        );

        // Test accessing different input positions
        assert_eq!(node.extract_input_operand(0), Some(0));
        assert_eq!(node.extract_input_operand(1), Some(1));
        assert_eq!(node.extract_input_operand(2), Some(2));
        assert_eq!(node.extract_input_operand(3), None); // Out of bounds
    }

    /// Test the refactored calculate_active_output_elements method
    #[test]
    fn test_calculate_active_output_elements() {
        // Test 1D tensor
        let node_1d = create_input_node(7, vec![5], 0, 1);
        assert_eq!(node_1d.calculate_active_output_elements(), 5);

        // Test 2D tensor
        let node_2d = create_input_node(7, vec![3, 4], 1, 1);
        assert_eq!(node_2d.calculate_active_output_elements(), 12);

        // Test 3D tensor
        let node_3d = create_input_node(7, vec![2, 3, 4], 2, 1);
        assert_eq!(node_3d.calculate_active_output_elements(), 24);
    }

    /// Test the refactored normalize_output_dimensions method
    #[test]
    fn test_normalize_output_dimensions() {
        // Test 1D tensor (should become [1, n])
        let node_1d = create_input_node(7, vec![5], 0, 1);
        assert_eq!(node_1d.normalize_output_dimensions(), [1, 5]);

        // Test 2D tensor (should preserve [m, n])
        let node_2d = create_input_node(7, vec![3, 4], 1, 1);
        assert_eq!(node_2d.normalize_output_dimensions(), [3, 4]);

        // Test 3D tensor (should take first two dimensions)
        let node_3d = create_input_node(7, vec![2, 3, 4], 2, 1);
        assert_eq!(node_3d.normalize_output_dimensions(), [2, 3]);
    }

    /// Test the refactored extract_immediate_value method
    #[test]
    fn test_extract_immediate_value() {
        // Test constant node
        let const_data = Tensor::<i32>::new(Some(&[1, 2, 3, 4]), &[2, 2]).unwrap();
        let raw_data = Tensor::<f32>::new(Some(&[1.0, 2.0, 3.0, 4.0]), &[2, 2]).unwrap();
        let const_node = create_const_node(const_data.clone(), raw_data, 7, vec![2, 2], 1, 1);

        let immediate = const_node.extract_immediate_value();
        assert!(immediate.is_some());
        let imm_tensor = immediate.unwrap();
        assert_eq!(imm_tensor.dims(), &[2, 2]);
        assert_eq!(imm_tensor.inner, const_data.inner);

        // Test non-constant node (should return None)
        let add_node = create_polyop_node(PolyOp::Add, 7, vec![(0, 0), (1, 0)], vec![2, 2], 2, 1);
        assert!(add_node.extract_immediate_value().is_none());
    }

    /// Test the refactored create_scalar_immediate_tensor method
    #[test]
    fn test_create_scalar_immediate_tensor() {
        let result = Node::create_scalar_immediate_tensor(42, 6);
        assert!(result.is_some());

        let tensor = result.unwrap();
        assert_eq!(tensor.inner.len(), 6);
        assert!(tensor.inner.iter().all(|&x| x == 42));
    }

    /// Test that the decode method still works correctly after refactoring
    #[test]
    fn test_decode_method_after_refactoring() {
        let node = create_polyop_node(PolyOp::Add, 7, vec![(0, 0), (1, 0)], vec![2, 2], 2, 1);

        let instr = node.decode(5);

        // Verify the decoded instruction has correct structure
        assert_eq!(instr.address, 5);
        assert_eq!(instr.ts1, Some(0));
        assert_eq!(instr.ts2, Some(1));
        assert_eq!(instr.ts3, None);
        assert_eq!(instr.td, Some(2));
        assert_eq!(instr.active_output_elements, 4);
        assert_eq!(instr.output_dims, [2, 2]);
    }
}

/// Tests for the refactored Model methods
#[cfg(test)]
mod model_refactor_tests {
    use onnx_tracer::{
        graph::{
            model::Model,
            node::{Node, RebaseScale, SupportedOp},
        },
        ops::{lookup::LookupOp, poly::PolyOp, Input, InputType},
    };
    use std::collections::{BTreeMap, HashMap};

    /// Test the extracted apply_input_scale_override method
    #[test]
    fn test_apply_input_scale_override() {
        let mut node = Node {
            idx: 0,
            opkind: SupportedOp::Input(Input {
                scale: 7,
                datum_type: InputType::F32,
            }),
            inputs: vec![],
            out_dims: vec![1, 4],
            out_scale: 7,
            num_uses: 1,
        };

        let override_scales = Some(vec![10]);
        let mut input_idx = 0;

        Model::apply_input_scale_override(&mut node, &override_scales, &mut input_idx);

        // Verify scale was overridden
        if let Some(input_op) = node.opkind.get_input() {
            assert_eq!(input_op.scale, 10);
        }
        assert_eq!(node.out_scale, 10);
        assert_eq!(input_idx, 1);
    }

    /// Test the extracted apply_output_scale_override method
    #[test]
    fn test_apply_output_scale_override() {
        let mut node = Node {
            idx: 1,
            opkind: SupportedOp::Linear(PolyOp::Add),
            inputs: vec![(0, 0)],
            out_dims: vec![2, 2],
            out_scale: 7,
            num_uses: 1,
        };

        let mut override_scales = HashMap::new();
        override_scales.insert(1, 6);

        Model::apply_output_scale_override(&mut node, 1, &Some(override_scales));

        // Verify that a RebaseScale node was created
        assert!(matches!(node.opkind, SupportedOp::RebaseScale(_)));
        assert_eq!(node.out_scale, 6);
    }

    /// Test the extracted expand_rebase_scale_node method
    #[test]
    fn test_expand_rebase_scale_node() {
        let inner_op = SupportedOp::Linear(PolyOp::Add);
        let rebase_scale = RebaseScale {
            inner: Box::new(inner_op),
            multiplier: 2.0,
            target_scale: 6,
            original_scale: 7,
        };

        let original_node = Node {
            idx: 1,
            opkind: SupportedOp::RebaseScale(rebase_scale.clone()),
            inputs: vec![(0, 0)],
            out_dims: vec![2, 2],
            out_scale: 6,
            num_uses: 2,
        };

        let (inner_node, div_node) =
            Model::expand_rebase_scale_node(&original_node, &rebase_scale, 10);

        // Verify inner node
        assert_eq!(inner_node.idx, 10);
        assert!(matches!(
            inner_node.opkind,
            SupportedOp::Linear(PolyOp::Add)
        ));
        assert_eq!(inner_node.out_scale, 7); // original_scale
        assert_eq!(inner_node.num_uses, 1);
        assert_eq!(inner_node.inputs, vec![(0, 0)]);

        // Verify division node
        assert_eq!(div_node.idx, 11);
        if let SupportedOp::Nonlinear(LookupOp::Div { denom }) = &div_node.opkind {
            assert_eq!(denom.0, 2.0);
        } else {
            panic!("Expected division operation");
        }
        assert_eq!(div_node.out_scale, 6); // target_scale
        assert_eq!(div_node.num_uses, 2); // inherited from original
        assert_eq!(div_node.inputs, vec![(10, 0)]); // input from inner node
    }

    /// Test that handle_node_insertion works correctly for normal nodes
    #[test]
    fn test_handle_node_insertion_normal() {
        let mut nodes = BTreeMap::new();
        let mut remappings = BTreeMap::new();

        let node = Node {
            idx: 1,
            opkind: SupportedOp::Linear(PolyOp::Add),
            inputs: vec![(0, 0)],
            out_dims: vec![2, 2],
            out_scale: 7,
            num_uses: 1,
        };

        Model::handle_node_insertion(&mut nodes, &mut remappings, 0, node);

        // Verify normal node insertion
        assert_eq!(remappings.get(&0), Some(&1));
        assert!(nodes.contains_key(&1));
        assert_eq!(nodes.len(), 1);
    }

    /// Test that handle_node_insertion expands RebaseScale nodes correctly
    #[test]
    fn test_handle_node_insertion_rebase_scale() {
        let mut nodes = BTreeMap::new();
        let mut remappings = BTreeMap::new();

        let inner_op = SupportedOp::Linear(PolyOp::Add);
        let rebase_scale = RebaseScale {
            inner: Box::new(inner_op),
            multiplier: 2.0,
            target_scale: 6,
            original_scale: 7,
        };

        let rebase_node = Node {
            idx: 1,
            opkind: SupportedOp::RebaseScale(rebase_scale),
            inputs: vec![(0, 0)],
            out_dims: vec![2, 2],
            out_scale: 6,
            num_uses: 1,
        };

        Model::handle_node_insertion(&mut nodes, &mut remappings, 0, rebase_node);

        // Verify RebaseScale expansion created two nodes
        assert_eq!(nodes.len(), 2);
        assert!(nodes.contains_key(&0)); // inner node
        assert!(nodes.contains_key(&1)); // div node
        assert_eq!(remappings.get(&0), Some(&1)); // remapping points to div node
    }
}
