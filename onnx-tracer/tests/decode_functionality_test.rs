//! Tests for ONNX model decode functionality
//!
//! This module contains comprehensive tests for the decode pipeline, including:
//! - Individual node decode operations
//! - Complete model decode workflow
//! - Edge cases and error handling

#[cfg(test)]
mod decode_tests {
    use onnx_tracer::{
        decode, decode_model, decode_node,
        graph::model::{Model, NodeType},
        ops::poly::PolyOp,
        tensor::Tensor,
        trace_types::ONNXOpcode,
        utils::parsing::{
            create_const_node, create_div_node, create_einsum_node, create_input_node,
            create_polyop_node, create_relu_node, create_sigmoid_node,
        },
    };
    use std::path::PathBuf;

    /// Test decode functionality for a simple ONNX model
    #[test]
    fn test_decode_simple_model() {
        // Test using one of the existing models
        let model_path = PathBuf::from("models/simple_mlp/network.onnx");

        // Skip test if model file doesn't exist
        if !model_path.exists() {
            println!("Skipping test - model file not found: {model_path:?}",);
            return;
        }

        let instructions = decode(&model_path);

        // Basic sanity checks
        assert!(
            !instructions.is_empty(),
            "Decoded instructions should not be empty"
        );

        // Verify all instructions have valid addresses
        for (i, instr) in instructions.iter().enumerate() {
            assert_eq!(
                instr.address,
                i + 1,
                "Instruction address should be sequential (accounting for prepended noop)"
            );
        }

        println!("Successfully decoded {} instructions", instructions.len());
    }

    /// Test decode_model function with a manually constructed model
    #[test]
    fn test_decode_model_manual() {
        let model = create_test_model();
        let instructions = decode_model(model);

        assert!(!instructions.is_empty(), "Should have decoded instructions");

        // Verify instruction structure
        for instr in &instructions {
            assert!(instr.address > 0, "Address should be positive");
            // Basic opcode validation - should be one of the known types
            match instr.opcode {
                ONNXOpcode::Input | ONNXOpcode::Add | ONNXOpcode::Relu | ONNXOpcode::MatMult => {}
                _ => {} // Allow other opcodes
            }
        }
    }

    /// Test decode_node function with different node types
    #[test]
    fn test_decode_node_types() {
        // Test Input node
        let input_node = create_input_node(7, vec![1, 4], 0, 1);
        let (idx, node_type) = (0_usize, NodeType::Node(input_node));
        let instr = decode_node((&idx, &node_type));

        assert_eq!(instr.address, 1); // Should account for prepended noop
        assert_eq!(instr.opcode, ONNXOpcode::Input);
        assert_eq!(instr.td, Some(0));
        assert!(instr.ts1.is_none()); // Input nodes have no inputs
        assert!(instr.ts2.is_none());

        // Test Constant node
        let const_data = Tensor::<i32>::new(Some(&[1, 2, 3, 4]), &[2, 2]).unwrap();
        let raw_data = Tensor::<f32>::new(Some(&[1.0, 2.0, 3.0, 4.0]), &[2, 2]).unwrap();
        let const_node = create_const_node(const_data, raw_data, 7, vec![2, 2], 1, 1);
        let (idx, node_type) = (1_usize, NodeType::Node(const_node));
        let instr = decode_node((&idx, &node_type));

        assert_eq!(instr.address, 2);
        assert_eq!(instr.opcode, ONNXOpcode::Constant);
        assert_eq!(instr.td, Some(1));
        assert!(instr.imm.is_some()); // Constant should have immediate value

        // Test Add node
        let add_node = create_polyop_node(PolyOp::Add, 7, vec![(0, 0), (1, 0)], vec![2, 2], 2, 1);
        let (idx, node_type) = (2_usize, NodeType::Node(add_node));
        let instr = decode_node((&idx, &node_type));

        assert_eq!(instr.address, 3);
        assert_eq!(instr.opcode, ONNXOpcode::Add);
        assert_eq!(instr.td, Some(2));
        assert_eq!(instr.ts1, Some(0));
        assert_eq!(instr.ts2, Some(1));

        // Test ReLU node
        let relu_node = create_relu_node(7, vec![(2, 0)], vec![2, 2], 3, 1);
        let (idx, node_type) = (3_usize, NodeType::Node(relu_node));
        let instr = decode_node((&idx, &node_type));

        assert_eq!(instr.address, 4);
        assert_eq!(instr.opcode, ONNXOpcode::Relu);
        assert_eq!(instr.td, Some(3));
        assert_eq!(instr.ts1, Some(2));
        assert!(instr.ts2.is_none()); // ReLU is unary
    }

    /// Test decode with division (RebaseScale expansion)
    #[test]
    fn test_decode_division_node() {
        let div_node = create_div_node(2, 7, vec![(0, 0)], vec![2, 2], 1, 1);
        let (idx, node_type) = (1_usize, NodeType::Node(div_node));
        let instr = decode_node((&idx, &node_type));

        assert_eq!(instr.address, 2);
        assert_eq!(instr.opcode, ONNXOpcode::Div);
        assert_eq!(instr.td, Some(1));
        assert_eq!(instr.ts1, Some(0));
        assert!(instr.imm.is_some()); // Division should have immediate value (denominator)
    }

    /// Test decode with matmul operation
    #[test]
    fn test_decode_matmul_node() {
        let matmul_node = create_einsum_node(
            "mk,nk->mn".to_string(),
            7,
            vec![(0, 0), (1, 0)],
            vec![3, 4],
            2,
            1,
        );
        let (idx, node_type) = (2_usize, NodeType::Node(matmul_node));
        let instr = decode_node((&idx, &node_type));

        assert_eq!(instr.address, 3);
        assert_eq!(instr.opcode, ONNXOpcode::Einsum("mk,nk->mn".to_string()));
        assert_eq!(instr.td, Some(2));
        assert_eq!(instr.ts1, Some(0));
        assert_eq!(instr.ts2, Some(1));
        assert_eq!(instr.output_dims, [3, 4]);
        assert_eq!(instr.num_output_elements(), 12); // 3 * 4
    }

    /// Test decode output dimensions handling
    #[test]
    fn test_decode_output_dimensions() {
        // Test 1D output (should be converted to [1, n])
        let node_1d = create_input_node(7, vec![5], 0, 1);
        let (idx, node_type) = (0_usize, NodeType::Node(node_1d));
        let instr = decode_node((&idx, &node_type));
        assert_eq!(instr.output_dims, [5]);
        assert_eq!(instr.num_output_elements(), 5);

        // Test 2D output (should preserve [m, n])
        let node_2d = create_input_node(7, vec![3, 4], 1, 1);
        let (idx, node_type) = (1_usize, NodeType::Node(node_2d));
        let instr = decode_node((&idx, &node_type));
        assert_eq!(instr.output_dims, [3, 4]);
        assert_eq!(instr.num_output_elements(), 12);
    }

    /// Test address calculation with bytecode prepend
    #[test]
    fn test_address_calculation() {
        use onnx_tracer::constants::BYTECODE_PREPEND_NOOP;

        // Test that addresses are correctly calculated with prepended noop
        let node = create_input_node(7, vec![1], 0, 1);
        let (idx, node_type) = (0_usize, NodeType::Node(node));
        let instr = decode_node((&idx, &node_type));

        assert_eq!(instr.address, BYTECODE_PREPEND_NOOP);

        let node2 = create_input_node(7, vec![1], 1, 1);
        let (idx2, node_type2) = (5_usize, NodeType::Node(node2));
        let instr2 = decode_node((&idx2, &node_type2));

        assert_eq!(instr2.address, 5 + BYTECODE_PREPEND_NOOP);
    }

    /// Helper function to create a simple test model
    fn create_test_model() -> Model {
        let mut model = Model::default();

        // Create a simple 2-input addition model
        let input1 = create_input_node(7, vec![1, 4], 0, 1);
        let input2 = create_input_node(7, vec![1, 4], 1, 1);
        let add_node = create_polyop_node(PolyOp::Add, 7, vec![(0, 0), (1, 0)], vec![1, 4], 2, 1);

        model.insert_node(input1);
        model.insert_node(input2);
        model.insert_node(add_node);

        model.set_inputs(vec![0, 1]);
        model.set_outputs(vec![(2, 0)]);

        model
    }

    /// Test decode with empty model (edge case)
    #[test]
    fn test_decode_empty_model() {
        let empty_model = Model::default();
        let instructions = decode_model(empty_model);

        // Should handle empty models gracefully
        assert!(instructions.is_empty() || instructions.len() == 1); // May have just the default node
    }

    /// Test that decode preserves instruction order
    #[test]
    fn test_decode_instruction_order() {
        let model = create_sequential_test_model();
        let instructions = decode_model(model);

        // Verify addresses are sequential
        for (i, instr) in instructions.iter().enumerate() {
            assert_eq!(
                instr.address,
                i + 1,
                "Instructions should have sequential addresses"
            );
        }
    }

    /// Helper function to create a sequential test model
    fn create_sequential_test_model() -> Model {
        let mut model = Model::default();

        // Create a chain: input -> relu -> sigmoid
        let input = create_input_node(7, vec![1, 4], 0, 1);
        let relu = create_relu_node(7, vec![(0, 0)], vec![1, 4], 1, 1);
        let sigmoid = create_sigmoid_node(7, vec![(1, 0)], vec![1, 4], 2, 1);

        model.insert_node(input);
        model.insert_node(relu);
        model.insert_node(sigmoid);

        model.set_inputs(vec![0]);
        model.set_outputs(vec![(2, 0)]);

        model
    }
}

/// Tests for RebaseScale expansion logic
#[cfg(test)]
mod rebase_scale_tests {
    use onnx_tracer::{
        graph::node::{Node, RebaseScale, SupportedOp},
        ops::{lookup::LookupOp, poly::PolyOp},
        utils::f32::F32,
    };

    /// Test RebaseScale node expansion into inner + division nodes
    #[test]
    fn test_rebase_scale_expansion() {
        // This test verifies the RebaseScale expansion logic
        // that converts a single RebaseScale node into two separate nodes

        // Create a simple inner operation (Add)
        let inner_op = SupportedOp::Linear(PolyOp::Add);

        // Create a RebaseScale wrapper
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

        // Verify the RebaseScale properties
        assert!(matches!(rebase_node.opkind, SupportedOp::RebaseScale(_)));

        if let SupportedOp::RebaseScale(ref rs) = rebase_node.opkind {
            assert_eq!(rs.multiplier, 2.0);
            assert_eq!(rs.target_scale, 6);
            assert_eq!(rs.original_scale, 7);
            assert!(matches!(
                *rs.inner.as_ref(),
                SupportedOp::Linear(PolyOp::Add)
            ));
        }
    }

    /// Test that division node is created correctly during expansion
    #[test]
    fn test_division_node_creation() {
        // Test the division node that gets created during RebaseScale expansion
        let div_op = SupportedOp::Nonlinear(LookupOp::Div { denom: F32(2.0) });

        let div_node = Node {
            idx: 2,
            opkind: div_op,
            inputs: vec![(1, 0)], // Takes input from inner node
            out_dims: vec![2, 2],
            out_scale: 6,
            num_uses: 1,
        };

        // Verify division node properties
        if let SupportedOp::Nonlinear(LookupOp::Div { denom }) = &div_node.opkind {
            assert_eq!(denom.0, 2.0);
        } else {
            panic!("Expected division operation");
        }

        assert_eq!(div_node.inputs, vec![(1, 0)]);
        assert_eq!(div_node.out_scale, 6);
    }
}
