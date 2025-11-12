use onnx_tracer::{
    graph::{
        model::NodeType,
        node::{map_outlet_indices, Node, SupportedOp},
        vars::VarScales,
    },
    ops::{poly::PolyOp, Constant, Input, InputType},
    tensor::Tensor,
    utils::parsing::create_input_node,
    RunArgs,
};
use std::collections::BTreeMap;
use tract_onnx::prelude::OutletId;

/// Test the helper method `process_and_prune_inputs`
#[test]
fn test_process_and_prune_inputs() {
    // Create test input nodes
    let input_node1 = create_input_node(7, vec![1, 3], 0, 1);
    let input_node2 = create_input_node(7, vec![1, 3], 1, 1);
    let input_node3 = create_input_node(7, vec![1, 3], 2, 1);
    let inputs = vec![
        NodeType::Node(input_node1),
        NodeType::Node(input_node2),
        NodeType::Node(input_node3),
    ];

    // Test case 1: No deleted indices
    let input_ids = vec![(0, 0), (1, 0), (2, 0)];
    let deleted_indices = vec![];

    let (pruned_ids, scales) = Node::process_and_prune_inputs(input_ids, &inputs, &deleted_indices);
    assert_eq!(pruned_ids.len(), 3);
    assert_eq!(scales.len(), 3);
    assert_eq!(scales[0], 7);
    assert_eq!(scales[1], 7);
    assert_eq!(scales[2], 7);

    // Test case 2: With deleted indices
    let input_ids = vec![(0, 0), (1, 0), (2, 0)];
    let deleted_indices = vec![1]; // Remove second input

    let (pruned_ids, scales) = Node::process_and_prune_inputs(input_ids, &inputs, &deleted_indices);
    assert_eq!(pruned_ids.len(), 2);
    assert_eq!(scales.len(), 2);
    assert_eq!(pruned_ids[0], (0, 0));
    assert_eq!(pruned_ids[1], (2, 0));

    // Test case 3: Multiple deleted indices
    let input_ids = vec![(0, 0), (1, 0), (2, 0)];
    let deleted_indices = vec![0, 2]; // Remove first and third inputs

    let (pruned_ids, scales) = Node::process_and_prune_inputs(input_ids, &inputs, &deleted_indices);
    assert_eq!(pruned_ids.len(), 1);
    assert_eq!(scales.len(), 1);
    assert_eq!(pruned_ids[0], (1, 0));
}

/// Test the helper method `update_node_map_with_inputs`
#[test]
fn test_update_node_map_with_inputs() {
    let mut other_nodes = BTreeMap::new();

    // Create test input nodes
    let input_node1 = create_input_node(7, vec![1, 3], 5, 1);
    let input_node2 = create_input_node(7, vec![1, 3], 10, 1);
    let inputs = vec![NodeType::Node(input_node1), NodeType::Node(input_node2)];

    // Update the node map
    Node::update_node_map_with_inputs(&mut other_nodes, &inputs);

    // Verify nodes were added
    assert_eq!(other_nodes.len(), 2);
    assert!(other_nodes.contains_key(&5));
    assert!(other_nodes.contains_key(&10));

    // Verify node data
    if let Some(NodeType::Node(node)) = other_nodes.get(&5) {
        assert_eq!(node.idx, 5);
        assert_eq!(node.out_dims, vec![1, 3]);
        assert_eq!(node.out_scale, 7);
    } else {
        panic!("Node 5 not found or wrong type");
    }
}

/// Test the helper method `determine_node_index`
#[test]
fn test_determine_node_index() {
    // Test case 1: Input operation should get index 0
    let input_op = SupportedOp::Input(Input {
        scale: 7,
        datum_type: InputType::F32,
    });
    let idx = Node::determine_node_index(&input_op, 42);
    assert_eq!(idx, 0);

    // Test case 2: Non-input operation should get default index
    let add_op = SupportedOp::Linear(PolyOp::Add);
    let idx = Node::determine_node_index(&add_op, 42);
    assert_eq!(idx, 42);

    // Test case 3: Constant operation should get default index
    let constant_op = SupportedOp::Constant(Constant {
        raw_values: Tensor::from([1.0, 2.0, 3.0].into_iter()),
        quantized_values: Tensor::from([1000, 2000, 3000].into_iter()),
    });
    let idx = Node::determine_node_index(&constant_op, 100);
    assert_eq!(idx, 100);
}

/// Test the mapping of outlet indices with various remapping scenarios
#[test]
fn test_outlet_mapping_comprehensive() {
    // Test case 1: Identity mapping
    let outlets = vec![
        OutletId::new(0, 0),
        OutletId::new(1, 1),
        OutletId::new(2, 0),
    ];
    let mut remappings = BTreeMap::new();
    remappings.insert(0, 0);
    remappings.insert(1, 1);
    remappings.insert(2, 2);

    let result = map_outlet_indices(&outlets, &remappings);
    assert_eq!(result, vec![(0, 0), (1, 1), (2, 0)]);

    // Test case 2: Complex remapping with gaps
    let outlets = vec![
        OutletId::new(0, 0),
        OutletId::new(3, 1),
        OutletId::new(5, 0),
    ];
    let mut remappings = BTreeMap::new();
    remappings.insert(0, 10);
    remappings.insert(3, 25);
    remappings.insert(5, 30);

    let result = map_outlet_indices(&outlets, &remappings);
    assert_eq!(result, vec![(10, 0), (25, 1), (30, 0)]);
}

/// Test input pruning behavior
#[test]
fn test_input_pruning_logic() {
    // Create test data for pruning behavior
    let mut input_ids = vec![(0, 0), (1, 0), (2, 0), (3, 0)];
    let _ = vec![
        NodeType::Node(create_input_node(7, vec![1, 3], 0, 1)),
        NodeType::Node(create_input_node(7, vec![1, 3], 1, 1)),
        NodeType::Node(create_input_node(7, vec![1, 3], 2, 1)),
        NodeType::Node(create_input_node(7, vec![1, 3], 3, 1)),
    ];

    // Test pruning with alternating deletions
    let deleted_indices = [1, 3];

    // Mark inputs for deletion
    input_ids.iter_mut().enumerate().for_each(|(i, (idx, _))| {
        if deleted_indices.contains(&i) {
            *idx = usize::MAX;
        }
    });

    // Verify marking worked correctly
    assert_eq!(input_ids[0], (0, 0)); // Not deleted
    assert_eq!(input_ids[1], (usize::MAX, 0)); // Deleted
    assert_eq!(input_ids[2], (2, 0)); // Not deleted
    assert_eq!(input_ids[3], (usize::MAX, 0)); // Deleted

    // Test retention
    input_ids.retain(|(idx, _)| *idx != usize::MAX);
    assert_eq!(input_ids.len(), 2);
    assert_eq!(input_ids[0], (0, 0));
    assert_eq!(input_ids[1], (2, 0));
}

/// Test scale computation and validation
#[test]
fn test_scale_computation() {
    // Create input nodes with different scales
    let input_node1 = create_input_node(5, vec![1, 3], 0, 1);
    let input_node2 = create_input_node(10, vec![1, 3], 1, 1);
    let input_node3 = create_input_node(15, vec![1, 3], 2, 1);
    let inputs = vec![
        NodeType::Node(input_node1),
        NodeType::Node(input_node2),
        NodeType::Node(input_node3),
    ];

    let input_ids = vec![(0, 0), (1, 0), (2, 0)];
    let deleted_indices = vec![];

    let (_, scales) = Node::process_and_prune_inputs(input_ids, &inputs, &deleted_indices);

    // Verify scale extraction
    assert_eq!(scales.len(), 3);
    assert_eq!(scales[0], 5);
    assert_eq!(scales[1], 10);
    assert_eq!(scales[2], 15);
}

/// Test edge cases for input processing
#[test]
fn test_input_processing_edge_cases() {
    // Test case 1: Empty inputs
    let input_ids = vec![];
    let inputs = vec![];
    let deleted_indices = vec![];

    let (pruned_ids, scales) = Node::process_and_prune_inputs(input_ids, &inputs, &deleted_indices);
    assert_eq!(pruned_ids.len(), 0);
    assert_eq!(scales.len(), 0);

    // Test case 2: All inputs deleted
    let input_ids = vec![(0, 0), (1, 0)];
    let inputs = vec![
        NodeType::Node(create_input_node(7, vec![1, 3], 0, 1)),
        NodeType::Node(create_input_node(7, vec![1, 3], 1, 1)),
    ];
    let deleted_indices = vec![0, 1]; // Delete all

    let (pruned_ids, scales) = Node::process_and_prune_inputs(input_ids, &inputs, &deleted_indices);
    assert_eq!(pruned_ids.len(), 0);
    assert_eq!(scales.len(), 0);

    // Test case 3: Single input, not deleted
    let input_ids = vec![(5, 0)]; // Node 5, outlet 0 (not 1)
    let inputs = vec![NodeType::Node(create_input_node(20, vec![2, 4], 5, 1))];
    let deleted_indices = vec![];

    let (pruned_ids, scales) = Node::process_and_prune_inputs(input_ids, &inputs, &deleted_indices);
    assert_eq!(pruned_ids.len(), 1);
    assert_eq!(pruned_ids[0], (5, 0));
    assert_eq!(scales.len(), 1);
    assert_eq!(scales[0], 20);
}

/// Test VarScales integration
#[test]
fn test_varscales_integration() {
    let run_args = RunArgs::default();
    let scales = VarScales::from_args(&run_args);

    // Test basic properties
    assert!(scales.get_max() > 0);
    assert!(scales.rebase_multiplier > 0);
}

/// Test node map update behavior
#[test]
fn test_node_map_update_behavior() {
    let mut other_nodes = BTreeMap::new();

    // Pre-populate with one node
    let existing_node = create_input_node(5, vec![2, 2], 100, 1);
    other_nodes.insert(100, NodeType::Node(existing_node));

    // Create new input nodes to add
    let input_node1 = create_input_node(7, vec![1, 3], 200, 1);
    let input_node2 = create_input_node(7, vec![1, 3], 300, 1);
    let inputs = vec![NodeType::Node(input_node1), NodeType::Node(input_node2)];

    // Update with new nodes
    Node::update_node_map_with_inputs(&mut other_nodes, &inputs);

    // Verify all nodes are present
    assert_eq!(other_nodes.len(), 3);
    assert!(other_nodes.contains_key(&100)); // Existing
    assert!(other_nodes.contains_key(&200)); // New
    assert!(other_nodes.contains_key(&300)); // New

    // Verify data integrity
    if let Some(NodeType::Node(node)) = other_nodes.get(&200) {
        assert_eq!(node.idx, 200);
        assert_eq!(node.out_scale, 7);
        assert_eq!(node.out_dims, vec![1, 3]);
    } else {
        panic!("Node 200 not found");
    }
}

/// Test operation type determination for node indices
#[test]
fn test_operation_type_node_index() {
    // Test different operation types
    let ops = vec![
        SupportedOp::Input(Input {
            scale: 7,
            datum_type: InputType::F32,
        }),
        SupportedOp::Linear(PolyOp::Add),
        SupportedOp::Linear(PolyOp::Mult),
        SupportedOp::Constant(Constant {
            raw_values: Tensor::from([1.0].into_iter()),
            quantized_values: Tensor::from([1000].into_iter()),
        }),
    ];

    let expected_indices = [0, 42, 42, 42]; // Only Input gets 0

    for (op, expected) in ops.iter().zip(expected_indices.iter()) {
        let idx = Node::determine_node_index(op, 42);
        assert_eq!(idx, *expected, "Failed for operation: {op:?}");
    }
}
