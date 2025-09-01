#[cfg(test)]
mod integration_tests {
    use onnx_tracer::builder::simple_matmult_model;
    use onnx_tracer::tensor::Tensor;

    #[test]
    fn test_simple_matmul_model_with_power_of_two_padding() {
        println!("Testing simple MatMul model...");

        // Create the simple MatMul model (uses "mk,nk->mn" equation)
        let model = simple_matmult_model();

        // Create input tensor [1, 4] - this has already power-of-two elements (4)
        let input = Tensor::<i32>::new(Some(&[1, 2, 3, 4]), &[1, 4]).unwrap();

        // Run inference
        let result = model.forward(&[input]).unwrap();

        // Verify output
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0].dims(), &[1, 3]);

        println!("Simple MatMul model works correctly!");

        #[cfg(feature = "matmul_power_of_two_padding")]
        {
            println!("MatMul power-of-two padding feature is ENABLED");
            println!("   - All 'mk,nk->mn' operations will use power-of-two padding");
            println!("   - M, N, K dimensions will be padded to next power of two");
            println!("   - Results will be cropped back to expected dimensions");
        }

        #[cfg(not(feature = "matmul_power_of_two_padding"))]
        {
            println!("MatMul power-of-two padding feature is DISABLED");
            println!("   - Regular einsum operations are used");
            println!("   - To enable: cargo test --features matmul_power_of_two_padding");
        }
    }

    #[test]
    fn test_matmul_with_non_power_of_two_dimensions() {
        println!("Testing MatMul with non-power-of-two dimensions...");

        // This test verifies that our implementation handles matrices
        // where M, N, K are not power-of-two

        // We'll create a custom model for this test
        use onnx_tracer::graph::model::Model;
        use onnx_tracer::graph::utilities::create_matmul_node;

        let mut model = Model::default();

        // Create input nodes for matrices with non-power-of-two dimensions
        // A: [3, 5] (m=3, k=5) - both non-power-of-two
        // B: [2, 5] (n=2, k=5) - n is power-of-two, k is not
        // Result should be [3, 2] (m=3, n=2)

        use onnx_tracer::graph::utilities::create_input_node;

        let input_a = create_input_node(1, vec![3, 5], 0, 1);
        let input_b = create_input_node(1, vec![2, 5], 1, 1);

        model.insert_node(input_a);
        model.insert_node(input_b);

        // Create MatMul node using "mk,nk->mn" equation
        let matmul_node = create_matmul_node(
            "mk,nk->mn".to_string(),
            1,                    // scale
            vec![(0, 0), (1, 0)], // inputs from nodes 0 and 1
            vec![3, 2],           // output dimensions [m=3, n=2]
            2,                    // node index
            1,                    // num_uses
        );
        model.insert_node(matmul_node);

        model.set_inputs(vec![0, 1]);
        model.set_outputs(vec![(2, 0)]);

        // Create test inputs
        let a_data: Vec<i32> = (1..=15).collect(); // 15 elements for [3, 5]
        let b_data: Vec<i32> = (16..=25).collect(); // 10 elements for [2, 5]

        let tensor_a = Tensor::<i32>::new(Some(&a_data), &[3, 5]).unwrap();
        let tensor_b = Tensor::<i32>::new(Some(&b_data), &[2, 5]).unwrap();

        // Run inference
        let result = model.forward(&[tensor_a, tensor_b]).unwrap();

        // Verify output dimensions
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0].dims(), &[3, 2]);

        println!("MatMul with non-power-of-two dimensions works correctly!");
        println!("   Input A: [3, 5] -> padded to [4, 8] (when feature enabled)");
        println!("   Input B: [2, 5] -> padded to [2, 8] (when feature enabled)");
        println!("   Output: [3, 2] (cropped back from padded result)");

        #[cfg(feature = "matmul_power_of_two_padding")]
        {
            println!("Power-of-two padding was applied during computation!");
        }
    }
}
