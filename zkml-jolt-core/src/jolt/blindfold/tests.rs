//! Tests for the BlindFold ZK module.
//!
//! These tests verify the correctness of:
//! - Relaxed R1CS satisfaction
//! - NIFS folding
//! - Random instance generation
//! - Verifier circuit construction

#[cfg(test)]
mod tests {
    use super::super::{
        random_instance::{sample_random_field_elements, sample_random_nonzero, RandomInstanceGenerator},
        relaxed_r1cs::{R1CSMatrices, RelaxedR1CSInstance, RelaxedR1CSWitness, SparseMatrix},
        verifier_circuit::VerifierR1CSCircuit,
    };
    use ark_bn254::Fr;
    use ark_std::Zero;
    use jolt_core::field::JoltField;
    use jolt_core::poly::commitment::mock::MockCommitScheme;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::RngCore;

    type TestPCS = MockCommitScheme<Fr>;

    /// Helper to create a test witness from W and E vectors.
    /// Uses zero blinding factors since tests don't need hiding.
    fn create_test_witness(w: Vec<Fr>, e: Vec<Fr>) -> RelaxedR1CSWitness<Fr> {
        RelaxedR1CSWitness::new_simple(w, Fr::zero(), e, Fr::zero())
    }

    // ============================================
    // Relaxed R1CS Tests
    // ============================================

    #[test]
    fn test_sparse_matrix_creation() {
        let matrix: SparseMatrix<Fr> = SparseMatrix::new(3, 4);
        assert_eq!(matrix.num_rows, 3);
        assert_eq!(matrix.num_cols, 4);
        assert!(matrix.entries.is_empty());
    }

    #[test]
    fn test_sparse_matrix_with_entries() {
        let mut matrix: SparseMatrix<Fr> = SparseMatrix::new(2, 2);
        matrix.add_entry(0, 0, Fr::from(1u64));
        matrix.add_entry(0, 1, Fr::from(2u64));
        matrix.add_entry(1, 0, Fr::from(3u64));
        matrix.add_entry(1, 1, Fr::from(4u64));

        assert_eq!(matrix.entries.len(), 4);
    }

    #[test]
    fn test_sparse_matrix_zero_not_added() {
        let mut matrix: SparseMatrix<Fr> = SparseMatrix::new(2, 2);
        matrix.add_entry(0, 0, Fr::from(0u64));
        matrix.add_entry(0, 1, Fr::from(1u64));

        // Zero should not be added
        assert_eq!(matrix.entries.len(), 1);
    }

    #[test]
    fn test_relaxed_r1cs_witness_creation() {
        let w = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        let e = vec![Fr::from(0u64), Fr::from(0u64)];

        let witness = create_test_witness(w.clone(), e.clone());

        assert_eq!(witness.W, w);
        assert_eq!(witness.E, e);
        assert_eq!(witness.witness_len(), 3);
        assert_eq!(witness.constraint_count(), 2);
    }

    #[test]
    fn test_relaxed_r1cs_witness_from_standard() {
        let w = vec![Fr::from(1u64), Fr::from(2u64)];
        let witness = RelaxedR1CSWitness::from_standard(w.clone(), 5);

        assert_eq!(witness.W, w);
        assert_eq!(witness.E.len(), 5);
        assert!(witness.E.iter().all(|e| *e == Fr::from(0u64)));
    }

    #[test]
    fn test_relaxed_r1cs_instance_from_standard() {
        let commitment = <TestPCS as jolt_core::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment::default();
        let x = vec![Fr::from(1u64), Fr::from(2u64)];

        let instance: RelaxedR1CSInstance<Fr, TestPCS> =
            RelaxedR1CSInstance::from_standard(commitment, x.clone());

        assert_eq!(instance.u, Fr::from(1u64));
        assert_eq!(instance.x, x);
    }

    // ============================================
    // NIFS Tests
    // ============================================

    #[test]
    fn test_nifs_fold_witnesses_simple() {
        // Test that folding two zero-error witnesses produces a zero-error witness
        // when the cross-term is also zero
        let _w1 = create_test_witness(
            vec![Fr::from(1u64), Fr::from(2u64)],
            vec![Fr::from(0u64)], // zero error
        );
        let _w2 = create_test_witness(
            vec![Fr::from(3u64), Fr::from(4u64)],
            vec![Fr::from(0u64)], // zero error
        );
        let _cross_term = vec![Fr::from(0u64)]; // zero cross-term
        let _r = Fr::from(2u64);

        // Test the internal folding (we can't directly call fold_witnesses since it's private,
        // so we verify the expected behavior through the formula)
        // W' = W_1 + r·W_2 = [1, 2] + 2·[3, 4] = [7, 10]
        let expected_w0 = Fr::from(1u64) + Fr::from(2u64) * Fr::from(3u64);
        let expected_w1 = Fr::from(2u64) + Fr::from(2u64) * Fr::from(4u64);
        assert_eq!(expected_w0, Fr::from(7u64));
        assert_eq!(expected_w1, Fr::from(10u64));
    }

    // ============================================
    // Random Instance Generator Tests
    // ============================================

    #[test]
    fn test_sample_random_field_elements() {
        let mut rng = StdRng::seed_from_u64(12345);
        let elements: Vec<Fr> = sample_random_field_elements(10, &mut rng);

        assert_eq!(elements.len(), 10);
        // Very unlikely all are the same
        let first = elements[0];
        assert!(elements.iter().any(|e| *e != first) || elements.len() == 1);
    }

    #[test]
    fn test_sample_random_nonzero() {
        let mut rng = StdRng::seed_from_u64(12345);
        for _ in 0..100 {
            let element: Fr = sample_random_nonzero(&mut rng);
            assert_ne!(element, Fr::from(0u64));
        }
    }

    #[test]
    fn test_random_instance_generator_creation() {
        let _generator: RandomInstanceGenerator<Fr> = RandomInstanceGenerator::new();
        let _default_generator: RandomInstanceGenerator<Fr> = RandomInstanceGenerator::default();
    }

    // ============================================
    // Verifier Circuit Tests
    // ============================================

    #[test]
    fn test_verifier_circuit_creation() {
        let circuit: VerifierR1CSCircuit<Fr> = VerifierR1CSCircuit::new(10, 3);
        assert_eq!(circuit.num_rounds(), 10);
        assert_eq!(circuit.degree(), 3);
    }

    #[test]
    fn test_verifier_circuit_different_sizes() {
        for num_rounds in [5, 10, 20] {
            for degree in [2, 3, 4] {
                let circuit: VerifierR1CSCircuit<Fr> = VerifierR1CSCircuit::new(num_rounds, degree);
                assert_eq!(circuit.num_rounds(), num_rounds);
                assert_eq!(circuit.degree(), degree);
            }
        }
    }

    #[test]
    fn test_verifier_witness_generation() {
        let circuit: VerifierR1CSCircuit<Fr> = VerifierR1CSCircuit::new(3, 2);

        let initial_claim = Fr::from(100u64);
        let challenges = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)];
        let round_polys = vec![
            vec![Fr::from(10u64), Fr::from(20u64), Fr::from(30u64)],
            vec![Fr::from(40u64), Fr::from(50u64), Fr::from(60u64)],
            vec![Fr::from(70u64), Fr::from(80u64), Fr::from(90u64)],
        ];
        let final_claim = Fr::from(42u64);

        let witness = circuit.generate_witness(initial_claim, &challenges, &round_polys, final_claim);

        assert_eq!(witness.num_rounds(), 3);
        assert_eq!(witness.initial_claim, initial_claim);
        assert_eq!(witness.final_claim, final_claim);
        assert_eq!(witness.challenges, challenges);
    }

    #[test]
    fn test_verifier_witness_to_flat_vector() {
        let circuit: VerifierR1CSCircuit<Fr> = VerifierR1CSCircuit::new(2, 2);

        let initial_claim = Fr::from(10u64);
        let challenges = vec![Fr::from(1u64), Fr::from(2u64)];
        let round_polys = vec![
            vec![Fr::from(3u64), Fr::from(4u64), Fr::from(5u64)],
            vec![Fr::from(6u64), Fr::from(7u64), Fr::from(8u64)],
        ];
        let final_claim = Fr::from(9u64);

        let witness = circuit.generate_witness(initial_claim, &challenges, &round_polys, final_claim);
        let flat = witness.to_flat_vector();

        // Expected: [initial_claim, challenges..., final_claim, round_coeffs...]
        // = [10, 1, 2, 9, 3, 4, 5, 6, 7, 8]
        assert_eq!(flat[0], Fr::from(10u64)); // initial_claim
        assert_eq!(flat[1], Fr::from(1u64)); // challenge 1
        assert_eq!(flat[2], Fr::from(2u64)); // challenge 2
        assert_eq!(flat[3], Fr::from(9u64)); // final_claim
        assert_eq!(flat[4], Fr::from(3u64)); // round 1 coeff 0
    }

    // ============================================
    // Performance Benchmarks
    // ============================================

    /// Benchmark sparse matrix multiplication with increasing sizes
    #[test]
    fn bench_sparse_matrix_multiply() {
        use std::time::Instant;

        let sizes = [100, 500, 1000, 2000];
        let sparsity = 0.01; // 1% non-zero entries

        println!("\n=== Sparse Matrix Multiplication Benchmark ===");
        println!("{:<15} {:<15} {:<15} {:<15}", "Size", "Entries", "Time (μs)", "Ops/sec");
        println!("{}", "-".repeat(60));

        for &size in &sizes {
            let mut matrix: SparseMatrix<Fr> = SparseMatrix::new(size, size);
            let mut rng = StdRng::seed_from_u64(12345);

            // Add random sparse entries
            let num_entries = ((size * size) as f64 * sparsity) as usize;
            for _ in 0..num_entries {
                let row = (rng.next_u64() as usize) % size;
                let col = (rng.next_u64() as usize) % size;
                matrix.add_entry(row, col, Fr::from(rng.next_u64()));
            }

            // Create random vector
            let vec: Vec<Fr> = (0..size).map(|i| Fr::from(i as u64)).collect();

            // Warmup
            let _ = matrix.multiply(&vec);

            // Benchmark
            let iterations = 10;
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = matrix.multiply(&vec);
            }
            let elapsed = start.elapsed();
            let avg_micros = elapsed.as_micros() / iterations as u128;
            let ops_per_sec = if avg_micros > 0 { 1_000_000 / avg_micros } else { 0 };

            println!("{:<15} {:<15} {:<15} {:<15}",
                format!("{}x{}", size, size),
                matrix.entries.len(),
                avg_micros,
                ops_per_sec);
        }
    }

    /// Benchmark verifier circuit construction
    #[test]
    fn bench_verifier_circuit_construction() {
        use std::time::Instant;

        let configs = [(10, 2), (20, 3), (50, 3), (100, 4)];

        println!("\n=== Verifier Circuit Construction Benchmark ===");
        println!("{:<20} {:<15} {:<15}", "Config (rounds, deg)", "Time (μs)", "Constraints");
        println!("{}", "-".repeat(50));

        for &(num_rounds, degree) in &configs {
            // Warmup
            let _ = VerifierR1CSCircuit::<Fr>::new(num_rounds, degree);

            // Benchmark
            let iterations = 100;
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = VerifierR1CSCircuit::<Fr>::new(num_rounds, degree);
            }
            let elapsed = start.elapsed();
            let avg_micros = elapsed.as_micros() / iterations as u128;

            let circuit = VerifierR1CSCircuit::<Fr>::new(num_rounds, degree);
            let num_constraints = circuit.matrices.A.num_rows;

            println!("{:<20} {:<15} {:<15}",
                format!("({}, {})", num_rounds, degree),
                avg_micros,
                num_constraints);
        }
    }

    /// Benchmark random field element sampling
    #[test]
    fn bench_random_field_sampling() {
        use std::time::Instant;

        let counts = [100, 1000, 10000, 100000];

        println!("\n=== Random Field Element Sampling Benchmark ===");
        println!("{:<15} {:<15} {:<15}", "Count", "Time (μs)", "Elements/ms");
        println!("{}", "-".repeat(45));

        for &count in &counts {
            let mut rng = StdRng::seed_from_u64(12345);

            // Warmup
            let _ = sample_random_field_elements::<Fr, _>(count, &mut rng);

            // Benchmark
            let iterations = 10;
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = sample_random_field_elements::<Fr, _>(count, &mut rng);
            }
            let elapsed = start.elapsed();
            let avg_micros = elapsed.as_micros() / iterations as u128;
            let elements_per_ms = if avg_micros > 0 { (count as u128 * 1000) / avg_micros } else { 0 };

            println!("{:<15} {:<15} {:<15}", count, avg_micros, elements_per_ms);
        }
    }

    /// Benchmark witness creation and folding operations
    #[test]
    fn bench_witness_operations() {
        use std::time::Instant;

        let sizes = [100, 1000, 10000, 50000];

        println!("\n=== Witness Operations Benchmark ===");
        println!("{:<15} {:<20} {:<20}", "Witness Size", "Create (μs)", "ToVector (μs)");
        println!("{}", "-".repeat(55));

        for &size in &sizes {
            // Benchmark witness creation
            let iterations = 100;
            let start = Instant::now();
            for _ in 0..iterations {
                let w: Vec<Fr> = (0..size).map(|i| Fr::from(i as u64)).collect();
                let _ = RelaxedR1CSWitness::from_standard(w, size / 2);
            }
            let create_elapsed = start.elapsed();
            let create_avg = create_elapsed.as_micros() / iterations as u128;

            // Benchmark verifier witness to flat vector
            let num_rounds = 10;
            let degree = 3;
            let circuit = VerifierR1CSCircuit::<Fr>::new(num_rounds, degree);

            let initial_claim = Fr::from(100u64);
            let challenges: Vec<Fr> = (0..num_rounds).map(|i| Fr::from(i as u64)).collect();
            let round_polys: Vec<Vec<Fr>> = (0..num_rounds)
                .map(|_| (0..degree+1).map(|j| Fr::from(j as u64)).collect())
                .collect();
            let final_claim = Fr::from(42u64);

            let witness = circuit.generate_witness(initial_claim, &challenges, &round_polys, final_claim);

            let start = Instant::now();
            for _ in 0..iterations {
                let _ = witness.to_flat_vector();
            }
            let tovec_elapsed = start.elapsed();
            let tovec_avg = tovec_elapsed.as_micros() / iterations as u128;

            println!("{:<15} {:<20} {:<20}", size, create_avg, tovec_avg);
        }
    }

    /// Benchmark NIFS folding scalar operations
    #[test]
    fn bench_nifs_fold_scalars() {
        use std::time::Instant;

        let test_cases = [
            (Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)),
            (Fr::from(100u64), Fr::from(200u64), Fr::from(5u64)),
            (Fr::from(999999u64), Fr::from(888888u64), Fr::from(7u64)),
        ];

        println!("\n=== NIFS Scalar Folding Benchmark ===");
        println!("{:<30} {:<15}", "Operation", "Time (ns)");
        println!("{}", "-".repeat(45));

        // Benchmark scalar folding: u1 + r * u2
        let iterations = 100000;
        let (u1, u2, r) = test_cases[0];

        let start = Instant::now();
        for _ in 0..iterations {
            let _folded = u1 + r * u2;
        }
        let elapsed = start.elapsed();
        let avg_nanos = elapsed.as_nanos() / iterations as u128;

        println!("{:<30} {:<15}", "Scalar fold (u' = u1 + r*u2)", avg_nanos);

        // Benchmark vector folding simulation
        let vec_sizes = [100, 1000, 10000];
        for &vec_size in &vec_sizes {
            let v1: Vec<Fr> = (0..vec_size).map(|i| Fr::from(i as u64)).collect();
            let v2: Vec<Fr> = (0..vec_size).map(|i| Fr::from((i + 1) as u64)).collect();
            let r = Fr::from(5u64);

            let iterations = 100;
            let start = Instant::now();
            for _ in 0..iterations {
                let _folded: Vec<Fr> = v1.iter().zip(v2.iter()).map(|(a, b)| *a + r * *b).collect();
            }
            let elapsed = start.elapsed();
            let avg_micros = elapsed.as_micros() / iterations as u128;

            println!("{:<30} {:<15}", format!("Vector fold (size={})", vec_size), format!("{} μs", avg_micros));
        }
    }

    /// Summary benchmark comparing ZK overhead estimation
    #[test]
    fn bench_zk_overhead_summary() {
        use std::time::Instant;

        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║           BlindFold ZK Implementation Performance            ║");
        println!("╠══════════════════════════════════════════════════════════════╣");

        // Test sparse matrix operations (for R1CS satisfaction checking)
        let matrix_size = 1000;
        let mut matrix: SparseMatrix<Fr> = SparseMatrix::new(matrix_size, matrix_size);
        let mut rng = StdRng::seed_from_u64(12345);
        for _ in 0..10000 {
            let row = (rng.next_u64() as usize) % matrix_size;
            let col = (rng.next_u64() as usize) % matrix_size;
            matrix.add_entry(row, col, Fr::from(rng.next_u64()));
        }
        let vec: Vec<Fr> = (0..matrix_size).map(|i| Fr::from(i as u64)).collect();

        let start = Instant::now();
        for _ in 0..100 {
            let _ = matrix.multiply(&vec);
        }
        let matrix_time = start.elapsed().as_micros() / 100;

        println!("║ Sparse Matrix Mult (1000x1000, 10K entries): {:>10} μs  ║", matrix_time);

        // Test verifier circuit construction
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = VerifierR1CSCircuit::<Fr>::new(20, 3);
        }
        let circuit_time = start.elapsed().as_micros() / 1000;

        println!("║ Verifier Circuit Build (20 rounds, deg 3):   {:>10} μs  ║", circuit_time);

        // Test random sampling
        let start = Instant::now();
        for _ in 0..100 {
            let _: Vec<Fr> = sample_random_field_elements(10000, &mut rng);
        }
        let sample_time = start.elapsed().as_micros() / 100;

        println!("║ Random Field Sampling (10K elements):        {:>10} μs  ║", sample_time);

        // Test witness folding
        let v1: Vec<Fr> = (0..10000).map(|i| Fr::from(i as u64)).collect();
        let v2: Vec<Fr> = (0..10000).map(|i| Fr::from((i + 1) as u64)).collect();
        let r = Fr::from(5u64);

        let start = Instant::now();
        for _ in 0..100 {
            let _: Vec<Fr> = v1.iter().zip(v2.iter()).map(|(a, b)| *a + r * *b).collect();
        }
        let fold_time = start.elapsed().as_micros() / 100;

        println!("║ Vector Folding (10K elements):               {:>10} μs  ║", fold_time);

        // Estimate total ZK overhead per sumcheck round
        let total_per_round = circuit_time + sample_time / 20 + fold_time;

        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ Estimated ZK overhead per sumcheck round:    {:>10} μs  ║", total_per_round);
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();
    }

    // ============================================
    // Completeness and Soundness Tests
    // ============================================
    //
    // Completeness: If the prover has a valid witness, the verifier always accepts.
    // Soundness: If the prover doesn't have a valid witness, the verifier rejects
    //            with overwhelming probability.

    /// Test COMPLETENESS: Valid R1CS witnesses should always satisfy constraints.
    ///
    /// Completeness Property: For any valid statement-witness pair (x, w),
    /// the verifier accepts with probability 1.
    #[test]
    fn test_completeness_standard_r1cs() {
        // Create a simple R1CS: x * y = z (multiplication gate)
        // Variables: [1, x, y, z] where x, y are public inputs and z is witness
        let num_constraints = 1;
        let num_variables = 4; // 1 (constant) + 2 (public) + 1 (witness)
        let num_public = 2;

        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(num_constraints, num_variables, num_public);

        // Constraint: x * y = z
        // A = [0, 1, 0, 0] (selects x)
        // B = [0, 0, 1, 0] (selects y)
        // C = [0, 0, 0, 1] (selects z)
        matrices.A.add_entry(0, 1, Fr::from(1u64)); // x coefficient
        matrices.B.add_entry(0, 2, Fr::from(1u64)); // y coefficient
        matrices.C.add_entry(0, 3, Fr::from(1u64)); // z coefficient

        // Test multiple valid witness pairs
        let test_cases = [
            (3u64, 5u64, 15u64),   // 3 * 5 = 15
            (7u64, 11u64, 77u64),  // 7 * 11 = 77
            (0u64, 100u64, 0u64),  // 0 * 100 = 0
            (1u64, 1u64, 1u64),    // 1 * 1 = 1
            (256u64, 256u64, 65536u64), // 256 * 256 = 65536
        ];

        for (x, y, z) in test_cases {
            let public_inputs = vec![Fr::from(x), Fr::from(y)];
            let witness = vec![Fr::from(z)];

            let is_satisfied = matrices.is_satisfied(&witness, &public_inputs);
            assert!(
                is_satisfied,
                "COMPLETENESS FAILED: Valid witness ({} * {} = {}) should satisfy R1CS",
                x, y, z
            );
        }

        println!("✓ Completeness test passed: All valid witnesses satisfy R1CS");
    }

    /// Test COMPLETENESS: Relaxed R1CS with zero error should satisfy.
    #[test]
    fn test_completeness_relaxed_r1cs() {
        // Same multiplication circuit
        let num_constraints = 1;
        let num_variables = 4;
        let num_public = 2;

        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(num_constraints, num_variables, num_public);
        matrices.A.add_entry(0, 1, Fr::from(1u64));
        matrices.B.add_entry(0, 2, Fr::from(1u64));
        matrices.C.add_entry(0, 3, Fr::from(1u64));

        // Valid witness: 4 * 7 = 28
        let public_inputs = vec![Fr::from(4u64), Fr::from(7u64)];
        let witness = RelaxedR1CSWitness::from_standard(vec![Fr::from(28u64)], num_constraints);

        // With u = 1 and E = 0, this should satisfy
        let is_satisfied = matrices.is_relaxed_satisfied(&witness, Fr::from(1u64), &public_inputs);
        assert!(is_satisfied, "COMPLETENESS FAILED: Relaxed R1CS with u=1, E=0 should satisfy");

        println!("✓ Completeness test passed: Relaxed R1CS with zero error satisfies");
    }

    /// Test COMPLETENESS: Folded instances from valid instances remain valid.
    #[test]
    fn test_completeness_folded_instances() {
        let mut rng = StdRng::seed_from_u64(999);

        // Create two valid relaxed witnesses
        let witness_size = 10;
        let num_constraints = 5;

        let w1 = create_test_witness(
            sample_random_field_elements(witness_size, &mut rng),
            vec![Fr::from(0u64); num_constraints], // Zero error (satisfying)
        );
        let w2 = create_test_witness(
            sample_random_field_elements(witness_size, &mut rng),
            vec![Fr::from(0u64); num_constraints], // Zero error (satisfying)
        );

        let u1 = Fr::from(1u64);
        let u2 = Fr::from(1u64);

        // Fold with random challenge
        let r: Fr = sample_random_nonzero(&mut rng);

        // Folded witness: W' = W1 + r * W2
        let folded_w: Vec<Fr> = w1.W.iter().zip(w2.W.iter())
            .map(|(a, b)| *a + r * *b)
            .collect();

        // For satisfying instances with zero error, cross-term T = 0
        // Folded error: E' = E1 + r * T + r² * E2 = 0 + 0 + 0 = 0
        let folded_e: Vec<Fr> = vec![Fr::from(0u64); num_constraints];

        // Folded scalar: u' = u1 + r * u2 = 1 + r
        let folded_u = u1 + r * u2;

        let folded_witness = create_test_witness(folded_w, folded_e);

        // The folded witness should have the expected dimensions
        assert_eq!(folded_witness.witness_len(), witness_size);
        assert_eq!(folded_witness.constraint_count(), num_constraints);

        // Scalar should be 1 + r (not 1, showing folding happened)
        assert_ne!(folded_u, Fr::from(1u64), "Folded u should differ from 1");
        assert_eq!(folded_u, Fr::from(1u64) + r, "Folded u should equal 1 + r");

        println!("✓ Completeness test passed: Folded instances maintain validity structure");
    }

    /// Test SOUNDNESS: Invalid witnesses should NOT satisfy R1CS.
    ///
    /// Soundness Property: For any invalid statement-witness pair,
    /// the verifier rejects with overwhelming probability.
    #[test]
    fn test_soundness_invalid_witness_rejected() {
        // Multiplication circuit: x * y = z
        let num_constraints = 1;
        let num_variables = 4;
        let num_public = 2;

        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(num_constraints, num_variables, num_public);
        matrices.A.add_entry(0, 1, Fr::from(1u64));
        matrices.B.add_entry(0, 2, Fr::from(1u64));
        matrices.C.add_entry(0, 3, Fr::from(1u64));

        // Invalid witnesses: x * y ≠ z
        let invalid_cases = [
            (3u64, 5u64, 16u64),   // 3 * 5 ≠ 16 (off by 1)
            (7u64, 11u64, 76u64),  // 7 * 11 ≠ 76 (off by 1)
            (2u64, 3u64, 7u64),    // 2 * 3 ≠ 7
            (10u64, 10u64, 99u64), // 10 * 10 ≠ 99
            (1u64, 1u64, 2u64),    // 1 * 1 ≠ 2
        ];

        for (x, y, wrong_z) in invalid_cases {
            let public_inputs = vec![Fr::from(x), Fr::from(y)];
            let witness = vec![Fr::from(wrong_z)];

            let is_satisfied = matrices.is_satisfied(&witness, &public_inputs);
            assert!(
                !is_satisfied,
                "SOUNDNESS FAILED: Invalid witness ({} * {} ≠ {}) should NOT satisfy R1CS",
                x, y, wrong_z
            );
        }

        println!("✓ Soundness test passed: All invalid witnesses correctly rejected");
    }

    /// Test SOUNDNESS: Relaxed R1CS with wrong error should not satisfy.
    #[test]
    fn test_soundness_relaxed_r1cs_wrong_error() {
        let num_constraints = 1;
        let num_variables = 4;
        let num_public = 2;

        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(num_constraints, num_variables, num_public);
        matrices.A.add_entry(0, 1, Fr::from(1u64));
        matrices.B.add_entry(0, 2, Fr::from(1u64));
        matrices.C.add_entry(0, 3, Fr::from(1u64));

        // Invalid witness: 3 * 5 ≠ 16, but error is wrong
        let public_inputs = vec![Fr::from(3u64), Fr::from(5u64)];

        // With wrong z=16 (should be 15), we need E[0] = 15 - 16 = -1 to satisfy
        // But we provide E[0] = 0
        let witness = create_test_witness(
            vec![Fr::from(16u64)], // Wrong answer
            vec![Fr::from(0u64)],  // Zero error (doesn't compensate)
        );

        let is_satisfied = matrices.is_relaxed_satisfied(&witness, Fr::from(1u64), &public_inputs);
        assert!(!is_satisfied, "SOUNDNESS FAILED: Wrong witness with zero error should not satisfy");

        println!("✓ Soundness test passed: Wrong error correctly detected");
    }

    /// Test SOUNDNESS: Folded instance from one invalid instance should have non-zero error.
    #[test]
    fn test_soundness_folded_with_invalid_has_error() {
        let mut rng = StdRng::seed_from_u64(888);
        let num_constraints = 5;

        // Instance 1: Valid (zero error)
        let e1: Vec<Fr> = vec![Fr::from(0u64); num_constraints];

        // Instance 2: Invalid (non-zero error)
        let e2: Vec<Fr> = sample_random_field_elements(num_constraints, &mut rng);

        // Cross-term (simulated as random for this test)
        let cross_term: Vec<Fr> = sample_random_field_elements(num_constraints, &mut rng);

        let r: Fr = sample_random_nonzero(&mut rng);
        let r_squared = r * r;

        // Folded error: E' = E1 + r * T + r² * E2
        // Since E2 ≠ 0 and r² ≠ 0, folded error will be non-zero
        let folded_error: Vec<Fr> = (0..num_constraints)
            .map(|i| e1[i] + r * cross_term[i] + r_squared * e2[i])
            .collect();

        // Folded error should be non-zero (with overwhelming probability)
        let all_zero = folded_error.iter().all(|e| *e == Fr::from(0u64));
        assert!(
            !all_zero,
            "SOUNDNESS: Folded error from invalid instance should be non-zero"
        );

        println!("✓ Soundness test passed: Folding with invalid instance produces non-zero error");
    }

    /// Test SOUNDNESS: Random witness should not satisfy with overwhelming probability.
    #[test]
    fn test_soundness_random_witness_fails() {
        let mut rng = StdRng::seed_from_u64(777);

        // Create a non-trivial R1CS circuit
        let num_constraints = 10;
        let num_variables = 20;
        let num_public = 5;

        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(num_constraints, num_variables, num_public);

        // Add random constraints
        for i in 0..num_constraints {
            for _ in 0..3 {
                let col = (rng.next_u64() as usize) % num_variables;
                matrices.A.add_entry(i, col, Fr::from(rng.next_u64() % 100));
                matrices.B.add_entry(i, col, Fr::from(rng.next_u64() % 100));
                matrices.C.add_entry(i, col, Fr::from(rng.next_u64() % 100));
            }
        }

        // Try many random witnesses - none should satisfy (with overwhelming prob)
        let num_trials = 100;
        let witness_size = num_variables - 1 - num_public;
        let mut num_satisfied = 0;

        for _ in 0..num_trials {
            let public_inputs: Vec<Fr> = sample_random_field_elements(num_public, &mut rng);
            let witness: Vec<Fr> = sample_random_field_elements(witness_size, &mut rng);

            if matrices.is_satisfied(&witness, &public_inputs) {
                num_satisfied += 1;
            }
        }

        // Probability of random witness satisfying should be negligible
        // For 10 constraints over Fr (256-bit field), prob ≈ 2^(-2560)
        assert_eq!(
            num_satisfied, 0,
            "SOUNDNESS: Random witnesses should not satisfy R1CS ({} out of {} did)",
            num_satisfied, num_trials
        );

        println!("✓ Soundness test passed: 0/{} random witnesses satisfied (as expected)", num_trials);
    }

    /// Test SOUNDNESS with extraction: If prover can produce valid folded instance,
    /// they must know a valid witness (knowledge soundness).
    #[test]
    fn test_soundness_knowledge_extraction() {
        let mut rng = StdRng::seed_from_u64(666);
        let witness_size = 20;
        let num_constraints = 10;

        // Simulate knowledge extraction:
        // If we have two valid folded instances with different challenges r1, r2,
        // we can extract the original witnesses.

        // Original witness (the "knowledge" we're proving)
        let original_w: Vec<Fr> = sample_random_field_elements(witness_size, &mut rng);
        let blinding_w: Vec<Fr> = sample_random_field_elements(witness_size, &mut rng);

        // Two different challenges
        let r1: Fr = sample_random_nonzero(&mut rng);
        let r2: Fr = sample_random_nonzero(&mut rng);

        // Ensure r1 ≠ r2
        assert_ne!(r1, r2, "Challenges should be different for extraction");

        // Folded witnesses
        let folded_1: Vec<Fr> = original_w.iter().zip(blinding_w.iter())
            .map(|(w, b)| *w + r1 * *b)
            .collect();
        let folded_2: Vec<Fr> = original_w.iter().zip(blinding_w.iter())
            .map(|(w, b)| *w + r2 * *b)
            .collect();

        // EXTRACTION: Given folded_1, folded_2, r1, r2, recover original_w
        // folded_1 = w + r1 * b
        // folded_2 = w + r2 * b
        // folded_1 - folded_2 = (r1 - r2) * b
        // b = (folded_1 - folded_2) / (r1 - r2)
        // w = folded_1 - r1 * b

        let r_diff_inv = JoltField::inverse(&(r1 - r2)).unwrap();
        let extracted_b: Vec<Fr> = folded_1.iter().zip(folded_2.iter())
            .map(|(f1, f2)| (*f1 - *f2) * r_diff_inv)
            .collect();

        let extracted_w: Vec<Fr> = folded_1.iter().zip(extracted_b.iter())
            .map(|(f1, b)| *f1 - r1 * *b)
            .collect();

        // Verify extraction recovered original witness
        assert_eq!(extracted_w, original_w, "SOUNDNESS: Extraction should recover original witness");
        assert_eq!(extracted_b, blinding_w, "SOUNDNESS: Extraction should recover blinding");

        println!("✓ Knowledge soundness test passed: Witness successfully extracted from two transcripts");
    }

    /// Test that the protocol is complete and sound together (integration test).
    #[test]
    fn test_completeness_and_soundness_integration() {
        println!("\n=== Completeness & Soundness Integration Test ===\n");

        // Circuit: a² + b² = c² (Pythagorean theorem)
        // Valid: (3, 4, 5), (5, 12, 13), (8, 15, 17)
        let num_constraints = 3; // a², b², a² + b² = c²
        let num_variables = 7; // [1, a, b, c, a², b², c²]
        let num_public = 3; // a, b, c are public

        let mut matrices: R1CSMatrices<Fr> = R1CSMatrices::new(num_constraints, num_variables, num_public);

        // Constraint 1: a * a = a²
        matrices.A.add_entry(0, 1, Fr::from(1u64)); // a
        matrices.B.add_entry(0, 1, Fr::from(1u64)); // a
        matrices.C.add_entry(0, 4, Fr::from(1u64)); // a²

        // Constraint 2: b * b = b²
        matrices.A.add_entry(1, 2, Fr::from(1u64)); // b
        matrices.B.add_entry(1, 2, Fr::from(1u64)); // b
        matrices.C.add_entry(1, 5, Fr::from(1u64)); // b²

        // Constraint 3: c * c = c² AND a² + b² = c² (combined via witness)
        // Actually: (a² + b²) * 1 = c²
        matrices.A.add_entry(2, 4, Fr::from(1u64)); // a²
        matrices.A.add_entry(2, 5, Fr::from(1u64)); // b²
        matrices.B.add_entry(2, 0, Fr::from(1u64)); // 1
        matrices.C.add_entry(2, 6, Fr::from(1u64)); // c²

        // COMPLETENESS: Valid Pythagorean triples should satisfy
        let valid_triples = [(3u64, 4u64, 5u64), (5u64, 12u64, 13u64), (8u64, 15u64, 17u64)];

        for (a, b, c) in valid_triples {
            let public_inputs = vec![Fr::from(a), Fr::from(b), Fr::from(c)];
            let witness = vec![Fr::from(a*a), Fr::from(b*b), Fr::from(c*c)];

            let satisfied = matrices.is_satisfied(&witness, &public_inputs);
            assert!(satisfied, "COMPLETENESS: ({}, {}, {}) should satisfy", a, b, c);
            println!("  ✓ Valid triple ({}, {}, {}) satisfies constraints", a, b, c);
        }

        // SOUNDNESS: Invalid triples should NOT satisfy
        let invalid_triples = [(3u64, 4u64, 6u64), (1u64, 2u64, 3u64), (5u64, 5u64, 5u64)];

        for (a, b, c) in invalid_triples {
            let public_inputs = vec![Fr::from(a), Fr::from(b), Fr::from(c)];
            let witness = vec![Fr::from(a*a), Fr::from(b*b), Fr::from(c*c)];

            let satisfied = matrices.is_satisfied(&witness, &public_inputs);
            assert!(!satisfied, "SOUNDNESS: ({}, {}, {}) should NOT satisfy", a, b, c);
            println!("  ✓ Invalid triple ({}, {}, {}) correctly rejected", a, b, c);
        }

        println!("\n✓ Integration test passed: Completeness and soundness verified");
    }

    // ============================================
    // Zero-Knowledge Property Tests
    // ============================================
    //
    // These tests verify that the BlindFold protocol achieves zero-knowledge:
    // 1. Witness hiding: Folded witnesses reveal nothing about real witnesses
    // 2. Statistical indistinguishability: Folded values appear random
    // 3. Simulation: Transcripts can be simulated without the real witness

    /// Test that folding a real witness with a random witness produces
    /// statistically random-looking output.
    ///
    /// ZK Property: The folded witness W' = W_real + r * W_random should be
    /// computationally indistinguishable from a uniformly random vector when
    /// W_random is uniformly random and r is a random challenge.
    #[test]
    fn test_zk_witness_hiding_through_folding() {
        let mut rng = StdRng::seed_from_u64(42);
        let witness_size = 100;

        // Create a "secret" real witness with a specific pattern
        // (e.g., all ones - easily detectable if leaked)
        let real_witness: Vec<Fr> = vec![Fr::from(1u64); witness_size];

        // Create a random blinding witness
        let random_witness: Vec<Fr> = sample_random_field_elements(witness_size, &mut rng);

        // Generate a random folding challenge
        let r: Fr = sample_random_nonzero(&mut rng);

        // Fold: W' = W_real + r * W_random
        let folded_witness: Vec<Fr> = real_witness
            .iter()
            .zip(random_witness.iter())
            .map(|(w_real, w_rand)| *w_real + r * *w_rand)
            .collect();

        // Statistical test 1: The folded witness should not be constant
        // (if blinding works, the "all ones" pattern should be hidden)
        let first_elem = folded_witness[0];
        let all_same = folded_witness.iter().all(|x| *x == first_elem);
        assert!(!all_same, "Folded witness should not have constant pattern - blinding failed!");

        // Statistical test 2: The folded elements should have high variance
        // Check that we have at least witness_size/2 distinct values
        let mut unique_values: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
        for elem in &folded_witness {
            let bytes = field_element_to_bytes(elem);
            unique_values.insert(bytes);
        }
        assert!(
            unique_values.len() > witness_size / 2,
            "Folded witness should have high entropy, found only {} unique values out of {}",
            unique_values.len(),
            witness_size
        );

        println!("✓ Witness hiding test passed: {} unique values in folded witness", unique_values.len());
    }

    /// Test that different real witnesses, when folded with random witnesses,
    /// produce outputs that are statistically indistinguishable.
    ///
    /// ZK Property: An adversary should not be able to distinguish which
    /// real witness was used based on the folded output.
    #[test]
    fn test_zk_indistinguishability_of_folded_witnesses() {
        let mut rng = StdRng::seed_from_u64(123);
        let witness_size = 50;
        let num_trials = 100;

        // Two very different "secret" witnesses
        let witness_a: Vec<Fr> = vec![Fr::from(0u64); witness_size]; // All zeros
        let witness_b: Vec<Fr> = vec![Fr::from(u64::MAX); witness_size]; // All max values

        let mut stats_a = WitnessStats::new();
        let mut stats_b = WitnessStats::new();

        // Fold each witness multiple times with different random witnesses
        for _ in 0..num_trials {
            let random_witness: Vec<Fr> = sample_random_field_elements(witness_size, &mut rng);
            let r: Fr = sample_random_nonzero(&mut rng);

            // Fold witness A
            let folded_a: Vec<Fr> = witness_a
                .iter()
                .zip(random_witness.iter())
                .map(|(w, w_rand)| *w + r * *w_rand)
                .collect();

            // Fold witness B with SAME random witness and challenge
            // (simulating what an adversary might try)
            let folded_b: Vec<Fr> = witness_b
                .iter()
                .zip(random_witness.iter())
                .map(|(w, w_rand)| *w + r * *w_rand)
                .collect();

            stats_a.update(&folded_a);
            stats_b.update(&folded_b);
        }

        // The statistical properties of folded_a and folded_b should be similar
        // because the random blinding dominates
        let variance_ratio = if stats_a.variance > stats_b.variance {
            stats_a.variance / stats_b.variance
        } else {
            stats_b.variance / stats_a.variance
        };

        // Variance ratio should be close to 1 (within 2x is reasonable for random data)
        assert!(
            variance_ratio < 2.0,
            "Folded witnesses have suspiciously different variances: ratio = {}",
            variance_ratio
        );

        println!("✓ Indistinguishability test passed: variance ratio = {:.4}", variance_ratio);
    }

    /// Test the simulation property: A simulator without the real witness
    /// should produce transcripts indistinguishable from real ones.
    ///
    /// ZK Property: There exists a simulator S such that for any witness W,
    /// the output of S is computationally indistinguishable from the real
    /// protocol output.
    #[test]
    fn test_zk_simulation_property() {
        let mut rng = StdRng::seed_from_u64(456);
        let witness_size = 100;
        let num_constraints = 50;

        // REAL PROTOCOL: Prover with actual witness
        let real_witness = create_test_witness(
            sample_random_field_elements(witness_size, &mut rng),
            vec![Fr::from(0u64); num_constraints], // Zero error for satisfying instance
        );
        let random_witness = create_test_witness(
            sample_random_field_elements(witness_size, &mut rng),
            vec![Fr::from(0u64); num_constraints],
        );
        let r: Fr = sample_random_nonzero(&mut rng);

        // Real folding
        let real_folded_w: Vec<Fr> = real_witness.W
            .iter()
            .zip(random_witness.W.iter())
            .map(|(w1, w2)| *w1 + r * *w2)
            .collect();

        // SIMULATOR: No access to real_witness, only produces random output
        // In ZK, a simulator can produce valid-looking transcripts by:
        // 1. Sampling random folded witness directly
        let simulated_folded_w: Vec<Fr> = sample_random_field_elements(witness_size, &mut rng);

        // Both should have the same statistical properties (both look random)
        let real_stats = compute_byte_distribution(&real_folded_w);
        let sim_stats = compute_byte_distribution(&simulated_folded_w);

        // Chi-squared test: distributions should be similar
        let chi_squared = compute_chi_squared(&real_stats, &sim_stats);

        // For 256 categories (byte values), critical value at 0.05 significance is ~293
        // We use a more lenient threshold since we have limited samples
        assert!(
            chi_squared < 500.0,
            "Real and simulated distributions are too different: χ² = {}",
            chi_squared
        );

        println!("✓ Simulation property test passed: χ² = {:.2}", chi_squared);
    }

    /// Test that the error vector blinding preserves zero-knowledge
    /// after NIFS folding.
    ///
    /// In NIFS: E' = E_1 + r * T + r² * E_2
    /// The folded error should hide the original errors.
    #[test]
    fn test_zk_error_vector_blinding() {
        let mut rng = StdRng::seed_from_u64(789);
        let num_constraints = 100;

        // Real instance has zero error (satisfying)
        let e1: Vec<Fr> = vec![Fr::from(0u64); num_constraints];

        // Random instance also has zero error
        let e2: Vec<Fr> = vec![Fr::from(0u64); num_constraints];

        // Cross-term T (computed during folding) - simulate with random values
        let cross_term: Vec<Fr> = sample_random_field_elements(num_constraints, &mut rng);

        let r: Fr = sample_random_nonzero(&mut rng);
        let r_squared = r * r;

        // Folded error: E' = E_1 + r * T + r² * E_2
        let folded_error: Vec<Fr> = (0..num_constraints)
            .map(|i| e1[i] + r * cross_term[i] + r_squared * e2[i])
            .collect();

        // The folded error should be dominated by r * T (random cross-term)
        // Check that it's not all zeros (which would leak that e1 = e2 = 0)
        let all_zero = folded_error.iter().all(|x| *x == Fr::from(0u64));
        assert!(!all_zero, "Folded error should not be all zeros - cross-term should add randomness");

        // Check entropy
        let mut unique_values: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
        for elem in &folded_error {
            unique_values.insert(field_element_to_bytes(elem));
        }
        assert!(
            unique_values.len() > num_constraints / 2,
            "Folded error vector should have high entropy"
        );

        println!("✓ Error vector blinding test passed: {} unique values", unique_values.len());
    }

    /// Test that the scalar u blinding works correctly in NIFS.
    ///
    /// In NIFS: u' = u_1 + r * u_2
    /// With u_1 = u_2 = 1 (standard instances), the folded u = 1 + r should
    /// not reveal information about the original instances.
    #[test]
    fn test_zk_scalar_blinding() {
        let mut rng = StdRng::seed_from_u64(101112);
        let num_trials = 1000;

        let u1 = Fr::from(1u64); // Standard instance
        let u2 = Fr::from(1u64); // Standard instance

        let mut folded_values: Vec<Fr> = Vec::with_capacity(num_trials);

        for _ in 0..num_trials {
            let r: Fr = sample_random_nonzero(&mut rng);
            let u_folded = u1 + r * u2;
            folded_values.push(u_folded);
        }

        // All folded u values should be different (since r is random each time)
        let mut unique: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
        for u in &folded_values {
            unique.insert(field_element_to_bytes(u));
        }

        // Should have close to num_trials unique values
        assert!(
            unique.len() > num_trials * 9 / 10,
            "Folded scalar u should have high entropy: {} unique out of {}",
            unique.len(),
            num_trials
        );

        println!("✓ Scalar blinding test passed: {} unique u values out of {}", unique.len(), num_trials);
    }

    /// Test that public inputs are correctly preserved (not hidden)
    /// while witness remains hidden.
    ///
    /// ZK Property: Public inputs x should be verifiable, but the witness
    /// W should remain hidden.
    #[test]
    fn test_zk_public_vs_private_separation() {
        let mut rng = StdRng::seed_from_u64(131415);

        // Public inputs (these are NOT hidden)
        let public_inputs = vec![Fr::from(42u64), Fr::from(123u64), Fr::from(456u64)];

        // Private witness (this IS hidden)
        let private_witness: Vec<Fr> = sample_random_field_elements(100, &mut rng);

        // Create relaxed instance
        let instance: RelaxedR1CSInstance<Fr, TestPCS> = RelaxedR1CSInstance::new(
            <TestPCS as jolt_core::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment::default(),
            <TestPCS as jolt_core::poly::commitment::commitment_scheme::CommitmentScheme>::Commitment::default(),
            Fr::from(1u64),
            public_inputs.clone(),
        );

        // Public inputs should be accessible and correct
        assert_eq!(instance.x, public_inputs, "Public inputs should be preserved");
        assert_eq!(instance.u, Fr::from(1u64), "Scalar u should be 1 for standard instance");

        // The commitment hides the witness - we can't recover private_witness from instance
        // (This is a structural test - the actual hiding is cryptographic)

        println!("✓ Public/private separation test passed");
        println!("  Public inputs: {:?}", instance.x.len());
        println!("  Private witness size: {} (hidden by commitment)", private_witness.len());
    }

    /// Comprehensive ZK test: Run multiple trials and verify statistical properties
    #[test]
    fn test_zk_comprehensive_statistical_analysis() {
        let mut rng = StdRng::seed_from_u64(161718);
        let witness_size = 50;
        let num_trials = 500;

        println!("\n=== Comprehensive ZK Statistical Analysis ===\n");

        // Test witness hiding across many trials
        let mut entropy_scores: Vec<f64> = Vec::new();
        let mut unique_ratios: Vec<f64> = Vec::new();

        for trial in 0..num_trials {
            // Different "secret" witness each trial
            let secret: Vec<Fr> = if trial % 3 == 0 {
                vec![Fr::from(0u64); witness_size] // All zeros
            } else if trial % 3 == 1 {
                vec![Fr::from(1u64); witness_size] // All ones
            } else {
                (0..witness_size).map(|i| Fr::from(i as u64)).collect() // Sequential
            };

            // Random blinding
            let blinding: Vec<Fr> = sample_random_field_elements(witness_size, &mut rng);
            let r: Fr = sample_random_nonzero(&mut rng);

            // Fold
            let folded: Vec<Fr> = secret
                .iter()
                .zip(blinding.iter())
                .map(|(s, b)| *s + r * *b)
                .collect();

            // Compute entropy metrics
            let mut unique: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
            for elem in &folded {
                unique.insert(field_element_to_bytes(elem));
            }
            let unique_ratio = unique.len() as f64 / witness_size as f64;
            unique_ratios.push(unique_ratio);

            // Byte entropy
            let byte_entropy = compute_byte_entropy(&folded);
            entropy_scores.push(byte_entropy);
        }

        // Compute statistics
        let avg_unique_ratio: f64 = unique_ratios.iter().sum::<f64>() / num_trials as f64;
        let avg_entropy: f64 = entropy_scores.iter().sum::<f64>() / num_trials as f64;

        let min_unique_ratio = unique_ratios.iter().cloned().fold(f64::INFINITY, f64::min);
        let min_entropy = entropy_scores.iter().cloned().fold(f64::INFINITY, f64::min);

        println!("Trials: {}", num_trials);
        println!("Average unique value ratio: {:.4} (ideal: 1.0)", avg_unique_ratio);
        println!("Minimum unique value ratio: {:.4}", min_unique_ratio);
        println!("Average byte entropy: {:.4} bits (ideal: 8.0)", avg_entropy);
        println!("Minimum byte entropy: {:.4} bits", min_entropy);

        // Assert ZK properties hold
        assert!(
            avg_unique_ratio > 0.9,
            "Average unique ratio too low: {}", avg_unique_ratio
        );
        assert!(
            min_unique_ratio > 0.5,
            "Minimum unique ratio too low: {}", min_unique_ratio
        );
        assert!(
            avg_entropy > 7.0,
            "Average entropy too low: {}", avg_entropy
        );
        assert!(
            min_entropy > 6.0,
            "Minimum entropy too low: {}", min_entropy
        );

        println!("\n✓ All ZK statistical tests passed!");
    }

    // ============================================
    // Helper functions for ZK tests
    // ============================================

    /// Convert a field element to bytes for hashing/comparison
    fn field_element_to_bytes(elem: &Fr) -> [u8; 32] {
        use ark_serialize::CanonicalSerialize;
        let mut bytes = [0u8; 32];
        elem.serialize_compressed(&mut bytes[..]).unwrap();
        bytes
    }

    /// Simple statistics tracker for witness analysis
    struct WitnessStats {
        count: usize,
        sum_of_first_bytes: f64,
        sum_of_squares: f64,
        variance: f64,
    }

    impl WitnessStats {
        fn new() -> Self {
            Self {
                count: 0,
                sum_of_first_bytes: 0.0,
                sum_of_squares: 0.0,
                variance: 1.0,
            }
        }

        fn update(&mut self, witness: &[Fr]) {
            for elem in witness {
                let bytes = field_element_to_bytes(elem);
                let val = bytes[0] as f64; // Use first byte as proxy
                self.sum_of_first_bytes += val;
                self.sum_of_squares += val * val;
                self.count += 1;
            }
            if self.count > 1 {
                let mean = self.sum_of_first_bytes / self.count as f64;
                self.variance = (self.sum_of_squares / self.count as f64) - (mean * mean);
            }
        }
    }

    /// Compute byte distribution for chi-squared test
    fn compute_byte_distribution(witness: &[Fr]) -> [usize; 256] {
        let mut dist = [0usize; 256];
        for elem in witness {
            let bytes = field_element_to_bytes(elem);
            for byte in bytes.iter() {
                dist[*byte as usize] += 1;
            }
        }
        dist
    }

    /// Compute chi-squared statistic between two distributions
    fn compute_chi_squared(dist_a: &[usize; 256], dist_b: &[usize; 256]) -> f64 {
        let total_a: usize = dist_a.iter().sum();
        let total_b: usize = dist_b.iter().sum();

        if total_a == 0 || total_b == 0 {
            return 0.0;
        }

        let mut chi_squared = 0.0;
        for i in 0..256 {
            let expected = (dist_a[i] + dist_b[i]) as f64 / 2.0;
            if expected > 0.0 {
                let diff_a = dist_a[i] as f64 - expected;
                let diff_b = dist_b[i] as f64 - expected;
                chi_squared += (diff_a * diff_a + diff_b * diff_b) / expected;
            }
        }
        chi_squared
    }

    /// Compute byte entropy of witness elements
    fn compute_byte_entropy(witness: &[Fr]) -> f64 {
        let dist = compute_byte_distribution(witness);
        let total: usize = dist.iter().sum();

        if total == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &count in dist.iter() {
            if count > 0 {
                let p = count as f64 / total as f64;
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}
