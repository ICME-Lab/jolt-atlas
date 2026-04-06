use ark_bn254::{Fq, Fq12, Fr};
use ark_ff::{One, UniformRand, Zero};
use ark_std::test_rng;
use jolt_optimizations::{
    fq12_to_multilinear_evals, get_g_mle, h_tilde_at_point, ExponentiationSteps,
};

#[test]
fn test_witness_generation_and_constraints() {
    let mut rng = test_rng();

    for test_idx in 0..100 {
        let base = Fq12::rand(&mut rng);

        let exponent = if test_idx == 0 {
            Fr::from(5u64)
        } else if test_idx == 1 {
            Fr::from(100u64)
        } else {
            Fr::rand(&mut rng)
        };

        let witness = ExponentiationSteps::new(base, exponent);

        assert!(
            witness.verify_result(),
            "Final result should match base^exponent"
        );

        assert_eq!(
            witness.rho_mles.len(),
            witness.quotient_mles.len() + 1,
            "Should have one more rho than quotients"
        );

        // Verify all MLEs have correct dimension (16 = 2^4 evaluations)
        for mle in &witness.rho_mles {
            assert_eq!(mle.len(), 16, "Rho MLEs should have 16 evaluations");
        }
        for mle in &witness.quotient_mles {
            assert_eq!(mle.len(), 16, "Quotient MLEs should have 16 evaluations");
        }

        // The constraint should be zero at all 16 cube vertices
        for cube_idx in 0..16 {
            for step in 1..=witness.num_steps() {
                assert!(
                    witness.verify_constraint_at_cube_point(step, cube_idx),
                    "Constraint failed at step {} for cube point {}",
                    step,
                    cube_idx
                );
            }
        }
    }
}

#[test]
fn test_trivial_cases() {
    let mut rng = test_rng();
    let base = Fq12::rand(&mut rng);

    // Test exponent = 0
    let witness_zero = ExponentiationSteps::new(base, Fr::from(0u64));
    assert_eq!(witness_zero.result, Fq12::one());
    assert!(witness_zero.verify_result());
    assert_eq!(witness_zero.bits.len(), 0);
    assert_eq!(witness_zero.rho_mles.len(), 1);
    assert_eq!(witness_zero.quotient_mles.len(), 0);

    // Test exponent = 1
    let witness_one = ExponentiationSteps::new(base, Fr::from(1u64));
    assert_eq!(witness_one.result, base);
    assert!(witness_one.verify_result());
    assert_eq!(witness_one.bits, vec![true]);
    assert_eq!(witness_one.rho_mles.len(), 2);
    // Test small known values to verify bit sequence
    let witness_five = ExponentiationSteps::new(base, Fr::from(5u64));
    assert_eq!(witness_five.bits, vec![true, false, true]);
    assert!(witness_five.verify_result());

    let witness_ten = ExponentiationSteps::new(base, Fr::from(10u64));
    assert_eq!(witness_ten.bits, vec![true, false, true, false]);
    assert!(witness_ten.verify_result());
}

#[test]
fn test_witness_soundness() {
    let mut rng = test_rng();

    // Test soundness: tampering with witness
    for test_idx in 0..20 {
        let base = Fq12::rand(&mut rng);
        let exponent = if test_idx == 0 {
            Fr::from(10u64)
        } else {
            Fr::rand(&mut rng)
        };

        let mut witness = ExponentiationSteps::new(base, exponent);

        // Verify original witness is valid
        assert!(witness.verify_result());
        let mut all_valid = true;
        for cube_idx in 0..16 {
            for step in 1..=witness.num_steps() {
                if !witness.verify_constraint_at_cube_point(step, cube_idx) {
                    all_valid = false;
                }
            }
        }
        assert!(all_valid, "Original witness should be valid");

        // Test 1: Tampering with ρ values
        if witness.rho_mles.len() > 1 {
            let tamper_idx = 1 + (test_idx % (witness.rho_mles.len() - 1));
            let point_idx = test_idx % 16;
            let original = witness.rho_mles[tamper_idx][point_idx];

            witness.rho_mles[tamper_idx][point_idx] += Fq::from(1u64);

            let mut soundness_broken = false;
            for cube_idx in 0..16 {
                if tamper_idx <= witness.num_steps() {
                    if !witness.verify_constraint_at_cube_point(tamper_idx, cube_idx) {
                        soundness_broken = true;
                        break;
                    }
                }
                if tamper_idx > 0 && tamper_idx + 1 <= witness.num_steps() {
                    if !witness.verify_constraint_at_cube_point(tamper_idx + 1, cube_idx) {
                        soundness_broken = true;
                        break;
                    }
                }
            }

            assert!(soundness_broken, "Tampering with ρ should break soundness");
            witness.rho_mles[tamper_idx][point_idx] = original;
        }

        // Test 2: Tampering with quotient
        if !witness.quotient_mles.is_empty() {
            let q_idx = test_idx % witness.quotient_mles.len();
            let point_idx = (test_idx * 7) % 16;
            let original = witness.quotient_mles[q_idx][point_idx];

            witness.quotient_mles[q_idx][point_idx] += Fq::from(1u64);

            let soundness_broken = !witness.verify_constraint_at_cube_point(q_idx + 1, point_idx);

            assert!(
                soundness_broken,
                "Tampering with quotient should break soundness"
            );
            witness.quotient_mles[q_idx][point_idx] = original;
        }

        // Test 3: Flipping bits
        if !witness.bits.is_empty() {
            let bit_idx = test_idx % witness.bits.len();
            witness.bits[bit_idx] = !witness.bits[bit_idx];

            let mut soundness_broken = false;
            for cube_idx in 0..16 {
                if !witness.verify_constraint_at_cube_point(bit_idx + 1, cube_idx) {
                    soundness_broken = true;
                    break;
                }
            }

            assert!(soundness_broken, "Flipping bits should break soundness");
            witness.bits[bit_idx] = !witness.bits[bit_idx];
        }

        // Test 4: Tampering with final result
        let original_result = witness.result;
        witness.result = witness.result + Fq12::one();
        assert!(
            !witness.verify_result(),
            "Modified result should fail verification"
        );
        witness.result = original_result;
    }
}

#[test]
fn test_constraint_at_random_field_element() {
    let mut rng = test_rng();

    // Create witness for a simple exponentiation
    let base = Fq12::rand(&mut rng);
    let exponent = Fr::from(10000300u64);
    let witness = ExponentiationSteps::new(base, exponent);

    let base_mle = fq12_to_multilinear_evals(&base);
    let g_mle = get_g_mle();

    for test_idx in 0..10000 {
        let z: Vec<Fq> = (0..4)
            .map(|_| {
                let mut val = Fq::rand(&mut rng);
                // Ensure it's not 0 or 1 (not on hypercube)
                while val == Fq::zero() || val == Fq::one() {
                    val = Fq::rand(&mut rng);
                }
                val
            })
            .collect();

        let step = 1 + (test_idx % witness.num_steps());
        let bit = witness.bits[step - 1];

        let h = h_tilde_at_point(
            &witness.rho_mles[step - 1],
            &witness.rho_mles[step],
            &base_mle,
            &witness.quotient_mles[step - 1],
            &g_mle,
            bit,
            &z,
        );

        // H̃(z) must be 0 at random z
        assert!(
            h.is_zero(),
            "H̃(z) must be 0 at random z (test {}, step {}). Got: {:?}",
            test_idx,
            step,
            h
        );
    }

    for step in 1..=witness.num_steps() {
        for cube_idx in 0..16 {
            assert!(
                witness.verify_constraint_at_cube_point(step, cube_idx),
                "Constraint should be zero at hypercube point {} step {}",
                cube_idx,
                step
            );
        }
    }

    println!("✓ Verified: Constraints are zero on hypercube (sanity check)");
}
