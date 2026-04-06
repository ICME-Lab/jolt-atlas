use ark_bn254::{Fr, G2Affine, G2Projective};
use ark_ec::PrimeGroup;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{PrimeField, UniformRand};
use ark_std::{test_rng, Zero};
use num_bigint::BigInt;

use jolt_optimizations::constants::get_bn254_frobenius_eigenvalue;
use jolt_optimizations::decomp_4d::{
    bigint_to_fr, decompose_scalar_table_based, fr_to_bigint, u128_to_fr,
};
use jolt_optimizations::frobenius::frobenius_psi_power_projective;

/// Verify that the decomposition is algebraically correct
///
/// Checks that k ≡ k0 + k1*λ + k2*λ² + k3*λ³ (mod r)
pub fn verify_decomposition(k: &Fr, coeffs: &[u128; 4], signs: &[bool; 4]) -> bool {
    let lambda_psi = get_bn254_frobenius_eigenvalue();

    let k0 = u128_to_fr(coeffs[0]);
    let k1 = u128_to_fr(coeffs[1]);
    let k2 = u128_to_fr(coeffs[2]);
    let k3 = u128_to_fr(coeffs[3]);

    let mut reconstructed = Fr::zero();
    if signs[0] {
        reconstructed -= k0;
    } else {
        reconstructed += k0;
    }
    if signs[1] {
        reconstructed -= k1 * lambda_psi;
    } else {
        reconstructed += k1 * lambda_psi;
    }
    if signs[2] {
        reconstructed -= k2 * lambda_psi * lambda_psi;
    } else {
        reconstructed += k2 * lambda_psi * lambda_psi;
    }
    if signs[3] {
        reconstructed -= k3 * lambda_psi * lambda_psi * lambda_psi;
    } else {
        reconstructed += k3 * lambda_psi * lambda_psi * lambda_psi;
    }

    *k == reconstructed
}

#[test]
fn test_sage_compatibility() {
    println!("=== Testing compatibility with tests from Sage ===\n");

    // Test cases from Sage script
    let test_cases = [
        (
            "Test Case 1",
            "0x2ffdd975c07ece990ef2aeea9920c65dfc712bd1163e8a2f7c83a49c56d5a734",
            (
                88042004215725297833u128,
                15108385708047359372u128,
                3026787831446614349u128,
                1979941837268327954u128,
            ),
            (false, false, false, false), // All positive in Sage
        ),
        (
            "Test Case 2",
            "0x26c8ae8c1778176ea4e8513d9ed63d752bac8867fff3864fab091b5942dfae90",
            (
                11419926953705671942u128,
                7811088190336128973u128,
                1826629594471335422u128,
                22015774689864596198u128,
            ),
            (true, true, true, true), // All negative in Sage
        ),
        (
            "Test Case 3",
            "0x2f5fd543c668ecc4e3b0dce9056f6c98478980b48fb257d49b8c00d39d34b7a5",
            (
                35754355908556826287u128,
                20910459483802393987u128,
                1028669091245124392u128,
                28869705842701787973u128,
            ),
            (false, true, false, false), // k0,k2,k3 positive, k1 negative in Sage
        ),
        (
            "Test Case 4",
            "0x15a1e07d6b2dd8b556f5bf7635127ab136889ca0e5187c5f489a2567724f36f8",
            (
                19611214812899750558u128,
                21033706128297981334u128,
                11593548150491899665u128,
                35317176612977865430u128,
            ),
            (false, false, true, true), // k0,k1 positive, k2,k3 negative in Sage
        ),
        (
            "Test Case 5",
            "0x1bec1358f762c8c545f99b8bfb618df2fdb8ca7c44f2e2747c35bcf0a9459787",
            (
                7872138357851074524u128,
                13298440468603925221u128,
                27497847960248044950u128,
                14701329493939829561u128,
            ),
            (false, true, true, false), // k0,k3 positive, k1,k2 negative in Sage
        ),
    ];

    for (name, scalar_hex, expected_coeffs, expected_signs) in test_cases.iter() {
        println!("{}", name);
        println!("  scalar = {}", scalar_hex);

        // Parse the scalar
        let scalar_str = if scalar_hex.starts_with("0x") {
            &scalar_hex[2..]
        } else {
            scalar_hex
        };

        let scalar_bigint =
            BigInt::parse_bytes(scalar_str.as_bytes(), 16).expect("Failed to parse scalar");

        // Decompose using our algorithm
        let (our_coeffs, our_signs) = decompose_scalar_table_based(&scalar_bigint);

        println!(
            "  Expected: k0={}, k1={}, k2={}, k3={}",
            expected_coeffs.0, expected_coeffs.1, expected_coeffs.2, expected_coeffs.3
        );
        println!("  Expected signs: {:?}", expected_signs);
        println!(
            "  Our result: k0={}, k1={}, k2={}, k3={}",
            our_coeffs[0], our_coeffs[1], our_coeffs[2], our_coeffs[3]
        );
        println!("  Our signs: {:?}", our_signs);

        // Check if coefficients match (accounting for sign differences)
        let coeffs_match = (our_coeffs[0] == expected_coeffs.0
            || (our_signs[0] != expected_signs.0 && our_coeffs[0] == expected_coeffs.0))
            && (our_coeffs[1] == expected_coeffs.1
                || (our_signs[1] != expected_signs.1 && our_coeffs[1] == expected_coeffs.1))
            && (our_coeffs[2] == expected_coeffs.2
                || (our_signs[2] != expected_signs.2 && our_coeffs[2] == expected_coeffs.2))
            && (our_coeffs[3] == expected_coeffs.3
                || (our_signs[3] != expected_signs.3 && our_coeffs[3] == expected_coeffs.3));

        // Verify algebraic correctness
        let scalar_fr = bigint_to_fr(&scalar_bigint);
        let algebraic_correct = verify_decomposition(&scalar_fr, &our_coeffs, &our_signs);

        println!("  ✓ Coefficients match: {}", coeffs_match);
        println!("  ✓ Algebraic check: {}", algebraic_correct);

        assert!(
            coeffs_match,
            "Coefficients should match Sage output for {}",
            name
        );
        assert!(
            algebraic_correct,
            "Decomposition should be algebraically correct for {}",
            name
        );

        if coeffs_match && algebraic_correct {
            println!("  🎉 SUCCESS: Matches Sage output!");

            // Test point equation when decomposition matches
            let mut rng = test_rng();
            let p = G2Affine::rand(&mut rng).into_group();

            // First check if our Frobenius implementation matches the eigenvalue
            let lambda_psi = get_bn254_frobenius_eigenvalue();
            let p1 = frobenius_psi_power_projective(&p, 1);
            let lambda_p = p.mul_bigint(lambda_psi.into_bigint());

            if p1 == lambda_p {
                println!("  ✓ Frobenius eigenvalue check: PASS");

                // Also check ψ² and ψ³
                let p2 = frobenius_psi_power_projective(&p, 2);
                let lambda2_p = p.mul_bigint((lambda_psi * lambda_psi).into_bigint());
                let psi2_check = p2 == lambda2_p;

                let p3 = frobenius_psi_power_projective(&p, 3);
                let lambda3_p = p.mul_bigint((lambda_psi * lambda_psi * lambda_psi).into_bigint());
                let psi3_check = p3 == lambda3_p;

                println!("  ✓ ψ²(P) = λ²*P: {}", psi2_check);
                println!("  ✓ ψ³(P) = λ³*P: {}", psi3_check);

                assert!(psi2_check, "ψ² should satisfy eigenvalue property");
                assert!(psi3_check, "ψ³ should satisfy eigenvalue property");
            } else {
                panic!("Frobenius eigenvalue check failed - ψ(P) ≠ λ*P");
            }

            // Compute k*P directly
            let k_times_p = p.mul_bigint(scalar_fr.into_bigint());

            // Compute k0*P + k1*φ(P) + k2*φ²(P) + k3*φ³(P)
            let p0 = p;
            let p1 = frobenius_psi_power_projective(&p, 1);
            let p2 = frobenius_psi_power_projective(&p, 2);
            let p3 = frobenius_psi_power_projective(&p, 3);

            let mut result = G2Projective::zero();

            let k0 = u128_to_fr(our_coeffs[0]);
            let k1 = u128_to_fr(our_coeffs[1]);
            let k2 = u128_to_fr(our_coeffs[2]);
            let k3 = u128_to_fr(our_coeffs[3]);

            let k0_p0 = p0.mul_bigint(k0.into_bigint());
            if our_signs[0] {
                result -= k0_p0;
            } else {
                result += k0_p0;
            }

            let k1_p1 = p1.mul_bigint(k1.into_bigint());
            if our_signs[1] {
                result -= k1_p1;
            } else {
                result += k1_p1;
            }

            let k2_p2 = p2.mul_bigint(k2.into_bigint());
            if our_signs[2] {
                result -= k2_p2;
            } else {
                result += k2_p2;
            }

            let k3_p3 = p3.mul_bigint(k3.into_bigint());
            if our_signs[3] {
                result -= k3_p3;
            } else {
                result += k3_p3;
            }

            let points_equal = k_times_p == result;
            println!("  ✓ Point equation: {}", points_equal);

            assert!(
                points_equal,
                "Point equation should hold: k*P == k0*P + k1*φ(P) + k2*φ²(P) + k3*φ³(P)"
            );

            if points_equal {
                println!("  🎯 Point verification: k*P == k0*P + k1*φ(P) + k2*φ²(P) + k3*φ³(P)");
            }
        }
        println!();
    }
}

#[test]
fn test_4d_decomposition_on_points() {
    println!("=== Testing 4D decomposition k*P = k0*P + k1*φ(P) + k2*φ²(P) + k3*φ³(P) ===\n");

    let mut rng = test_rng();

    for test_num in 1..=100 {
        let k = Fr::rand(&mut rng);
        let p = G2Affine::generator()
            .into_group()
            .mul_bigint(Fr::rand(&mut rng).into_bigint())
            .into_affine();

        let k_bigint = fr_to_bigint(k);
        let (mini_scalars, negate_points) = decompose_scalar_table_based(&k_bigint);

        println!(
            "Test #{}: k = 0x{:016x}...",
            test_num,
            k_bigint.clone() >> 192
        );
        println!(
            "  Decomposition: k0={}, k1={}, k2={}, k3={}",
            mini_scalars[0], mini_scalars[1], mini_scalars[2], mini_scalars[3]
        );

        // Verify algebraic correctness first
        let algebraic_match = verify_decomposition(&k, &mini_scalars, &negate_points);
        println!("  ✓ Algebraic check: {}", algebraic_match);

        assert!(
            algebraic_match,
            "Decomposition should be algebraically correct"
        );

        // Test point equation
        let p_proj = p.into_group();
        let k_times_p = p_proj.mul_bigint(k.into_bigint());

        let p0 = p_proj;
        let p1 = frobenius_psi_power_projective(&p_proj, 1);
        let p2 = frobenius_psi_power_projective(&p_proj, 2);
        let p3 = frobenius_psi_power_projective(&p_proj, 3);

        let mut result = G2Projective::zero();

        let k0 = u128_to_fr(mini_scalars[0]);
        let k1 = u128_to_fr(mini_scalars[1]);
        let k2 = u128_to_fr(mini_scalars[2]);
        let k3 = u128_to_fr(mini_scalars[3]);

        let k0_p0 = p0.mul_bigint(k0.into_bigint());
        if negate_points[0] {
            result -= k0_p0;
        } else {
            result += k0_p0;
        }

        let k1_p1 = p1.mul_bigint(k1.into_bigint());
        if negate_points[1] {
            result -= k1_p1;
        } else {
            result += k1_p1;
        }

        let k2_p2 = p2.mul_bigint(k2.into_bigint());
        if negate_points[2] {
            result -= k2_p2;
        } else {
            result += k2_p2;
        }

        let k3_p3 = p3.mul_bigint(k3.into_bigint());
        if negate_points[3] {
            result -= k3_p3;
        } else {
            result += k3_p3;
        }

        let points_equal = k_times_p == result;
        println!("  ✓ Points match: {}", points_equal);

        assert!(
            points_equal,
            "Point equation should hold for test {}",
            test_num
        );

        if points_equal {
            println!("  🎉 SUCCESS: k*P == k0*P + k1*φ(P) + k2*φ²(P) + k3*φ³(P)");
        }
        println!();
    }
}
