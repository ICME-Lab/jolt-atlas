use ark_bn254::{G2Affine, G2Projective};
use ark_ec::AffineRepr;
use ark_ff::UniformRand;
use ark_std::{test_rng, Zero};

use jolt_optimizations::frobenius_psi_power_projective;

#[test]
fn test_frobenius_identity() {
    let mut rng = test_rng();
    let p = G2Affine::rand(&mut rng).into_group();

    // ψ^4 should be the identity
    let psi4_p = frobenius_psi_power_projective(&p, 4);
    assert_eq!(p, psi4_p);

    // ψ^0 should also be the identity
    let psi0_p = frobenius_psi_power_projective(&p, 0);
    assert_eq!(p, psi0_p);
}

#[test]
fn test_frobenius_composition() {
    let mut rng = test_rng();
    let p = G2Affine::rand(&mut rng).into_group();

    // ψ^2 = ψ ∘ ψ
    let psi1_p = frobenius_psi_power_projective(&p, 1);
    let psi1_psi1_p = frobenius_psi_power_projective(&psi1_p, 1);
    let psi2_p = frobenius_psi_power_projective(&p, 2);

    assert_eq!(psi1_psi1_p, psi2_p);
}

#[test]
fn test_frobenius_zero_point() {
    let zero = G2Projective::zero();

    for k in 0..8 {
        let result = frobenius_psi_power_projective(&zero, k);
        assert_eq!(result, zero);
    }
}
