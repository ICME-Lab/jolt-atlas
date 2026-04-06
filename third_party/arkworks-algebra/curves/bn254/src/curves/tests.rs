use ark_algebra_test_templates::*;
use ark_ff::fields::Field;

use crate::{Bn254, G1Projective, G2Projective};

test_group!(g1; G1Projective; sw);
test_group!(g2; G2Projective; sw);
test_group!(pairing_output; ark_ec::pairing::PairingOutput<Bn254>; msm);
test_pairing!(pairing; crate::Bn254);
test_group!(g1_glv; G1Projective; glv);
test_group!(g2_glv; G2Projective; glv);

// Test compressed pairing.
#[cfg(test)]
mod test {
    use ark_ec::pairing::{CompressedPairing, Pairing};
    use ark_ff::{AdditiveGroup, CyclotomicMultSubgroup, Field, UniformRand};
    use ark_std::{test_rng, vec::Vec};

    use crate::{
        compressible_fq12_to_fq12, fq12_to_compressible_fq12, torus_compress_fq6,
        torus_compress_psi_6_pow_to_two_fq2, torus_decompress_fq6, Bn254, CompressibleFq12, Fq12,
        Fq2, Fq6, G1Projective, G2Projective,
    };

    const PSI_6: [u64; 32] = [
        0x72ab9d4cf9110b20,
        0x973cd2a37a1b4236,
        0x1ba98b1bc336a6d5,
        0x2dfcbbaca5d60846,
        0xd5439cc568e9448a,
        0xe1db0edd8549f5a5,
        0x6f93e1753d4af731,
        0x49c8a9c36c6fee88,
        0x15f3f80815d4b9fa,
        0x1ba650a506fe3ed0,
        0xaefe7d8fff4f3dff,
        0x6ac0657132447def,
        0xf1f3ac284dd0152c,
        0x37018c9976be9e50,
        0x22c60855715fb2d0,
        0x208847a342f135b0,
        0x3ee1e2cb94f42a1d,
        0x3c4a7ccbcfffdab3,
        0xad8c7c82ecaf3f33,
        0x6ca3fffc37db0759,
        0x37cd075497b1b668,
        0xea10af7b063fb6d2,
        0x4c05f030809f6505,
        0x2a2ce31f8f0d0104,
        0x8cabad13097c3c76,
        0x4d2a25ff065eb903,
        0xa3df56ffaff05a83,
        0x2d5b0867d19408a1,
        0x223eaadf9b381c71,
        0xe8809c2c51000097,
        0x0e97a02f2739dcf3,
        0x1b59e685800b,
    ];

    #[test]
    fn test_compression() {
        let q2: [u64; 8] = [
            0x3b5458a2275d69b1,
            0xa602072d09eac101,
            0x4a50189c6d96cadc,
            0x04689e957a1242c8,
            0x26edfa5c34c6b38d,
            0xb00b855116375606,
            0x599a6f7c0348d21c,
            0x925c4b8763cbf9c,
        ];

        let num_trials = 5;
        let mut rng = test_rng();

        // Step by step testing of the compression algorithm described in
        // https://eprint.iacr.org/2007/429.pdf Proposition 1
        for _ in 0..num_trials {
            let fq12_ele = CompressibleFq12::rand(&mut rng);
            let c1 = fq12_ele.torus_compress_base_order_minus_one_pow();
            let c1_pow = -c1.pow(q2);
            let compressed_prod = CompressibleFq12::mul_torus_compressed_elements(c1_pow, c1);

            assert_eq!(
                CompressibleFq12::torus_decompress(compressed_prod),
                fq12_ele.pow(PSI_6)
            );
            let compressed_fq6 = torus_compress_fq6(compressed_prod);
            let decompressed_fq6 = torus_decompress_fq6(compressed_fq6);
            assert_eq!(compressed_prod, decompressed_fq6);
        }

        // Test that the compression does not work with Fq12 as expected because the generator of
        // the quadratic extension inside Fq12 is not of degree 2 over Fq2.
        for _ in 0..num_trials {
            let fq12_ele = Fq12::rand(&mut rng);
            let c1 = fq12_ele.torus_compress_base_order_minus_one_pow();
            let c1_pow = -c1.pow(q2);
            let compressed_prod = Fq12::mul_torus_compressed_elements(c1_pow, c1);

            assert_ne!(Fq12::torus_decompress(compressed_prod), fq12_ele.pow(PSI_6));
        }

        // Test compression of an CompressibleFq12 element e2e.
        for _ in 0..num_trials {
            let compressible_fq12 = CompressibleFq12::rand(&mut rng);
            let compressed_fq12 = torus_compress_psi_6_pow_to_two_fq2(compressible_fq12);
            let decompressed_fq12 = compressed_fq12.decompress();
            assert_eq!(compressible_fq12.pow(PSI_6), decompressed_fq12);
        }
    }

    #[test]
    fn test_compressible_fq12_to_fq12_conversion() {
        let num_trials = 100;
        let mut rng = test_rng();

        let x = Fq6 {
            c0: Fq2::ZERO,
            c1: Fq2::ONE,
            c2: Fq2::ZERO,
        };

        for _ in 0..num_trials {
            // a = c0 + c1 * residue^(1/2)
            let a = CompressibleFq12::rand(&mut rng);
            // a_prime = c0 + c1 * x * residue^(1/6), where x = residue^(1/3)
            let a_prime = compressible_fq12_to_fq12(a);
            assert_eq!(a_prime.c1, x * a.c1);
        }

        for _ in 0..num_trials {
            // a = c0 + c1 * residue^(1/6)
            let a = Fq12::rand(&mut rng);
            // a_prime = c0 + c1 / x * residue^(1/2), where x = residue^(1/3)
            let a_prime = fq12_to_compressible_fq12(a);
            assert_eq!(a_prime.c1, x.inverse().unwrap() * a.c1);
        }

        // Check inverses
        for _ in 0..num_trials {
            let fq12_ele = Fq12::rand(&mut rng);
            let compressible_fq12 = fq12_to_compressible_fq12(fq12_ele);
            let fq12_back = compressible_fq12_to_fq12(compressible_fq12);
            assert_eq!(fq12_ele, fq12_back);

            let compressible_fq12 = CompressibleFq12::rand(&mut rng);
            let fq12_ele = compressible_fq12_to_fq12(compressible_fq12);
            let compressible_fq12_back = fq12_to_compressible_fq12(fq12_ele);
            assert_eq!(compressible_fq12, compressible_fq12_back);
        }

        // Homomorphic property
        for _ in 0..num_trials {
            let a = CompressibleFq12::rand(&mut rng);
            let b = CompressibleFq12::rand(&mut rng);
            let c = a * b;
            let a_prime = compressible_fq12_to_fq12(a);
            let b_prime = compressible_fq12_to_fq12(b);
            let c_prime = compressible_fq12_to_fq12(c);
            assert_eq!(a_prime * b_prime, c_prime);

            let a = Fq12::rand(&mut rng);
            let b = Fq12::rand(&mut rng);
            let c = a * b;
            let a_prime = fq12_to_compressible_fq12(a);
            let b_prime = fq12_to_compressible_fq12(b);
            let c_prime = fq12_to_compressible_fq12(c);
            assert_eq!(a_prime * b_prime, c_prime);
        }

        // Check that converting an Fq12 element to a CompressibleFq12 element before compressing
        // its psi_6 power works, where decompressing is simply converting back to an Fq12 element.
        for _ in 0..num_trials {
            let a = Fq12::rand(&mut rng);
            let a_prime = fq12_to_compressible_fq12(a);
            let compressed = torus_compress_psi_6_pow_to_two_fq2(a_prime);

            let decompressed = compressed.decompress();
            let decompressed_fq12 = compressible_fq12_to_fq12(decompressed);
            assert_eq!(a.pow(PSI_6), decompressed_fq12);
        }
    }

    #[test]
    fn test_compressed_pairing() {
        let num_trials = 10;
        let mut rng = test_rng();

        // Test pairing e2e.
        for _ in 0..num_trials {
            let g1 = G1Projective::rand(&mut rng);
            let g2 = G2Projective::rand(&mut rng);
            let pairing_value = Bn254::pairing(g1, g2).0;
            let compressed_pairing_value = Bn254::compressed_pairing(g1, g2);
            assert_eq!(pairing_value, compressed_pairing_value.decompress_to_fq12());
        }

        // Test multi-pairing e2e.
        let num_pairs = 10;
        for _ in 0..num_trials {
            let g1 = (0..num_pairs)
                .map(|_| G1Projective::rand(&mut rng))
                .collect::<Vec<_>>();
            let g2 = (0..num_pairs)
                .map(|_| G2Projective::rand(&mut rng))
                .collect::<Vec<_>>();
            let pairing_value = Bn254::multi_pairing(g1.iter().cloned(), g2.iter().cloned()).0;
            let compressed_pairing_value =
                Bn254::compressed_multi_pairing(g1.iter().cloned(), g2.iter().cloned());
            assert_eq!(pairing_value, compressed_pairing_value.decompress_to_fq12());
        }

        // This is more for documentation purposes. For the compressed pairing calculation, we swap
        // the order of computing exponentiation by \Psi_6(q^2) and \Phi_6(q^2) (see documentation
        // for their definitions). The Miller loop output is not in the cyclotomic subgroup of the
        // right order where the optimization (defined in the CyclotomicMultSubgroup trait) can be
        // applied.
        for _ in 0..num_trials {
            let g1 = G1Projective::rand(&mut rng);
            let g2 = G2Projective::rand(&mut rng);

            let miller_loop_output = Bn254::multi_miller_loop([g1], [g2]).0;
            assert_ne!(
                miller_loop_output.cyclotomic_inverse(),
                miller_loop_output.inverse()
            );
            assert_ne!(
                miller_loop_output.cyclotomic_exp(PSI_6),
                miller_loop_output.pow(PSI_6)
            );
        }
    }
}
