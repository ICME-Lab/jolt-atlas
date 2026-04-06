use ark_algebra_bench_templates::{bench, criterion_main, field_common, paste, prime_field, sqrt};
// Import BN254 types
use ark_test_curves::bn254::{Fq, Fr, G1Projective as G1};

// Instantiate benchmarks for BN254
bench!(Name = "BN254", Group = G1, ScalarField = Fr, BaseField = Fq,);
