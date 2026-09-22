use super::*;
use ark_bn254::{Bn254, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_std::{
    rand::{rngs::StdRng, SeedableRng},
    UniformRand,
};

fn bases() -> Vec<DoryCommitment> {
    let mut rng = StdRng::seed_from_u64(641);
    let mut result = vec![DoryCommitment::default()];
    result.extend((0..16).map(|_| {
        DoryCommitment(ArkGT(Bn254::pairing(
            G1Affine::rand(&mut rng),
            G2Affine::rand(&mut rng),
        )))
    }));
    result
}

fn reference(commitments: &[DoryCommitment], coefficients: &[Fr]) -> DoryCommitment {
    DoryCommitment(
        commitments
            .iter()
            .zip(coefficients)
            .map(|(c, s)| ArkFr(*s) * c.0)
            .fold(<ArkGT as DoryGroup>::identity(), |a, b| a + b),
    )
}

#[test]
fn gt_msm_matches_independent_exponents_at_scalar_boundaries() {
    let bases = bases();
    let mut rng = StdRng::seed_from_u64(642);
    // Exercise every scalar bit and each arkworks small-scalar group.
    let mut scalars = vec![Fr::from(0u64), Fr::from(1u64), -Fr::from(1u64)];
    for bit in 0..Fr::MODULUS_BIT_SIZE {
        let mut bytes = [0u8; 32];
        bytes[bit as usize / 8] = 1 << (bit % 8);
        let power = Fr::from_le_bytes_mod_order(&bytes);
        scalars.extend([power - Fr::from(1u64), power, power + Fr::from(1u64)]);
    }
    scalars.extend((0..1024).map(|_| Fr::rand(&mut rng)));
    for length in [0, 1, 7, 127, 128, 129, 255, 256, 257, scalars.len()] {
        let commitments: Vec<_> = (0..length).map(|i| bases[i % bases.len()]).collect();
        let coefficients = &scalars[..length];
        let expected = reference(&commitments, coefficients);
        for workers in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            assert_eq!(
                pool.install(|| DoryScheme::combine_commitments(&commitments, coefficients)),
                expected
            );
            let borrowed: Vec<_> = commitments.iter().collect();
            assert_eq!(
                pool.install(|| DoryScheme::combine_commitments(&borrowed, coefficients)),
                expected
            );
        }
    }
}

#[test]
fn gt_msm_handles_zero_identity_repeated_and_canceling_terms() {
    let bases = bases();
    for length in [128, 129, 1024] {
        let repeated = vec![bases[1]; length];
        let identity = vec![DoryCommitment::default(); length];
        for value in [Fr::from(0u64), Fr::from(1u64), -Fr::from(1u64)] {
            let scalars = vec![value; length];
            assert_eq!(
                DoryScheme::combine_commitments(&repeated, &scalars),
                reference(&repeated, &scalars)
            );
            assert_eq!(
                DoryScheme::combine_commitments(&identity, &scalars),
                DoryCommitment::default()
            );
        }
        let scalars: Vec<_> = (0..length)
            .map(|i| {
                if i % 2 == 0 {
                    Fr::from(1u64)
                } else {
                    -Fr::from(1u64)
                }
            })
            .collect();
        assert_eq!(
            DoryScheme::combine_commitments(&repeated, &scalars),
            reference(&repeated, &scalars)
        );
    }
}

#[test]
fn gt_msm_rejects_length_mismatch_on_both_paths() {
    for (commitments, scalars) in [(0, 1), (1, 0), (127, 128), (128, 127), (128, 129)] {
        let points = vec![DoryCommitment::default(); commitments];
        let coefficients = vec![Fr::from(0u64); scalars];
        assert!(std::panic::catch_unwind(|| DoryScheme::combine_commitments(
            &points,
            &coefficients
        ))
        .is_err());
    }
}

#[test]
#[ignore = "isolated commitment combination benchmark"]
fn benchmark_gt_msm() {
    let bases = bases();
    let commitments: Vec<_> = (0..65536).map(|i| bases[1 + i % 16]).collect();
    let mut rng = StdRng::seed_from_u64(643);
    let coefficients: Vec<_> = (0..commitments.len()).map(|_| Fr::rand(&mut rng)).collect();
    let mut expected = None;
    for mode in ["independent", "msm", "msm", "independent"] {
        let start = std::time::Instant::now();
        let result = if mode == "independent" {
            DoryCommitment(
                commitments
                    .par_iter()
                    .zip(&coefficients)
                    .map(|(c, s)| ArkFr(*s) * c.0)
                    .reduce(<ArkGT as DoryGroup>::identity, |a, b| a + b),
            )
        } else {
            DoryScheme::combine_commitments(&commitments, &coefficients)
        };
        let elapsed = start.elapsed().as_secs_f64();
        if let Some(old) = expected {
            assert_eq!(old, result);
        } else {
            expected = Some(result);
        }
        println!(
            "GT_MSM_BENCH mode={mode} terms={} workers={} seconds={elapsed:.9}",
            commitments.len(),
            rayon::current_num_threads()
        );
    }
}
