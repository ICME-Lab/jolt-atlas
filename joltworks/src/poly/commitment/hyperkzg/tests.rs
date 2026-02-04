use super::*;
use crate::{
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::PolynomialEvaluation, rlc_polynomial::build_materialized_rlc,
    },
    transcripts::{Blake2bTranscript, Transcript},
};
use ark_bn254::Bn254;
use ark_std::UniformRand;
use common::CommittedPolynomial;
use rand::Rng;
use rand_core::SeedableRng;
use std::collections::BTreeMap;

type Fr = <Bn254 as Pairing>::ScalarField;
type Challenge = <Fr as JoltField>::Challenge;

#[test]
fn test_hyperkzg_eval() {
    // Test with poly(X1, X2) = 1 + X1 + X2 + X1*X2
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let srs = HyperKZGSRS::setup(&mut rng, 3);
    let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(3);

    // poly is in eval. representation; evaluated at [(0,0), (0,1), (1,0), (1,1)]
    let poly =
        MultilinearPolynomial::from(vec![Fr::from(1), Fr::from(2), Fr::from(2), Fr::from(4)]);

    let c = HyperKZG::commit(&poly, &pk).0;

    let test_inner = |point: Vec<Challenge>| -> Result<(), ProofVerifyError> {
        // Compute the expected evaluation dynamically
        let eval = poly.evaluate(&point);
        let mut tr = Blake2bTranscript::new(b"TestEval");
        let proof = HyperKZG::open(&pk, &poly, &point, &mut tr).unwrap();
        let mut tr = Blake2bTranscript::new(b"TestEval");
        HyperKZG::verify(&proof, &vk, &mut tr, &point, &eval, &c)
    };

    let test_inner_wrong =
        |point: Vec<Challenge>, wrong_eval: Fr| -> Result<(), ProofVerifyError> {
            let mut tr = Blake2bTranscript::new(b"TestEval");
            let proof = HyperKZG::open(&pk, &poly, &point, &mut tr).unwrap();
            let mut tr = Blake2bTranscript::new(b"TestEval");
            HyperKZG::verify(&proof, &vk, &mut tr, &point, &wrong_eval, &c)
        };

    // Test various evaluation points - eval is computed dynamically
    let point = vec![Challenge::from(0u128), Challenge::from(0u128)];
    assert!(test_inner(point).is_ok());

    let point = vec![Challenge::from(0u128), Challenge::from(1u128)];
    assert!(test_inner(point).is_ok());

    let point = vec![Challenge::from(1u128), Challenge::from(1u128)];
    assert!(test_inner(point).is_ok());

    let point = vec![Challenge::from(0u128), Challenge::from(2u128)];
    assert!(test_inner(point).is_ok());

    let point = vec![Challenge::from(2u128), Challenge::from(2u128)];
    assert!(test_inner(point).is_ok());

    // Random points
    let point = vec![Challenge::from(12345u128), Challenge::from(67890u128)];
    assert!(test_inner(point).is_ok());

    // Try incorrect evaluations and expect failure
    let point = vec![Challenge::from(2u128), Challenge::from(2u128)];
    let correct_eval = poly.evaluate(&point);
    assert!(test_inner_wrong(point, correct_eval + Fr::from(1)).is_err());
}

#[test]
fn test_hyperkzg_small() {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

    // poly = [1, 2, 1, 4]
    let poly =
        MultilinearPolynomial::from(vec![Fr::from(1), Fr::from(2), Fr::from(1), Fr::from(4)]);

    // point = [4,3] using MontU128Challenge
    let point = vec![Challenge::from(4u128), Challenge::from(3u128)];

    // Compute eval dynamically
    let eval = poly.evaluate(&point);

    let srs = HyperKZGSRS::setup(&mut rng, 3);
    let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(3);

    // make a commitment
    let c = HyperKZG::commit(&poly, &pk).0;

    // prove an evaluation
    let mut tr = Blake2bTranscript::new(b"TestEval");
    let proof = HyperKZG::open(&pk, &poly, &point, &mut tr).unwrap();
    let post_c_p = tr.challenge_scalar::<Fr>();

    // verify the evaluation
    let mut verifier_transcript = Blake2bTranscript::new(b"TestEval");
    assert!(HyperKZG::verify(&proof, &vk, &mut verifier_transcript, &point, &eval, &c).is_ok());
    let post_c_v = verifier_transcript.challenge_scalar::<Fr>();

    // check if the prover transcript and verifier transcript are kept in the same state
    assert_eq!(post_c_p, post_c_v);

    let mut proof_bytes = Vec::new();
    proof.serialize_compressed(&mut proof_bytes).unwrap();
    assert_eq!(proof_bytes.len(), 368);

    // Change the proof and expect verification to fail
    let mut bad_proof = proof.clone();
    let v1 = bad_proof.v[1].clone();
    bad_proof.v[0].clone_from(&v1);
    let mut verifier_transcript2 = Blake2bTranscript::new(b"TestEval");
    assert!(HyperKZG::verify(
        &bad_proof,
        &vk,
        &mut verifier_transcript2,
        &point,
        &eval,
        &c
    )
    .is_err());
}

#[test]
fn test_hyperkzg_large() {
    // test the hyperkzg prover and verifier with random instances (derived from a seed)
    for ell in [8, 9, 10] {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(ell as u64);

        let n = 1 << ell; // n = 2^ell

        let poly_raw = (0..n)
            .map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng))
            .collect::<Vec<_>>();
        let poly = MultilinearPolynomial::from(poly_raw.clone());
        let point = (0..ell)
            .map(|_| {
                <<Bn254 as Pairing>::ScalarField as JoltField>::Challenge::from(rng.gen::<u128>())
            })
            .collect::<Vec<_>>();
        let eval = poly.evaluate(&point);

        let srs = HyperKZGSRS::setup(&mut rng, n);
        let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(n);

        // make a commitment
        let C = HyperKZG::commit(&poly, &pk).0;

        // prove an evaluation
        let mut prover_transcript = Blake2bTranscript::new(b"TestEval");
        let proof: HyperKZGProof<Bn254> =
            HyperKZG::open(&pk, &poly, &point, &mut prover_transcript).unwrap();

        // verify the evaluation
        let mut verifier_tr = Blake2bTranscript::new(b"TestEval");
        assert!(HyperKZG::verify(&proof, &vk, &mut verifier_tr, &point, &eval, &C).is_ok());

        // Change the proof and expect verification to fail
        let mut bad_proof = proof.clone();
        let v1 = bad_proof.v[1].clone();
        bad_proof.v[0].clone_from(&v1);
        let mut verifier_tr2 = Blake2bTranscript::new(b"TestEval");
        assert!(HyperKZG::verify(&bad_proof, &vk, &mut verifier_tr2, &point, &eval, &C).is_err());
    }
}

#[test]
fn test_hyperkzg_batch_dense_diff() {
    pub struct OpeningData {
        poly: MultilinearPolynomial<Fr>,
        eval: Fr,
        commitment: HyperKZGCommitment<Bn254>,
    }

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0x12344321);
    let sizes = [8, 9, 10];
    let max_size = 1 << sizes[sizes.len() - 1];
    let srs = HyperKZGSRS::setup(&mut rng, max_size);
    let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(max_size);
    let mut prover_tr = Blake2bTranscript::new(b"TestEval");

    let mut point = (0..sizes[0] - 1)
        .map(|_| <<Bn254 as Pairing>::ScalarField as JoltField>::Challenge::from(rng.gen::<u128>()))
        .collect::<Vec<_>>();
    let mut openings_data: Vec<OpeningData> = Vec::with_capacity(sizes.len());
    for ell in sizes {
        // extend point for iteration
        point.insert(
            0,
            <<Bn254 as Pairing>::ScalarField as JoltField>::Challenge::from(rng.gen::<u128>()),
        );

        let n = 1 << ell; // n = 2^ell
        let poly_raw = (0..n)
            .map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng))
            .collect::<Vec<_>>();
        let poly = MultilinearPolynomial::from(poly_raw.clone());
        let eval = poly.evaluate(&point);

        // make a commitment
        let C = HyperKZG::commit(&poly, &pk).0;
        openings_data.push(OpeningData {
            poly,
            eval,
            commitment: C,
        });
    }

    let coeffs: Vec<Fr> = prover_tr.challenge_scalar_powers(openings_data.len());
    let polynomials: Vec<&MultilinearPolynomial<Fr>> =
        openings_data.iter().map(|data| &data.poly).collect();
    let joint_poly = DensePolynomial::linear_combination(&polynomials, &coeffs);
    let joint_eval: Fr = openings_data
        .iter()
        .zip(coeffs.iter())
        .zip(sizes.iter())
        .map(|((data, coeff), size)| {
            let (r_lo, _) = point.split_at(point.len() - size);
            let lagrange_eval: Fr = r_lo.iter().map(|r| Fr::one() - r).product();
            data.eval * coeff * lagrange_eval
        })
        .sum();

    // prove an evaluation
    let proof: HyperKZGProof<Bn254> = HyperKZG::open(
        &pk,
        &MultilinearPolynomial::LargeScalars(joint_poly),
        &point,
        &mut prover_tr,
    )
    .unwrap();

    // verify the evaluation
    let mut verifier_tr = Blake2bTranscript::new(b"TestEval");
    let coeffs: Vec<Fr> = verifier_tr.challenge_scalar_powers(openings_data.len());
    let joint_commitment = HyperKZG::combine_commitments(
        &openings_data
            .iter()
            .map(|data| &data.commitment)
            .collect::<Vec<&HyperKZGCommitment<Bn254>>>(),
        &coeffs,
    );
    assert!(HyperKZG::verify(
        &proof,
        &vk,
        &mut verifier_tr,
        &point,
        &joint_eval,
        &joint_commitment,
    )
    .is_ok());
}

#[test]
fn test_hyperkzg_batch_dense_oh() {
    let log_N = 8;
    let N = 1 << log_N;
    let log_K = 4;
    let log_T = 6;
    let K: usize = 1 << log_K;
    let T: usize = 1 << log_T;
    let oh_total_size = K * T;
    let oh_num_vars = log_K + log_T;

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0x7654);

    let srs = HyperKZGSRS::setup(&mut rng, oh_total_size);
    let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(oh_total_size);
    let mut prover_tr = Blake2bTranscript::new(b"TestEval");

    let dense_poly_raw = (0..N)
        .map(|_| <Bn254 as Pairing>::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>();
    let dense_poly = MultilinearPolynomial::from(dense_poly_raw.clone());
    let mut point = (0..log_N)
        .map(|_| <<Bn254 as Pairing>::ScalarField as JoltField>::Challenge::from(rng.gen::<u128>()))
        .collect::<Vec<_>>();
    let dense_poly_eval = dense_poly.evaluate(&point);
    let dense_commitment = HyperKZG::commit(&dense_poly, &pk).0;

    // Generate random nonzero indices (simulating random addresses per timestep)
    let nonzero_indices: Vec<Option<u16>> = (0..T)
        .map(|_| Some((rng.gen::<u64>() % K as u64) as u16))
        .collect();

    // Create OneHotPolynomial
    let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices.clone(), K);

    (0..oh_num_vars - log_N).for_each(|_| {
        point.insert(
            0,
            <<Bn254 as Pairing>::ScalarField as JoltField>::Challenge::from(rng.gen::<u128>()),
        );
    });
    let one_hot_eval = one_hot_poly.evaluate(&point);
    let one_hot_commitment =
        HyperKZG::commit(&MultilinearPolynomial::OneHot(one_hot_poly.clone()), &pk).0;

    let gamma_powers = prover_tr.challenge_scalar_powers::<Fr>(2);

    // Build RLCPolynomial manually
    let polynomial_map = BTreeMap::from([
        (CommittedPolynomial::DivNodeQuotient(0), dense_poly.clone()),
        (
            CommittedPolynomial::DivNodeQuotient(1),
            MultilinearPolynomial::OneHot(one_hot_poly.clone()),
        ),
    ]);
    let rlc = build_materialized_rlc(&gamma_powers, &polynomial_map);
    // Compute RLC evaluation
    let (r_lo, _) = point.split_at(point.len() - log_N);
    let lagrange_eval: Fr = r_lo.iter().map(|r| Fr::one() - r).product();
    let rlc_eval = dense_poly_eval * lagrange_eval + gamma_powers[1] * one_hot_eval;

    // prove an evaluation
    let proof: HyperKZGProof<Bn254> = HyperKZG::open(&pk, &rlc, &point, &mut prover_tr).unwrap();

    // verify the evaluation
    let mut verifier_tr = Blake2bTranscript::new(b"TestEval");
    let gamma_powers = verifier_tr.challenge_scalar_powers::<Fr>(2);
    let rlc_commitment =
        HyperKZG::combine_commitments(&[&dense_commitment, &one_hot_commitment], &gamma_powers);
    HyperKZG::verify(
        &proof,
        &vk,
        &mut verifier_tr,
        &point,
        &rlc_eval,
        &rlc_commitment,
    )
    .unwrap();
}

#[test]
fn test_hyperkzg_batch_3dense_3oh() {
    struct PolyData {
        poly: MultilinearPolynomial<Fr>,
        num_vars: usize,
        eval: Fr,
        commitment: HyperKZGCommitment<Bn254>,
    }

    struct OneHotPolyData {
        poly: OneHotPolynomial<Fr>,
        num_vars: usize,
        eval: Fr,
        commitment: HyperKZGCommitment<Bn254>,
    }

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0x7654);

    // Define dense polynomial sizes (log_N values)
    let dense_configs = [6, 7, 8]; // 64, 128, 256 elements

    // Define OneHot polynomial parameters (log_K, log_T)
    let oh_configs = [(3, 5), (4, 6), (5, 7)]; // (K, T) pairs

    // Calculate max size needed
    let max_dense_size = 1 << dense_configs.iter().max().unwrap();
    let max_oh_size = oh_configs
        .iter()
        .map(|(log_k, log_t)| (1 << log_k) * (1 << log_t))
        .max()
        .unwrap();
    let max_size = max_dense_size.max(max_oh_size);
    let max_num_vars = oh_configs
        .iter()
        .map(|(log_k, log_t)| log_k + log_t)
        .max()
        .unwrap();

    // Setup
    let srs = HyperKZGSRS::setup(&mut rng, max_size);
    let (pk, vk): (HyperKZGProverKey<Bn254>, HyperKZGVerifierKey<Bn254>) = srs.trim(max_size);
    let mut prover_tr = Blake2bTranscript::new(b"TestEval");

    // Generate evaluation point based on largest polynomial
    let point: Vec<Challenge> = (0..max_num_vars)
        .map(|_| Challenge::from(rng.gen::<u128>()))
        .collect();

    // Create dense polynomials
    let dense_polys: Vec<PolyData> = dense_configs
        .iter()
        .map(|&log_n| {
            let n = 1 << log_n;
            let raw: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
            let poly = MultilinearPolynomial::from(raw);
            let eval = poly.evaluate(&point[point.len() - log_n..]);
            let commitment = HyperKZG::commit(&poly, &pk).0;
            PolyData {
                poly,
                num_vars: log_n,
                eval,
                commitment,
            }
        })
        .collect();

    // Create OneHot polynomials
    let oh_polys: Vec<OneHotPolyData> = oh_configs
        .iter()
        .map(|&(log_k, log_t)| {
            let k = 1 << log_k;
            let t = 1 << log_t;
            let num_vars = log_k + log_t;
            let nonzero_indices: Vec<Option<u16>> = (0..t)
                .map(|_| Some((rng.gen::<u64>() % k as u64) as u16))
                .collect();
            let one_hot = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, k);
            let eval = one_hot.evaluate(&point[point.len() - num_vars..]);
            let poly_wrapped = MultilinearPolynomial::OneHot(one_hot.clone());
            let commitment = HyperKZG::commit(&poly_wrapped, &pk).0;
            OneHotPolyData {
                poly: one_hot,
                num_vars,
                eval,
                commitment,
            }
        })
        .collect();

    // Combine all polynomials for RLC
    let mut all_polys: Vec<MultilinearPolynomial<Fr>> =
        dense_polys.iter().map(|data| data.poly.clone()).collect();
    all_polys.extend(
        oh_polys
            .iter()
            .map(|data| MultilinearPolynomial::OneHot(data.poly.clone())),
    );
    let mut all_polys: Vec<(CommittedPolynomial, MultilinearPolynomial<Fr>)> = dense_polys
        .iter()
        .enumerate()
        .map(|(i, data)| (CommittedPolynomial::DivNodeQuotient(i), data.poly.clone()))
        .collect();
    all_polys.extend(oh_polys.iter().enumerate().map(|(i, data)| {
        (
            CommittedPolynomial::RsqrtNodeInv(i), // make sure RsqrtNodeInv comes after DivNodeQuotient in Ordering
            MultilinearPolynomial::OneHot(data.poly.clone()),
        )
    }));
    let num_polys = all_polys.len();

    // Get RLC powers
    let gamma_powers = prover_tr.challenge_scalar_powers::<Fr>(num_polys);

    // Build RLC
    let polynomial_map = BTreeMap::from_iter(all_polys);
    let rlc = build_materialized_rlc(&gamma_powers, &polynomial_map);

    // Compute RLC evaluation with Lagrange corrections
    let mut rlc_eval = Fr::zero();

    // Dense polynomials
    for (i, data) in dense_polys.iter().enumerate() {
        let (r_lo, _) = point.split_at(point.len() - data.num_vars);
        let lagrange_eval: Fr = r_lo.iter().map(|r| Fr::one() - r).product();
        rlc_eval += data.eval * lagrange_eval * gamma_powers[i];
    }

    // OneHot polynomials
    for (i, data) in oh_polys.iter().enumerate() {
        let idx = i + dense_polys.len();
        let (r_lo, _) = point.split_at(point.len() - data.num_vars);
        let lagrange_eval: Fr = r_lo.iter().map(|r| Fr::one() - r).product();
        rlc_eval += data.eval * lagrange_eval * gamma_powers[idx];
    }

    // Prove
    let proof: HyperKZGProof<Bn254> = HyperKZG::open(&pk, &rlc, &point, &mut prover_tr).unwrap();

    // Verify
    let mut verifier_tr = Blake2bTranscript::new(b"TestEval");
    let gamma_powers = verifier_tr.challenge_scalar_powers::<Fr>(num_polys);
    let all_commitments: Vec<&HyperKZGCommitment<Bn254>> = dense_polys
        .iter()
        .map(|data| &data.commitment)
        .chain(oh_polys.iter().map(|data| &data.commitment))
        .collect();
    let rlc_commitment = HyperKZG::combine_commitments(&all_commitments, &gamma_powers);

    HyperKZG::verify(
        &proof,
        &vk,
        &mut verifier_tr,
        &point,
        &rlc_eval,
        &rlc_commitment,
    )
    .unwrap();
}

#[test]
fn test_hyperkzg_srs_file_roundtrip() {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);
    let max_degree = 1 << 8; // 256 elements

    // Create SRS
    let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, max_degree);
    assert_eq!(srs.max_degree(), max_degree);

    // Save to temp file
    let temp_dir = std::env::temp_dir();
    let srs_path = temp_dir.join("test_hyperkzg_srs.bin");
    srs.save_to_file(&srs_path).unwrap();

    // Load from file
    let loaded_srs = HyperKZGSRS::<Bn254>::load_from_file(&srs_path).unwrap();
    assert_eq!(loaded_srs.max_degree(), max_degree);

    // Verify they produce the same keys
    let (pk1, _vk1) = srs.trim(max_degree);
    let (pk2, vk2) = loaded_srs.trim(max_degree);

    // Create a test polynomial and verify both setups produce the same commitment
    let poly = MultilinearPolynomial::from(
        (0..max_degree)
            .map(|i| Fr::from(i as u64))
            .collect::<Vec<_>>(),
    );

    let c1 = HyperKZG::commit(&poly, &pk1).0;
    let c2 = HyperKZG::commit(&poly, &pk2).0;
    assert_eq!(c1, c2);

    // Verify proof works with loaded SRS
    let point: Vec<Challenge> = (0..8).map(|i| Challenge::from((i * 7) as u128)).collect();
    let eval = poly.evaluate(&point);

    let mut tr1 = Blake2bTranscript::new(b"TestSRS");
    let proof = HyperKZG::open(&pk2, &poly, &point, &mut tr1).unwrap();

    let mut tr2 = Blake2bTranscript::new(b"TestSRS");
    assert!(HyperKZG::verify(&proof, &vk2, &mut tr2, &point, &eval, &c2).is_ok());

    // Cleanup
    std::fs::remove_file(&srs_path).ok();
}

#[test]
fn test_hyperkzg_one_hot_commit() {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);

    // Test parameters: K (address space) and T (trace length)
    // Using small values for fast testing
    let K: usize = 16; // 2^4 addresses
    let T: usize = 64; // 2^6 timesteps
    let total_size = K * T; // 1024 elements

    // Setup SRS large enough for K * T
    let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, total_size);
    let (pk, _vk) = srs.trim(total_size);

    // Generate random nonzero indices (simulating random addresses per timestep)
    let nonzero_indices: Vec<Option<u16>> = (0..T)
        .map(|_| Some((rng.gen::<u64>() % K as u64) as u16))
        .collect();

    // Create OneHotPolynomial
    let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices.clone(), K);

    // Commit using sparse method
    let sparse_commitment = HyperKZG::<Bn254>::commit_one_hot(&pk, &one_hot_poly).unwrap();

    // Create equivalent dense polynomial for comparison
    let mut dense_coeffs = vec![Fr::zero(); total_size];
    for (t, k) in nonzero_indices.iter().enumerate() {
        if let Some(k) = k {
            let idx = *k as usize * T + t;
            dense_coeffs[idx] = Fr::one();
        }
    }
    let dense_poly = MultilinearPolynomial::from(dense_coeffs);

    // Commit using dense method
    let dense_commitment = HyperKZG::<Bn254>::commit(&dense_poly, &pk).0;

    // Both commitments should be equal
    assert_eq!(
        sparse_commitment.0, dense_commitment.0,
        "Sparse and dense OneHot commitments should match"
    );
}

#[test]
fn test_hyperkzg_one_hot_commit_2() {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);

    // Test parameters: K (address space) and T (trace length)
    // Using small values for fast testing
    let K: usize = 16; // 2^4 addresses
    let T: usize = 64; // 2^6 timesteps
    let total_size = K * T; // 1024 elements

    // Setup SRS large enough for K * T
    let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, total_size);
    let (pk, _vk) = srs.trim(total_size);

    // Generate random nonzero indices (simulating random addresses per timestep)
    let nonzero_indices: Vec<Option<u16>> = (0..T)
        .map(|_| Some((rng.gen::<u64>() % K as u64) as u16))
        .collect();

    // Create OneHotPolynomial
    let one_hot_poly = MultilinearPolynomial::OneHot(OneHotPolynomial::<Fr>::from_indices(
        nonzero_indices.clone(),
        K,
    ));

    // Commit using sparse method
    let sparse_commitment = HyperKZG::<Bn254>::commit(&one_hot_poly, &pk).0;

    // Create equivalent dense polynomial for comparison
    let mut dense_coeffs = vec![Fr::zero(); total_size];
    for (t, k) in nonzero_indices.iter().enumerate() {
        if let Some(k) = k {
            let idx = *k as usize * T + t;
            dense_coeffs[idx] = Fr::one();
        }
    }
    let dense_poly = MultilinearPolynomial::from(dense_coeffs);

    // Commit using dense method
    let dense_commitment = HyperKZG::<Bn254>::commit(&dense_poly, &pk).0;

    // Both commitments should be equal
    assert_eq!(
        sparse_commitment.0, dense_commitment.0,
        "Sparse and dense OneHot commitments should match"
    );
}

#[test]
fn test_hyperkzg_one_hot_commit_with_none_indices() {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(123);

    let K: usize = 8;
    let T: usize = 32;
    let total_size = K * T;

    let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, total_size);
    let (pk, _vk) = srs.trim(total_size);

    // Generate indices with some None values (simulating no memory access at some timesteps)
    let nonzero_indices: Vec<Option<u16>> = (0..T)
        .map(|i| {
            if i % 3 == 0 {
                None // No access at every 3rd timestep
            } else {
                Some((rng.gen::<u64>() % K as u64) as u16)
            }
        })
        .collect();

    let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices.clone(), K);

    // Commit using sparse method
    let sparse_commitment = HyperKZG::<Bn254>::commit_one_hot(&pk, &one_hot_poly).unwrap();

    // Create equivalent dense polynomial
    let mut dense_coeffs = vec![Fr::zero(); total_size];
    for (t, k) in nonzero_indices.iter().enumerate() {
        if let Some(k) = k {
            let idx = *k as usize * T + t;
            dense_coeffs[idx] = Fr::one();
        }
    }
    let dense_poly = MultilinearPolynomial::from(dense_coeffs);

    let dense_commitment = HyperKZG::<Bn254>::commit(&dense_poly, &pk).0;

    assert_eq!(
        sparse_commitment.0, dense_commitment.0,
        "Sparse and dense commitments should match even with None indices"
    );
}

#[test]
fn test_hyperkzg_batch_one_hot_commit() {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(999);

    let K: usize = 16;
    let T: usize = 64;
    let total_size = K * T;
    let num_polys = 5;

    let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, total_size);
    let (pk, _vk) = srs.trim(total_size);

    // Generate multiple OneHotPolynomials
    let polys: Vec<OneHotPolynomial<Fr>> = (0..num_polys)
        .map(|_| {
            let nonzero_indices: Vec<Option<u16>> = (0..T)
                .map(|_| Some((rng.gen::<u64>() % K as u64) as u16))
                .collect();
            OneHotPolynomial::from_indices(nonzero_indices, K)
        })
        .collect();

    // Batch commit
    let batch_commitments = HyperKZG::<Bn254>::batch_commit_one_hot(&pk, &polys).unwrap();

    // Individual commits
    let individual_commitments: Vec<_> = polys
        .iter()
        .map(|p| HyperKZG::<Bn254>::commit_one_hot(&pk, p).unwrap())
        .collect();

    // All should match
    for (batch, individual) in batch_commitments.iter().zip(individual_commitments.iter()) {
        assert_eq!(
            batch.0, individual.0,
            "Batch and individual commits should match"
        );
    }
}

#[test]
fn test_hyperkzg_one_hot_empty() {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(456);

    let K: usize = 8;
    let T: usize = 16;
    let total_size = K * T;

    let srs = HyperKZGSRS::<Bn254>::setup(&mut rng, total_size);
    let (pk, _vk) = srs.trim(total_size);

    // All None indices (completely zero polynomial)
    let nonzero_indices: Vec<Option<u16>> = vec![None; T];

    let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, K);

    let commitment = HyperKZG::<Bn254>::commit_one_hot(&pk, &one_hot_poly).unwrap();

    // Commitment to zero polynomial should be the identity element
    assert!(
        commitment.0.is_zero(),
        "Commitment to all-None OneHot should be zero"
    );
}
