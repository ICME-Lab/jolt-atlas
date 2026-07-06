//! Dory polynomial commitment scheme adapter.
//!
//! Wraps the external `dory-pcs` crate so it satisfies joltworks'
//! [`CommitmentScheme`](super::commitment_scheme::CommitmentScheme) trait,
//! letting the ONNX prover/verifier select Dory instead of HyperKZG.
//!
//! Dory is asymptotically faster than HyperKZG for large instances
//! (`num_vars` past ~30), which is exactly the regime slms models hit.
//!
//! # Combine-compatible commitments
//!
//! The ONNX prover reduces every committed polynomial to a single joint opening
//! of the index-0 overlap RLC (`build_materialized_rlc`), then combines the
//! per-polynomial commitments homomorphically. Dory commitments are matrix
//! (tier-1 rows → tier-2 pairing) structured, so they only add across arities if
//! every polynomial shares a column width. This adapter therefore commits with a
//! *fixed* column count derived from the SRS (see [`DoryScheme::split`]) rather
//! than a per-polynomial balanced split; with that, `Σ_i γ_i · C_i` equals the
//! commitment of the overlap RLC and Dory reuses HyperKZG's exact opening path
//! (`REQUIRES_MATERIALIZED_POLYS = true`). The joint polynomial stays
//! `max_num_vars`-sized, never the sum of all committed sizes.
//!
//! One-hot committed polynomials (the bulk of a model's witness) are committed
//! sparsely in `O(nonzeros)` via [`DoryScheme::commit_one_hot`], bit-identical
//! to the dense path so they still combine homomorphically.

mod transcript;
mod types;

pub use types::{DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryVerifierSetup};

use std::borrow::Borrow;

use ark_bn254::Fr;
use dory::{
    backends::arkworks::{ArkFr, ArkG1, ArkGT, ArkworksPolynomial, G1Routines, G2Routines, BN254},
    primitives::{
        arithmetic::{Field as DoryField, Group as DoryGroup, PairingCurve},
        poly::Polynomial as DoryPolynomial,
    },
    prove as dory_prove, setup as dory_setup, verify as dory_verify, Transparent,
};
use rayon::prelude::*;

use self::transcript::LocalToDoryTranscript;
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

/// Dory PCS over BN254, implementing joltworks' [`CommitmentScheme`].
#[derive(Clone, Debug)]
pub struct DoryScheme;

impl DoryScheme {
    /// Log column count shared by *every* commitment: `sigma_J = log2(#g1
    /// generators)`. The SRS from `setup(N)` has `2^{ceil(N/2)}` g1 generators,
    /// so this is the balanced column width of the largest committable
    /// polynomial (`N = max_num_vars`).
    #[inline]
    fn column_log(setup: &DoryProverSetup) -> usize {
        setup.prover.g1_vec.len().log_2()
    }

    /// Matrix split for a polynomial of `num_vars` variables using the shared
    /// column width `sigma_j`: columns take `min(num_vars, sigma_j)`.
    ///
    /// Committing every polynomial with the *same* column width — rather than a
    /// per-polynomial balanced split — is what makes Dory commitments additively
    /// homomorphic across *different* arities. With a shared, prefix-aligned
    /// column basis the row-major reshape ([`ark_poly`] `row i = coeffs[i*cols..]`,
    /// tier-2 pairs rows with `g2[..num_rows]`) embeds a smaller polynomial into
    /// the top rows of a larger one, so `Σ_i γ_i · C_i` equals the commitment of
    /// the index-0 overlap RLC that [`build_materialized_rlc`] forms. That lets
    /// Dory reuse the same per-polynomial-commit + `combine_commitments` opening
    /// path as HyperKZG (see [`combine_commitments`]).
    ///
    /// [`build_materialized_rlc`]: crate::poly::rlc_polynomial::build_materialized_rlc
    /// [`combine_commitments`]: CommitmentScheme::combine_commitments
    /// [`ark_poly`]: dory::backends::arkworks
    #[inline]
    fn split(num_vars: usize, sigma_j: usize) -> (usize, usize) {
        let sigma = num_vars.min(sigma_j);
        let nu = num_vars - sigma;
        (nu, sigma)
    }

    /// Materialize a joltworks polynomial into a dense `Vec<ArkFr>` padded to a
    /// power-of-two length (dory requires exactly `2^(nu+sigma)` coefficients).
    ///
    /// One-hot polynomials are sparse and `get_coeff` panics on them, so they are
    /// densified directly: a `1` at each `k*T+t`, matching the index layout
    /// [`build_materialized_rlc`](crate::poly::rlc_polynomial::build_materialized_rlc)
    /// uses so the per-polynomial commitment agrees with the joint overlap RLC.
    /// (This dense expansion is `O(K*T)`; one-hot polynomials instead use
    /// [`Self::commit_one_hot`] to keep commitment generation `O(nonzeros)`.)
    fn materialize(poly: &MultilinearPolynomial<Fr>) -> Vec<ArkFr> {
        if let MultilinearPolynomial::OneHot(one_hot) = poly {
            let t_len = one_hot.nonzero_indices.len();
            let padded = (one_hot.K * t_len).next_power_of_two();
            let mut coeffs = vec![<ArkFr as DoryField>::zero(); padded];
            let one = <ArkFr as DoryField>::one();
            for (t, k_opt) in one_hot.nonzero_indices.iter().enumerate() {
                if let Some(k) = k_opt {
                    coeffs[*k as usize * t_len + t] = one;
                }
            }
            return coeffs;
        }
        let padded = poly.len().next_power_of_two();
        let mut coeffs: Vec<ArkFr> = (0..poly.len()).map(|i| ArkFr(poly.get_coeff(i))).collect();
        coeffs.resize(padded, <ArkFr as DoryField>::zero());
        coeffs
    }

    /// Sparse commitment for a one-hot polynomial: `O(nonzeros)` group additions
    /// instead of the `O(K*T)` dense MSM in [`Self::materialize`] + [`commit`].
    ///
    /// The one-hot has a single `1` at flat index `k*T+t` per set cycle. In the
    /// row-major (`nu`, `sigma`) layout that entry lands in row `idx/cols`,
    /// column `idx%cols`, so its row commitment picks up exactly `g1[col]`. We
    /// accumulate those generator picks per row (tier-1), then multi-pair with
    /// the g2 setup (tier-2) — the *same* computation the dense Transparent
    /// commit performs (`blind = 0`, `mask = identity`), so the resulting GT
    /// commitment is bit-identical and stays combine-compatible with the dense
    /// commitments of the other polynomials.
    ///
    /// [`commit`]: CommitmentScheme::commit
    fn commit_one_hot(
        one_hot: &crate::poly::one_hot_polynomial::OneHotPolynomial<Fr>,
        setup: &DoryProverSetup,
    ) -> (DoryCommitment, DoryHint) {
        let t_len = one_hot.nonzero_indices.len();
        let padded = (one_hot.K * t_len).next_power_of_two();
        let num_vars = padded.log_2();
        let (nu, sigma) = Self::split(num_vars, Self::column_log(setup));
        let cols = 1usize << sigma;
        let num_rows = 1usize << nu;

        // Tier-1: each set entry adds its column generator into its row. Rows
        // with no set entry stay the identity (contribute nothing to tier-2).
        let mut row_commitments = vec![<ArkG1 as DoryGroup>::identity(); num_rows];
        for (t, k_opt) in one_hot.nonzero_indices.iter().enumerate() {
            if let Some(k) = k_opt {
                let idx = *k as usize * t_len + t;
                row_commitments[idx / cols] =
                    row_commitments[idx / cols] + setup.prover.g1_vec[idx % cols];
            }
        }

        // Tier-2: identical to the dense commit's `multi_pair_g2_setup`.
        let tier_2 = <BN254 as PairingCurve>::multi_pair_g2_setup(
            &row_commitments,
            &setup.prover.g2_vec[..num_rows],
        );
        (
            DoryCommitment(tier_2),
            DoryHint {
                row_commitments,
                commit_blind: <ArkFr as DoryField>::zero(),
            },
        )
    }

    fn dory_point(opening_point: &[<Fr as JoltField>::Challenge]) -> Vec<ArkFr> {
        opening_point
            .iter()
            .rev()
            .map(|c| ArkFr((*c).into()))
            .collect()
    }
}

impl CommitmentScheme for DoryScheme {
    type Field = Fr;
    type ProverSetup = DoryProverSetup;
    type VerifierSetup = DoryVerifierSetup;
    type Commitment = DoryCommitment;
    type Proof = DoryProof;
    type BatchedProof = DoryProof;
    type OpeningProofHint = DoryHint;

    // Every polynomial is committed with the same (SRS-derived) column width, so
    // commitments of *different* arities combine additively — `Σ_i γ_i · C_i` is
    // the commitment of the index-0 overlap RLC. That lets Dory reuse the same
    // per-polynomial-commit + `combine_commitments` opening path as HyperKZG
    // (one joint 2^{max_num_vars} opening) via `build_materialized_rlc`.
    const REQUIRES_MATERIALIZED_POLYS: bool = true;

    #[tracing::instrument(skip_all, name = "DoryScheme::setup_prover")]
    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        let (prover, verifier) = dory_setup::<BN254>(max_num_vars);
        DoryProverSetup { prover, verifier }
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        DoryVerifierSetup(setup.verifier.clone())
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::commit")]
    fn commit(
        poly: &MultilinearPolynomial<Self::Field>,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        // One-hot polynomials commit sparsely (O(nonzeros)); the result is
        // bit-identical to the dense path so it still combines homomorphically.
        if let MultilinearPolynomial::OneHot(one_hot) = poly {
            return Self::commit_one_hot(one_hot, setup);
        }
        let coeffs = Self::materialize(poly);
        let num_vars = coeffs.len().log_2();
        let (nu, sigma) = Self::split(num_vars, Self::column_log(setup));
        let ark_poly = ArkworksPolynomial::new(coeffs);
        let (commitment, row_commitments, commit_blind) = ark_poly
            .commit::<BN254, Transparent, G1Routines>(nu, sigma, &setup.prover)
            .expect("dory commit");
        (
            DoryCommitment(commitment),
            DoryHint {
                row_commitments,
                commit_blind,
            },
        )
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::batch_commit")]
    fn batch_commit<U>(
        polys: &[U],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync,
    {
        polys
            .par_iter()
            .map(|p| Self::commit(p.borrow(), gens))
            .collect()
    }

    /// Homomorphically combine per-polynomial commitments into the commitment of
    /// their index-0 overlap RLC `Σ_i γ_i · f_i` (the joint polynomial
    /// [`build_materialized_rlc`] opens).
    ///
    /// Because every polynomial is committed with the same column width (see
    /// [`Self::split`]), the tier-1 row commitments and the tier-2 multi-pairing
    /// are both linear over a shared, prefix-aligned generator basis, so the GT
    /// commitments add: `C_joint = Σ_i γ_i · C_i`.
    ///
    /// [`build_materialized_rlc`]: crate::poly::rlc_polynomial::build_materialized_rlc
    fn combine_commitments<C: Borrow<Self::Commitment>>(
        commitments: &[C],
        coeffs: &[Self::Field],
    ) -> Self::Commitment {
        assert_eq!(
            commitments.len(),
            coeffs.len(),
            "combine_commitments: {} commitments but {} coefficients",
            commitments.len(),
            coeffs.len(),
        );
        let combined = commitments
            .iter()
            .zip(coeffs.iter())
            .map(|(commitment, gamma)| ArkFr(*gamma) * commitment.borrow().0)
            .fold(<ArkGT as DoryGroup>::identity(), |acc, term| acc + term);
        DoryCommitment(combined)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::prove")]
    fn prove<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        let num_vars = opening_point.len();
        let (nu, sigma) = Self::split(num_vars, Self::column_log(setup));
        let coeffs = Self::materialize(poly);
        assert_eq!(
            coeffs.len(),
            1usize << num_vars,
            "polynomial size (2^{}) must match opening point length ({num_vars})",
            coeffs.len().log_2(),
        );
        let ark_poly = ArkworksPolynomial::new(coeffs);
        let point = Self::dory_point(opening_point);

        let (row_commitments, commit_blind) = match hint {
            Some(h) => (h.row_commitments, h.commit_blind),
            None => {
                let (_commitment, rows, blind) = ark_poly
                    .commit::<BN254, Transparent, G1Routines>(nu, sigma, &setup.prover)
                    .expect("dory commit (hint recompute)");
                (rows, blind)
            }
        };

        let mut dory_transcript = LocalToDoryTranscript::new(transcript);
        let (proof, _blind) = dory_prove::<_, BN254, G1Routines, G2Routines, _, _, Transparent>(
            &ark_poly,
            &point,
            row_commitments,
            commit_blind,
            nu,
            sigma,
            &setup.prover,
            &mut dory_transcript,
        )
        .expect("dory prove");
        DoryProof(proof)
    }

    #[tracing::instrument(skip_all, name = "DoryScheme::verify")]
    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        opening: &Self::Field,
        commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        let point = Self::dory_point(opening_point);
        let mut dory_transcript = LocalToDoryTranscript::new(transcript);
        dory_verify::<_, BN254, G1Routines, G2Routines, _>(
            commitment.0,
            ArkFr(*opening),
            &point,
            &proof.0,
            setup.0.clone(),
            &mut dory_transcript,
        )
        .map_err(|_| ProofVerifyError::InternalError)
    }

    fn protocol_name() -> &'static [u8] {
        b"dory"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::{dense_mlpoly::DensePolynomial, multilinear_polynomial::PolynomialEvaluation},
        transcripts::Blake2bTranscript,
    };

    /// Bare-metal round trip against the native `dory-pcs` API. Proves the crate
    /// links and runs inside this workspace's `dev/twist-shout` arkworks fork.
    #[test]
    fn dory_native_round_trip() {
        use dory::backends::arkworks::Blake2bTranscript as DoryBlake2b;

        let (prover_setup, verifier_setup) = dory_setup::<BN254>(10);
        let (nu, sigma) = (4usize, 4usize);
        let num_vars = nu + sigma;
        let poly_size = 1usize << num_vars;

        let coefficients: Vec<ArkFr> = (0..poly_size)
            .map(|_| <ArkFr as DoryField>::random())
            .collect();
        let poly = ArkworksPolynomial::new(coefficients);

        let (tier_2, tier_1, commit_blind) = poly
            .commit::<BN254, Transparent, G1Routines>(nu, sigma, &prover_setup)
            .unwrap();

        let point: Vec<ArkFr> = (0..num_vars)
            .map(|_| <ArkFr as DoryField>::random())
            .collect();
        let evaluation = poly.evaluate(&point);

        let mut prover_transcript = DoryBlake2b::new(b"dory-native-roundtrip");
        let (proof, _) = dory_prove::<_, BN254, G1Routines, G2Routines, _, _, Transparent>(
            &poly,
            &point,
            tier_1,
            commit_blind,
            nu,
            sigma,
            &prover_setup,
            &mut prover_transcript,
        )
        .unwrap();

        let mut verifier_transcript = DoryBlake2b::new(b"dory-native-roundtrip");
        dory_verify::<_, BN254, G1Routines, G2Routines, _>(
            tier_2,
            evaluation,
            &point,
            &proof,
            verifier_setup,
            &mut verifier_transcript,
        )
        .expect("dory verification should succeed");
    }

    /// End-to-end round trip through the joltworks [`CommitmentScheme`] trait:
    /// materialize a joltworks polynomial, commit, open at a joltworks
    /// `F::Challenge` point, and verify against the claimed evaluation computed
    /// with joltworks' own MLE (this is what pins the point-reversal convention).
    #[test]
    fn dory_scheme_round_trip() {
        let num_vars = 8;
        let n = 1usize << num_vars;
        let coeffs: Vec<Fr> = (0..n).map(|i| Fr::from_u64(i as u64 + 1)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs));

        let setup = DoryScheme::setup_prover(num_vars);
        let vsetup = DoryScheme::setup_verifier(&setup);

        let (commitment, hint) = DoryScheme::commit(&poly, &setup);

        let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|i| <Fr as JoltField>::Challenge::from((i as u128) + 7))
            .collect();
        let opening = poly.evaluate(&point);

        let mut pt = Blake2bTranscript::new(b"dory-adapter-rt");
        let proof = DoryScheme::prove(&setup, &poly, &point, Some(hint), &mut pt);

        let mut vt = Blake2bTranscript::new(b"dory-adapter-rt");
        DoryScheme::verify(&proof, &vsetup, &mut vt, &point, &opening, &commitment)
            .expect("dory adapter verify should succeed");
    }

    /// The homomorphism that lets Dory reuse HyperKZG's opening path:
    /// committing every polynomial with the shared column width makes
    /// `Σ_i γ_i · C_i` equal to the commitment of the index-0 overlap RLC — even
    /// across *different* arities — and a joint opening verifies against it.
    #[test]
    fn combine_matches_overlap_rlc() {
        fn dense(coeffs: Vec<u64>) -> MultilinearPolynomial<Fr> {
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                coeffs.into_iter().map(Fr::from_u64).collect(),
            ))
        }

        // Mixed arities 3, 1, 0 — the overlap RLC is a single 3-variable poly.
        let setup = DoryScheme::setup_prover(3);
        let polys = [
            dense(vec![11, 12, 13, 14, 15, 16, 17, 18]),
            dense(vec![21, 22]),
            dense(vec![31]),
        ];
        let gammas: Vec<Fr> = (0..polys.len())
            .map(|i| Fr::from_u64(i as u64 + 2))
            .collect();

        // Per-polynomial commitments (fixed column width) combined homomorphically.
        let commitments: Vec<DoryCommitment> = polys
            .iter()
            .map(|p| DoryScheme::commit(p, &setup).0)
            .collect();
        let combined = DoryScheme::combine_commitments(&commitments, &gammas);

        // The index-0 overlap RLC, committed directly.
        let joint_len = polys.iter().map(|p| p.len()).max().unwrap();
        let mut joint = vec![Fr::from_u64(0); joint_len];
        for (p, gamma) in polys.iter().zip(&gammas) {
            for i in 0..p.len() {
                joint[i] += *gamma * p.get_coeff(i);
            }
        }
        let joint_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(joint));
        let (joint_commitment, _) = DoryScheme::commit(&joint_poly, &setup);
        assert_eq!(
            combined, joint_commitment,
            "Σ γ_i · C_i must equal the commitment of the overlap RLC",
        );

        // A joint opening verifies against the *combined* commitment.
        let point: Vec<<Fr as JoltField>::Challenge> = (0..3)
            .map(|i| <Fr as JoltField>::Challenge::from((i as u128) + 5))
            .collect();
        let opening = joint_poly.evaluate(&point);
        let mut pt = Blake2bTranscript::new(b"dory-combine");
        let proof = DoryScheme::prove(&setup, &joint_poly, &point, None, &mut pt);
        let vsetup = DoryScheme::setup_verifier(&setup);
        let mut vt = Blake2bTranscript::new(b"dory-combine");
        DoryScheme::verify(&proof, &vsetup, &mut vt, &point, &opening, &combined)
            .expect("opening must verify against the combined commitment");
    }

    /// The sparse one-hot commit must be *bit-identical* to committing the dense
    /// expansion (1s at `k*T+t`), otherwise it would not combine homomorphically
    /// with the other polynomials' commitments.
    #[test]
    fn sparse_one_hot_commit_matches_dense() {
        use crate::poly::one_hot_polynomial::OneHotPolynomial;

        // K=8, T=16 (K*T=128 → 7 vars); a mix of set and unset cycles.
        let k = 8usize;
        let t = 16usize;
        let indices: Vec<Option<u16>> = (0..t)
            .map(|i| {
                if i % 3 == 0 {
                    None
                } else {
                    Some(((i * 5) % k) as u16)
                }
            })
            .collect();
        let one_hot = MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(indices, k));

        // Dense reference: expand to a LargeScalars poly with 1s at k*T+t and commit
        // it through the ordinary (non-one-hot) path.
        let dense_coeffs = DoryScheme::materialize(&one_hot);
        let dense_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
            dense_coeffs.iter().map(|c| c.0).collect(),
        ));

        let setup = DoryScheme::setup_prover((k * t).log_2());
        let (sparse_commitment, _) = DoryScheme::commit(&one_hot, &setup);
        let (dense_commitment, _) = DoryScheme::commit(&dense_poly, &setup);
        assert_eq!(
            sparse_commitment, dense_commitment,
            "sparse one-hot commitment must equal the dense expansion's commitment",
        );
    }
}
