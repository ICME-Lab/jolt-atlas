//! This is a port of https://github.com/microsoft/Nova/blob/main/src/provider/hyperkzg.rs
//! and such code is Copyright (c) Microsoft Corporation.
//!
//! This module implements `HyperKZG`, a KZG-based polynomial commitment for multilinear polynomials
//! HyperKZG is based on the transformation from univariate PCS to multilinear PCS in the Gemini paper (section 2.4.2 in <https://eprint.iacr.org/2022/420.pdf>).
//! However, there are some key differences:
//! (1) HyperKZG works with multilinear polynomials represented in evaluation form (rather than in coefficient form in Gemini's transformation).
//! This means that Spartan's polynomial IOP can use commit to its polynomials as-is without incurring any interpolations or FFTs.
//! (2) HyperKZG is specialized to use KZG as the univariate commitment scheme, so it includes several optimizations (both during the transformation of multilinear-to-univariate claims
//! and within the KZG commitment scheme implementation itself).

use crate::{
    field::JoltField,
    msm::VariableBaseMSM,
    poly::{
        dense_mlpoly::DensePolynomial, multilinear_polynomial::MultilinearPolynomial,
        one_hot_polynomial::OneHotPolynomial, unipoly::UniPoly,
    },
    transcripts::{AppendToTranscript, Transcript},
    utils::errors::ProofVerifyError,
};
use ark_ec::{pairing::Pairing, AffineRepr};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{One, Zero};
use kzg::{KZGProverKey, KZGVerifierKey, UnivariateKZG, SRS};
use rand_core::{CryptoRng, RngCore};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use std::{
    io::{Read as IoRead, Write as IoWrite},
    marker::PhantomData,
    path::Path,
    sync::Arc,
};

pub mod commitment_scheme;
pub mod kzg;
#[cfg(test)]
pub mod tests;

pub struct HyperKZGSRS<P: Pairing>(Arc<SRS<P>>);

impl<P: Pairing> HyperKZGSRS<P> {
    pub fn setup<R: RngCore + CryptoRng>(rng: &mut R, max_degree: usize) -> Self
    where
        P::ScalarField: JoltField,
    {
        Self(Arc::new(SRS::setup(rng, max_degree, 2)))
    }

    pub fn trim(self, max_degree: usize) -> (HyperKZGProverKey<P>, HyperKZGVerifierKey<P>) {
        let (kzg_pk, kzg_vk) = SRS::trim(self.0, max_degree);
        (HyperKZGProverKey { kzg_pk }, HyperKZGVerifierKey { kzg_vk })
    }

    /// Load SRS from a file using compressed serialization.
    pub fn load_from_file<Pth: AsRef<Path>>(
        path: Pth,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let srs = SRS::<P>::deserialize_compressed(reader)?;
        Ok(Self(Arc::new(srs)))
    }

    /// Save SRS to a file using compressed serialization.
    pub fn save_to_file<Pth: AsRef<Path>>(
        &self,
        path: Pth,
    ) -> Result<(), ark_serialize::SerializationError> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        self.0.serialize_compressed(writer)?;
        Ok(())
    }

    /// Load SRS from a reader using compressed serialization.
    pub fn load_from_reader<R: IoRead>(
        reader: R,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let srs = SRS::<P>::deserialize_compressed(reader)?;
        Ok(Self(Arc::new(srs)))
    }

    /// Save SRS to a writer using compressed serialization.
    pub fn save_to_writer<W: IoWrite>(
        &self,
        writer: W,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.0.serialize_compressed(writer)?;
        Ok(())
    }

    /// Get the maximum degree this SRS supports.
    pub fn max_degree(&self) -> usize {
        self.0.g1_powers.len().saturating_sub(1)
    }
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyperKZGProverKey<P: Pairing> {
    pub kzg_pk: KZGProverKey<P>,
}

#[derive(Copy, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyperKZGVerifierKey<P: Pairing> {
    pub kzg_vk: KZGVerifierKey<P>,
}

#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct HyperKZGCommitment<P: Pairing>(pub P::G1Affine);

impl<P: Pairing> Default for HyperKZGCommitment<P> {
    fn default() -> Self {
        Self(P::G1Affine::zero())
    }
}

impl<P: Pairing> AppendToTranscript for HyperKZGCommitment<P> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_point(&self.0.into_group());
    }
}

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct HyperKZGProof<P: Pairing> {
    pub com: Vec<P::G1Affine>,
    pub w: Vec<P::G1Affine>,
    pub v: Vec<Vec<P::ScalarField>>,
}

// On input f(x) and u compute the witness polynomial used to prove
// that f(u) = v. The main part of this is to compute the
// division (f(x) - f(u)) / (x - u), but we don't use a general
// division algorithm, we make use of the fact that the division
// never has a remainder, and that the denominator is always a linear
// polynomial. The cost is (d-1) mults + (d-1) adds in P::ScalarField, where
// d is the degree of f.
//
// We use the fact that if we compute the quotient of f(x)/(x-u),
// there will be a remainder, but it'll be v = f(u).  Put another way
// the quotient of f(x)/(x-u) and (f(x) - f(v))/(x-u) is the
// same.  One advantage is that computing f(u) could be decoupled
// from kzg_open, it could be done later or separate from computing W.
fn kzg_batch_open_no_rem<P: Pairing>(
    f: &MultilinearPolynomial<P::ScalarField>,
    u: &[P::ScalarField],
    pk: &HyperKZGProverKey<P>,
) -> Vec<P::G1Affine>
where
    <P as Pairing>::ScalarField: JoltField,
{
    let f: &DensePolynomial<P::ScalarField> = f.try_into().unwrap();
    let h = u
        .par_iter()
        .map(|ui| {
            let h = compute_witness_polynomial::<P>(&f.evals(), *ui);
            MultilinearPolynomial::from(h)
        })
        .collect::<Vec<_>>();

    UnivariateKZG::commit_batch(&pk.kzg_pk, &h).unwrap()
}

fn compute_witness_polynomial<P: Pairing>(
    f: &[P::ScalarField],
    u: P::ScalarField,
) -> Vec<P::ScalarField>
where
    <P as Pairing>::ScalarField: JoltField,
{
    let d = f.len();

    // Compute h(x) = f(x)/(x - u)
    let mut h = vec![P::ScalarField::zero(); d];
    for i in (1..d).rev() {
        h[i - 1] = f[i] + h[i] * u;
    }

    h
}

fn kzg_open_batch<P: Pairing, ProofTranscript: Transcript>(
    f: &[MultilinearPolynomial<P::ScalarField>],
    u: &[P::ScalarField],
    pk: &HyperKZGProverKey<P>,
    transcript: &mut ProofTranscript,
) -> (Vec<P::G1Affine>, Vec<Vec<P::ScalarField>>)
where
    <P as Pairing>::ScalarField: JoltField,
{
    let k = f.len();
    let t = u.len();

    // The verifier needs f_i(u_j), so we compute them here
    // (V will compute B(u_j) itself)
    let mut v = vec![vec!(P::ScalarField::zero(); k); t];
    v.par_iter_mut().enumerate().for_each(|(i, v_i)| {
        // for each point u
        v_i.par_iter_mut().zip_eq(f).for_each(|(v_ij, f)| {
            // for each poly f
            *v_ij = UniPoly::eval_as_univariate(f, &u[i]);
        });
    });

    // TODO(moodlezoup): Avoid cloned()
    let scalars = v.iter().flatten().collect::<Vec<&P::ScalarField>>();
    transcript.append_scalars::<P::ScalarField>(&scalars);
    let q_powers: Vec<P::ScalarField> = transcript.challenge_scalar_powers(f.len());
    let f_arc: Vec<Arc<MultilinearPolynomial<P::ScalarField>>> =
        f.iter().map(|poly| Arc::new(poly.clone())).collect();

    let B = {
        let poly_refs: Vec<&MultilinearPolynomial<P::ScalarField>> =
            f_arc.iter().map(|arc| arc.as_ref()).collect();
        let dense_result = DensePolynomial::linear_combination(&poly_refs, &q_powers);
        MultilinearPolynomial::from(dense_result.Z)
    };

    // Now open B at u0, ..., u_{t-1}
    let w = kzg_batch_open_no_rem(&B, u, pk);

    // The prover computes the challenge to keep the transcript in the same
    // state as that of the verifier
    transcript.append_points(&w.iter().map(|g| g.into_group()).collect::<Vec<P::G1>>());
    let _d_0: P::ScalarField = transcript.challenge_scalar();

    (w, v)
}

// vk is hashed in transcript already, so we do not add it here
fn kzg_verify_batch<P: Pairing, ProofTranscript: Transcript>(
    vk: &HyperKZGVerifierKey<P>,
    C: &[P::G1Affine],
    W: &[P::G1Affine],
    u: &[P::ScalarField],
    v: &[Vec<P::ScalarField>],
    transcript: &mut ProofTranscript,
) -> bool
where
    <P as Pairing>::ScalarField: JoltField,
{
    let k = C.len();
    let t = u.len();

    let scalars = v.iter().flatten().collect::<Vec<&P::ScalarField>>();
    transcript.append_scalars::<P::ScalarField>(&scalars);
    let q_powers: Vec<P::ScalarField> = transcript.challenge_scalar_powers(k);

    transcript.append_points(&W.iter().map(|g| g.into_group()).collect::<Vec<P::G1>>());
    let d_0: P::ScalarField = transcript.challenge_scalar();
    let d_1 = d_0 * d_0;

    assert_eq!(t, 3);
    assert_eq!(W.len(), 3);
    // We write a special case for t=3, since this what is required for
    // hyperkzg. Following the paper directly, we must compute:
    // let L0 = C_B - vk.G * B_u[0] + W[0] * u[0];
    // let L1 = C_B - vk.G * B_u[1] + W[1] * u[1];
    // let L2 = C_B - vk.G * B_u[2] + W[2] * u[2];
    // let R0 = -W[0];
    // let R1 = -W[1];
    // let R2 = -W[2];
    // let L = L0 + L1*d_0 + L2*d_1;
    // let R = R0 + R1*d_0 + R2*d_1;
    //
    // We group terms to reduce the number of scalar mults (to seven):
    // In Rust, we could use MSMs for these, and speed up verification.
    //
    // Note, that while computing L, the intermediate computation of C_B together with computing
    // L0, L1, L2 can be replaced by single MSM of C with the powers of q multiplied by (1 + d_0 + d_1)
    // with additionally concatenated inputs for scalars/bases.

    let q_power_multiplier: P::ScalarField = P::ScalarField::one() + d_0 + d_1;

    let q_powers_multiplied: Vec<P::ScalarField> = q_powers
        .par_iter()
        .map(|q_power| *q_power * q_power_multiplier)
        .collect();

    // Compute the batched openings
    // compute B(u_i) = v[i][0] + q*v[i][1] + ... + q^(t-1) * v[i][t-1]
    let B_u = v
        .into_par_iter()
        .map(|v_i| {
            v_i.into_par_iter()
                .zip(q_powers.par_iter())
                .map(|(a, b)| *a * *b)
                .sum()
        })
        .collect::<Vec<P::ScalarField>>();

    let L = <P::G1 as VariableBaseMSM>::msm_field_elements(
        &[&C[..k], &[W[0], W[1], W[2], vk.kzg_vk.g1]].concat(),
        &[
            &q_powers_multiplied[..k],
            &[
                u[0],
                (u[1] * d_0),
                (u[2] * d_1),
                -(B_u[0] + d_0 * B_u[1] + d_1 * B_u[2]),
            ],
        ]
        .concat(),
    )
    .unwrap();

    let R = W[0] + W[1] * d_0 + W[2] * d_1;

    // Check that e(L, vk.H) == e(R, vk.tau_H)
    P::multi_pairing([L, -R], [vk.kzg_vk.g2, vk.kzg_vk.beta_g2]).is_zero()
}

#[derive(Clone)]
pub struct HyperKZG<P: Pairing> {
    _phantom: PhantomData<P>,
}

impl<P: Pairing> HyperKZG<P>
where
    <P as Pairing>::ScalarField: JoltField,
{
    pub fn protocol_name() -> &'static [u8] {
        b"HyperKZG"
    }

    /// Setup prover key from a file containing the SRS.
    pub fn setup_prover_from_file<Pth: AsRef<Path>>(
        path: Pth,
        max_degree: usize,
    ) -> Result<HyperKZGProverKey<P>, ark_serialize::SerializationError> {
        let srs = HyperKZGSRS::load_from_file(path)?;
        Ok(srs.trim(max_degree).0)
    }

    /// Setup prover key from a reader containing the SRS.
    pub fn setup_prover_from_reader<R: IoRead>(
        reader: R,
        max_degree: usize,
    ) -> Result<HyperKZGProverKey<P>, ark_serialize::SerializationError> {
        let srs = HyperKZGSRS::load_from_reader(reader)?;
        Ok(srs.trim(max_degree).0)
    }

    #[tracing::instrument(skip_all, name = "HyperKZG::open")]
    pub fn open<ProofTranscript: Transcript>(
        pk: &HyperKZGProverKey<P>,
        poly: &MultilinearPolynomial<P::ScalarField>,
        point: &[<P::ScalarField as JoltField>::Challenge],
        transcript: &mut ProofTranscript,
    ) -> Result<HyperKZGProof<P>, ProofVerifyError> {
        let ell = point.len();
        let n = poly.len();
        assert_eq!(n, 1 << ell); // Below we assume that n is a power of two

        // Phase 1  -- create commitments com_1, ..., com_\ell
        // We do not compute final Pi (and its commitment) as it is constant and equals to 'eval'
        // also known to verifier, so can be derived on its side as well
        let mut polys: Vec<MultilinearPolynomial<P::ScalarField>> = Vec::new();
        polys.push(poly.clone());
        for i in 0..ell - 1 {
            let previous_poly: &DensePolynomial<P::ScalarField> = (&polys[i]).try_into().unwrap();
            let Pi_len = previous_poly.len() / 2;
            let mut Pi = vec![P::ScalarField::zero(); Pi_len];
            Pi.par_iter_mut().enumerate().for_each(|(j, Pi_j)| {
                *Pi_j = point[ell - i - 1] * (previous_poly[2 * j + 1] - previous_poly[2 * j])
                    + previous_poly[2 * j];
            });

            polys.push(MultilinearPolynomial::from(Pi));
        }

        assert_eq!(polys.len(), ell);
        assert_eq!(polys[ell - 1].len(), 2);

        // We do not need to commit to the first polynomial as it is already committed.
        let com: Vec<P::G1Affine> = UnivariateKZG::commit_variable_batch(&pk.kzg_pk, &polys[1..])?;

        // Phase 2
        // We do not need to add x to the transcript, because in our context x was obtained from the transcript.
        // We also do not need to absorb `C` and `eval` as they are already absorbed by the transcript by the caller
        transcript.append_points(&com.iter().map(|g| g.into_group()).collect::<Vec<P::G1>>());
        let r: <P as Pairing>::ScalarField = transcript.challenge_scalar();
        let u = vec![r, -r, r * r];

        // Phase 3 -- create response
        let (w, v) = kzg_open_batch(&polys, &u, pk, transcript);

        Ok(HyperKZGProof { com, w, v })
    }

    /// A method to verify purported evaluations of a batch of polynomials
    #[tracing::instrument(skip_all, name = "HyperKZG::verify")]
    pub fn verify_inner<ProofTranscript: Transcript>(
        vk: &HyperKZGVerifierKey<P>,
        C: &HyperKZGCommitment<P>,
        point: &[<P::ScalarField as JoltField>::Challenge],
        P_of_x: &P::ScalarField,
        pi: &HyperKZGProof<P>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let y = P_of_x;

        let ell = point.len();

        let mut com = pi.com.clone();

        // we do not need to add x to the transcript, because in our context x was
        // obtained from the transcript
        transcript.append_points(&com.iter().map(|g| g.into_group()).collect::<Vec<P::G1>>());
        let r: <P as Pairing>::ScalarField = transcript.challenge_scalar();

        if r == P::ScalarField::zero() || C.0 == P::G1Affine::zero() {
            return Err(ProofVerifyError::InternalError);
        }
        com.insert(0, C.0); // set com_0 = C, shifts other commitments to the right

        let u = vec![r, -r, r * r];

        // Setup vectors (Y, ypos, yneg) from pi.v
        let v = &pi.v;
        if v.len() != 3 {
            return Err(ProofVerifyError::InternalError);
        }
        if v[0].len() != ell || v[1].len() != ell || v[2].len() != ell {
            return Err(ProofVerifyError::InternalError);
        }
        let ypos = &v[0];
        let yneg = &v[1];
        let mut Y = v[2].to_vec();
        Y.push(*y);

        // Check consistency of (Y, ypos, yneg)
        let two = P::ScalarField::from(2u64);
        for i in 0..ell {
            if two * r * Y[i + 1]
                != r * (P::ScalarField::one() - point[ell - i - 1]) * (ypos[i] + yneg[i])
                    + point[ell - i - 1] * (ypos[i] - yneg[i])
            {
                return Err(ProofVerifyError::InternalError);
            }
            // Note that we don't make any checks about Y[0] here, but our batching
            // check below requires it
        }

        // Check commitments to (Y, ypos, yneg) are valid
        if !kzg_verify_batch(vk, &com, &pi.w, &u, &pi.v, transcript) {
            return Err(ProofVerifyError::InternalError);
        }

        Ok(())
    }
}

/// Specialized implementation for BN254 that provides optimized sparse OneHot commitment.
impl HyperKZG<ark_bn254::Bn254> {
    /// Commit to a OneHotPolynomial without materializing the dense representation.
    /// This exploits the sparsity of OneHot polynomials (only T nonzero coefficients
    /// out of K*T total) by performing T point additions instead of a full MSM.
    ///
    /// The polynomial layout is: coefficient at index `k * T + t` is 1 if `nonzero_indices[t] == Some(k)`.
    #[tracing::instrument(skip_all, name = "HyperKZG::commit_one_hot")]
    pub fn commit_one_hot(
        pk: &HyperKZGProverKey<ark_bn254::Bn254>,
        poly: &OneHotPolynomial<ark_bn254::Fr>,
    ) -> Result<HyperKZGCommitment<ark_bn254::Bn254>, ProofVerifyError> {
        let T = poly.nonzero_indices.len();
        let K = poly.K;
        let required_size = K * T;

        if pk.kzg_pk.g1_powers().len() < required_size {
            return Err(ProofVerifyError::KeyLengthError(
                pk.kzg_pk.g1_powers().len(),
                required_size,
            ));
        }

        // Collect all indices where the coefficient is 1
        // Index formula: k * T + t (where k is the address at timestep t)
        let indices: Vec<usize> = poly
            .nonzero_indices
            .iter()
            .enumerate()
            .filter_map(|(t, k)| k.map(|k| k as usize * T + t))
            .collect();

        if indices.is_empty() {
            return Ok(HyperKZGCommitment(ark_bn254::G1Affine::zero()));
        }

        // Use optimized batch point addition (all coefficients are 1)
        let g1_bases = pk.kzg_pk.g1_powers();
        let indices_slice = [indices];
        let results = jolt_optimizations::batch_g1_additions_multi(g1_bases, &indices_slice);

        Ok(HyperKZGCommitment(results[0]))
    }

    /// Batch commit to multiple OneHotPolynomials efficiently.
    #[tracing::instrument(skip_all, name = "HyperKZG::batch_commit_one_hot")]
    pub fn batch_commit_one_hot(
        pk: &HyperKZGProverKey<ark_bn254::Bn254>,
        polys: &[OneHotPolynomial<ark_bn254::Fr>],
    ) -> Result<Vec<HyperKZGCommitment<ark_bn254::Bn254>>, ProofVerifyError> {
        if polys.is_empty() {
            return Ok(vec![]);
        }

        // Verify SRS is large enough for all polynomials
        let max_required = polys
            .iter()
            .map(|p| p.K * p.nonzero_indices.len())
            .max()
            .unwrap_or(0);
        if pk.kzg_pk.g1_powers().len() < max_required {
            return Err(ProofVerifyError::KeyLengthError(
                pk.kzg_pk.g1_powers().len(),
                max_required,
            ));
        }

        // Collect indices for all polynomials
        let all_indices: Vec<Vec<usize>> = polys
            .iter()
            .map(|poly| {
                let T = poly.nonzero_indices.len();
                poly.nonzero_indices
                    .iter()
                    .enumerate()
                    .filter_map(|(t, k)| k.map(|k| k as usize * T + t))
                    .collect()
            })
            .collect();

        let g1_bases = pk.kzg_pk.g1_powers();
        let results = jolt_optimizations::batch_g1_additions_multi(g1_bases, &all_indices);

        Ok(results.into_iter().map(HyperKZGCommitment).collect())
    }
}
