//! A never-materialized random linear combination of committed polynomials for
//! Dory's joint opening.
//!
//! The ONNX prover opens `Σ_i γ_i · f_i` (all `f_i` overlapped at index 0) once.
//! Almost every `f_i` is a one-hot (`O(T)` nonzeros in a `K·T` domain), so the
//! dense joint (`2^{max_num_vars}` coefficients) is both far larger than the
//! actual data and, for model-global lookup polynomials, impossible to hold in
//! memory. Dory's opening only needs three things from the polynomial —
//! `L^T·M` (vector-matrix product), the evaluation, and (absent a hint) the row
//! commitments — and all three are linear, so this type computes them straight
//! from the one-hots in `O(Σ nonzeros)` plus `O(dense)` for the few dense polys.
use ark_bn254::Fr;
use ark_ff::{One, Zero};
use common::{parallel::par_enabled, CommittedPoly};
use dory::{
    backends::arkworks::{ArkFr, ArkG1, G1Routines},
    error::DoryError,
    mode::Mode,
    primitives::{
        arithmetic::{DoryRoutines, Group as DoryGroup, PairingCurve},
        poly::{MultilinearLagrange, Polynomial as DoryPolynomial},
    },
    setup::ProverSetup,
};
use rayon::prelude::*;
use std::collections::BTreeMap;

use crate::{
    poly::{
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        one_hot_polynomial::OneHotPolynomial,
    },
    utils::math::Math,
};

/// `Σ_i γ_i · f_i` with every `f_i` embedded at index 0 of a `2^num_vars` domain.
pub struct SparseRlc<'a> {
    num_vars: usize,
    /// Dense contributions, already combined: `dense[j] = Σ_{dense i} γ_i · f_i[j]`.
    dense: Vec<ArkFr>,
    /// One-hot contributions kept sparse: `(γ_i, f_i)`.
    one_hots: Vec<(Fr, &'a OneHotPolynomial<Fr>)>,
}

impl<'a> SparseRlc<'a> {
    /// Build from the committed polynomial map and its RLC coefficients (same
    /// `BTreeMap` order as `build_materialized_rlc`).
    pub fn new(
        coeffs: &[Fr],
        polynomials: &'a BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>>,
        num_vars: usize,
    ) -> Self {
        assert_eq!(coeffs.len(), polynomials.len());
        let mut dense_polys = vec![];
        let mut one_hots = vec![];
        for ((_, poly), gamma) in polynomials.iter().zip(coeffs) {
            match poly {
                MultilinearPolynomial::OneHot(oh) => one_hots.push((*gamma, oh)),
                p => dense_polys.push((*gamma, p)),
            }
        }
        let dense_len = dense_polys
            .iter()
            .map(|(_, p)| p.original_len())
            .max()
            .unwrap_or(0);
        assert!(
            dense_len <= 1 << num_vars,
            "dense committed polynomial larger than the joint domain"
        );
        let dense: Vec<ArkFr> = (0..dense_len)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|j| {
                let mut acc = Fr::zero();
                for (gamma, p) in &dense_polys {
                    if j < p.original_len() {
                        acc += *gamma * p.get_scaled_coeff(j, Fr::one());
                    }
                }
                ArkFr(acc)
            })
            .collect();
        for (_, oh) in &one_hots {
            assert!(
                oh.K * oh.nonzero_indices.len() <= 1 << num_vars,
                "one-hot polynomial larger than the joint domain"
            );
        }
        Self {
            num_vars,
            dense,
            one_hots,
        }
    }

    /// Row commitments of the joint (tier-1), computed from scratch. Prefer
    /// combining the per-polynomial commit hints; this is the hint-less fallback.
    fn row_commitments<E, M1>(&self, nu: usize, sigma: usize, setup: &ProverSetup<E>) -> Vec<E::G1>
    where
        E: PairingCurve,
        M1: DoryRoutines<E::G1>,
        E::G1: DoryGroup<Scalar = ArkFr>,
    {
        let num_rows = 1usize << nu;
        let num_cols = 1usize << sigma;
        let g1 = &setup.g1_vec[..num_cols];

        // Dense part: an MSM per (partial) row.
        let mut rows: Vec<E::G1> = (0..num_rows)
            .into_par_iter()
            .map(|i| {
                let start = i * num_cols;
                if start >= self.dense.len() {
                    return E::G1::identity();
                }
                let end = (start + num_cols).min(self.dense.len());
                M1::msm(&g1[..end - start], &self.dense[start..end])
            })
            .collect();

        // One-hots: per polynomial, sum generator picks per row (additions only),
        // then scale each touched row by γ_i once.
        let contributions: Vec<E::G1> = self
            .one_hots
            .par_iter()
            .fold(
                || vec![E::G1::identity(); num_rows],
                |mut local, (gamma, oh)| {
                    let t_len = oh.nonzero_indices.len();
                    let touched_rows = (oh.K * t_len).div_ceil(num_cols);
                    let mut picks = vec![E::G1::identity(); touched_rows];
                    for (t, k_opt) in oh.nonzero_indices.iter().enumerate() {
                        if let Some(k) = k_opt {
                            let idx = *k as usize * t_len + t;
                            picks[idx / num_cols] = picks[idx / num_cols] + g1[idx % num_cols];
                        }
                    }
                    let gamma = ArkFr(*gamma);
                    for (r, pick) in picks.into_iter().enumerate() {
                        local[r] = local[r] + pick.scale(&gamma);
                    }
                    local
                },
            )
            .reduce(
                || vec![E::G1::identity(); num_rows],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x = *x + y;
                    }
                    a
                },
            );
        for (r, c) in rows.iter_mut().zip(contributions) {
            *r = *r + c;
        }
        rows
    }
}

impl DoryPolynomial<ArkFr> for SparseRlc<'_> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Evaluate at a point in *dory* variable order (`point[0]` ↔ lowest index
    /// bit). A sub-polynomial of `m` variables embedded at index 0 contributes
    /// `f_i(point[..m]) · Π_{j≥m} (1 − point[j])`; joltworks evaluates with the
    /// reversed convention, hence the `rev()`.
    fn evaluate(&self, point: &[ArkFr]) -> ArkFr {
        assert_eq!(point.len(), self.num_vars);
        let high_factor = |m: usize| -> Fr {
            point[m..]
                .iter()
                .fold(Fr::one(), |acc, p| acc * (Fr::one() - p.0))
        };
        let mut total = Fr::zero();
        if !self.dense.is_empty() {
            let m = self.dense.len().log_2();
            let r: Vec<Fr> = point[..m].iter().rev().map(|p| p.0).collect();
            let dense: Vec<Fr> = self.dense.iter().map(|c| c.0).collect();
            total += MultilinearPolynomial::from(dense).evaluate(&r) * high_factor(m);
        }
        let one_hot_total: Fr = self
            .one_hots
            .par_iter()
            .map(|(gamma, oh)| {
                let m = oh.get_num_vars();
                let r: Vec<Fr> = point[..m].iter().rev().map(|p| p.0).collect();
                *gamma * oh.evaluate(&r) * high_factor(m)
            })
            .sum();
        ArkFr(total + one_hot_total)
    }

    fn commit<E, Mo, M1>(
        &self,
        nu: usize,
        sigma: usize,
        setup: &ProverSetup<E>,
    ) -> Result<(E::GT, Vec<E::G1>, ArkFr), DoryError>
    where
        E: PairingCurve,
        Mo: Mode,
        M1: DoryRoutines<E::G1>,
        E::G1: DoryGroup<Scalar = ArkFr>,
        E::GT: DoryGroup<Scalar = ArkFr>,
    {
        if nu + sigma != self.num_vars {
            return Err(DoryError::InvalidSize {
                expected: 1 << (nu + sigma),
                actual: 1 << self.num_vars,
            });
        }
        let row_commitments = self.row_commitments::<E, M1>(nu, sigma, setup);
        let tier_2 = E::multi_pair_g2_setup(&row_commitments, &setup.g2_vec[..1 << nu]);
        let r_d1: ArkFr = Mo::sample();
        let commitment = Mo::mask(tier_2, &setup.ht, &r_d1);
        Ok((commitment, row_commitments, r_d1))
    }
}

impl MultilinearLagrange<ArkFr> for SparseRlc<'_> {
    /// `v[col] = Σ_row L[row] · M[row][col]` over the row-major `2^nu × 2^sigma`
    /// matrix, in `O(dense) + O(Σ nonzeros)`.
    #[tracing::instrument(skip_all, name = "SparseRlc::vector_matrix_product")]
    fn vector_matrix_product(&self, left_vec: &[ArkFr], _nu: usize, sigma: usize) -> Vec<ArkFr> {
        let num_cols = 1usize << sigma;
        let mut v: Vec<Fr> = (0..num_cols)
            .into_par_iter()
            .with_min_len(par_enabled())
            .map(|j| {
                let mut acc = Fr::zero();
                let mut idx = j;
                let mut row = 0;
                while idx < self.dense.len() {
                    acc += left_vec[row].0 * self.dense[idx].0;
                    idx += num_cols;
                    row += 1;
                }
                acc
            })
            .collect();

        let sparse: Vec<Fr> = self
            .one_hots
            .par_iter()
            .fold(
                || vec![Fr::zero(); num_cols],
                |mut local, (gamma, oh)| {
                    let t_len = oh.nonzero_indices.len();
                    let touched_rows = (oh.K * t_len).div_ceil(num_cols);
                    // γ_i · L[row] for every row this polynomial touches.
                    let lg: Vec<Fr> = left_vec[..touched_rows]
                        .iter()
                        .map(|l| l.0 * *gamma)
                        .collect();
                    for (t, k_opt) in oh.nonzero_indices.iter().enumerate() {
                        if let Some(k) = k_opt {
                            let idx = *k as usize * t_len + t;
                            local[idx % num_cols] += lg[idx / num_cols];
                        }
                    }
                    local
                },
            )
            .reduce(
                || vec![Fr::zero(); num_cols],
                |mut a, b| {
                    a.par_iter_mut()
                        .zip(b.par_iter())
                        .with_min_len(par_enabled())
                        .for_each(|(x, y)| *x += *y);
                    a
                },
            );
        v.par_iter_mut()
            .zip(sparse.par_iter())
            .with_min_len(par_enabled())
            .for_each(|(x, y)| *x += *y);
        v.into_iter().map(ArkFr).collect()
    }
}

/// Combine per-polynomial tier-1 hints into the joint's: `rows[r] = Σ_i γ_i · rows_i[r]`.
pub fn combine_row_commitments(hints: &[Vec<ArkG1>], coeffs: &[Fr]) -> Vec<ArkG1> {
    assert_eq!(hints.len(), coeffs.len());
    let num_rows = hints.iter().map(|h| h.len()).max().unwrap_or(0);
    // Row r collects one term per polynomial with more than r rows. Row 0 has
    // every polynomial (a real MSM); rows near the top belong to a handful of
    // large polynomials. arkworks' MSM builds its own thread pools, so it must
    // not be invoked from inside a rayon job: large rows run sequentially
    // (each MSM parallel internally), small rows in parallel by scale-and-add.
    const MSM_THRESHOLD: usize = 64;
    let row_terms = |r: usize| -> (Vec<ArkG1>, Vec<ArkFr>) {
        hints
            .iter()
            .zip(coeffs)
            .filter(|(h, _)| r < h.len())
            .map(|(h, g)| (h[r], ArkFr(*g)))
            .unzip()
    };
    let mut rows: Vec<ArkG1> = (0..num_rows)
        .into_par_iter()
        .map(|r| {
            let (bases, scalars) = row_terms(r);
            if bases.len() > MSM_THRESHOLD {
                <ArkG1 as DoryGroup>::identity()
            } else {
                bases
                    .iter()
                    .zip(&scalars)
                    .fold(<ArkG1 as DoryGroup>::identity(), |acc, (b, s)| {
                        acc + b.scale(s)
                    })
            }
        })
        .collect();
    for (r, row) in rows.iter_mut().enumerate() {
        let (bases, scalars) = row_terms(r);
        if bases.len() > MSM_THRESHOLD {
            *row = G1Routines::msm(&bases, &scalars);
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        field::JoltField,
        poly::{
            commitment::{
                commitment_scheme::CommitmentScheme,
                dory::{DoryHint, DoryScheme},
            },
            dense_mlpoly::DensePolynomial,
        },
        transcripts::{Blake2bTranscript, Transcript},
    };
    use dory::{backends::arkworks::BN254, Transparent};

    fn one_hot(k: usize, t: usize, seed: u64) -> MultilinearPolynomial<Fr> {
        let idx: Vec<Option<u16>> = (0..t)
            .map(|i| {
                let v = (seed + 7 * i as u64 + (i as u64 * i as u64) % 5) % (k as u64 + 1);
                (v < k as u64).then_some(v as u16)
            })
            .collect();
        MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(idx, k))
    }

    fn dense(coeffs: Vec<u64>) -> MultilinearPolynomial<Fr> {
        MultilinearPolynomial::LargeScalars(DensePolynomial::new(
            coeffs.into_iter().map(Fr::from_u64).collect(),
        ))
    }

    /// The sparse joint must agree with the dense overlap RLC on everything the
    /// Dory opening touches, and the resulting proof must verify against the
    /// homomorphically combined commitment.
    #[test]
    fn sparse_rlc_matches_dense_and_verifies() {
        use crate::poly::rlc_polynomial::build_materialized_rlc;
        // Joint domain: 2^9 (K=8,T=64); mixed arities incl. tiny dense polys.
        let num_vars = 9;
        let setup = DoryScheme::setup_prover(num_vars);
        let mut polys: BTreeMap<CommittedPoly, MultilinearPolynomial<Fr>> = BTreeMap::new();
        polys.insert(CommittedPoly::ClampRaD(0, 0), one_hot(8, 64, 3));
        polys.insert(CommittedPoly::ClampRaD(0, 1), one_hot(8, 64, 11));
        polys.insert(CommittedPoly::ClampRaD(1, 0), one_hot(4, 16, 5));
        polys.insert(CommittedPoly::ClampRaD(2, 0), one_hot(2, 2, 1));
        polys.insert(
            CommittedPoly::DivNodeQuotient(3),
            dense(vec![5, 6, 7, 8, 9, 10, 11, 12]),
        );
        polys.insert(CommittedPoly::DivNodeQuotient(4), dense(vec![42]));
        let gammas: Vec<Fr> = (0..polys.len())
            .map(|i| Fr::from_u64(3 * i as u64 + 2))
            .collect();

        let joint_dense = build_materialized_rlc(&gammas, &polys);
        let joint_dense_ark = dory::backends::arkworks::ArkworksPolynomial::new(
            (0..1 << num_vars)
                .map(|i| ArkFr(joint_dense.get_coeff(i)))
                .collect(),
        );
        let sparse = SparseRlc::new(&gammas, &polys, num_vars);
        let (nu, sigma) = DoryScheme::split(num_vars, DoryScheme::column_log(&setup));

        // Evaluation (dory var order).
        let point: Vec<ArkFr> = (0..num_vars)
            .map(|i| ArkFr(Fr::from_u64(17 + 13 * i as u64)))
            .collect();
        assert_eq!(sparse.evaluate(&point), joint_dense_ark.evaluate(&point));

        // Vector-matrix product.
        let left: Vec<ArkFr> = (0..1 << nu)
            .map(|i| ArkFr(Fr::from_u64(101 + i as u64)))
            .collect();
        assert_eq!(
            sparse.vector_matrix_product(&left, nu, sigma),
            joint_dense_ark.vector_matrix_product(&left, nu, sigma)
        );

        // Row commitments: combined hints == sparse recompute == dense commit.
        let (commitments, hints): (Vec<_>, Vec<DoryHint>) = polys
            .values()
            .map(|p| DoryScheme::commit(p, &setup))
            .unzip();
        let combined_hint = DoryScheme::combine_hints(hints.clone(), &gammas);
        let (_, dense_rows, _) = joint_dense_ark
            .commit::<BN254, Transparent, G1Routines>(nu, sigma, &setup.prover)
            .unwrap();
        let (_, sparse_rows, _) = sparse
            .commit::<BN254, Transparent, G1Routines>(nu, sigma, &setup.prover)
            .unwrap();
        assert_eq!(sparse_rows, dense_rows);
        assert_eq!(combined_hint.row_commitments, dense_rows);

        // End to end: sparse prove_rlc verifies against Σ γ_i C_i.
        let combined = DoryScheme::combine_commitments(&commitments, &gammas);
        let opening_point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|i| <Fr as JoltField>::Challenge::from((i as u128) + 5))
            .collect();
        let opening = joint_dense.evaluate(&opening_point);
        let mut pt = Blake2bTranscript::new(b"sparse-rlc");
        let proof = DoryScheme::prove_rlc(&setup, &polys, &gammas, hints, &opening_point, &mut pt);
        let vsetup = DoryScheme::setup_verifier(&setup);
        let mut vt = Blake2bTranscript::new(b"sparse-rlc");
        DoryScheme::verify(
            &proof,
            &vsetup,
            &mut vt,
            &opening_point,
            &opening,
            &combined,
        )
        .expect("sparse joint opening must verify against the combined commitment");

        // And without hints (sparse recompute path).
        let mut pt = Blake2bTranscript::new(b"sparse-rlc");
        let proof = DoryScheme::prove_rlc(&setup, &polys, &gammas, vec![], &opening_point, &mut pt);
        let mut vt = Blake2bTranscript::new(b"sparse-rlc");
        DoryScheme::verify(
            &proof,
            &vsetup,
            &mut vt,
            &opening_point,
            &opening,
            &combined,
        )
        .expect("hint-less sparse joint opening must verify");
    }
}
