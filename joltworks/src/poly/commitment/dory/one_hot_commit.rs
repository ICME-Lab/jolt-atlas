//! Fast tier-1 / tier-2 for one-hot polynomial commitments.
//!
//! A one-hot polynomial's tier-1 row commitments are plain sums of column
//! generators (no scalars), so they can be built with *batched affine
//! addition*: pair the points of every row level by level, invert all the
//! slope denominators of a level with one Montgomery batch inversion, and add
//! in affine coordinates — roughly half the field multiplications of a mixed
//! projective add, with no projective bookkeeping.
//!
//! Tier-2 pairs each row with its G2 generator; rows that received no entry
//! are the identity and contribute nothing, so they are skipped (bucketed
//! lookup polynomials leave whole slabs of empty rows in their gap tails).
use ark_bn254::{Bn254, Fq, G1Affine, G1Projective};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{batch_inversion, Field, Zero};
use dory::backends::arkworks::{ArkG1, ArkG2, ArkGT, BN254};
use dory::primitives::arithmetic::{Group as DoryGroup, PairingCurve};

/// Sum, for every row, the affine points it was assigned (`(row, point)` pairs
/// grouped by row via `row_offsets`), returning projective row sums.
///
/// `points` is consumed level by level: each level pairs adjacent points of the
/// same row and replaces them by their sum, leaving at most one point per row.
pub(super) fn batched_affine_row_sums(
    mut points: Vec<G1Affine>,
    mut row_offsets: Vec<usize>, // len = num_rows + 1, row r owns points[row_offsets[r]..row_offsets[r+1]]
) -> Vec<G1Projective> {
    let num_rows = row_offsets.len() - 1;
    loop {
        // Pairs to add this level: (a, b) index pairs; carry singles through.
        let mut dens: Vec<Fq> = Vec::new();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let mut next_offsets = Vec::with_capacity(row_offsets.len());
        next_offsets.push(0usize);
        let mut any_pair = false;
        for r in 0..num_rows {
            let (s, e) = (row_offsets[r], row_offsets[r + 1]);
            let n = e - s;
            let mut i = s;
            while i + 1 < e {
                pairs.push((i, i + 1));
                any_pair = true;
                i += 2;
            }
            let survivors = n.div_ceil(2);
            next_offsets.push(next_offsets.last().unwrap() + survivors);
        }
        if !any_pair {
            break;
        }
        // Denominators: x2 - x1 (distinct x), 2*y1 (doubling), or 1 (a point
        // at infinity / inverse pair — handled without a slope).
        dens.reserve(pairs.len());
        for &(a, b) in &pairs {
            let (p, q) = (&points[a], &points[b]);
            let d = if p.infinity || q.infinity {
                Fq::ONE
            } else if p.x == q.x {
                if p.y == q.y {
                    p.y + p.y
                } else {
                    Fq::ONE
                }
            } else {
                q.x - p.x
            };
            dens.push(d);
        }
        batch_inversion(&mut dens);
        let mut next: Vec<G1Affine> = Vec::with_capacity(*next_offsets.last().unwrap());
        let mut pair_iter = pairs.iter().zip(dens.iter());
        for r in 0..num_rows {
            let (s, e) = (row_offsets[r], row_offsets[r + 1]);
            let mut i = s;
            while i + 1 < e {
                let (&(a, b), inv) = pair_iter.next().unwrap();
                debug_assert_eq!((a, b), (i, i + 1));
                let (p, q) = (points[a], points[b]);
                let sum = if p.infinity {
                    q
                } else if q.infinity {
                    p
                } else if p.x == q.x {
                    if p.y == q.y {
                        // Doubling: λ = 3x² / 2y.
                        let lambda = p.x.square() * Fq::from(3u64) * inv;
                        let x3 = lambda.square() - p.x - p.x;
                        let y3 = lambda * (p.x - x3) - p.y;
                        G1Affine::new_unchecked(x3, y3)
                    } else {
                        G1Affine::identity()
                    }
                } else {
                    // λ = (y2 - y1) / (x2 - x1).
                    let lambda = (q.y - p.y) * inv;
                    let x3 = lambda.square() - p.x - q.x;
                    let y3 = lambda * (p.x - x3) - p.y;
                    G1Affine::new_unchecked(x3, y3)
                };
                next.push(sum);
                i += 2;
            }
            if i < e {
                next.push(points[i]);
            }
        }
        points = next;
        row_offsets = next_offsets;
    }
    (0..num_rows)
        .map(|r| {
            if row_offsets[r + 1] > row_offsets[r] {
                points[row_offsets[r]].into_group()
            } else {
                G1Projective::zero()
            }
        })
        .collect()
}

/// Tier-1 row commitments of a one-hot polynomial: row `idx / cols` receives
/// generator `g1[idx % cols]` for every set entry `idx = k*T + t`.
pub(super) fn one_hot_row_commitments(
    nonzero_indices: &[Option<u16>],
    t_len: usize,
    cols: usize,
    num_rows: usize,
    g1: &[G1Affine],
) -> Vec<ArkG1> {
    // Counting sort of the set entries by row.
    let mut counts = vec![0usize; num_rows + 1];
    for (t, k_opt) in nonzero_indices.iter().enumerate() {
        if let Some(k) = k_opt {
            counts[(*k as usize * t_len + t) / cols + 1] += 1;
        }
    }
    for r in 0..num_rows {
        counts[r + 1] += counts[r];
    }
    let offsets = counts;
    let mut fill = offsets.clone();
    let mut points = vec![G1Affine::identity(); *offsets.last().unwrap()];
    for (t, k_opt) in nonzero_indices.iter().enumerate() {
        if let Some(k) = k_opt {
            let idx = *k as usize * t_len + t;
            let row = idx / cols;
            points[fill[row]] = g1[idx % cols];
            fill[row] += 1;
        }
    }
    batched_affine_row_sums(points, offsets)
        .into_iter()
        .map(ArkG1)
        .collect()
}

/// Tier-2 `Σ_i e(row_i, g2_i)` skipping identity rows, with `g2_prepared`
/// the setup's Miller-loop-ready G2 generators.
pub(super) fn tier_2_skip_identity(
    rows: &[ArkG1],
    g2: &[ArkG2],
    g2_prepared: &[<Bn254 as Pairing>::G2Prepared],
) -> ArkGT {
    let live: Vec<usize> = (0..rows.len()).filter(|&i| !rows[i].0.is_zero()).collect();
    if live.is_empty() {
        return <ArkGT as DoryGroup>::identity();
    }
    if live.len() == rows.len() {
        return <BN254 as PairingCurve>::multi_pair_g2_setup(rows, g2);
    }
    let ps: Vec<<Bn254 as Pairing>::G1Prepared> = live
        .iter()
        .map(|&i| rows[i].0.into_affine().into())
        .collect();
    let qs: Vec<<Bn254 as Pairing>::G2Prepared> =
        live.iter().map(|&i| g2_prepared[i].clone()).collect();
    let ml = Bn254::multi_miller_loop(ps, qs);
    ArkGT(Bn254::final_exponentiation(ml).expect("final exponentiation"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::UniformRand;

    #[test]
    fn batched_affine_sums_match_projective() {
        let mut rng = ark_std::test_rng();
        let gens: Vec<G1Affine> = (0..64)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect();
        // 5 rows with 0, 1, 2, 7 and 33 points (33 includes a repeated generator).
        let sizes = [0usize, 1, 2, 7, 33];
        let mut offsets = vec![0usize];
        let mut points = Vec::new();
        let mut expected = Vec::new();
        for (r, &n) in sizes.iter().enumerate() {
            let mut acc = G1Projective::zero();
            for j in 0..n {
                let g = gens[(r * 7 + j * 3) % 64];
                points.push(g);
                acc += g;
            }
            expected.push(acc);
            offsets.push(points.len());
        }
        let got = batched_affine_row_sums(points, offsets);
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!(g.into_affine(), e.into_affine());
        }
    }
}
