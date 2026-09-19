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
use crate::poly::compact_indices::IndexSlice;
use ark_bn254::{Bn254, Fq, G1Affine, G1Projective};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{batch_inversion, Field, Zero};
use dory::backends::arkworks::{ArkG1, ArkG2, ArkGT, BN254};
use dory::primitives::arithmetic::{Group as DoryGroup, PairingCurve};
use rayon::prelude::*;

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
    nonzero_indices: IndexSlice<'_, u16>,
    t_len: usize,
    cols: usize,
    num_rows: usize,
    g1: &[G1Affine],
    generator_sum: impl Fn() -> G1Projective + Sync,
) -> Vec<ArkG1> {
    // If a time vector spans several commitment columns, each aligned
    // segment contributes to disjoint rows for every address. Compute those
    // segments independently without collecting all nonzero points at once.
    // Only one column of temporary points is live per running segment.
    if t_len >= cols {
        assert!(t_len.is_power_of_two() && cols.is_power_of_two());
        assert_eq!(nonzero_indices.len(), t_len);
        assert_eq!(g1.len(), cols);
        let segments = t_len / cols;
        assert!(num_rows.is_multiple_of(segments));
        let buckets = num_rows / segments;
        let pieces = nonzero_indices
            .par_chunks(cols)
            .map(|indices| {
                let mut offsets = vec![0usize; buckets + 1];
                let mut present = 0;
                for k in indices.iter().flatten() {
                    offsets[usize::from(*k) + 1] += 1;
                    present += 1;
                }
                // A complete segment partitions exactly the public column
                // generators. Recover a frequent row from that fixed sum.
                // For small or uniform groups, keep the existing additions.
                let recovered = if present == cols {
                    offsets[1..]
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, count)| *count)
                        .filter(|(_, count)| **count >= 32 && **count >= cols / 2)
                        .map(|(bucket, _)| bucket)
                } else {
                    None
                };
                if let Some(bucket) = recovered {
                    offsets[bucket + 1] = 0;
                }
                for k in 0..buckets {
                    offsets[k + 1] += offsets[k];
                }
                let mut fill = offsets.clone();
                let mut points = vec![G1Affine::identity(); offsets[buckets]];
                for (column, k) in indices.iter().enumerate() {
                    if let Some(k) = k {
                        let bucket = usize::from(*k);
                        if Some(bucket) == recovered {
                            continue;
                        }
                        points[fill[bucket]] = g1[column];
                        fill[bucket] += 1;
                    }
                }
                let mut rows = batched_affine_row_sums(points, offsets);
                if let Some(bucket) = recovered {
                    let mut sum = generator_sum();
                    for row in &rows {
                        sum -= row;
                    }
                    rows[bucket] = sum;
                }
                rows
            })
            .collect::<Vec<_>>();
        return (0..num_rows)
            .into_par_iter()
            .map(|row| ArkG1(pieces[row % segments][row / segments]))
            .collect();
    }
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
    fn segmented_one_hot_rows_match_direct_group_sums() {
        let mut rng = ark_std::test_rng();
        let mut generators = (0..64)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect::<Vec<_>>();
        generators[0] = G1Affine::identity();
        generators[2] = generators[1];
        generators[3] = -generators[1];
        for workers in [1, 8] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    for t_len in [1usize, 4, 16, 64, 256] {
                        for cols in [1usize, 4, 16, 64] {
                            for buckets in [1usize, 2, 16] {
                                if buckets * t_len < cols {
                                    continue;
                                }
                                let num_rows = buckets * t_len / cols;
                                for mode in 0..3 {
                                    let indices = (0..t_len)
                                        .map(|t| {
                                            if mode == 0 || (mode == 1 && t % 5 == 0) {
                                                None
                                            } else {
                                                Some(((t * 7 + t / 3) % buckets) as u16)
                                            }
                                        })
                                        .collect::<Vec<_>>();
                                    let mut expected = vec![G1Projective::zero(); num_rows];
                                    for (t, k) in indices.iter().enumerate() {
                                        if let Some(k) = k {
                                            let index = usize::from(*k) * t_len + t;
                                            expected[index / cols] += generators[index % cols];
                                        }
                                    }
                                    let actual = one_hot_row_commitments(
                                        indices.as_slice(),
                                        t_len,
                                        cols,
                                        num_rows,
                                        &generators[..cols],
                                        || {
                                            generators[..cols]
                                                .iter()
                                                .fold(G1Projective::zero(), |sum, g| sum + g)
                                        },
                                    );
                                    assert_eq!(
                                        actual.iter().map(|p| p.0).collect::<Vec<_>>(),
                                        expected
                                    );
                                }
                            }
                        }
                    }
                });
        }
    }

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

#[cfg(test)]
mod recovery_tests {
    use super::*;
    use ark_ec::PrimeGroup;
    use ark_serialize::CanonicalSerialize;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    };

    fn bases(cols: usize) -> Vec<G1Affine> {
        let generator = G1Projective::generator();
        let mut point = generator;
        let points = (0..cols)
            .map(|_| {
                point += generator;
                point
            })
            .collect::<Vec<_>>();
        G1Projective::normalize_batch(&points)
    }

    fn sum(gens: &[G1Affine]) -> G1Projective {
        gens.iter().fold(G1Projective::zero(), |s, p| s + p)
    }

    fn literal(
        indices: &[Option<u16>],
        cols: usize,
        buckets: usize,
        gens: &[G1Affine],
    ) -> Vec<ArkG1> {
        let t_len = indices.len();
        let mut rows = vec![ArkG1(G1Projective::zero()); buckets * t_len / cols];
        for (t, k) in indices.iter().enumerate() {
            if let Some(k) = k {
                let flat = usize::from(*k) * t_len + t;
                rows[flat / cols].0 += gens[flat % cols];
            }
        }
        rows
    }

    #[test]
    fn recovered_rows_match_literal_sums_and_original_kernel() {
        let mut gens = bases(128);
        gens[0] = G1Affine::identity();
        gens[2] = gens[1];
        gens[3] = -gens[1];
        for workers in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    for cols in [16, 64, 128] {
                        for t_len in [cols / 2, cols, cols * 4] {
                            for buckets in [2, 16, 256] {
                                for mode in 0..5 {
                                    let indices = (0..t_len)
                                        .map(|t| match mode {
                                            0 => None,
                                            1 => Some(0),
                                            2 => Some((t % 2) as u16),
                                            3 => Some(
                                                (if t % 16 == 0 {
                                                    t % buckets
                                                } else {
                                                    buckets - 1
                                                })
                                                    as u16,
                                            ),
                                            _ => Some((t % buckets) as u16),
                                        })
                                        .collect::<Vec<_>>();
                                    let compact = indices.clone();
                                    let expected = literal(&indices, cols, buckets, &gens);
                                    let cached = OnceLock::new();
                                    let calls = AtomicUsize::new(0);
                                    let generator_sum = || {
                                        *cached.get_or_init(|| {
                                            calls.fetch_add(1, Ordering::Relaxed);
                                            sum(&gens[..cols])
                                        })
                                    };
                                    let actual = one_hot_row_commitments(
                                        compact.as_slice(),
                                        t_len,
                                        cols,
                                        expected.len(),
                                        &gens[..cols],
                                        generator_sum,
                                    );
                                    assert_eq!(
                                        actual, expected,
                                        "cols={cols} t={t_len} k={buckets} mode={mode}"
                                    );
                                    assert_eq!(
                                        actual,
                                        original_one_hot_row_commitments(
                                            indices.as_slice(),
                                            t_len,
                                            cols,
                                            expected.len(),
                                            &gens[..cols]
                                        )
                                    );
                                    assert!(calls.load(Ordering::Relaxed) <= 1);
                                    if t_len < cols || mode == 0 {
                                        assert_eq!(calls.load(Ordering::Relaxed), 0);
                                    }
                                    if t_len >= cols && cols >= 64 && mode == 1 {
                                        assert_eq!(calls.load(Ordering::Relaxed), 1);
                                    }
                                }
                            }
                        }
                    }
                });
        }
    }

    #[test]
    fn recovered_rows_preserve_every_absent_position() {
        let gens = bases(64);
        for absent in 0..64 {
            let mut indices = vec![Some(1); 64];
            indices[absent] = None;
            let actual = one_hot_row_commitments(indices.as_slice(), 64, 64, 2, &gens, || {
                panic!("A partial segment must not use the full generator sum")
            });
            assert_eq!(actual, literal(&indices, 64, 2, &gens));
        }
    }

    #[test]
    fn recovered_rows_use_each_segments_own_presence_condition() {
        let gens = bases(64);
        let mut indices = vec![Some(1); 256];
        indices[65] = None;
        indices[193] = None;
        let calls = AtomicUsize::new(0);
        let actual = one_hot_row_commitments(indices.as_slice(), 256, 64, 8, &gens, || {
            calls.fetch_add(1, Ordering::Relaxed);
            sum(&gens)
        });
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(actual, literal(&indices, 64, 2, &gens));
    }

    #[test]
    #[should_panic]
    fn recovered_rows_reject_an_address_beyond_the_matrix() {
        let gens = bases(64);
        let mut indices = vec![Some(0); 64];
        indices[63] = Some(2);
        one_hot_row_commitments(indices.as_slice(), 64, 64, 2, &gens, || sum(&gens));
    }

    #[test]
    #[ignore = "Isolated row kernel with synthetic public bases, not complete proof time"]
    fn recovered_rows_benchmark() {
        use sha3::{Digest, Keccak256};
        use std::time::Instant;
        let mode = std::env::var("ROW_MODE").unwrap();
        assert!(matches!(mode.as_str(), "original" | "candidate"));
        let distribution = std::env::var("ROW_DISTRIBUTION").unwrap();
        let cols = 1usize << 16;
        let t_len = 1usize << 22;
        let buckets = 256;
        let gens = bases(cols);
        let indices = (0..t_len)
            .map(|t| match distribution.as_str() {
                "constant" => Some(0),
                "skewed" => Some((if t % 8 == 0 { t / 8 % buckets } else { 0 }) as u16),
                "uniform" => Some((t % buckets) as u16),
                "missing" => {
                    if t % 17 == 0 {
                        None
                    } else {
                        Some(0)
                    }
                }
                _ => panic!("Unknown fixture"),
            })
            .collect::<Vec<_>>();
        let cached = OnceLock::new();
        let cache_nanos = std::sync::atomic::AtomicU64::new(0);
        let generator_sum = || {
            *cached.get_or_init(|| {
                let start = Instant::now();
                let value = sum(&gens);
                cache_nanos.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                value
            })
        };
        let call = || {
            if mode == "original" {
                original_one_hot_row_commitments(
                    indices.as_slice(),
                    t_len,
                    cols,
                    buckets * t_len / cols,
                    &gens,
                )
            } else {
                one_hot_row_commitments(
                    indices.as_slice(),
                    t_len,
                    cols,
                    buckets * t_len / cols,
                    &gens,
                    generator_sum,
                )
            }
        };
        let start = Instant::now();
        let cold = call();
        let cold_seconds = start.elapsed().as_secs_f64();
        let mut cold_bytes = Vec::new();
        cold.serialize_compressed(&mut cold_bytes).unwrap();
        drop(cold);
        let start = Instant::now();
        let warm = call();
        let seconds = start.elapsed().as_secs_f64();
        let mut bytes = Vec::new();
        warm.serialize_compressed(&mut bytes).unwrap();
        assert_eq!(cold_bytes, bytes);
        let digest = format!("{:x}", Keccak256::digest(&bytes));
        println!("ROW_BENCH {{\"mode\":\"{}\",\"distribution\":\"{}\",\"rows_seconds\":{},\"cold_rows_seconds\":{},\"cache_seconds\":{},\"cache_initialized\":{},\"row_bytes\":{},\"row_digest\":\"{}\",\"cols\":{},\"t_len\":{},\"buckets\":{},\"workers\":{}}}", mode, distribution, seconds, cold_seconds, cache_nanos.load(Ordering::Relaxed) as f64 / 1e9, cached.get().is_some(), bytes.len(), digest, cols, t_len, buckets, rayon::current_num_threads());
    }

    // The exact accepted control kernel is inserted below during preparation.
    fn original_one_hot_row_commitments(
        nonzero_indices: &[Option<u16>],
        t_len: usize,
        cols: usize,
        num_rows: usize,
        g1: &[G1Affine],
    ) -> Vec<ArkG1> {
        // If a time vector spans several commitment columns, each aligned
        // segment contributes to disjoint rows for every address. Compute those
        // segments independently without collecting all nonzero points at once.
        // Only one column of temporary points is live per running segment.
        if t_len > cols {
            assert!(t_len.is_power_of_two() && cols.is_power_of_two());
            assert_eq!(nonzero_indices.len(), t_len);
            assert_eq!(g1.len(), cols);
            let segments = t_len / cols;
            assert!(num_rows.is_multiple_of(segments));
            let buckets = num_rows / segments;
            let pieces = nonzero_indices
                .par_chunks(cols)
                .map(|indices| {
                    let mut offsets = vec![0usize; buckets + 1];
                    for k in indices.iter().flatten() {
                        offsets[usize::from(*k) + 1] += 1;
                    }
                    for k in 0..buckets {
                        offsets[k + 1] += offsets[k];
                    }
                    let mut fill = offsets.clone();
                    let mut points = vec![G1Affine::identity(); offsets[buckets]];
                    for (column, k) in indices.iter().enumerate() {
                        if let Some(k) = k {
                            let bucket = usize::from(*k);
                            points[fill[bucket]] = g1[column];
                            fill[bucket] += 1;
                        }
                    }
                    batched_affine_row_sums(points, offsets)
                })
                .collect::<Vec<_>>();
            return (0..num_rows)
                .into_par_iter()
                .map(|row| ArkG1(pieces[row % segments][row / segments]))
                .collect();
        }
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
}
