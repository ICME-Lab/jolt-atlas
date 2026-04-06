//! Row-wise binary MSM using Bucket (XYZZ coordinates)

use ark_bn254::g1::Config as G1Config;
use ark_bn254::{G1Affine, G1Projective};
use ark_ec::{models::short_weierstrass::Bucket, CurveGroup};
use rayon::prelude::*;

use crate::small_row::SmallRow;

/// Computes row-wise binary MSM using Bucket (XYZZ) coordinates, returning projective points.
///
/// # Arguments
/// * `key` - Fixed G1Affine key of length n (column basis points)
/// * `rows` - Slice of SmallRow; each entry holds indices into `key`
/// * `k_hint` - Runtime hint for max row size (used to tune ILP)
///
/// # Returns
/// Vector of G1Projective of length n (row sums in projective form)
pub fn msm_rows_bucket_projective(
    key: &[G1Affine],
    rows: &[SmallRow],
    k_hint: usize,
) -> Vec<G1Projective> {
    #[inline(always)]
    fn ilp_from_k(k: usize) -> usize {
        match k {
            0..=64 => 2,
            65..=256 => 4,
            257..=1024 => 6,
            _ => 8,
        }
    }

    let ilp = ilp_from_k(k_hint);

    rows.par_iter()
        .map(|row| {
            let mut acc = Bucket::<G1Config>::ZERO;

            if row.is_u16() {
                let s = row.as_u16_slice();
                let mut chunks = s.chunks_exact(ilp);

                for ch in &mut chunks {
                    acc += key[ch[0] as usize];
                    if ilp > 1 {
                        acc += key[ch[1] as usize];
                    }
                    if ilp > 2 {
                        acc += key[ch[2] as usize];
                    }
                    if ilp > 3 {
                        acc += key[ch[3] as usize];
                    }
                    if ilp > 4 {
                        acc += key[ch[4] as usize];
                    }
                    if ilp > 5 {
                        acc += key[ch[5] as usize];
                    }
                    if ilp > 6 {
                        acc += key[ch[6] as usize];
                    }
                    if ilp > 7 {
                        acc += key[ch[7] as usize];
                    }
                }

                for &j in chunks.remainder() {
                    acc += key[j as usize];
                }
            } else {
                let s = row.as_u32_slice();
                let mut chunks = s.chunks_exact(ilp);

                for ch in &mut chunks {
                    acc += key[ch[0] as usize];
                    if ilp > 1 {
                        acc += key[ch[1] as usize];
                    }
                    if ilp > 2 {
                        acc += key[ch[2] as usize];
                    }
                    if ilp > 3 {
                        acc += key[ch[3] as usize];
                    }
                    if ilp > 4 {
                        acc += key[ch[4] as usize];
                    }
                    if ilp > 5 {
                        acc += key[ch[5] as usize];
                    }
                    if ilp > 6 {
                        acc += key[ch[6] as usize];
                    }
                    if ilp > 7 {
                        acc += key[ch[7] as usize];
                    }
                }

                for &j in chunks.remainder() {
                    acc += key[j as usize];
                }
            }

            acc.into()
        })
        .collect()
}

/// Computes row-wise binary MSM using Bucket (XYZZ), returning affine points.
///
/// # Arguments
/// * `key` - Fixed G1Affine key of length n (column basis points)
/// * `rows` - Slice of SmallRow; each entry holds indices into `key`
/// * `k_hint` - Runtime hint for max row size (used to tune ILP)
///
/// # Returns
/// Vector of G1Affine of length n (row sums in affine form)
pub fn msm_rows_bucket_affine(key: &[G1Affine], rows: &[SmallRow], k_hint: usize) -> Vec<G1Affine> {
    let proj = msm_rows_bucket_projective(key, rows, k_hint);
    G1Projective::normalize_batch(&proj)
}

/// Computes row-wise binary MSM from sparse one-hot indices, returning projective points.
///
/// # Returns
/// Vector of G1Projective (row sums in projective form)
pub fn batch_addition_matrix(
    bases: &[G1Affine],
    nonzero_indices: &[Option<usize>],
    cycles_per_row: usize,
    k: usize,
    row_len: usize,
) -> Vec<G1Projective> {
    #[inline(always)]
    fn ilp_from_k(k: usize) -> usize {
        match k {
            0..=64 => 2,
            65..=256 => 4,
            257..=1024 => 6,
            _ => 8,
        }
    }

    let ilp = ilp_from_k(k);
    let use_u16 = row_len <= 65_536;

    nonzero_indices
        .par_chunks(cycles_per_row)
        .map(|chunk| {
            let mut acc = Bucket::<G1Config>::ZERO;

            if use_u16 {
                let indices: Vec<u16> = chunk
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &opt)| opt.map(|kopt| (i * k + kopt) as u16))
                    .collect();

                let mut chunks_iter = indices.chunks_exact(ilp);
                for ch in &mut chunks_iter {
                    acc += bases[ch[0] as usize];
                    if ilp > 1 {
                        acc += bases[ch[1] as usize];
                    }
                    if ilp > 2 {
                        acc += bases[ch[2] as usize];
                    }
                    if ilp > 3 {
                        acc += bases[ch[3] as usize];
                    }
                    if ilp > 4 {
                        acc += bases[ch[4] as usize];
                    }
                    if ilp > 5 {
                        acc += bases[ch[5] as usize];
                    }
                    if ilp > 6 {
                        acc += bases[ch[6] as usize];
                    }
                    if ilp > 7 {
                        acc += bases[ch[7] as usize];
                    }
                }
                for &j in chunks_iter.remainder() {
                    acc += bases[j as usize];
                }
            } else {
                let indices: Vec<u32> = chunk
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &opt)| opt.map(|kopt| (i * k + kopt) as u32))
                    .collect();

                let mut chunks_iter = indices.chunks_exact(ilp);
                for ch in &mut chunks_iter {
                    acc += bases[ch[0] as usize];
                    if ilp > 1 {
                        acc += bases[ch[1] as usize];
                    }
                    if ilp > 2 {
                        acc += bases[ch[2] as usize];
                    }
                    if ilp > 3 {
                        acc += bases[ch[3] as usize];
                    }
                    if ilp > 4 {
                        acc += bases[ch[4] as usize];
                    }
                    if ilp > 5 {
                        acc += bases[ch[5] as usize];
                    }
                    if ilp > 6 {
                        acc += bases[ch[6] as usize];
                    }
                    if ilp > 7 {
                        acc += bases[ch[7] as usize];
                    }
                }
                for &j in chunks_iter.remainder() {
                    acc += bases[j as usize];
                }
            }

            acc.into()
        })
        .collect()
}

/// Computes row-wise binary MSM from sparse one-hot u8 indices, returning projective points.
///
/// Memory-efficient variant for small k values (k < 256). Uses u8 for nonzero indices
/// to reduce memory footprint.
///
/// # Returns
/// Vector of G1Projective (row sums in projective form)
pub fn batch_addition_matrix_u8(
    bases: &[G1Affine],
    nonzero_indices: &[Option<u8>],
    cycles_per_row: usize,
    k: usize,
    row_len: usize,
) -> Vec<G1Projective> {
    #[inline(always)]
    fn ilp_from_k(k: usize) -> usize {
        match k {
            0..=64 => 2,
            65..=256 => 4,
            257..=1024 => 6,
            _ => 8,
        }
    }

    let ilp = ilp_from_k(k);
    let use_u16 = row_len <= 65_536;

    nonzero_indices
        .par_chunks(cycles_per_row)
        .map(|chunk| {
            let mut acc = Bucket::<G1Config>::ZERO;

            if use_u16 {
                let indices: Vec<u16> = chunk
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &opt)| opt.map(|kopt| (i * k + kopt as usize) as u16))
                    .collect();

                let mut chunks_iter = indices.chunks_exact(ilp);
                for ch in &mut chunks_iter {
                    acc += bases[ch[0] as usize];
                    if ilp > 1 {
                        acc += bases[ch[1] as usize];
                    }
                    if ilp > 2 {
                        acc += bases[ch[2] as usize];
                    }
                    if ilp > 3 {
                        acc += bases[ch[3] as usize];
                    }
                    if ilp > 4 {
                        acc += bases[ch[4] as usize];
                    }
                    if ilp > 5 {
                        acc += bases[ch[5] as usize];
                    }
                    if ilp > 6 {
                        acc += bases[ch[6] as usize];
                    }
                    if ilp > 7 {
                        acc += bases[ch[7] as usize];
                    }
                }
                for &j in chunks_iter.remainder() {
                    acc += bases[j as usize];
                }
            } else {
                let indices: Vec<u32> = chunk
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &opt)| opt.map(|kopt| (i * k + kopt as usize) as u32))
                    .collect();

                let mut chunks_iter = indices.chunks_exact(ilp);
                for ch in &mut chunks_iter {
                    acc += bases[ch[0] as usize];
                    if ilp > 1 {
                        acc += bases[ch[1] as usize];
                    }
                    if ilp > 2 {
                        acc += bases[ch[2] as usize];
                    }
                    if ilp > 3 {
                        acc += bases[ch[3] as usize];
                    }
                    if ilp > 4 {
                        acc += bases[ch[4] as usize];
                    }
                    if ilp > 5 {
                        acc += bases[ch[5] as usize];
                    }
                    if ilp > 6 {
                        acc += bases[ch[6] as usize];
                    }
                    if ilp > 7 {
                        acc += bases[ch[7] as usize];
                    }
                }
                for &j in chunks_iter.remainder() {
                    acc += bases[j as usize];
                }
            }

            acc.into()
        })
        .collect()
}

/// Computes row-wise binary MSM from sparse one-hot u8 indices with variable row lengths.
///
/// Memory-efficient variant for small k values (k < 256) that supports variable row lengths.
///
/// # Arguments
/// * `bases` - Basis points to accumulate
/// * `nonzero_indices` - Flat array of sparse one-hot indices
/// * `row_lengths` - Length of each row (number of cycles per row)
/// * `k` - Size of one-hot encoding space
/// * `row_len` - Maximum addressable index (for choosing u16 vs u32 internally)
///
/// # Returns
/// Vector of G1Projective (row sums in projective form)
pub fn batch_addition_matrix_u8_variable(
    bases: &[G1Affine],
    nonzero_indices: &[Option<u8>],
    row_lengths: &[usize],
    k: usize,
    row_len: usize,
) -> Vec<G1Projective> {
    #[inline(always)]
    fn ilp_from_k(k: usize) -> usize {
        match k {
            0..=64 => 2,
            65..=256 => 4,
            257..=1024 => 6,
            _ => 8,
        }
    }

    let ilp = ilp_from_k(k);
    let use_u16 = row_len <= 65_536;

    let mut offsets = Vec::with_capacity(row_lengths.len() + 1);
    offsets.push(0);
    for &len in row_lengths {
        offsets.push(offsets.last().unwrap() + len);
    }

    row_lengths
        .par_iter()
        .enumerate()
        .map(|(row_idx, _)| {
            let start = offsets[row_idx];
            let end = offsets[row_idx + 1];
            let chunk = &nonzero_indices[start..end];
            let mut acc = Bucket::<G1Config>::ZERO;

            if use_u16 {
                let indices: Vec<u16> = chunk
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &opt)| opt.map(|kopt| (i * k + kopt as usize) as u16))
                    .collect();

                let mut chunks_iter = indices.chunks_exact(ilp);
                for ch in &mut chunks_iter {
                    acc += bases[ch[0] as usize];
                    if ilp > 1 {
                        acc += bases[ch[1] as usize];
                    }
                    if ilp > 2 {
                        acc += bases[ch[2] as usize];
                    }
                    if ilp > 3 {
                        acc += bases[ch[3] as usize];
                    }
                    if ilp > 4 {
                        acc += bases[ch[4] as usize];
                    }
                    if ilp > 5 {
                        acc += bases[ch[5] as usize];
                    }
                    if ilp > 6 {
                        acc += bases[ch[6] as usize];
                    }
                    if ilp > 7 {
                        acc += bases[ch[7] as usize];
                    }
                }
                for &j in chunks_iter.remainder() {
                    acc += bases[j as usize];
                }
            } else {
                let indices: Vec<u32> = chunk
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &opt)| opt.map(|kopt| (i * k + kopt as usize) as u32))
                    .collect();

                let mut chunks_iter = indices.chunks_exact(ilp);
                for ch in &mut chunks_iter {
                    acc += bases[ch[0] as usize];
                    if ilp > 1 {
                        acc += bases[ch[1] as usize];
                    }
                    if ilp > 2 {
                        acc += bases[ch[2] as usize];
                    }
                    if ilp > 3 {
                        acc += bases[ch[3] as usize];
                    }
                    if ilp > 4 {
                        acc += bases[ch[4] as usize];
                    }
                    if ilp > 5 {
                        acc += bases[ch[5] as usize];
                    }
                    if ilp > 6 {
                        acc += bases[ch[6] as usize];
                    }
                    if ilp > 7 {
                        acc += bases[ch[7] as usize];
                    }
                }
                for &j in chunks_iter.remainder() {
                    acc += bases[j as usize];
                }
            }

            acc.into()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::{rand::RngCore, UniformRand};

    #[test]
    fn test_msm_bucket_correctness() {
        let mut rng = ark_std::test_rng();

        let n = 100;
        let k = 50;

        let key: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();

        let rows: Vec<SmallRow> = (0..n)
            .map(|_| {
                let num_indices = (rng.next_u64() as usize) % k + 1;
                let indices: Vec<u16> = (0..num_indices)
                    .map(|_| (rng.next_u64() as u16) % (n as u16))
                    .collect();
                SmallRow::from_u16(indices)
            })
            .collect();

        let result = msm_rows_bucket_affine(&key, &rows, k);

        for (row_idx, row) in rows.iter().enumerate() {
            let mut expected = G1Affine::identity();
            for idx in row.iter_usize() {
                expected = (expected + key[idx]).into();
            }
            assert_eq!(
                result[row_idx], expected,
                "Bucket MSM mismatch at row {}",
                row_idx
            );
        }
    }

    #[test]
    fn test_msm_bucket_projective() {
        let mut rng = ark_std::test_rng();

        let n = 50;
        let k = 30;

        let key: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();

        let rows: Vec<SmallRow> = (0..n)
            .map(|_| {
                let num_indices = (rng.next_u64() as usize) % k + 1;
                let indices: Vec<u16> = (0..num_indices)
                    .map(|_| (rng.next_u64() as u16) % (n as u16))
                    .collect();
                SmallRow::from_u16(indices)
            })
            .collect();

        let proj = msm_rows_bucket_projective(&key, &rows, k);
        let result = G1Projective::normalize_batch(&proj);

        for (row_idx, row) in rows.iter().enumerate() {
            let mut expected = G1Affine::identity();
            for idx in row.iter_usize() {
                expected = (expected + key[idx]).into();
            }
            assert_eq!(
                result[row_idx], expected,
                "Bucket projective MSM mismatch at row {}",
                row_idx
            );
        }
    }

    #[test]
    fn test_msm_sparse_streaming() {
        let mut rng = ark_std::test_rng();

        let k = 8;
        let cycles_per_row = 100;
        let num_rows = 50;
        let row_len = k * cycles_per_row;

        let bases: Vec<G1Affine> = (0..row_len).map(|_| G1Affine::rand(&mut rng)).collect();

        let nonzero_indices: Vec<Option<usize>> = (0..num_rows * cycles_per_row)
            .map(|_| {
                if rng.next_u64() % 2 == 0 {
                    Some((rng.next_u64() as usize) % k)
                } else {
                    None
                }
            })
            .collect();

        let result = batch_addition_matrix(&bases, &nonzero_indices, cycles_per_row, k, row_len);

        for (row_idx, chunk) in nonzero_indices.chunks(cycles_per_row).enumerate() {
            let mut expected = G1Affine::identity();
            for (i, &opt) in chunk.iter().enumerate() {
                if let Some(kopt) = opt {
                    let idx = i * k + kopt;
                    expected = (expected + bases[idx]).into();
                }
            }
            assert_eq!(
                result[row_idx].into_affine(),
                expected,
                "Sparse streaming MSM mismatch at row {}",
                row_idx
            );
        }
    }

    #[test]
    fn test_msm_sparse_streaming_vs_bucket() {
        let mut rng = ark_std::test_rng();

        let k = 16;
        let cycles_per_row = 200;
        let num_rows = 30;
        let row_len = k * cycles_per_row;

        let bases: Vec<G1Affine> = (0..row_len).map(|_| G1Affine::rand(&mut rng)).collect();

        let nonzero_indices: Vec<Option<usize>> = (0..num_rows * cycles_per_row)
            .map(|_| {
                if rng.next_u64() % 3 == 0 {
                    Some((rng.next_u64() as usize) % k)
                } else {
                    None
                }
            })
            .collect();

        let sparse_result =
            batch_addition_matrix(&bases, &nonzero_indices, cycles_per_row, k, row_len);

        let rows: Vec<SmallRow> = nonzero_indices
            .chunks(cycles_per_row)
            .map(|chunk| {
                let indices: Vec<u16> = chunk
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &opt)| opt.map(|kopt| (i * k + kopt) as u16))
                    .collect();
                SmallRow::from_u16(indices)
            })
            .collect();

        let bucket_result = msm_rows_bucket_projective(&bases, &rows, k);

        for (idx, (sparse, bucket)) in sparse_result.iter().zip(bucket_result.iter()).enumerate() {
            assert_eq!(
                sparse.into_affine(),
                bucket.into_affine(),
                "Sparse vs bucket mismatch at row {}",
                idx
            );
        }
    }

    #[test]
    fn test_batch_addition_matrix_u8() {
        let mut rng = ark_std::test_rng();

        let k = 16;
        let cycles_per_row = 150;
        let num_rows = 40;
        let row_len = k * cycles_per_row;

        let bases: Vec<G1Affine> = (0..row_len).map(|_| G1Affine::rand(&mut rng)).collect();

        let nonzero_indices_u8: Vec<Option<u8>> = (0..num_rows * cycles_per_row)
            .map(|_| {
                if rng.next_u64() % 3 == 0 {
                    Some((rng.next_u64() as u8) % (k as u8))
                } else {
                    None
                }
            })
            .collect();

        let nonzero_indices_usize: Vec<Option<usize>> = nonzero_indices_u8
            .iter()
            .map(|&opt| opt.map(|x| x as usize))
            .collect();

        let result_u8 =
            batch_addition_matrix_u8(&bases, &nonzero_indices_u8, cycles_per_row, k, row_len);
        let result_usize =
            batch_addition_matrix(&bases, &nonzero_indices_usize, cycles_per_row, k, row_len);

        for (idx, (r_u8, r_usize)) in result_u8.iter().zip(result_usize.iter()).enumerate() {
            assert_eq!(
                r_u8.into_affine(),
                r_usize.into_affine(),
                "u8 vs usize mismatch at row {}",
                idx
            );
        }
    }

    #[test]
    fn test_batch_addition_matrix_u8_variable() {
        let mut rng = ark_std::test_rng();

        let k = 16;
        let num_rows = 40;

        let row_lengths: Vec<usize> = (0..num_rows)
            .map(|_| 50 + (rng.next_u64() as usize) % 200)
            .collect();

        let total_elements: usize = row_lengths.iter().sum();
        let max_row_len = *row_lengths.iter().max().unwrap();
        let row_len = k * max_row_len;

        let bases: Vec<G1Affine> = (0..row_len).map(|_| G1Affine::rand(&mut rng)).collect();

        let nonzero_indices_u8: Vec<Option<u8>> = (0..total_elements)
            .map(|_| {
                if rng.next_u64() % 3 == 0 {
                    Some((rng.next_u64() as u8) % (k as u8))
                } else {
                    None
                }
            })
            .collect();

        let result_variable = batch_addition_matrix_u8_variable(
            &bases,
            &nonzero_indices_u8,
            &row_lengths,
            k,
            row_len,
        );

        let mut offset = 0;
        for (row_idx, &row_len_cur) in row_lengths.iter().enumerate() {
            let chunk = &nonzero_indices_u8[offset..offset + row_len_cur];
            let mut expected = G1Affine::identity();
            for (i, &opt) in chunk.iter().enumerate() {
                if let Some(kopt) = opt {
                    let idx = i * k + kopt as usize;
                    expected = (expected + bases[idx]).into();
                }
            }
            assert_eq!(
                result_variable[row_idx].into_affine(),
                expected,
                "Variable version mismatch at row {}",
                row_idx
            );
            offset += row_len_cur;
        }
    }

    #[test]
    fn test_batch_addition_matrix_u8_variable_vs_fixed() {
        let mut rng = ark_std::test_rng();

        let k = 16;
        let cycles_per_row = 150;
        let num_rows = 40;
        let row_len = k * cycles_per_row;

        let bases: Vec<G1Affine> = (0..row_len).map(|_| G1Affine::rand(&mut rng)).collect();

        let nonzero_indices_u8: Vec<Option<u8>> = (0..num_rows * cycles_per_row)
            .map(|_| {
                if rng.next_u64() % 3 == 0 {
                    Some((rng.next_u64() as u8) % (k as u8))
                } else {
                    None
                }
            })
            .collect();

        let result_fixed =
            batch_addition_matrix_u8(&bases, &nonzero_indices_u8, cycles_per_row, k, row_len);

        let row_lengths = vec![cycles_per_row; num_rows];
        let result_variable = batch_addition_matrix_u8_variable(
            &bases,
            &nonzero_indices_u8,
            &row_lengths,
            k,
            row_len,
        );

        for (idx, (fixed, variable)) in result_fixed.iter().zip(result_variable.iter()).enumerate()
        {
            assert_eq!(
                fixed.into_affine(),
                variable.into_affine(),
                "Fixed vs variable mismatch at row {} (uniform row lengths)",
                idx
            );
        }
    }
}
