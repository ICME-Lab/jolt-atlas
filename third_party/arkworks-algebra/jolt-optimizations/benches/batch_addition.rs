use ark_bn254::G1Affine;
use ark_std::rand::RngCore;
use ark_std::UniformRand;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jolt_optimizations::{
    batch_addition_matrix, batch_addition_matrix_u8_variable, batch_g1_additions_multi,
    msm_rows_bucket_affine, msm_rows_bucket_projective, SmallRow,
};

fn bench_msm_bucket_vs_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm_comparison");
    let mut rng = ark_std::test_rng();
    group.sample_size(10);

    for &(n, k) in &[(4096, 128), (1 << 16, 512)] {
        let key: Vec<G1Affine> = (0..n).map(|_| G1Affine::rand(&mut rng)).collect();

        let rows_new: Vec<SmallRow> = if n < 65536 {
            (0..n)
                .map(|_| {
                    let num_indices = (rng.next_u64() as usize) % k + 1;
                    let indices: Vec<u16> = (0..num_indices)
                        .map(|_| (rng.next_u64() as usize % n) as u16)
                        .collect();
                    SmallRow::from_u16(indices)
                })
                .collect()
        } else {
            (0..n)
                .map(|_| {
                    let num_indices = (rng.next_u64() as usize) % k + 1;
                    let indices: Vec<u32> = (0..num_indices)
                        .map(|_| (rng.next_u64() as usize % n) as u32)
                        .collect();
                    SmallRow::from_u32(indices)
                })
                .collect()
        };

        let name = format!("n={}_k={}", n, k);

        group.bench_with_input(
            BenchmarkId::new("bucket_projective", &name),
            &(&key, &rows_new, k),
            |b, (key, rows, k)| {
                b.iter(|| black_box(msm_rows_bucket_projective(key, rows, *k)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bucket_affine", &name),
            &(&key, &rows_new, k),
            |b, (key, rows, k)| {
                b.iter(|| black_box(msm_rows_bucket_affine(key, rows, *k)));
            },
        );

        let indices_sets: Vec<Vec<usize>> = rows_new
            .iter()
            .map(|row| row.iter_usize().collect())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_additions_multi", &name),
            &(&key, &indices_sets),
            |b, (key, indices_sets)| {
                b.iter(|| black_box(batch_g1_additions_multi(key, indices_sets)));
            },
        );
    }

    group.finish();
}

fn bench_sparse_matrix_approaches(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matrix_large");
    let mut rng = ark_std::test_rng();
    group.sample_size(10);

    let cycles_per_row = 1 << 8;
    let k = 512;
    let row_len = k * cycles_per_row;
    let num_rows = 1 << 15;

    let bases: Vec<G1Affine> = (0..row_len).map(|_| G1Affine::rand(&mut rng)).collect();

    let nonzero_indices: Vec<Option<usize>> = (0..num_rows * cycles_per_row)
        .map(|_| {
            if rng.next_u64() % 4 == 0 {
                Some((rng.next_u64() as usize) % k)
            } else {
                None
            }
        })
        .collect();

    let indices_sets: Vec<Vec<usize>> = nonzero_indices
        .chunks(cycles_per_row)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .filter_map(|(i, &opt)| opt.map(|kopt| i * k + kopt))
                .collect()
        })
        .collect();

    group.bench_function("batch_addition_matrix", |b| {
        b.iter(|| {
            black_box(batch_addition_matrix(
                &bases,
                &nonzero_indices,
                cycles_per_row,
                k,
                row_len,
            ))
        });
    });

    group.bench_function("batch_g1_additions_multi", |b| {
        b.iter(|| black_box(batch_g1_additions_multi(&bases, &indices_sets)));
    });

    group.finish();
}

fn bench_row_dense_column_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_dense_column_sparse");
    let mut rng = ark_std::test_rng();
    group.sample_size(10);

    for &(num_rows, total_cols, k, density) in &[(1000, 1000, 32, 0.7)] {
        let bases: Vec<G1Affine> = (0..total_cols).map(|_| G1Affine::rand(&mut rng)).collect();

        let mut row_lengths = Vec::with_capacity(num_rows);
        let mut all_indices_u8 = Vec::new();
        let mut indices_sets = Vec::with_capacity(num_rows);

        for _ in 0..num_rows {
            let elements_in_row = (total_cols as f64 * density) as usize;
            let mut row_indices = Vec::with_capacity(elements_in_row);

            for _ in 0..elements_in_row {
                let col_idx = (rng.next_u64() as usize) % total_cols;
                row_indices.push(col_idx);
            }
            row_indices.sort_unstable();
            row_indices.dedup();

            let actual_row_len = row_indices.len();

            let max_cycle = row_indices.last().map(|&idx| idx / k).unwrap_or(0);
            let cycles_for_row = max_cycle + 1;
            row_lengths.push(cycles_for_row);

            let mut chunk_u8 = vec![None; cycles_for_row];
            for &idx in &row_indices {
                let cycle = idx / k;
                let kopt = (idx % k) as u8;
                chunk_u8[cycle] = Some(kopt);
            }

            all_indices_u8.extend(chunk_u8);
            indices_sets.push(row_indices);
        }

        let row_len = total_cols;

        let name = format!(
            "rows={}_cols={}_k={}_dens={}",
            num_rows, total_cols, k, density
        );

        group.bench_with_input(
            BenchmarkId::new("batch_addition_matrix_u8_variable", &name),
            &(&bases, &all_indices_u8, &row_lengths, k, row_len),
            |b, (bases, all_indices_u8, row_lengths, k, row_len)| {
                b.iter(|| {
                    black_box(batch_addition_matrix_u8_variable(
                        bases,
                        all_indices_u8,
                        row_lengths,
                        *k,
                        *row_len,
                    ))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batch_g1_additions_multi", &name),
            &(&bases, &indices_sets),
            |b, (bases, indices_sets)| {
                b.iter(|| black_box(batch_g1_additions_multi(bases, indices_sets)));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    // bench_msm_bucket_vs_batch,
    // bench_sparse_matrix_approaches,
    bench_row_dense_column_sparse
);
criterion_main!(benches);
