//! Immutable lookup indices. Width depends on the table size and whether
//! the vector contains absent entries; no representation choice is serialized.
use allocative::Allocative;
use rayon::prelude::*;
use std::{
    ops::{Index, Range},
    sync::Arc,
};

#[derive(Clone, Debug, PartialEq, Allocative)]
pub enum CompactIndices<I> {
    Direct(Vec<Option<I>>),
    Bytes {
        rows: Vec<u8>,
        decode: Vec<Option<I>>,
    },
    Shorts {
        rows: Vec<u16>,
        decode: Vec<Option<I>>,
    },
}

impl CompactIndices<u16> {
    pub fn new(indices: Vec<Option<u16>>, table_size: usize) -> Self {
        assert!((1..=65536).contains(&table_size));
        let mut all_present = true;
        for index in &indices {
            match index {
                Some(i) => assert!(usize::from(*i) < table_size),
                None => all_present = false,
            }
        }
        // At an encoding boundary, a fully populated vector needs no absent
        // code. Sparse vectors retain the original representation below.
        if all_present && matches!(table_size, 256 | 65536) {
            let width = if table_size == 256 { 1 } else { 2 };
            let direct = indices.len() * std::mem::size_of::<Option<u16>>();
            let encoded = indices.len() * width + table_size * std::mem::size_of::<Option<u16>>();
            if encoded < direct {
                let decode = (0..table_size).map(|i| Some(i as u16)).collect();
                if table_size == 256 {
                    let mut rows = Vec::with_capacity(indices.len());
                    rows.extend(indices.into_iter().map(|i| i.unwrap() as u8));
                    return Self::Bytes { rows, decode };
                }
                let mut rows = Vec::with_capacity(indices.len());
                rows.extend(indices.into_iter().map(|i| i.unwrap()));
                return Self::Shorts { rows, decode };
            }
        }
        let width = if table_size < 256 {
            1
        } else if table_size < 65536 {
            2
        } else {
            4
        };
        let direct = indices.len() * std::mem::size_of::<Option<u16>>();
        let encoded = indices.len() * width + (table_size + 1) * std::mem::size_of::<Option<u16>>();
        if encoded >= direct {
            return Self::Direct(indices);
        }
        // Code zero is the absent entry. The largest address remains distinct.
        if table_size < 256 {
            let decode = std::iter::once(None)
                .chain((0..table_size).map(|i| Some(i as u16)))
                .collect();
            let mut rows = Vec::with_capacity(indices.len());
            rows.extend(indices.into_iter().map(|i| i.map_or(0, |i| (i + 1) as u8)));
            Self::Bytes { rows, decode }
        } else if table_size < 65536 {
            let decode = std::iter::once(None)
                .chain((0..table_size).map(|i| Some(i as u16)))
                .collect();
            let mut rows = Vec::with_capacity(indices.len());
            rows.extend(indices.into_iter().map(|i| i.map_or(0, |i| i + 1)));
            Self::Shorts { rows, decode }
        } else {
            Self::Direct(indices)
        }
    }
}
impl<I> CompactIndices<I> {
    pub fn len(&self) -> usize {
        match self {
            Self::Direct(v) => v.len(),
            Self::Bytes { rows, .. } => rows.len(),
            Self::Shorts { rows, .. } => rows.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, index: usize) -> Option<&Option<I>> {
        match self {
            Self::Direct(v) => v.get(index),
            Self::Bytes { rows, decode } => rows.get(index).map(|i| &decode[usize::from(*i)]),
            Self::Shorts { rows, decode } => rows.get(index).map(|i| &decode[usize::from(*i)]),
        }
    }
    pub fn allocated_bytes(&self) -> usize {
        match self {
            Self::Direct(v) => v.capacity() * std::mem::size_of::<Option<I>>(),
            Self::Bytes { rows, decode } => {
                rows.capacity() + decode.capacity() * std::mem::size_of::<Option<I>>()
            }
            Self::Shorts { rows, decode } => {
                rows.capacity() * 2 + decode.capacity() * std::mem::size_of::<Option<I>>()
            }
        }
    }
    pub fn as_slice(&self) -> IndexSlice<'_, I> {
        IndexSlice::Compact(self, 0, self.len())
    }
    pub fn slice(&self, range: Range<usize>) -> IndexSlice<'_, I> {
        self.as_slice().slice(range)
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Option<I>> + DoubleEndedIterator {
        (0..self.len()).map(|i| &self[i])
    }
}
impl<I: Sync> CompactIndices<I> {
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = &Option<I>> {
        (0..self.len()).into_par_iter().map(|i| &self[i])
    }
    pub fn par_chunks(
        &self,
        size: usize,
    ) -> impl IndexedParallelIterator<Item = IndexSlice<'_, I>> {
        self.as_slice().par_chunks(size)
    }
}
impl<I> Index<usize> for CompactIndices<I> {
    type Output = Option<I>;
    fn index(&self, i: usize) -> &Self::Output {
        self.get(i).expect("lookup index out of bounds")
    }
}

#[derive(Clone, Copy)]
pub enum IndexSlice<'a, I> {
    Direct(&'a [Option<I>]),
    Compact(&'a CompactIndices<I>, usize, usize),
}
impl<'a, I> From<&'a [Option<I>]> for IndexSlice<'a, I> {
    fn from(v: &'a [Option<I>]) -> Self {
        Self::Direct(v)
    }
}
impl<'a, I> IndexSlice<'a, I> {
    pub fn len(&self) -> usize {
        match self {
            Self::Direct(v) => v.len(),
            Self::Compact(_, start, end) => end - start,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, i: usize) -> Option<&'a Option<I>> {
        match self {
            Self::Direct(v) => v.get(i),
            Self::Compact(v, start, end) => {
                if i < end - start {
                    v.get(start + i)
                } else {
                    None
                }
            }
        }
    }
    pub fn slice(&self, range: Range<usize>) -> Self {
        assert!(range.start <= range.end && range.end <= self.len());
        match self {
            Self::Direct(v) => Self::Direct(&v[range]),
            Self::Compact(v, start, _) => Self::Compact(v, start + range.start, start + range.end),
        }
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &'a Option<I>> + DoubleEndedIterator + '_ {
        (0..self.len()).map(|i| self.get(i).unwrap())
    }
}
impl<'a, I: Sync> IndexSlice<'a, I> {
    pub fn par_chunks(self, size: usize) -> impl IndexedParallelIterator<Item = Self> {
        assert!(size > 0);
        (0..self.len().div_ceil(size))
            .into_par_iter()
            .map(move |i| self.slice(i * size..((i + 1) * size).min(self.len())))
    }
}
impl<I> Index<usize> for IndexSlice<'_, I> {
    type Output = Option<I>;
    fn index(&self, i: usize) -> &Self::Output {
        self.get(i).expect("lookup slice index out of bounds")
    }
}

/// Existing generic sumchecks may keep their original index vectors. Sparse
/// openings can share the compact vector without decoding another full copy.
#[derive(Clone, Debug, PartialEq, Allocative)]
pub enum SharedIndices<I> {
    Direct(Arc<Vec<Option<I>>>),
    Compact(Arc<CompactIndices<I>>),
}
impl<I> From<Arc<Vec<Option<I>>>> for SharedIndices<I> {
    fn from(v: Arc<Vec<Option<I>>>) -> Self {
        Self::Direct(v)
    }
}
impl<I> From<Arc<CompactIndices<I>>> for SharedIndices<I> {
    fn from(v: Arc<CompactIndices<I>>) -> Self {
        Self::Compact(v)
    }
}
impl<I> SharedIndices<I> {
    pub fn len(&self) -> usize {
        match self {
            Self::Direct(v) => v.len(),
            Self::Compact(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, i: usize) -> Option<&Option<I>> {
        match self {
            Self::Direct(v) => v.get(i),
            Self::Compact(v) => v.get(i),
        }
    }
}
impl<I> Index<usize> for SharedIndices<I> {
    type Output = Option<I>;
    fn index(&self, i: usize) -> &Self::Output {
        self.get(i).expect("shared lookup index out of bounds")
    }
}

impl<I> Default for SharedIndices<I> {
    fn default() -> Self {
        Self::Direct(Arc::new(Vec::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        field::JoltField,
        poly::{
            multilinear_polynomial::{BindingOrder, PolynomialBinding},
            one_hot_polynomial::OneHotPolynomial,
            ra_poly::RaPolynomial,
        },
        transcripts::{Blake2bTranscript, Transcript},
    };
    use ark_bn254::Fr;
    use std::time::Instant;

    #[test]
    fn compact_indices_preserve_absence_all_addresses_and_views() {
        for table in [1, 2, 4, 16, 255, 256, 257, 65535, 65536] {
            let input: Vec<_> = std::iter::once(None)
                .chain((0..table).map(|i| Some(i as u16)))
                .chain([None, Some(0), Some((table - 1) as u16)])
                .cycle()
                .take((4 * table + 4).max(8192))
                .collect();
            let packed = CompactIndices::new(input.clone(), table);
            assert_eq!(packed.len(), input.len());
            assert_eq!(packed.iter().copied().collect::<Vec<_>>(), input);
            assert_eq!(packed.par_iter().copied().collect::<Vec<_>>(), input);
            assert_eq!(packed.get(input.len()), None);
            for chunk in [1, 7, 257, 8192] {
                let parts: Vec<Vec<_>> = packed
                    .par_chunks(chunk)
                    .map(|v| v.iter().copied().collect())
                    .collect();
                assert_eq!(parts.into_iter().flatten().collect::<Vec<_>>(), input);
            }
            assert_eq!(
                packed
                    .slice(1..input.len() - 1)
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                input[1..input.len() - 1]
            );
            assert!(packed.allocated_bytes() <= input.len() * std::mem::size_of::<Option<u16>>());
            if table == 65535 {
                assert!(matches!(packed, CompactIndices::Shorts { .. }));
                assert_eq!(packed[65535], Some(65534));
            }
            if table == 65536 {
                assert!(matches!(packed, CompactIndices::Direct(_)));
                assert_eq!(packed[65536], Some(65535));
            }
        }
    }

    #[test]
    fn compact_indices_match_independent_polynomial_evaluations_and_all_binding_rounds() {
        for workers in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(|| {
                    for (table, all_present) in [2usize, 16, 256, 65536]
                        .into_iter()
                        .flat_map(|table| [(table, false), (table, true)])
                    {
                        let rows = if table == 65536 && all_present {
                            1usize << 18
                        } else {
                            8192usize
                        };
                        let input: Vec<_> = (0..rows)
                            .map(|i| match i % 5 {
                                0 if !all_present => None,
                                1 => Some(0),
                                2 => Some((table - 1) as u16),
                                _ => Some((i % table) as u16),
                            })
                            .collect();
                        let polynomial = OneHotPolynomial::<Fr>::from_indices(input.clone(), table);
                        let mut tr = Blake2bTranscript::new(b"compact-indices-independent");
                        let point: Vec<Fr> =
                            tr.challenge_vector(table.ilog2() as usize + rows.ilog2() as usize);
                        let (address, cycle) = point.split_at(table.ilog2() as usize);
                        let weight = |bits: &[Fr], index: usize| {
                            bits.iter()
                                .enumerate()
                                .map(|(j, r)| {
                                    if index >> (bits.len() - 1 - j) & 1 == 1 {
                                        *r
                                    } else {
                                        Fr::from(1u64) - r
                                    }
                                })
                                .product::<Fr>()
                        };
                        let expected: Fr = input
                            .iter()
                            .enumerate()
                            .filter_map(|(t, k)| {
                                k.map(|k| weight(address, usize::from(k)) * weight(cycle, t))
                            })
                            .sum();
                        assert_eq!(polynomial.evaluate(&point), expected);
                        let values: Vec<Fr> = (0..table).map(|k| weight(address, k)).collect();
                        let mut dense: Vec<Fr> = input
                            .iter()
                            .map(|k| k.map_or(Fr::from(0u64), |k| values[usize::from(k)]))
                            .collect();
                        let mut compact = RaPolynomial::<u16, Fr>::new(
                            polynomial.nonzero_indices.clone(),
                            values.clone(),
                        );
                        let mut original = RaPolynomial::<u16, Fr>::new(Arc::new(input), values);
                        for round in 0..rows.ilog2() {
                            for (i, v) in dense.iter().enumerate() {
                                assert_eq!(compact.get_bound_coeff(i), *v);
                                assert_eq!(original.get_bound_coeff(i), *v);
                            }
                            let challenge = if round == 0 {
                                <Fr as JoltField>::Challenge::from(0u128)
                            } else if round == 1 {
                                <Fr as JoltField>::Challenge::from(1u128)
                            } else {
                                tr.challenge_scalar_optimized::<Fr>()
                            };
                            let half = dense.len() / 2;
                            dense = (0..half)
                                .map(|i| dense[i] + (dense[half + i] - dense[i]) * challenge)
                                .collect();
                            compact.bind_parallel(challenge, BindingOrder::HighToLow);
                            original.bind_parallel(challenge, BindingOrder::HighToLow);
                        }
                        assert_eq!(compact.final_claim(), dense[0]);
                        assert_eq!(original.final_claim(), dense[0]);
                    }
                });
        }
    }

    #[test]
    fn compact_indices_keep_shared_ownership_and_public_capacity_bounds() {
        let packed = Arc::new(CompactIndices::new(vec![Some(255); 1 << 16], 256));
        let cloned = packed.clone();
        let source = SharedIndices::from(packed.clone());
        assert!(Arc::ptr_eq(&packed, &cloned));
        assert!(matches!(&source,SharedIndices::Compact(x) if Arc::ptr_eq(x,&packed)));
        assert_eq!(
            packed.allocated_bytes(),
            (1 << 16) + 256 * std::mem::size_of::<Option<u16>>()
        );
        let small = CompactIndices::new(vec![None, Some(255)], 256);
        assert!(matches!(small, CompactIndices::Direct(_)));
    }

    #[test]
    fn compact_indices_present_boundaries_and_sparse_fallback() {
        for table in [1, 255, 256, 257, 65535, 65536] {
            let n = (table * 4).max(8192);
            let input: Vec<_> = (0..n).map(|i| Some((i % table) as u16)).collect();
            for absent in [None, Some(0), Some(n / 2), Some(n - 1)] {
                let mut input = input.clone();
                if let Some(i) = absent {
                    input[i] = None;
                }
                let packed = CompactIndices::new(input.clone(), table);
                assert_eq!(packed.iter().copied().collect::<Vec<_>>(), input);
                assert_eq!(packed.par_iter().copied().collect::<Vec<_>>(), input);
                assert_eq!(
                    packed.slice(1..n - 1).iter().copied().collect::<Vec<_>>(),
                    input[1..n - 1]
                );
                let chunks: Vec<Vec<_>> = packed
                    .par_chunks(257)
                    .map(|x| x.iter().copied().collect())
                    .collect();
                assert_eq!(chunks.into_iter().flatten().collect::<Vec<_>>(), input);
                if table == 256 {
                    assert_eq!(
                        matches!(packed, CompactIndices::Bytes { .. }),
                        absent.is_none()
                    );
                }
                if table == 65536 {
                    assert_eq!(
                        matches!(packed, CompactIndices::Shorts { .. }),
                        absent.is_none()
                    );
                }
                assert!(packed.allocated_bytes() <= n * std::mem::size_of::<Option<u16>>());
            }
            for n in [0, 1, 2] {
                let p = CompactIndices::new(vec![Some((table - 1) as u16); n], table);
                assert!(matches!(p, CompactIndices::Direct(_)));
            }
        }
        // Presence validation cannot turn an out-of-range value into a byte.
        for input in [vec![Some(256); 8192], vec![None, Some(256)]] {
            assert!(std::panic::catch_unwind(|| CompactIndices::new(input, 256)).is_err());
        }
        // One missing row at every position must preserve None and Some(255).
        for absent in 0..1024 {
            let mut input = vec![Some(255); 1024];
            input[absent] = None;
            let p = CompactIndices::new(input.clone(), 256);
            assert!(matches!(p, CompactIndices::Shorts { .. }));
            assert!(p.iter().copied().eq(input.into_iter()));
        }
    }

    #[test]
    fn compact_indices_present_commitment_matches_independent_dense() {
        use crate::poly::{
            commitment::{commitment_scheme::CommitmentScheme, dory::DoryScheme},
            dense_mlpoly::DensePolynomial,
            multilinear_polynomial::MultilinearPolynomial,
        };
        use dory::{backends::arkworks::ArkFr, primitives::arithmetic::Group};
        let table = 256;
        let rows = 1024;
        let input: Vec<_> = (0..rows).map(|i| Some(((i * 17) % table) as u16)).collect();
        let packed = OneHotPolynomial::<Fr>::from_indices(input.clone(), table);
        assert!(matches!(
            &*packed.nonzero_indices,
            CompactIndices::Bytes { .. }
        ));
        let mut direct = packed.clone();
        direct.nonzero_indices = Arc::new(CompactIndices::Direct(input.clone()));
        let mut dense = vec![Fr::from(0u64); table * rows];
        for (i, k) in input.iter().enumerate() {
            dense[usize::from(k.unwrap()) * rows + i] = Fr::from(1u64);
        }
        let setup = DoryScheme::setup_prover((table * rows).ilog2() as usize);
        let reference = MultilinearPolynomial::LargeScalars(DensePolynomial::new(dense));
        let (expected, _) = DoryScheme::commit(&reference, &setup);
        let fixed_blind = setup.prover.ht.scale(&ArkFr(Fr::from(19u64)));
        for p in [packed, direct] {
            let (actual, _) = DoryScheme::commit(&MultilinearPolynomial::OneHot(p), &setup);
            assert_eq!(actual, expected);
            assert_eq!(actual.0 + fixed_blind, expected.0 + fixed_blind);
        }
    }

    #[test]
    #[ignore = "isolated matched representation benchmark"]
    fn compact_indices_present_process_benchmark() {
        let mode = std::env::var("PRESENT_INDEX_MODE").unwrap();
        assert!(mode == "control" || mode == "candidate");
        let start = Instant::now();
        let n = 1usize << 22;
        let mut arrays = Vec::new();
        let mut stored = 0usize;
        for j in 0..32 {
            let input: Vec<_> = (0..n).map(|i| Some(((i + j) % 256) as u16)).collect();
            let p = if mode == "control" {
                let decode = std::iter::once(None).chain((0..256).map(Some)).collect();
                let mut rows = Vec::with_capacity(input.len());
                rows.extend(input.into_iter().map(|i| i.unwrap() + 1));
                CompactIndices::Shorts { rows, decode }
            } else {
                CompactIndices::new(input, 256)
            };
            stored += p.allocated_bytes();
            arrays.push(p);
        }
        let construction = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut sum = 0u64;
        for _ in 0..4 {
            sum += arrays
                .par_iter()
                .map(|p| p.iter().flatten().map(|x| u64::from(*x)).sum::<u64>())
                .sum::<u64>();
        }
        assert_eq!(sum, 4 * 32 * (n as u64 / 256) * (255 * 256 / 2));
        println!("PRESENT_INDEX_BENCH mode={mode} construction={construction} scan={} stored_bytes={stored} sum={sum}", start.elapsed().as_secs_f64());
    }

    #[test]
    #[ignore = "isolated process comparison with explicit storage mode"]
    fn compact_indices_process_benchmark() {
        let mode = std::env::var("LOOKUP_INDEX_BENCH_MODE").unwrap();
        assert!(mode == "original" || mode == "compact");
        let start = Instant::now();
        let n = 1usize << 22;
        let mut arrays = Vec::new();
        let mut expected = 0u64;
        let mut stored = 0usize;
        for j in 0..32 {
            let input: Vec<Option<u16>> = (0..n)
                .map(|i| {
                    if i % 257 == 256 {
                        None
                    } else {
                        Some(((i + j) % 256) as u16)
                    }
                })
                .collect();
            expected += input
                .iter()
                .flatten()
                .map(|i| u64::from(*i) + 1)
                .sum::<u64>();
            let array = if mode == "original" {
                CompactIndices::Direct(input)
            } else {
                CompactIndices::new(input, 256)
            };
            stored += array.allocated_bytes();
            arrays.push(array);
        }
        let build_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut sum = 0u64;
        for _ in 0..4 {
            sum += arrays
                .par_iter()
                .map(|a| a.iter().flatten().map(|i| u64::from(*i) + 1).sum::<u64>())
                .sum::<u64>();
        }
        let scan_seconds = start.elapsed().as_secs_f64();
        assert_eq!(sum, 4 * expected);
        println!("LOOKUP_INDEX_BENCH mode={mode} build_seconds={build_seconds} scan_seconds={scan_seconds} entries={} stored_bytes={stored} sum={sum}",32*n);
    }
}
