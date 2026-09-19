//! Immutable lookup indices with an encoding selected by the public table size.
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
        assert!(indices
            .iter()
            .flatten()
            .all(|i| usize::from(*i) < table_size));
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
                    for table in [2usize, 16, 256, 65536] {
                        let rows = 8192usize;
                        let input: Vec<_> = (0..rows)
                            .map(|i| match i % 5 {
                                0 => None,
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
            (1 << 17) + 257 * std::mem::size_of::<Option<u16>>()
        );
        let small = CompactIndices::new(vec![None, Some(255)], 256);
        assert!(matches!(small, CompactIndices::Direct(_)));
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
