//! Reduce sparse addresses and pairs of rows before allocating field columns.

use super::*;
use crate::protocol::session::Equality;

const ROW_CHUNK: usize = 512;
pub(super) const SPARSE_ROUNDS: usize = 4;

/// Row-major K=16 indices. Code zero is an empty row; 1 through 16 select a column.
pub(super) struct Indices {
    bytes: Vec<u8>,
    len: usize,
    pub(super) prefix: usize,
    shared: Vec<(usize, SharedRows)>,
}

enum SharedRows {
    Lookup(Arc<Addresses>, usize, u32),
    Digits(Arc<Vec<u8>>, usize, usize),
}

impl Indices {
    pub(super) fn new(len: usize) -> Self {
        Self {
            bytes: Vec::new(),
            len,
            prefix: 0,
            shared: Vec::new(),
        }
    }

    pub(super) fn attach(&mut self, row: usize, source: (Arc<Addresses>, usize, u32)) {
        let (data, shift, mask) = source;
        self.prefix = self.prefix.max(row + data.len());
        self.shared
            .push((row, SharedRows::Lookup(data, shift, mask)));
    }

    pub(super) fn attach_digits(
        &mut self,
        row: usize,
        data: Arc<Vec<u8>>,
        offset: usize,
        len: usize,
    ) {
        if len != 0 {
            self.prefix = self.prefix.max(row + len);
            self.shared
                .push((row, SharedRows::Digits(data, offset, len)));
        }
    }

    pub(super) fn prepare(&mut self) {
        self.shared.sort_unstable_by_key(|s| s.0);
    }

    pub(super) fn insert(&mut self, row: usize, column: u8) {
        debug_assert!(row < self.len && usize::from(column) < K);
        let offset = row * 5;
        let code = u16::from(column + 1) << (offset % 8);
        let [low, high] = code.to_le_bytes();
        if self.bytes.len() < offset / 8 + 2 {
            self.bytes.resize(offset / 8 + 2, 0);
        }
        self.bytes[offset / 8] |= low;
        self.bytes[offset / 8 + 1] |= high;
        self.prefix = self.prefix.max(row + 1);
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub(super) fn get(&self, row: usize) -> Option<u8> {
        let offset = row * 5;
        if let Some((start, source)) = self
            .shared
            .partition_point(|block| block.0 <= row)
            .checked_sub(1)
            .map(|i| &self.shared[i])
        {
            let i = row - start;
            match source {
                SharedRows::Lookup(data, shift, mask) if i < data.len() => {
                    return Some(data.digit(i, *shift, *mask));
                }
                SharedRows::Digits(data, offset, len) if i < *len => return Some(data[offset + i]),
                _ => (),
            }
        }
        if offset / 8 + 1 >= self.bytes.len() {
            return None;
        }
        let pair = u16::from_le_bytes([self.bytes[offset / 8], self.bytes[offset / 8 + 1]]);
        (((pair >> (offset % 8)) as u8) & 31).checked_sub(1)
    }
}

impl MultilinearPoly<F> for Indices {
    fn num_vars(&self) -> usize {
        self.len.ilog2() as usize + ADDRESS_BITS
    }

    fn evaluate(&self, point: &[F]) -> F {
        assert_eq!(point.len(), self.num_vars());
        let (rows, address) = point.split_at(point.len() - ADDRESS_BITS);
        RowWeights::new(rows)
            .by_address(self, 0)
            .into_iter()
            .zip(address_weights(address))
            .map(|(r, a)| r * a)
            .sum()
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        let cols = 1 << sigma;
        let mut values = vec![F::zero(); cols];
        for row in 0..(self.len * K / cols) {
            let start = row * cols;
            let end = start + cols;
            values.fill(F::zero());
            for cycle in start / K..end.div_ceil(K) {
                if let Some(column) = self.get(cycle) {
                    let index = cycle * K + usize::from(column);
                    if (start..end).contains(&index) {
                        values[index - start] = F::one();
                    }
                }
            }
            f(row, &values);
        }
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        let cols = 1 << sigma;
        let mut result = vec![F::zero(); cols];
        self.for_each_one(&mut |index| result[index % cols] += left[index / cols]);
        result
    }

    fn is_one_hot(&self) -> bool {
        true
    }

    fn one_hot_k(&self) -> Option<usize> {
        Some(K)
    }

    fn one_hot_index_order(&self) -> Option<jolt_poly::OneHotIndexOrder> {
        Some(jolt_poly::OneHotIndexOrder::RowMajor)
    }

    fn for_each_one(&self, f: &mut dyn FnMut(usize)) {
        for cycle in 0..self.prefix {
            if let Some(column) = self.get(cycle) {
                f(cycle * K + usize::from(column));
            }
        }
    }
}

struct ReadFold {
    address: Vec<F>,
    rows: RowWeights,
    start: usize,
    totals: [F; K],
}

pub(super) struct PackedFold<'a> {
    indices: &'a Indices,
    hot_weights: [F; K],
    reads: Vec<ReadFold>,
    bit_address: Vec<F>,
    bit_rows: RowWeights,
    bit_mix: F,
    bit_totals: [F; K],
    row_weights: Vec<[F; K]>,
    values: Option<Vec<F>>,
    remaining: usize,
}

impl<'a> PackedFold<'a> {
    pub(super) fn new(indices: &'a Indices, reads: &[Read], bit_check: &[F], bit_mix: F) -> Self {
        let bit_rows = RowWeights::new(&bit_check[ADDRESS_BITS..]);
        let bit_totals = bit_rows.by_address(indices, 0).map(|w| w * bit_mix);
        let reads = reads
            .iter()
            .map(|read| {
                let rows = RowWeights::new(&read.rows);
                let totals = rows.by_address(indices, read.start);
                ReadFold {
                    address: read.address.to_vec(),
                    rows,
                    start: read.start,
                    totals,
                }
            })
            .collect();
        Self {
            indices,
            hot_weights: [F::one(); K],
            reads,
            bit_address: weights(&bit_check[..ADDRESS_BITS]),
            bit_rows,
            bit_mix,
            bit_totals,
            row_weights: Vec::new(),
            values: None,
            remaining: indices.len(),
        }
    }

    pub(super) fn bind_address(&mut self, r: F, bit: usize) {
        for (address, weight) in self.hot_weights.iter_mut().enumerate() {
            *weight *= if address & (1 << bit) == 0 {
                F::one() - r
            } else {
                r
            };
        }
        for read in &mut self.reads {
            bind_halves(&mut read.address, r);
        }
        bind_halves(&mut self.bit_address, r);
        if bit == 0 {
            self.row_weights.push(self.hot_weights);
        }
    }

    pub(super) fn address_samples(&self, round: usize) -> [F; 4] {
        let half = 1 << (ADDRESS_BITS - round - 1);
        let mut samples = [F::zero(); 4];
        for (i, sample) in samples.iter_mut().enumerate() {
            let x = F::from_u64(i as u64);
            let evaluate = |column: &[F], address: usize| {
                let base = address & (half - 1);
                column[base] + x * (column[base + half] - column[base])
            };
            let hot = |address: usize| {
                self.hot_weights[address] * if address & half == 0 { F::one() - x } else { x }
            };
            for (address, &weight) in self.bit_totals.iter().enumerate() {
                let value = hot(address);
                *sample +=
                    weight * evaluate(&self.bit_address, address) * value * (value - F::one());
            }
            for read in &self.reads {
                for (address, &weight) in read.totals.iter().enumerate() {
                    *sample += hot(address) * evaluate(&read.address, address) * weight;
                }
            }
        }
        samples
    }

    /// Evaluate a folded row from its original indices until the fourth row bind.
    #[inline]
    pub(super) fn value(&self, row: usize) -> F {
        if let Some(values) = &self.values {
            return values[row];
        }
        self.row_weights
            .iter()
            .enumerate()
            .filter_map(|(prefix, weights)| {
                self.indices
                    .get(prefix * self.remaining + row)
                    .map(|address| weights[usize::from(address)])
            })
            .sum()
    }

    fn rows(&self, start: usize, len: usize) -> Vec<[F; 3]> {
        let bits = self
            .bit_rows
            .scaled_range(self.bit_mix * self.bit_address[0], start, len);
        let mut rows: Vec<_> = (start..start + len)
            .zip(bits)
            .map(|(row, bit)| [self.value(row), F::zero(), bit])
            .collect();
        for read in &self.reads {
            let lo = start.max(read.start);
            let hi = (start + len).min(read.start + read.rows.len());
            if lo < hi {
                let weights = read
                    .rows
                    .scaled_range(read.address[0], lo - read.start, hi - lo);
                for (row, weight) in rows[lo - start..hi - start].iter_mut().zip(weights) {
                    row[1] += weight;
                }
            }
        }
        rows
    }

    fn row_pairs(&self, start: usize, len: usize) -> impl Iterator<Item = ([F; 3], [F; 3])> {
        self.rows(start, len)
            .into_iter()
            .zip(self.rows(start + self.remaining / 2, len))
    }

    pub(super) fn row_samples(&self) -> [F; 4] {
        let half = self.remaining / 2;
        (0..half)
            .into_par_iter()
            .step_by(ROW_CHUNK)
            .map(|start| {
                let mut samples = [F::zero(); 4];
                for (mut values, high) in self.row_pairs(start, ROW_CHUNK.min(half - start)) {
                    let delta = std::array::from_fn::<_, 3, _>(|i| high[i] - values[i]);
                    samples[0] += equation(&values);
                    for sample in &mut samples[1..] {
                        for (value, &delta) in values.iter_mut().zip(&delta) {
                            *value += delta;
                        }
                        *sample += equation(&values);
                    }
                }
                samples
            })
            .reduce(
                || [F::zero(); 4],
                |a, b| std::array::from_fn(|i| a[i] + b[i]),
            )
    }

    pub(super) fn bind_row(&mut self, challenge: F) {
        let half = self.remaining / 2;
        self.bit_rows.0.bind(challenge);
        for read in &mut self.reads {
            // An aligned interval either spans the whole axis or lies in one half.
            let weight = if read.rows.len() == self.remaining {
                read.rows.0.bind(challenge);
                F::one()
            } else if read.start < half {
                F::one() - challenge
            } else {
                read.start -= half;
                challenge
            };
            read.address[0] *= weight;
        }
        if let Some(values) = &mut self.values {
            bind_halves(values, challenge);
        } else {
            self.row_weights = self
                .row_weights
                .iter()
                .flat_map(|weights| {
                    [
                        weights.map(|w| w * (F::one() - challenge)),
                        weights.map(|w| w * challenge),
                    ]
                })
                .collect();
        }
        self.remaining = half;
        if self.values.is_none() && self.row_weights.len() == 1 << SPARSE_ROUNDS {
            self.values = Some(
                (0..half)
                    .into_par_iter()
                    .with_min_len(ROW_CHUNK)
                    .map(|row| self.value(row))
                    .collect(),
            );
        }
    }
}

// An equality weight factors into one value per row block and one per offset.
struct RowWeights(Equality<F>);

impl RowWeights {
    fn new(point: &[F]) -> Self {
        Self(Equality::new(point))
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn scaled_range(&self, scale: F, start: usize, len: usize) -> impl Iterator<Item = F> + '_ {
        let (blocks, offsets) = self.0.factors();
        let width = offsets.len();
        let end = start + len;
        (start / width..end.div_ceil(width)).flat_map(move |block| {
            let base = block * width;
            let lo = start.saturating_sub(base);
            let hi = width.min(end - base);
            let scale = scale * blocks[block];
            offsets[lo..hi].iter().map(move |&offset| scale * offset)
        })
    }

    fn by_address(&self, indices: &Indices, start: usize) -> [F; K] {
        let mut totals = [F::zero(); K];
        let (blocks, offsets) = self.0.factors();
        let width = offsets.len();
        for (i, &block) in blocks.iter().enumerate() {
            let mut local = [F::zero(); K];
            for (offset, &weight) in offsets.iter().enumerate() {
                if let Some(address) = indices.get(start + i * width + offset) {
                    local[usize::from(address)] += weight;
                }
            }
            for (total, weight) in totals.iter_mut().zip(local) {
                *total += block * weight;
            }
        }
        totals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::protocol::session::{Column, Fold};

    fn pack(indices: &[Option<u8>]) -> Indices {
        let mut packed = Indices::new(indices.len());
        for (row, &column) in indices.iter().enumerate() {
            if let Some(column) = column {
                packed.insert(row, column);
            }
        }
        packed
    }

    #[test]
    fn row_rounds_and_final_value_match_dense_fold() {
        let rows = 12;
        let indices: Vec<_> = (0..1 << rows)
            .map(|i| (i % 5 != 0).then_some((i % K) as u8))
            .collect();
        let point: Vec<_> = (2..2 + ADDRESS_BITS + rows)
            .map(|i| F::from_u64(i as u64))
            .collect();
        let address = std::array::from_fn(|i| F::from_u64((i + 4) as u64));
        let reads = [
            Read::new(0, &point[..rows], address),
            Read::new(128, &point[..7], address),
            Read::new(2048, &point[..11], address),
        ];
        let mix = F::from_u64(17);
        let packed = pack(&indices);
        assert!((0..indices.len()).all(|i| packed.get(i) == indices[i]));
        let mut sparse = PackedFold::new(&packed, &reads, &point, mix);
        for bit in (0..ADDRESS_BITS).rev() {
            sparse.bind_address(F::from_u64((bit + 3) as u64), bit);
        }

        let values: Vec<_> = indices
            .iter()
            .map(|index| index.map_or(F::zero(), |a| sparse.hot_weights[usize::from(a)]))
            .collect();
        let mut combined = vec![F::zero(); indices.len()];
        for read in &sparse.reads {
            for (value, weight) in combined[read.start..]
                .iter_mut()
                .zip(read.rows.scaled_range(read.address[0], 0, read.rows.len()))
            {
                *value += weight;
            }
        }
        let bits: Vec<_> = weights(&point[ADDRESS_BITS..])
            .into_iter()
            .map(|v| v * mix * sparse.bit_address[0])
            .collect();
        let mut claim = values
            .iter()
            .zip(&combined)
            .zip(&bits)
            .map(|((&value, &weight), &bit)| equation(&[value, weight, bit]))
            .sum();
        let mut dense = Fold::new(
            [values, combined, bits].map(Column::owned),
            rows,
            3,
            equation,
        );
        let mut challenge = None;
        for round in 0..rows {
            let expected = dense.prove_round(challenge, round, claim).unwrap();
            assert_eq!(UnivariatePoly::from_evals(&sparse.row_samples()), expected);
            let next = F::from_u64(if round == 4 { 0 } else { round as u64 + 11 });
            claim = expected.evaluate(next);
            sparse.bind_row(next);
            challenge = Some(next);
            assert_eq!(
                sparse.values.as_ref().map(Vec::len),
                (round + 1 >= SPARSE_ROUNDS).then_some(1 << (rows - round - 1))
            );
        }
        dense.finish_rounds(challenge.unwrap()).unwrap();
        assert_eq!(sparse.value(0), dense.columns[0].at(0));
    }

    #[test]
    fn factored_row_weights_match_dense_weights() {
        let point = [0, 7, 2, 1, 9].map(F::from_u64);
        let scale = F::from_u64(13);
        let factored = RowWeights::new(&point);
        let dense = weights(&point);
        let indices: Vec<_> = (0..dense.len())
            .map(|i| (i % 3 != 0).then_some((i % K) as u8))
            .collect();
        let mut expected = [F::zero(); K];
        for (&index, &weight) in indices.iter().zip(&dense) {
            if let Some(address) = index {
                expected[usize::from(address)] += weight;
            }
        }
        assert_eq!(factored.by_address(&pack(&indices), 0), expected);
        assert_eq!(
            factored
                .scaled_range(scale, 0, dense.len())
                .collect::<Vec<_>>(),
            dense
                .into_iter()
                .map(|weight| scale * weight)
                .collect::<Vec<_>>()
        );
    }
}
