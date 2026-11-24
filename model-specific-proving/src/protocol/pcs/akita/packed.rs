//! Reduce sparse addresses and pairs of rows before allocating field columns.

use super::*;

const ROW_CHUNK: usize = 512;

struct ReadFold {
    address: Vec<F>,
    rows: RowWeights,
    start: usize,
    totals: [F; K],
}

pub(super) struct PackedFold<'a> {
    indices: &'a [Option<u8>],
    hot_weights: [F; K],
    reads: Vec<ReadFold>,
    bit_address: Vec<F>,
    bit_rows: RowWeights,
    bit_mix: F,
    bit_totals: [F; K],
}

impl<'a> PackedFold<'a> {
    pub(super) fn new(
        indices: &'a [Option<u8>],
        reads: &[Read],
        bit_check: &[F],
        bit_mix: F,
    ) -> Self {
        let bit_rows = RowWeights::new(&bit_check[ADDRESS_BITS..]);
        let bit_totals = bit_rows.by_address(indices).map(|w| w * bit_mix);
        let reads = reads
            .iter()
            .map(|read| {
                let rows = RowWeights::new(&read.rows);
                let totals = rows.by_address(&indices[read.start..][..rows.len()]);
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

    fn rows(&self, start: usize, len: usize) -> Vec<[F; 3]> {
        let bits = self
            .bit_rows
            .scaled_range(self.bit_mix * self.bit_address[0], start, len);
        let mut rows: Vec<_> = self.indices[start..start + len]
            .iter()
            .zip(bits)
            .map(|(&index, bit)| {
                let value =
                    index.map_or(F::zero(), |address| self.hot_weights[usize::from(address)]);
                [value, F::zero(), bit]
            })
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
            .zip(self.rows(start + self.indices.len() / 2, len))
    }

    pub(super) fn row_samples(&self) -> [F; 4] {
        let half = self.indices.len() / 2;
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

    pub(super) fn bind_rows(self, challenge: F) -> [Vec<F>; 3] {
        let half = self.indices.len() / 2;
        let [mut values, mut weights, mut bits] = std::array::from_fn(|_| vec![F::zero(); half]);
        values
            .par_chunks_mut(ROW_CHUNK)
            .zip(weights.par_chunks_mut(ROW_CHUNK))
            .zip(bits.par_chunks_mut(ROW_CHUNK))
            .enumerate()
            .for_each(|(chunk, ((values, weights), bits))| {
                let pairs = self.row_pairs(chunk * ROW_CHUNK, values.len());
                for (((value, weight), bit), (low, high)) in
                    values.iter_mut().zip(weights).zip(bits).zip(pairs)
                {
                    [*value, *weight, *bit] =
                        std::array::from_fn(|i| low[i] + challenge * (high[i] - low[i]));
                }
            });
        [values, weights, bits]
    }
}

// An equality weight factors into one value per row block and one per offset.
struct RowWeights {
    blocks: Vec<F>,
    offsets: Vec<F>,
}

impl RowWeights {
    fn new(point: &[F]) -> Self {
        let (high, low) = point.split_at(point.len() / 2);
        Self {
            blocks: weights(high),
            offsets: weights(low),
        }
    }

    fn len(&self) -> usize {
        self.blocks.len() * self.offsets.len()
    }

    fn scaled_range(&self, scale: F, start: usize, len: usize) -> impl Iterator<Item = F> + '_ {
        let width = self.offsets.len();
        let end = start + len;
        (start / width..end.div_ceil(width)).flat_map(move |block| {
            let base = block * width;
            let lo = start.saturating_sub(base);
            let hi = width.min(end - base);
            let scale = scale * self.blocks[block];
            self.offsets[lo..hi]
                .iter()
                .map(move |&offset| scale * offset)
        })
    }

    fn by_address(&self, indices: &[Option<u8>]) -> [F; K] {
        let mut totals = [F::zero(); K];
        for (indices, &block) in indices.chunks_exact(self.offsets.len()).zip(&self.blocks) {
            let mut local = [F::zero(); K];
            for (&index, &weight) in indices.iter().zip(&self.offsets) {
                if let Some(address) = index {
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

    #[test]
    fn first_row_round_matches_dense_fold() {
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
        let mut sparse = PackedFold::new(&indices, &reads, &point, mix);
        for bit in (0..ADDRESS_BITS).rev() {
            sparse.bind_address(F::from_u64((bit + 3) as u64), bit);
        }

        let values = indices
            .iter()
            .map(|index| index.map_or(F::zero(), |a| sparse.hot_weights[usize::from(a)]))
            .collect();
        let mut combined = vec![F::zero(); indices.len()];
        for read in &sparse.reads {
            for (value, weight) in
                combined[read.start..]
                    .iter_mut()
                    .zip(read.rows.blocks.iter().flat_map(|&block| {
                        read.rows
                            .offsets
                            .iter()
                            .map(move |&offset| read.address[0] * block * offset)
                    }))
            {
                *value += weight;
            }
        }
        let bits = weights(&point[ADDRESS_BITS..])
            .into_iter()
            .map(|v| v * mix * sparse.bit_address[0])
            .collect();
        let mut dense = Fold::new(
            [values, combined, bits].map(Column::owned),
            rows,
            3,
            equation,
        );
        let first = dense.prove_round(None, 0, F::zero()).unwrap();
        assert_eq!(UnivariatePoly::from_evals(&sparse.row_samples()), first);
        let challenge = F::from_u64(11);
        let next = dense.prove_round(Some(challenge), 1, F::zero()).unwrap();
        let mut folded = Fold::new(
            sparse.bind_rows(challenge).map(Column::owned),
            rows - 1,
            3,
            equation,
        );
        assert_eq!(folded.prove_round(None, 0, F::zero()).unwrap(), next);
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
        assert_eq!(factored.by_address(&indices), expected);
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
