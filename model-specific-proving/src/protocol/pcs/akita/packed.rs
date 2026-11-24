//! Fold the sparse address axis before allocating field columns for the rows.

use super::*;

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

    pub(super) fn row_columns(self) -> [Vec<F>; 3] {
        let values = self
            .indices
            .iter()
            .map(|index| index.map_or(F::zero(), |address| self.hot_weights[usize::from(address)]))
            .collect();
        let mut weights = vec![F::zero(); self.indices.len()];
        for read in self.reads {
            for (weight, row) in weights[read.start..]
                .iter_mut()
                .zip(read.rows.scaled(read.address[0]))
            {
                *weight += row;
            }
        }
        let mut bits = Vec::with_capacity(self.bit_rows.len());
        bits.extend(self.bit_rows.scaled(self.bit_mix * self.bit_address[0]));
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

    fn scaled(&self, scale: F) -> impl Iterator<Item = F> + '_ {
        self.blocks.iter().flat_map(move |&block| {
            let scale = scale * block;
            self.offsets.iter().map(move |&offset| scale * offset)
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
            factored.scaled(scale).collect::<Vec<_>>(),
            dense
                .into_iter()
                .map(|weight| scale * weight)
                .collect::<Vec<_>>()
        );
    }
}
