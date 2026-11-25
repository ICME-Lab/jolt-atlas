use super::*;

/// Tables of weighted deviations from one. A zero nibble contributes nothing.
pub(super) struct PackedRound<'a> {
    source: &'a LutLayers,
    level: usize,
    stride: usize,
    unit: usize,
    columns: usize,
    groups: Vec<[Field; 16]>,
    bindings: usize,
}

impl<'a> PackedRound<'a> {
    pub(super) fn new(
        source: &'a LutLayers,
        level: usize,
        stride: usize,
        weights: &[Field],
    ) -> Self {
        let nodes = source.nodes::<LAYER2>() << (LAYER2 - level);
        let columns = nodes / 4;
        let unit = stride.min(1 << source.log_trees).min(columns);
        let groups = (0..nodes / unit)
            .map(|group| {
                let index = group * unit;
                let part = index / columns;
                let position = index % columns + STREAM[level][part] * columns;
                let lut = &source.luts[position >> source.log_trees];
                let weight = weights[index / stride % weights.len()];
                std::array::from_fn(|n| (lut[n & MASK[level][part]] - Field::ONE) * weight)
            })
            .collect();
        Self {
            source,
            level,
            stride,
            unit,
            columns,
            groups,
            bindings: weights.len(),
        }
    }

    /// Fill one aligned tile with the folded values minus one.
    pub(super) fn fill(&self, start: usize, dst: &mut [Field]) {
        dst.fill(Field::ZERO);
        for k in 0..self.bindings {
            let mut index = start + k * self.stride;
            let mut output = &mut *dst;
            while !output.is_empty() {
                let size = (self.unit - index % self.unit).min(output.len());
                let lut = &self.groups[index / self.unit];
                let position =
                    index % self.columns + STREAM[self.level][index / self.columns] * self.columns;
                let (block, rest) = output.split_at_mut(size);
                if position.is_multiple_of(NIBBLES_IN_WORD) && size.is_multiple_of(NIBBLES_IN_WORD)
                {
                    let words = &self.source.words()[position / NIBBLES_IN_WORD..];
                    for (bits, chunk) in words.iter().zip(block.chunks_exact_mut(NIBBLES_IN_WORD)) {
                        let mut word = *bits;
                        while word != 0 {
                            let lane = word.trailing_zeros() as usize / 4;
                            chunk[lane] += lut[((word >> (4 * lane)) & 15) as usize];
                            word &= !(15u64 << (4 * lane));
                        }
                    }
                } else {
                    for (offset, value) in block.iter_mut().enumerate() {
                        *value += lut[self.source.nibble(position + offset)];
                    }
                }
                index += size;
                output = rest;
            }
        }
    }
}
