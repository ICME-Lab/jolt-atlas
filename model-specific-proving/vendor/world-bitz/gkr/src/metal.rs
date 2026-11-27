use super::*;

unsafe extern "C" {
    fn bitz_leaf_buckets(
        library: *const u8,
        library_len: usize,
        packed: *const u64,
        weights: *const Field,
        trees: u32,
        count: u32,
        result: *mut Field,
    ) -> bool;
}

pub(super) struct Leaves<'a> {
    lut: &'a LutLayers,
    point: &'a [Field],
    bins: Vec<[Field; 256]>,
}

impl<'a> Leaves<'a> {
    pub(super) fn new(lut: &'a LutLayers, point: &'a [Field]) -> Option<Self> {
        const LIBRARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/leaf.metallib"));
        let trees = 1 << lut.log_trees;
        let count = 1usize << (point.len() - 2);
        if trees < 16384 || count < trees {
            return None;
        }

        let weights = poly::eq_table(&point[..lut.log_trees]);
        let mut bins = vec![[Field::ZERO; 256]; count / trees];
        // Metal reads exactly two count-nibble blocks and writes one histogram per row pair.
        let _span = tracing::info_span!("Metal BitZ leaf histograms").entered();
        let ok = unsafe {
            bitz_leaf_buckets(
                LIBRARY.as_ptr(),
                LIBRARY.len(),
                lut.words().as_ptr(),
                weights.as_ptr(),
                trees.try_into().ok()?,
                count.try_into().ok()?,
                bins.as_mut_ptr().cast(),
            )
        };
        if !ok {
            tracing::warn!("Metal BitZ histogram failed");
        }
        ok.then_some(Self { lut, point, bins })
    }

    pub(super) fn first(&self, at_one: bool) -> (Wide256, Wide256) {
        let eq = poly::eq_table(&self.point[self.lut.log_trees..self.point.len() - 1]);
        let rows = self.bins.len();
        self.bins
            .par_iter()
            .enumerate()
            .map(|(row, bins)| {
                let mut weights = [[Field::ZERO; 16]; 2];
                for (key, &w) in bins.iter().enumerate() {
                    weights[0][key & 15] += w;
                    weights[1][key >> 4] += w;
                }
                let mut sums = (Wide256::zero(), Wide256::zero());
                for (side, weights) in weights.into_iter().enumerate() {
                    let i = row + side * rows;
                    let image = &self.lut.luts[i];
                    for (key, &w) in weights.iter().enumerate().skip(1) {
                        let [l0, l1, r0, r1] = [1, 2, 4, 8].map(|mask| image[key & mask]);
                        let (l, r) = if at_one { (l1, r1) } else { (l0, r0) };
                        let w = w * eq[i];
                        sums.0 += Wide256::mul(w, l * r + Field::ONE);
                        sums.1 += Wide256::mul(w, (l0 + l1) * (r0 + r1));
                    }
                }
                sums
            })
            .reduce(zero, add)
    }

    pub(super) fn second(&self, first: Field, at_one: bool) -> (Wide256, Wide256) {
        let eq = poly::eq_table(&self.point[self.lut.log_trees..self.point.len() - 2]);
        let trees = 1 << self.lut.log_trees;
        let half = self.bins.len() * trees;
        self.bins
            .par_iter()
            .zip(eq)
            .enumerate()
            .map(|(row, (bins, eq))| {
                let mut weights = *bins;
                // The all-zero nibble pair contributes its remaining equality mass.
                weights[0] = Field::ONE + weights[1..].iter().copied().sum::<Field>();
                let (b, inf) = leaf_pair_eval(self.lut, row * trees, half, first, at_one, &weights);
                (Wide256::mul(b.reduce(), eq), Wide256::mul(inf.reduce(), eq))
            })
            .reduce(zero, add)
    }
}

fn zero() -> (Wide256, Wide256) {
    (Wide256::zero(), Wide256::zero())
}

fn add((a, b): (Wide256, Wide256), (c, d): (Wide256, Wide256)) -> (Wide256, Wide256) {
    (a + c, b + d)
}
