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
        third: *mut Field,
    ) -> bool;
}

pub(super) struct Leaves<'a> {
    lut: &'a LutLayers,
    point: &'a [Field],
    bins: Vec<[Field; 256]>,
    third: Option<Vec<[Field; 81]>>,
}

impl<'a> Leaves<'a> {
    pub(super) fn new(lut: &'a LutLayers, point: &'a [Field], try_third: bool) -> Option<Self> {
        const LIBRARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/leaf.metallib"));
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let trees = 1 << lut.log_trees;
        let count = 1usize << (point.len() - 2);
        if trees < 16384
            || count < trees
            || !*ENABLED.get_or_init(|| std::env::var("ATLAS_BITZ_METAL").as_deref() != Ok("0"))
        {
            return None;
        }

        let weights = poly::eq_table(&point[..lut.log_trees]);
        let mut bins = vec![[Field::ZERO; 256]; count / trees];
        static THIRD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let mut third = (try_third
            && count >= 2 * trees
            && *THIRD
                .get_or_init(|| std::env::var("ATLAS_BITZ_METAL_ROUND3").as_deref() != Ok("0")))
        .then(|| vec![[Field::ZERO; 81]; count / (2 * trees)]);
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
                third
                    .as_mut()
                    .map_or(std::ptr::null_mut(), |m| m.as_mut_ptr().cast()),
            )
        };
        if !ok {
            tracing::warn!("Metal BitZ histogram failed");
        }
        ok.then_some(Self {
            lut,
            point,
            bins,
            third,
        })
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
    pub(super) fn third(&self, prefix: &[Field], at_one: bool) -> Option<(Wide256, Wide256)> {
        let moments = self.third.as_ref()?;
        let _span = tracing::info_span!("Metal BitZ leaf round three").entered();
        let high = poly::eq_table(&self.point[self.lut.log_trees..self.point.len() - 3]);
        let rows = moments.len();
        Some(
            moments
                .par_iter()
                .zip(high)
                .enumerate()
                .map(|(row, (moments, high))| {
                    let mut left = [Field::ONE; 9];
                    let mut right = [Field::ONE; 9];
                    let mut b = Wide256::zero();
                    let mut inf = Wide256::zero();
                    for i in 0..8 {
                        let (stream, bit) = (i / 2, i % 2);
                        let image = &self.lut.luts[row + stream * rows];
                        let scale = prefix[stream / 2 + 2 * bit];
                        left[i] = scale * (image[1 << bit] + Field::ONE);
                        right[i] = scale * (image[4 << bit] + Field::ONE);
                    }
                    for (i, &l) in left.iter().enumerate() {
                        for (j, &r) in right.iter().enumerate() {
                            let term = Wide256::mul(moments[i * 9 + j], l * r);
                            let selected = |k| k == 8 || ((k / 2) % 2 == usize::from(at_one));
                            if selected(i) && selected(j) {
                                b += term;
                            }
                            if i < 8 && j < 8 {
                                inf += term;
                            }
                        }
                    }
                    (
                        Wide256::mul(b.reduce(), high),
                        Wide256::mul(inf.reduce(), high),
                    )
                })
                .reduce(zero, add),
        )
    }
}

fn zero() -> (Wide256, Wide256) {
    (Wide256::zero(), Wide256::zero())
}

fn add((a, b): (Wide256, Wide256), (c, d): (Wide256, Wide256)) -> (Wide256, Wide256) {
    (a + c, b + d)
}
