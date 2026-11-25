//! Random instances whose claims hold by construction.

use field::F128;
use num_traits::ConstZero;
use poly::{DenseMultilinearExtension, eq_table};
use rand_core::{Rng, SeedableRng};
use rand_pcg::Pcg64;

pub fn rng(seed: u64) -> Pcg64 {
    Pcg64::seed_from_u64(seed)
}

pub fn random(rng: &mut Pcg64) -> F128 {
    F128::new(rng.next_u64(), rng.next_u64())
}

pub fn random_elements(rng: &mut Pcg64, count: usize) -> Vec<F128> {
    (0..count).map(|_| random(rng)).collect()
}

/// A claim `<rows (x) columns, f> = target` with the witness it is about,
/// `f` packed column major with `2^log_rows` rows.
pub struct Leaf {
    pub log_rows: usize,
    pub log_columns: usize,
    pub rows: Vec<F128>,
    pub columns: Vec<F128>,
    /// From the definition.
    pub target: F128,
    pub packed: Vec<F128>,
}

impl Leaf {
    /// Random weights over a random witness.
    pub fn random(log_rows: usize, log_columns: usize, seed: u64) -> Self {
        let mut rng = rng(seed);
        let packed = random_elements(&mut rng, 1 << (log_rows + log_columns - 7));
        Self::with_witness(log_rows, log_columns, packed, &mut rng)
    }

    /// Random weights over a witness of a few set bits, so that the
    /// extension stays cheap to evaluate at large shapes.
    pub fn sparse(log_rows: usize, log_columns: usize, seed: u64) -> Self {
        let mut rng = rng(seed);
        let mut packed = vec![F128::ZERO; 1 << (log_rows + log_columns - 7)];
        for _ in 0..64 {
            let index = rng.next_u64() as usize % packed.len();
            packed[index] = random(&mut rng);
        }
        Self::with_witness(log_rows, log_columns, packed, &mut rng)
    }

    fn with_witness(
        log_rows: usize,
        log_columns: usize,
        packed: Vec<F128>,
        rng: &mut Pcg64,
    ) -> Self {
        let rows = random_elements(rng, 1 << log_rows);
        let columns = random_elements(rng, 1 << log_columns);
        let mut leaf = Self {
            log_rows,
            log_columns,
            rows,
            columns,
            target: F128::ZERO,
            packed,
        };
        leaf.target = leaf
            .set_bits()
            .map(|index| leaf.rows[index % (1 << log_rows)] * leaf.columns[index >> log_rows])
            .sum();
        leaf
    }

    /// Bit `index` of the packed witness.
    pub fn bit(&self, index: usize) -> bool {
        let element = self.packed[index >> 7];
        let half = if index % 128 < 64 {
            element.lo
        } else {
            element.hi
        };
        (half >> (index % 64)) & 1 == 1
    }

    fn set_bits(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.packed.len() << 7).filter(|&index| self.bit(index))
    }

    /// `rows (x) columns` written out per bit.
    pub fn weights(&self) -> Vec<F128> {
        (0..self.packed.len() << 7)
            .map(|index| {
                self.rows[index % (1 << self.log_rows)] * self.columns[index >> self.log_rows]
            })
            .collect()
    }

    /// `MLE[f]` over all `t + s` variables. Dense, so small instances only.
    pub fn bits_extension(&self) -> DenseMultilinearExtension<F128> {
        let log_bits = self.log_rows + self.log_columns;
        let table = (0..1usize << log_bits)
            .map(|index| F128::from(self.bit(index)))
            .collect();
        DenseMultilinearExtension::from_evaluations(log_bits, table).unwrap()
    }

    /// `MLE[f](point)` from the set bits alone.
    pub fn evaluate(&self, point: &[F128]) -> F128 {
        let (low, high) = point.split_at(7);
        let eq_low = eq_table(low);
        let eq_high = eq_table(high);
        self.set_bits()
            .map(|index| eq_high[index >> 7] * eq_low[index % 128])
            .sum()
    }
}

#[test]
fn the_leaf_is_its_own_extension_on_the_cube() {
    let leaf = Leaf::random(7, 2, 98);
    let extension = leaf.bits_extension();
    assert_eq!(extension.num_vars(), 9);
    assert_eq!(extension[(3 << 7) | 5], F128::from(leaf.bit((3 << 7) | 5)));
    let written_out: F128 = extension
        .iter()
        .zip(leaf.weights())
        .map(|(f, weight)| weight * *f)
        .sum();
    assert_eq!(written_out, leaf.target);
    let point: Vec<F128> = (0..9).map(|i| F128::new(i + 3, 7)).collect();
    assert_eq!(leaf.evaluate(&point), extension.evaluate(&point).unwrap());
}
