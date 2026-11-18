//! Multilinear polynomials over `F_{2^128}` given by their values on the Boolean cube.
//!
//! Variable `j` is bit `j` of a value's index, so the first variable is the least significant.

use super::gf128::Gf128;
use rayon::prelude::*;

/// The fewest field elements a parallel pass gives one task: below it a task costs more to
/// hand out than to run.
pub const PAR_MIN_LEN: usize = 1 << 10;

/// `eq(point, b)` for every `b` in the cube: the product over variables of `point_j` where
/// `b_j = 1` and `1 + point_j` where `b_j = 0`.
pub fn eq_table(point: &[Gf128]) -> Vec<Gf128> {
    let mut table = Vec::with_capacity(1 << point.len());
    table.push(Gf128::ONE);
    for &coordinate in point {
        let len = table.len();
        table.extend_from_within(..);
        let (low, high) = table.split_at_mut(len);
        low.par_iter_mut()
            .zip(high.par_iter_mut())
            .with_min_len(PAR_MIN_LEN)
            .for_each(|(lo, hi)| {
                *hi = *lo * coordinate;
                *lo += *hi;
            });
    }
    table
}

/// The inner product of two vectors of equal length.
pub fn inner_product(a: &[Gf128], b: &[Gf128]) -> Gf128 {
    debug_assert_eq!(a.len(), b.len());
    a.par_iter()
        .zip(b.par_iter())
        .with_min_len(PAR_MIN_LEN)
        .map(|(&x, &y)| x * y)
        .reduce(|| Gf128::ZERO, |x, y| x + y)
}

/// The polynomial with its first variable fixed to `r`.
pub fn fold_first(values: &[Gf128], r: Gf128) -> Vec<Gf128> {
    values
        .par_chunks_exact(2)
        .with_min_len(PAR_MIN_LEN)
        .map(|pair| pair[0] + r * (pair[0] + pair[1]))
        .collect()
}

/// The vector `⊗_j (1, factors_j)`: entry `i` is the product of the factors at the set bits
/// of `i`.
pub fn tensor_expand(factors: &[Gf128]) -> Vec<Gf128> {
    let mut table = Vec::with_capacity(1 << factors.len());
    table.push(Gf128::ONE);
    for &factor in factors {
        for i in 0..table.len() {
            let scaled = table[i] * factor;
            table.push(scaled);
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize) -> Vec<Gf128> {
        (0..1u128 << n)
            .map(|i| Gf128(i * i * 0x1_0000_0001 + 3))
            .collect()
    }

    /// The multilinear extension from its definition, one cube point at a time.
    fn evaluate_reference(values: &[Gf128], point: &[Gf128]) -> Gf128 {
        values
            .iter()
            .enumerate()
            .fold(Gf128::ZERO, |acc, (b, &value)| {
                let weight = point.iter().enumerate().fold(Gf128::ONE, |w, (j, &p)| {
                    w * if (b >> j) & 1 == 1 { p } else { Gf128::ONE + p }
                });
                acc + weight * value
            })
    }

    #[test]
    fn eq_table_and_folding_agree_with_the_definition() {
        let point = [Gf128(5), Gf128(0xabc), Gf128(1 << 100), Gf128(7)];
        let poly = values(4);
        let expected = evaluate_reference(&poly, &point);
        assert_eq!(inner_product(&eq_table(&point), &poly), expected);
        let folded = point.iter().fold(poly.clone(), |p, &r| fold_first(&p, r));
        assert_eq!(folded, vec![expected]);
        // On a cube point the table is an indicator.
        let cube = [Gf128::ONE, Gf128::ZERO, Gf128::ONE, Gf128::ONE];
        let table = eq_table(&cube);
        assert!(table
            .iter()
            .enumerate()
            .all(|(b, &w)| w == if b == 0b1101 { Gf128::ONE } else { Gf128::ZERO }));
    }

    #[test]
    fn tensor_expansion_multiplies_the_factors_at_set_bits() {
        let factors = [Gf128(3), Gf128(10), Gf128(77)];
        let table = tensor_expand(&factors);
        for (i, &entry) in table.iter().enumerate() {
            let expected = (0..3)
                .filter(|j| (i >> j) & 1 == 1)
                .fold(Gf128::ONE, |p, j| p * factors[j]);
            assert_eq!(entry, expected);
        }
    }
}
