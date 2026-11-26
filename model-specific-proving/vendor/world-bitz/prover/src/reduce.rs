//! Grand-product reduction to a factored inner-product claim over committed bits.
//!
//! Each leaf is `1 + (fold.row_images[row] - 1) * table.bit(column, row)`.
//! At the terminal GKR point, subtracting one leaves the weighted sum of bits:
//! `sum(row, column) u1[row] * u2[column] * table.bit(column, row)`.
//! The caller must discharge this claim through the commitment opening.

use common::{BitTable, ClaimError, Fold, LinearClaim, OpeningQuery, SecurityLevel};
use field::F128;
use gkr::{gpgkr_prove, grand_product_witness};
use num_traits::ConstOne;
use poly::eq_table;
use transcript::ProverState;

/// Reduces the grand-product circuit to a factored claim on the committed bits.
#[tracing::instrument(name = "Reduce grand products", skip_all)]
pub fn gkr_reduce<S>(
    transcript: &mut ProverState,
    fold: &Fold<S>,
    table: &BitTable,
    security: SecurityLevel,
) -> Result<OpeningQuery, ClaimError> {
    reduce_rows(transcript, &fold.row_images, &fold.zeta, table, security)
}

#[tracing::instrument(name = "Reduce grand products from row images", skip_all)]
pub(crate) fn reduce_rows(
    transcript: &mut ProverState,
    row_images: &[F128],
    zeta: &[F128],
    table: &BitTable<'_>,
    security: SecurityLevel,
) -> Result<OpeningQuery, ClaimError> {
    let witnesses = grand_product_witness(row_images, table);
    let (mut point, claim) = gpgkr_prove(transcript, zeta, witnesses, security);

    // The multilinear extension of the constant-one table is one at every point.
    let inner_product_claim = claim - F128::ONE;

    let r1 = row_images.len().max(1).ilog2() as usize;
    let r2 = point.len() - r1;
    // GKR returns coordinates in low-index-bit-first order: columns, then rows.
    let alfa_b = point.split_off(r2);
    let alfa_c = point;

    let u1: Vec<_> = row_images
        .iter()
        .zip(poly::eq_table(&alfa_b))
        .map(|(a, b)| (*a - F128::ONE) * b)
        .collect();

    let u2 = eq_table(&alfa_c);
    let claim = LinearClaim::from_shape(table.shape(), u1, u2, inner_product_claim)?;
    Ok(OpeningQuery::InnerProduct { claim })
}

#[cfg(test)]
mod order_check_ai_test {
    use super::*;
    use common::{BitZParams, Fold, Shape};
    use field::gf128::smallest_generator;
    use num_traits::ConstZero;

    const Q114: u128 = (1 << 114) - 11;
    type F = field::Fq<Q114>;

    type F2 = field::FqDefault;

    fn shape() -> Shape {
        Shape::new(7, 15).unwrap()
    }

    fn params() -> BitZParams<F> {
        BitZParams::new(shape(), smallest_generator()).unwrap()
    }

    fn packed_witness(shape: &Shape, bit_fn: impl Fn(usize, usize) -> bool) -> Vec<F128> {
        let mut packed = vec![F128::ZERO; (1 << shape.log_bits()) / 128];
        for column in 0..shape.columns() {
            for row in 0..shape.rows() {
                if bit_fn(column, row) {
                    let index = (column << shape.log_rows()) | row;
                    let element = &mut packed[index >> 7];
                    let offset = index % 128;
                    if offset < 64 {
                        element.lo |= 1u64 << offset;
                    } else {
                        element.hi |= 1u64 << (offset - 64);
                    }
                }
            }
        }
        packed
    }

    #[test]
    fn factored_claim_matches_the_bit_table() {
        let shape = shape();

        // Vary both dimensions to exercise the row/column split and coordinate order.
        let packed = packed_witness(&shape, |c, b| {
            let h =
                (b as u64).wrapping_mul(2654435761) ^ (c as u64).wrapping_mul(0x9E3779B97F4A7C15);
            (h >> 5) & 1 == 1
        });
        let table = params().table(&packed).unwrap();

        let row_images: Vec<F128> = (0..shape.rows())
            .map(|b| F128::from(((b as u128) + 1) * 0x9E3779B97F4A7C15u128 + 7))
            .collect();
        let zeta: Vec<F128> = (0..shape.log_columns())
            .map(|i| F128::from((i as u128 + 3) * 0xABCDEF12345u128 + 1))
            .collect();

        let images = vec![F128::ONE; shape.columns()];
        let folds = vec![0u128; shape.columns()];
        let fold = Fold::new(&shape, folds, images, row_images, zeta).unwrap();

        let mut prover = transcript::build_prover("order-check", &F128::ZERO);
        let query = gkr_reduce(&mut prover, &fold, &table, SecurityLevel::Bits100).unwrap();
        check_query(&query, &table);
    }

    #[test]
    fn factored_claim_matches_for_narrow_tables() {
        // The narrowest tables the transpose takes: one block of 128 columns,
        // then two.
        for log_columns in [7, 8] {
            let shape = Shape::new(22 - log_columns, log_columns).unwrap();
            let params = BitZParams::<F2>::new(shape, smallest_generator()).unwrap();
            let packed = packed_witness(&shape, |column, row| {
                let bits = (row as u64).wrapping_mul(0x9E3779B97F4A7C15)
                    ^ (column as u64).wrapping_mul(0xD1B54A32D192ED03);
                bits >> 63 & 1 == 1
            });
            let table = params.table(&packed).unwrap();
            let row_images = (0..shape.rows())
                .map(|row| F128::from((row as u128 + 1) * 0x9E3779B97F4A7C15 + 7))
                .collect();
            let zeta = (0..shape.log_columns())
                .map(|index| F128::from((index as u128 + 3) * 0xABCDEF12345 + 1))
                .collect();
            let fold = Fold::new(
                &shape,
                vec![0_u64; shape.columns()],
                vec![F128::ONE; shape.columns()],
                row_images,
                zeta,
            )
            .unwrap();

            let mut prover = transcript::build_prover("narrow-table", &F128::ZERO);
            let query = gkr_reduce(&mut prover, &fold, &table, SecurityLevel::Bits100).unwrap();
            check_query(&query, &table);
        }
    }

    // Identical columns make the leaf evaluation independent of column coordinates.
    #[test]
    fn b_only_pattern_isolates_the_row_pathway() {
        let shape = shape();

        let packed = packed_witness(&shape, |_c, b| (b * 2654435761) >> 5 & 1 == 1);
        let table = params().table(&packed).unwrap();

        let row_images: Vec<F128> = (0..shape.rows())
            .map(|b| F128::from(((b as u128) + 1) * 0x9E3779B97F4A7C15u128 + 7))
            .collect();
        let zeta: Vec<F128> = (0..shape.log_columns())
            .map(|i| F128::from((i as u128 + 3) * 0xABCDEF12345u128 + 1))
            .collect();

        let images = vec![F128::ONE; shape.columns()];
        let folds = vec![0u128; shape.columns()];
        let fold = Fold::new(&shape, folds, images, row_images, zeta).unwrap();

        let mut prover = transcript::build_prover("order-check-b", &F128::ZERO);
        let query = gkr_reduce(&mut prover, &fold, &table, SecurityLevel::Bits100).unwrap();
        check_query(&query, &table);
    }

    fn check_query(query: &OpeningQuery, table: &BitTable<'_>) {
        let OpeningQuery::InnerProduct { claim } = query else {
            panic!("expected a factored inner-product claim");
        };
        assert_eq!(claim.row_weights().len(), table.shape().rows());
        assert_eq!(claim.column_weights().len(), table.shape().columns());
        let mut target = F128::ZERO;
        for (column, &column_weight) in claim.column_weights().iter().enumerate() {
            for (row, &row_weight) in claim.row_weights().iter().enumerate() {
                if table.bit(column, row) {
                    target += row_weight * column_weight;
                }
            }
        }
        assert_eq!(target, claim.target());
    }
}
