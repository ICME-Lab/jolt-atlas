use ark_bn254::Fr;
use ark_ff::{Field, One};
use itertools::Itertools;
use joltworks::field::JoltField;
use joltworks::poly::{
    multilinear_polynomial::BindingOrder, split_eq_poly::GruenSplitEqPolynomial,
};
use qwen3_common::MatrixShape;

pub(crate) fn eq_point_eval(left: &[Fr], right: &[<Fr as JoltField>::Challenge]) -> Option<Fr> {
    (left.len() == right.len()).then_some(())?;
    let mut eq = GruenSplitEqPolynomial::new(left, BindingOrder::LowToHigh);
    for challenge in right.iter().rev() {
        eq.bind(*challenge);
    }
    (eq.len() == 1).then_some(eq.get_current_scalar())
}

pub(crate) fn bits_to_rem<const FRAC_BITS: usize>(bits: &[Fr; FRAC_BITS]) -> Fr {
    bits.iter()
        .enumerate()
        .map(|(bit, value)| Fr::from(1_u64 << bit) * *value)
        .sum()
}

pub(crate) fn combine_eq_points(left: &[Fr], right: &[Fr]) -> Option<(Fr, Vec<Fr>)> {
    (left.len() == right.len()).then_some(())?;
    let mut scalar = Fr::one();
    let mut point = Vec::with_capacity(left.len());
    for (left, right) in left.iter().zip_eq(right) {
        let denom = (Fr::one() - left) * (Fr::one() - right) + *left * *right;
        let inv = Field::inverse(&denom)?;
        scalar *= denom;
        point.push(*left * *right * inv);
    }
    Some((scalar, point))
}

pub(crate) fn log2(value: usize) -> usize {
    value.ilog2() as usize
}

pub(crate) fn eval_i32_mle_at_point(values: &[i32], point: &[Fr]) -> Option<Fr> {
    (values.len() == (1_usize << point.len())).then_some(())?;
    let split_eq_point = point.iter().rev().copied().collect::<Vec<_>>();
    let eq = GruenSplitEqPolynomial::<Fr>::new(&split_eq_point, BindingOrder::LowToHigh).merge();
    Some(
        values
            .iter()
            .zip_eq(eq.evals_ref())
            .map(|(value, eq)| *eq * Fr::from_i32(*value))
            .sum(),
    )
}

pub(crate) fn eval_i32_matrix_at_point(
    values: &[i32],
    shape: MatrixShape,
    point: &[Fr],
) -> Option<Fr> {
    (values.len() == shape.len()).then_some(())?;
    (point.len() == shape.point_len()).then_some(())?;
    eval_i32_mle_at_point(values, point)
}
