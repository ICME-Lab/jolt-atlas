use ark_bn254::Fr;
use joltworks::{field::JoltField, transcripts::Transcript};

use crate::pedersen::{Commitment, PedersenParams};

pub(crate) fn validate_tables(num_vars: usize, tables: &[&[Fr]]) -> Option<()> {
    if num_vars == 0 {
        return None;
    }
    let expected_len = 1_usize.checked_shl(num_vars as u32)?;
    if expected_len < 2 {
        return None;
    }
    tables
        .iter()
        .all(|table| table.len() == expected_len && table.len().is_power_of_two())
        .then_some(())
}

pub(crate) fn hadamard_values(lhs: &[Fr], rhs: &[Fr]) -> Vec<Fr> {
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "Hadamard inputs must have same length"
    );
    lhs.iter().zip(rhs).map(|(lhs, rhs)| *lhs * rhs).collect()
}

pub(crate) fn evaluate_mle(values: &[Fr], point: &[<Fr as JoltField>::Challenge]) -> Fr {
    assert_eq!(values.len(), 1 << point.len(), "point/table size mismatch");
    let mut bound = values.to_vec();
    for r in point {
        let r: Fr = (*r).into();
        let len = bound.len() / 2;
        for i in 0..len {
            let lo = bound[2 * i];
            let hi = bound[2 * i + 1];
            bound[i] = lo + r * (hi - lo);
        }
        bound.truncate(len);
    }
    bound[0]
}

pub(crate) fn evaluate_eq(
    point: &[<Fr as JoltField>::Challenge],
    eval_point: &[<Fr as JoltField>::Challenge],
) -> Fr {
    assert_eq!(point.len(), eval_point.len(), "eq point length mismatch");
    point
        .iter()
        .zip(eval_point)
        .map(|(x, y)| {
            let x: Fr = (*x).into();
            let y: Fr = (*y).into();
            Fr::from(1_u64) - x - y + Fr::from(2_u64) * x * y
        })
        .product()
}

pub(crate) fn absorb_gkr_statement<T: Transcript>(
    params: &PedersenParams,
    output_point: &[<Fr as JoltField>::Challenge],
    output_commitment: &Commitment,
    transcript: &mut T,
) {
    transcript.append_message(b"cs/three-product-gkr/v1");
    transcript.append_point(&params.value_generator);
    transcript.append_point(&params.blinding_generator);
    transcript.append_u64(output_point.len() as u64);
    for r in output_point {
        let r: Fr = (*r).into();
        transcript.append_scalar(&r);
    }
    transcript.append_point(&output_commitment.0);
}

pub(crate) fn absorb_layer_label<T: Transcript>(label: &'static [u8], transcript: &mut T) {
    transcript.append_message(b"cs/product-layer/v1");
    transcript.append_message(label);
}
