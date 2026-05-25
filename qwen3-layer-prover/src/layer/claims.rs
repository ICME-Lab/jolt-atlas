use joltworks::{field::JoltField, poly::eq_poly::EqPolynomial, transcripts::Transcript};

use crate::claim::{Claim, Poly, Shape};

use super::types::LayerShape;

// Claims that start or leave the layer IOP.
//
// The output claim is not caller-provided.  It is sampled after all layer
// commitments have been absorbed, then evaluated from the committed
// `hidden_out` witness.  The verifier repeats the same challenge draw and
// checks that the proof uses that point.

pub(crate) fn claim_hidden_out_after_commitments<F, T>(
    transcript: &mut T,
    hidden_out: &[i32],
    shape: &LayerShape,
) -> Claim<F>
where
    F: JoltField,
    T: Transcript,
{
    let point = draw_hidden_out_point(transcript, shape);
    hidden_out_claim(hidden_out, shape, point)
}

pub(crate) fn draw_hidden_out_point<F, T>(transcript: &mut T, shape: &LayerShape) -> Vec<F>
where
    F: JoltField,
    T: Transcript,
{
    transcript.challenge_vector::<F>(shape.hidden_shape().padded_power_of_two().point_len())
}

pub(crate) fn hidden_out_claim<F: JoltField>(
    hidden_out: &[i32],
    shape: &LayerShape,
    point: Vec<F>,
) -> Claim<F> {
    let hidden_shape = shape.hidden_shape();
    let poly = Poly::new(
        joltworks::poly::multilinear_polynomial::MultilinearPolynomial::from(padded_i32(
            hidden_out,
            &hidden_shape,
        )),
        None,
    );
    Claim {
        value: evaluate_i32_mle(hidden_out, &hidden_shape, &point),
        point,
        poly,
    }
}

pub(crate) fn evaluate_i32_mle<F: JoltField>(values: &[i32], shape: &Shape, point: &[F]) -> F {
    let domain_shape = shape.padded_power_of_two();
    assert_eq!(point.len(), domain_shape.point_len());
    let eq = EqPolynomial::<F>::evals(point);
    (0..domain_shape.numel())
        .map(|idx| {
            let value = values.get(idx).copied().unwrap_or(0);
            F::from_i32(value) * eq[idx]
        })
        .sum()
}

pub(crate) fn point_matches_claim<F: JoltField>(claim: &Claim<F>, point: &[F]) -> bool {
    claim.point == point
}

fn padded_i32(values: &[i32], shape: &Shape) -> Vec<i32> {
    let len = shape.padded_power_of_two().numel();
    let mut out = vec![0; len];
    out[..values.len()].copy_from_slice(values);
    out
}
