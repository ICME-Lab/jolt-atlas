use crate::{
    advice::{Advice, Requests},
    channel::{Channel, Result},
    field::F,
    fold::basis,
    norm::Claim,
    projection::{Input, Projection, Query},
    rounding::Rounded,
    tensor::Shape,
};
use jolt_field::{Ring, Zero};

pub(crate) fn project(
    advice: &mut Advice,
    shape: Shape,
    bits: u32,
    p: Projection<'_>,
    values: Option<&[i32]>,
) -> Result<Rounded> {
    let output = Shape::new(shape.rows, p.bias.len())?;
    let (out, remainder) = values
        .map(|input| {
            Input {
                rows: shape.rows,
                width: shape.cols,
                bits,
            }
            .execute(input, p)
        })
        .transpose()?
        .map_or((None, None), |(a, b)| (Some(a), Some(b)));
    Rounded::record(advice, output, bits, true, out, remainder)
}

pub(crate) fn reverse(
    io: &mut Channel<'_, '_>,
    requests: &mut Requests,
    shape: Shape,
    bits: u32,
    data: &[(&Rounded, Projection<'_>, Claim<F>)],
    input: Option<&[i32]>,
) -> Result<Vec<Claim<F>>> {
    let parameters = Input {
        rows: shape.rows,
        width: shape.cols,
        bits,
    };
    let queries = data
        .iter()
        .map(|(rounded, projection, claim)| {
            Ok(Query {
                projection: *projection,
                output: claim,
                remainder: rounded.remainder(io, requests, claim)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    match io {
        Channel::Send(t, messages, _) => {
            let (proof, claims) = parameters.prove(input.unwrap(), &queries, *t)?;
            messages.projections.push(proof);
            Ok(claims)
        }
        Channel::Check(t, _, _, _, projections) => {
            let (proof, rest) = projections
                .split_first()
                .ok_or("missing projection proof")?;
            *projections = rest;
            parameters.verify(&queries, proof, *t)
        }
    }
}

#[derive(Clone, Copy)]
pub struct Rotary<'a> {
    pub cos: &'a [i32],
    pub sin: &'a [i32],
}

impl Rotary<'_> {
    fn pair(self, shape: Shape, row: usize, col: usize) -> (i128, i128) {
        let half = shape.cols / 2;
        let index = row * half + (col % half);
        (
            i128::from(self.cos[index]),
            i128::from(self.sin[index]) * if col < half { -1 } else { 1 },
        )
    }

    pub(crate) fn apply(
        self,
        advice: &mut Advice,
        shape: Shape,
        bits: u32,
        values: Option<&[i32]>,
    ) -> Result<Rounded> {
        if shape.cols < 2
            || !shape.cols.is_power_of_two()
            || self.cos.len() != shape.rows * shape.cols / 2
            || self.sin.len() != self.cos.len()
        {
            return Err("invalid RoPE table");
        }
        Rounded::evaluate(
            advice,
            shape,
            bits,
            true,
            values.map(|x| {
                move |row: usize, col: usize| {
                    let (a, b) = self.pair(shape, row, col);
                    let i = row * shape.cols;
                    a * i128::from(x[i + col]) + b * i128::from(x[i + (col ^ (shape.cols / 2))])
                }
            }),
        )
    }

    pub(crate) fn reverse(
        self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        rounded: &Rounded,
        claims: &[Claim<F>],
        values: Option<&[i32]>,
    ) -> Result<Claim<F>> {
        let shape = rounded.shape;
        let mixing = io.sample(claims.len(), claims.len())?;
        let mut sum = F::zero();
        let mut weights = vec![F::zero(); shape.padded()];
        for (claim, a) in claims.iter().zip(mixing) {
            sum += a * rounded.raw(io, requests, claim)?;
            for (x, w) in weights.iter_mut().zip(basis(&claim.point)) {
                *x += a * w;
            }
        }
        let half = shape.cols / 2;
        for (row, coeff) in weights
            .chunks_exact_mut(shape.cols)
            .take(shape.rows)
            .enumerate()
        {
            for col in 0..half {
                let (a, b) = self.pair(shape, row, col);
                let (x, y) = (coeff[col], coeff[col + half]);
                coeff[col] = x * F::from_i128(a) - y * F::from_i128(b);
                coeff[col + half] = y * F::from_i128(a) + x * F::from_i128(b);
            }
        }
        io.dot(sum, weights, || {
            (0..shape.padded())
                .map(|i| shape.at(values.unwrap(), i))
                .collect()
        })
    }
}
