use crate::{
    advice::{Advice, Requests, Slot},
    channel::{Channel, Result},
    field::F,
    fold::{agreement, Column},
    layers::{Activation, Mlp},
    linear,
    lookup::{Lookup, Lookups, Table},
    norm::{Claim, Width},
    rounding::Rounded,
    tensor::Shape,
};
use jolt_field::{One, Ring, Zero};

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Feed {
    gate: Option<Rounded>,
    up: Rounded,
    activations: Activated,
    product: Option<Rounded>,
    pub output: Rounded,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Activated {
    output: Option<Vec<i32>>,
    lookup: Lookup,
    ranges: [Slot; 4],
}

fn config(p: Activation<'_>) -> (&[i32], bool) {
    let half = p.correction.len() / 2;
    let symmetric = half > 1
        && i64::from(p.lower) == -(half as i64)
        && p.correction.first() == p.correction.last()
        && (1..half).all(|i| p.correction[half - i] == p.correction[half + i]);
    (
        if symmetric {
            &p.correction[half..]
        } else {
            p.correction
        },
        symmetric,
    )
}

fn address(value: i32, lower: i32, last: i64, symmetric: bool) -> (usize, [i64; 4]) {
    let x = i64::from(value);
    let abs = x.abs();
    let sign = i64::from(value >= 0);
    if symmetric {
        let address = abs.min(last);
        (address as usize, [sign, 0, 0, abs - address])
    } else {
        let x = x - i64::from(lower);
        let address = x.clamp(0, last);
        (
            address as usize,
            [sign, abs, (address - x).max(0), (x - address).max(0)],
        )
    }
}

impl Activated {
    fn prepare(
        advice: &mut Advice,
        p: Activation<'_>,
        shape: Shape,
        input: Option<&[i32]>,
    ) -> Result<Self> {
        let (table, symmetric) = config(p);
        if table.is_empty() || !table.len().is_power_of_two() {
            return Err("invalid activation table");
        }
        let last = table.len() as i64 - 1;
        let addresses = input.map(|x| {
            (0..shape.padded())
                .map(|i| {
                    shape
                        .index(i)
                        .map_or(0, |j| address(x[j], p.lower, last, symmetric).0 as u32)
                })
                .collect()
        });
        let lookup = Lookup::record(advice, shape, table.len(), addresses)?;
        let widths = [
            1,
            if symmetric { 0 } else { 32 },
            if symmetric { 0 } else { 33 },
            33,
        ];
        let slots = widths
            .into_iter()
            .enumerate()
            .map(|(i, width)| {
                advice.column(shape, Width::Unsigned(width), |j| {
                    i128::from(address(input.unwrap()[j], p.lower, last, symmetric).1[i])
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let output = input
            .map(|x| {
                x.iter()
                    .map(|&value| {
                        let index = address(value, p.lower, last, symmetric).0;
                        value
                            .max(0)
                            .checked_add(table[index])
                            .ok_or("activation exceeds i32")
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        Ok(Self {
            output,
            lookup,
            ranges: slots.try_into().unwrap(),
        })
    }

    fn reverse<'a>(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        lookups: &mut Lookups<'a>,
        p: Activation<'a>,
        shape: Shape,
        claim: Claim<F>,
        input: Option<&[i32]>,
        gated: bool,
        up: Option<&[i32]>,
    ) -> Result<(Claim<F>, Claim<F>)> {
        let (table, symmetric) = config(p);
        let vars = shape.bits();
        let len = shape.padded();
        let draw = io.sample(vars + 4, vars + 4)?;
        let (check, coefficients) = draw.split_at(vars);
        let last = table.len() as i64 - 1;
        let active = |i| shape.index(i).is_some();
        let relation = |v: &[F; 11]| {
            let [x, index, corr, sign, abs, low, high, mask, where_output, spot, up] = *v;
            let f_last = F::from_i64(last);
            let magnitude = if symmetric { index + high } else { abs };
            let signed = x - (sign + sign - F::one()) * magnitude;
            let clamp = if symmetric {
                coefficients[1] * high * (f_last - index)
            } else {
                coefficients[1] * (x - F::from_i32(p.lower) - index + low - high)
                    + coefficients[2] * low * index
                    + coefficients[3] * high * (f_last - index)
            };
            let y = sign * magnitude + corr;
            mask * (where_output * y * if gated { up } else { F::one() }
                + spot * (coefficients[0] * signed + clamp))
        };
        let (end, values) = io.fold(
            claim.value,
            vars,
            if gated { 5 } else { 4 },
            || {
                let x = input.unwrap();
                let indexes = self.lookup.addresses.as_ref().unwrap();
                let q = crate::fold::basis(&claim.point);
                let spot = crate::fold::basis(check);
                [
                    Column::source(len, |i| shape.at(x, i)),
                    Column::source(len, |i| F::from_u32(indexes[i])),
                    Column::source(len, |i| F::from_i32(table[indexes[i] as usize])),
                    Column::source(len, |i| {
                        shape.index(i).map_or(F::zero(), |j| {
                            F::from_i64(address(x[j], p.lower, last, symmetric).1[0])
                        })
                    }),
                    Column::source(len, |i| {
                        shape.index(i).map_or(F::zero(), |j| {
                            F::from_i64(address(x[j], p.lower, last, symmetric).1[1])
                        })
                    }),
                    Column::source(len, |i| {
                        shape.index(i).map_or(F::zero(), |j| {
                            F::from_i64(address(x[j], p.lower, last, symmetric).1[2])
                        })
                    }),
                    Column::source(len, |i| {
                        shape.index(i).map_or(F::zero(), |j| {
                            F::from_i64(address(x[j], p.lower, last, symmetric).1[3])
                        })
                    }),
                    Column::source(len, |i| F::from_bool(active(i))),
                    Column::table(q, len, 1),
                    Column::table(spot, len, 1),
                    Column::source(len, |i| up.map_or(F::zero(), |x| shape.at(x, i))),
                ]
            },
            [0, 1, 2, 3, 4, 5, 6, 10],
            relation,
        )?;
        let [x, a, t, sign, abs, low, high, up] = values;
        let point = end.challenges;
        if relation(&[
            x,
            a,
            t,
            sign,
            abs,
            low,
            high,
            shape.active(&point),
            agreement(&claim.point, &point),
            agreement(check, &point),
            up,
        ]) != end.output_claim
        {
            return Err("invalid activation terminal evaluation");
        }
        for (slot, value) in self.ranges.into_iter().zip([sign, abs, low, high]) {
            requests.push(slot, &point, value);
        }
        lookups.read(
            &self.lookup,
            Table::Values(table),
            point.clone(),
            a,
            (t, F::zero()),
        );
        Ok((
            Claim {
                point: point.clone(),
                value: x,
            },
            Claim { point, value: up },
        ))
    }
}

impl Feed {
    pub fn prepare(
        advice: &mut Advice,
        p: Mlp<'_>,
        shape: Shape,
        bits: u32,
        input: Option<&[i32]>,
    ) -> Result<Self> {
        let up = linear::project(advice, shape, bits, p.up, input)?;
        let gate = p
            .gate
            .map(|g| linear::project(advice, shape, bits, g, input))
            .transpose()?;
        let x = gate.as_ref().unwrap_or(&up);
        if x.shape != up.shape {
            return Err("invalid MLP gate width");
        }
        let activated = Activated::prepare(advice, p.activation, up.shape, x.values.as_deref())?;
        let product = gate
            .as_ref()
            .map(|_| {
                Rounded::evaluate(
                    advice,
                    up.shape,
                    bits,
                    true,
                    activated
                        .output
                        .as_ref()
                        .zip(up.values.as_ref())
                        .map(|(a, u)| {
                            move |row, col| {
                                let i = row * up.shape.cols + col;
                                i128::from(a[i]) * i128::from(u[i])
                            }
                        }),
                )
            })
            .transpose()?;
        let out = product
            .as_ref()
            .map_or(activated.output.as_deref(), |x| x.values.as_deref());
        let output = linear::project(advice, up.shape, bits, p.down, out)?;
        Ok(Self {
            gate,
            up,
            activations: activated,
            product,
            output,
        })
    }

    pub fn reverse<'a>(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        lookups: &mut Lookups<'a>,
        p: Mlp<'a>,
        shape: Shape,
        bits: u32,
        claim: Claim<F>,
        input: Option<&[i32]>,
    ) -> Result<Vec<Claim<F>>> {
        let output = self
            .product
            .as_ref()
            .map_or(self.activations.output.as_deref(), |x| x.values.as_deref());
        let mut claim = linear::reverse(
            io,
            requests,
            self.up.shape,
            bits,
            &[(&self.output, p.down, claim)],
            output,
        )?
        .remove(0);
        if let Some(product) = &self.product {
            claim.value = product.raw(io, requests, &claim)?;
        }
        let x = self.gate.as_ref().unwrap_or(&self.up).values.as_deref();
        let (activated, up) = self.activations.reverse(
            io,
            requests,
            lookups,
            p.activation,
            self.up.shape,
            claim,
            x,
            self.gate.is_some(),
            self.up.values.as_deref(),
        )?;
        let data = match (&self.gate, p.gate) {
            (Some(g), Some(projection)) => vec![(g, projection, activated), (&self.up, p.up, up)],
            _ => vec![(&self.up, p.up, activated)],
        };
        linear::reverse(io, requests, shape, bits, &data, input)
    }
}
