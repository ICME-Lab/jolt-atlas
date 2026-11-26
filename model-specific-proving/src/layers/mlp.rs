use std::sync::Arc;

use jolt_field::{One, Ring, Zero};

use super::{
    linear::{Hadamard, Linear, Rounded},
    Activation, Mlp,
};
use crate::protocol::{
    field::F,
    lookup::{Addresses, Lookup, Lookups, Table},
    opening::{Layout, Packed, Request},
    range::{Ranges, Width},
    sumcheck::{Column, Session},
    tensor::{equality, inner, weights, Claim, Shape},
    Result,
};

struct Activate<'a> {
    shape: Shape,
    correction: &'a [i32],
    lower: i32,
    symmetric: bool,
    lookup: Lookup,
    constraints: Ranges<3>,
    excess: Ranges<1>,
}

struct ActivationTrace {
    output: Vec<i32>,
    addresses: Arc<Addresses>,
}

impl<'a> Activate<'a> {
    fn new(layout: &mut Layout, shape: Shape, table: Activation<'a>) -> Result<Self> {
        let size = table.correction.len();
        if size < 2
            || !size.is_power_of_two()
            || i64::from(table.lower) + (size - 1) as i64 > i64::from(i32::MAX)
        {
            return Err("invalid activation table");
        }
        let half = size / 2;
        let symmetric = half > 1
            && i64::from(table.lower) == -(half as i64)
            && table.correction[0] == table.correction[size - 1]
            && (1..half).all(|i| table.correction[half - i] == table.correction[half + i]);
        let correction = if symmetric {
            &table.correction[half..]
        } else {
            table.correction
        };
        let unsigned = Width::unsigned;
        let constraints = Ranges::new(
            layout,
            shape,
            [
                unsigned(1),
                unsigned(if symmetric { 0 } else { 32 }),
                unsigned(if symmetric { 0 } else { 32 }),
            ],
        );
        let lookup = Lookup::new(layout, shape.bits(), correction.len())?;
        Ok(Self {
            shape,
            correction,
            lower: table.lower,
            symmetric,
            lookup,
            constraints,
            excess: Ranges::adaptive(layout, shape, 32),
        })
    }

    fn cell(&self, x: i32) -> (usize, [i64; 4]) {
        let x = i64::from(x);
        let magnitude = x.abs();
        let sign = i64::from(x >= 0);
        let last = self.correction.len() as i64 - 1;
        if self.symmetric {
            let address = magnitude.min(last);
            (address as usize, [sign, 0, 0, magnitude - address])
        } else {
            let shifted = x - i64::from(self.lower);
            let address = shifted.clamp(0, last);
            (
                address as usize,
                [
                    sign,
                    magnitude,
                    (address - shifted).max(0),
                    (shifted - address).max(0),
                ],
            )
        }
    }

    fn execute(&self, x: &[i32], packed: &mut Packed<'_>) -> Result<ActivationTrace> {
        let stride = self.shape.width.next_power_of_two();
        let mut addresses = Addresses::new(1 << self.shape.bits(), self.lookup.bits);
        let mut columns: [Vec<i64>; 4] = std::array::from_fn(|_| Vec::with_capacity(x.len()));
        let mut output = Vec::with_capacity(x.len());
        for (i, &value) in x.iter().enumerate() {
            let (address, range) = self.cell(value);
            addresses.set(
                i / self.shape.width * stride + i % self.shape.width,
                address,
            );
            for (column, number) in columns.iter_mut().zip(range) {
                column.push(number);
            }
            output.push(
                value
                    .max(0)
                    .checked_add(self.correction[address])
                    .ok_or("activation overflow")?,
            );
        }
        let [sign, absolute, low, excess] = &columns;
        self.constraints.record(packed, [sign, absolute, low])?;
        self.excess.record_adaptive(packed, excess)?;
        self.lookup.record(packed, &addresses)?;
        Ok(ActivationTrace {
            output,
            addresses: Arc::new(addresses),
        })
    }

    fn reverse<'w>(
        &self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        lookups: &mut Lookups<'a>,
        claim: Claim,
        witness: impl Fn() -> (&'w ActivationTrace, &'w [i32]),
    ) -> Result<Claim> {
        let shape = self.shape;
        let vars = shape.bits();
        let stride = shape.width.next_power_of_two();
        let len = 1 << vars;
        let random = io.random(vars + 4, vars + 4)?;
        let (p, coefficients) = random.split_at(vars);
        let last = F::from_u64((self.correction.len() - 1) as u64);
        let lower = F::from_i32(self.lower);
        let formula = |v: &[F]| {
            let [x, index, correction, sign, absolute, low, high, active, q, p] =
                v.try_into().expect("activation columns");
            let magnitude = if self.symmetric {
                index + high
            } else {
                absolute
            };
            let signed = x - (sign + sign - F::one()) * magnitude;
            let clamp = if self.symmetric {
                coefficients[1] * high * (last - index)
            } else {
                inner(
                    coefficients[1..].iter().copied(),
                    [
                        x - lower - index + low - high,
                        low * index,
                        high * (last - index),
                    ],
                )
            };
            active * (q * (sign * magnitude + correction) + p * (coefficients[0] * signed + clamp))
        };
        let active = |i| i / stride < shape.rows && i % stride < shape.width;
        let (end, v) = io.fold(
            claim.value,
            vars,
            4,
            || {
                let (trace, x) = witness();
                let mut columns = vec![
                    Column::source(len, move |i| shape.at(x, i)),
                    Column::source(len, move |i| F::from_u64(trace.addresses.get(i) as u64)),
                    Column::source(len, move |i| {
                        F::from_i32(self.correction[trace.addresses.get(i)])
                    }),
                ];
                for j in 0..4 {
                    if self.symmetric && matches!(j, 1 | 2) {
                        columns.push(Column::axis(len, vars, vec![F::zero()]));
                        continue;
                    }
                    columns.push(Column::source(len, move |i| {
                        if active(i) {
                            F::from_i64(self.cell(x[i / stride * shape.width + i % stride]).1[j])
                        } else {
                            F::zero()
                        }
                    }));
                }
                columns.extend([
                    Column::source(len, move |i| F::from_bool(active(i))),
                    Column::split_equality(&claim.point),
                    Column::split_equality(p),
                ]);
                columns
            },
            [0, 1, 2, 3, 4, 5, 6],
            formula,
        )?;
        io.expect(&end, || {
            let (r, c) = end.challenges.split_at(shape.row_bits());
            let rows: F = weights(r).into_iter().take(shape.rows).sum();
            let cols: F = weights(c).into_iter().take(shape.width).sum();
            let mut values = v.to_vec();
            values.extend([
                rows * cols,
                equality(&claim.point, &end.challenges),
                equality(p, &end.challenges),
            ]);
            formula(&values)
        })?;
        self.constraints
            .open(io, requests, &end.challenges, [v[3], v[4], v[5]])?;
        self.excess.open(io, requests, &end.challenges, [v[6]])?;
        lookups.read(
            &self.lookup,
            &[Table::Direct(self.correction)],
            end.challenges.clone(),
            v[1],
            vec![v[2]],
            io.proving().then(|| witness().0.addresses.clone()),
        );
        Ok(Claim::new(end.challenges, v[0]))
    }
}

pub(crate) struct FeedForward<'a> {
    gate: Linear<'a>,
    activation: Activate<'a>,
    up: Option<(Linear<'a>, Hadamard)>,
    down: Linear<'a>,
    input: Shape,
}

pub(crate) struct FeedTrace {
    gate: Rounded,
    activation: ActivationTrace,
    up: Option<(Rounded, Rounded)>,
    down: Rounded,
}

impl FeedTrace {
    pub(crate) fn values(&self) -> &[i32] {
        &self.down.values
    }
}

impl<'a> FeedForward<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        input: Shape,
        bits: usize,
        mlp: Mlp<'a>,
    ) -> Result<Self> {
        let gate = Linear::new(layout, input, mlp.gate.unwrap_or(mlp.up), bits)?;
        let shape = gate.shape();
        let activation = Activate::new(layout, shape, mlp.activation)?;
        let up = if mlp.gate.is_some() {
            let up = Linear::new(layout, input, mlp.up, bits)?;
            if up.shape() != shape {
                return Err("invalid gated MLP");
            }
            Some((up, Hadamard::new(layout, shape, bits)))
        } else {
            None
        };
        let down = Linear::new(layout, shape, mlp.down, bits)?;
        if down.shape() != input {
            return Err("invalid MLP output");
        }
        Ok(Self {
            gate,
            activation,
            up,
            down,
            input,
        })
    }

    pub(crate) fn execute(&self, input: &[i32], packed: &mut Packed<'_>) -> Result<FeedTrace> {
        let gate = self.gate.execute(input, packed)?;
        let activation = self.activation.execute(&gate.values, packed)?;
        let up = self
            .up
            .as_ref()
            .map(|(up, mul)| {
                let up = up.execute(input, packed)?;
                let product = mul.execute(&activation.output, &up.values, packed)?;
                Ok((up, product))
            })
            .transpose()?;
        let values = up
            .as_ref()
            .map_or(activation.output.as_slice(), |(_, y)| &y.values);
        let down = self.down.execute(values, packed)?;
        Ok(FeedTrace {
            gate,
            activation,
            up,
            down,
        })
    }

    pub(crate) fn reverse<'w>(
        &self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        lookups: &mut Lookups<'a>,
        claim: Claim,
        witness: impl Fn() -> (&'w FeedTrace, &'w [i32]),
    ) -> Result<Claim> {
        let claim = self.down.reverse(io, requests, claim, || {
            let (w, _) = witness();
            (
                &w.down,
                w.up.as_ref()
                    .map_or(w.activation.output.as_slice(), |(_, y)| &y.values),
            )
        })?;
        let (claim, up) = if let Some((up, mul)) = &self.up {
            let [gate, branch] = mul.reverse(io, requests, claim, || {
                let (w, _) = witness();
                let (up, product) = w.up.as_ref().expect("gated trace");
                (product, &w.activation.output, &up.values)
            })?;
            (
                gate,
                Some(up.reverse(io, requests, branch, || {
                    let (w, x) = witness();
                    (&w.up.as_ref().expect("up trace").0, x)
                })?),
            )
        } else {
            (claim, None)
        };
        let claim = self.activation.reverse(io, requests, lookups, claim, || {
            let (w, _) = witness();
            (&w.activation, &w.gate.values)
        })?;
        let gate = self.gate.reverse(io, requests, claim, || {
            let (w, x) = witness();
            (&w.gate, x)
        })?;
        if let Some(up) = up {
            io.combine(&[gate, up], || {
                let x = witness().1;
                (0..1 << self.input.bits())
                    .map(|i| self.input.at(x, i))
                    .collect()
            })
        } else {
            Ok(gate)
        }
    }
}
