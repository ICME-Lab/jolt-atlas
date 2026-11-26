//! Fixed-table activation and normalization constraints.

use std::sync::Arc;

use jolt_poly::MultilinearPoly;
use jolt_transcript::Transcript;

use crate::protocol::{
    lookup::{Addresses, Batches, Lookup, Read, Table},
    pcs::{bind_i32, Advice, Kind, Layout, Opening, Part, Slot},
    range::{Range, Ranges, I32},
    session::{instance, query_column, query_weight, Claim, Column, ProofField, Session},
    tensor::{dot, eq, weights, Shape},
    Result,
};

#[derive(Clone, Copy)]
pub struct Activation<'a> {
    pub lower: i32,
    pub correction: &'a [i32],
}

pub(crate) struct Activate<'a> {
    shape: Shape,
    table: Activation<'a>,
    proof: ActivationProof,
    lookup: Lookup,
}

enum ActivationProof {
    Direct(Box<[i32]>),
    Clamp {
        ranges: Ranges,
        sign: Slot,
        symmetric: bool,
    },
}

pub(crate) struct ActivationTrace {
    pub(crate) output: Vec<i32>,
    indices: Arc<Addresses>,
    signs: Option<Arc<Vec<u8>>>,
}

impl<'a> Activate<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        shape: Shape,
        table: Activation<'a>,
        input_bound: Option<u128>,
    ) -> Result<Self> {
        let size = table.correction.len();
        if size < 2 || !size.is_power_of_two() {
            return Err("invalid activation table size");
        }
        let upper = i64::from(table.lower) + (table.correction.len() - 1) as i64;
        if upper > i64::from(i32::MAX) {
            return Err("invalid activation bounds");
        }
        let direct = input_bound
            .filter(|&bound| {
                i128::from(table.lower) <= -(bound as i128) && bound as i128 <= i128::from(upper)
            })
            .and_then(|_| {
                table
                    .correction
                    .iter()
                    .enumerate()
                    .map(|(index, &correction)| {
                        let input = i64::from(table.lower) + index as i64;
                        i32::try_from(input.max(0) + i64::from(correction))
                    })
                    .collect::<std::result::Result<Box<[_]>, _>>()
                    .ok()
            });
        let half = size / 2;
        let symmetric = direct.is_none()
            && half >= 2
            && i64::from(table.lower) == -(half as i64)
            && table.correction[0] == table.correction[size - 1]
            && (1..half).all(|i| table.correction[half - i] == table.correction[half + i]);
        let lookup = Lookup::new(layout, shape.vars(), if symmetric { half } else { size })?;
        let proof = match direct {
            Some(values) => ActivationProof::Direct(values),
            None => ActivationProof::Clamp {
                ranges: Ranges::new(
                    layout,
                    shape,
                    if symmetric {
                        [Range::unsigned(0), Range::unsigned(31)]
                    } else {
                        [Range::unsigned(32); 2]
                    },
                )?,
                sign: layout.add(shape.vars(), Kind::Bits { planes: 1 })?,
                symmetric,
            },
        };
        Ok(Self {
            shape,
            table,
            proof,
            lookup,
        })
    }

    pub(crate) fn bind(&self, t: &mut impl Transcript) {
        t.append(&(i64::from(self.table.lower) as u64));
        bind_i32(t, self.table.correction);
        t.append(&match self.proof {
            ActivationProof::Direct(_) => 1u64,
            ActivationProof::Clamp {
                symmetric: true, ..
            } => 2,
            ActivationProof::Clamp { .. } => 0,
        });
    }

    pub(crate) fn execute(&self, x: &[i32], a: &mut Advice) -> Result<ActivationTrace> {
        self.shape.check(x)?;
        let lower = i64::from(self.table.lower);
        let upper = lower + (self.table.correction.len() - 1) as i64;
        let (ranges, sign, symmetric) = match &self.proof {
            ActivationProof::Direct(table) => {
                let zero = (-lower) as u32;
                let stride = self.shape.width.next_power_of_two();
                let mut indices = vec![zero; 1 << self.shape.vars()];
                let mut output = Vec::with_capacity(x.len());
                for (row, input) in x.chunks_exact(self.shape.width).enumerate() {
                    for (col, &value) in input.iter().enumerate() {
                        let index = usize::try_from(i64::from(value) - lower)
                            .map_err(|_| "activation input below table")?;
                        output.push(*table.get(index).ok_or("activation input above table")?);
                        indices[row * stride + col] = index as u32;
                    }
                }
                let indices = Arc::new(Addresses::Words(indices));
                self.lookup.record(a, Arc::clone(&indices))?;
                return Ok(ActivationTrace {
                    output,
                    indices,
                    signs: None,
                });
            }
            ActivationProof::Clamp {
                ranges,
                sign,
                symmetric,
            } => (ranges, *sign, *symmetric),
        };
        if lower > 0 || upper < 0 {
            return Err("activation domain must contain zero");
        }
        let mut output = Vec::with_capacity(x.len());
        let mut magnitude = Vec::with_capacity(x.len());
        let mut excess = Vec::with_capacity(x.len());
        let size = 1 << self.shape.vars();
        let stride = self.shape.width.next_power_of_two();
        let table = self.correction(symmetric);
        let mut indices = vec![if symmetric { 0 } else { (-lower) as u32 }; size];
        let mut signs = vec![0; size];
        for (row, values) in x.chunks_exact(self.shape.width).enumerate() {
            for (col, &value) in values.iter().enumerate() {
                let x = i64::from(value);
                let index = if symmetric {
                    x.unsigned_abs().min((table.len() - 1) as u64) as u32
                } else {
                    (x.clamp(lower, upper) - lower) as u32
                };
                let correction = i64::from(table[index as usize]);
                indices[row * stride + col] = index;
                signs[row * stride + col] = u8::from(x >= 0);
                output
                    .push(i32::try_from(x.max(0) + correction).map_err(|_| "activation overflow")?);
                if symmetric {
                    excess.push(x.unsigned_abs() as u32 - index);
                } else {
                    magnitude.push(x.unsigned_abs() as u32);
                    excess.push((x - lower - i64::from(index)).unsigned_abs() as u32);
                }
            }
        }
        if symmetric {
            ranges.record_packed(a, &[&magnitude, &excess])?;
        } else {
            ranges.record(a, &[&magnitude, &excess])?;
        }
        let signs = Arc::new(signs);
        a.put(sign, Part::Digits(Arc::clone(&signs)))?;
        let indices = Arc::new(Addresses::Words(indices));
        self.lookup.record(a, Arc::clone(&indices))?;
        Ok(ActivationTrace {
            output,
            indices,
            signs: Some(signs),
        })
    }

    fn correction(&self, symmetric: bool) -> &'a [i32] {
        let values = self.table.correction;
        if symmetric {
            &values[values.len() / 2..]
        } else {
            values
        }
    }

    pub(crate) fn reverse<'s, 'w, F: ProofField>(
        &'s self,
        q: Claim<F>,
        trace: impl Fn() -> (&'w [i32], &'w ActivationTrace) + Sync,
        io: &mut Session<'_, '_, F>,

        lookups: &mut Batches<'s, F>,
    ) -> Result<Claim<F>> {
        let (ranges, sign, symmetric) = match &self.proof {
            ActivationProof::Direct(table) => {
                let (row, col) = q.point.split_at(self.shape.row_vars());
                let rows = weights(row);
                let cols = weights(col);
                let active = rows.iter().take(self.shape.rows).copied().sum::<F>()
                    * cols.iter().take(self.shape.width).copied().sum::<F>();
                let [address] = io.values(|| {
                    [dot(
                        (0..rows.len()).map(|row| {
                            dot(
                                (0..cols.len()).map(|col| {
                                    F::from_u64(u64::from(
                                        trace().1.indices.get(row * cols.len() + col),
                                    ))
                                }),
                                cols.iter().copied(),
                            )
                        }),
                        rows.iter().copied(),
                    )]
                })?;
                // Tensor claims have zero padding; the table evaluates padded
                // cells at input zero, which can have a nonzero table output.
                let at_zero = table[(-i64::from(self.table.lower)) as usize];
                let output = q.value + (F::one() - active) * F::from_i32(at_zero);
                let claim = Claim {
                    point: q.point.clone(),
                    value: address + F::from_i32(self.table.lower),
                };
                lookups.push(
                    &[Table::Slice(table)],
                    &self.lookup,
                    Read {
                        cell: q.point,
                        address,
                        output: Box::new([output]),
                        witness: io.proving().then(|| Arc::clone(&trace().1.indices)),
                    },
                );
                return Ok(claim);
            }
            ActivationProof::Clamp {
                ranges,
                sign,
                symmetric,
            } => (ranges, *sign, *symmetric),
        };
        let vars = self.shape.vars();
        io.pow(vars + 3)?;
        let mix = io.challenge(3)?;
        let random = io.challenge(vars)?;
        let lower = F::from_i32(self.table.lower);
        let table = self.correction(symmetric);
        let last = F::from_u64((table.len() - 1) as u64);
        let magnitude = |v: &[F]| if symmetric { v[1] + v[5] } else { v[3] };
        let clamp = |v: &[F]| {
            let signed = v[0] - (v[4] + v[4] - F::one()) * magnitude(v);
            if symmetric {
                mix[0] * signed + mix[1] * v[5] * (v[1] - last)
            } else {
                dot(
                    [
                        last * (v[0] - v[1] - lower) - v[5] * (v[1] + v[1] - last),
                        v[5] * v[1] * (v[1] - last),
                        signed,
                    ],
                    mix.iter().copied(),
                )
            }
        };
        let shape = self.shape;
        let width = shape.width.next_power_of_two();
        let size = 1 << vars;
        let active = |i: usize| i / width < shape.rows && i % width < shape.width;
        let replay = io.source_replay_rounds(vars);
        let (point, v) = io.relation(
            instance(q.value, vars, 4),
            || {
                let (input, trace) = trace();
                let x = move |i: usize| {
                    if active(i) {
                        i64::from(input[(i / width) * shape.width + i % width])
                    } else {
                        0
                    }
                };
                let address = move |i| trace.indices.get(i);
                [
                    Column::source(size, move |i| F::from_i64(x(i))),
                    Column::source(size, move |i| F::from_u64(u64::from(address(i)))),
                    Column::source(size, move |i| F::from_i32(table[address(i) as usize])),
                    if symmetric {
                        Column::repeat(vec![F::zero()].into(), size)
                    } else {
                        Column::source(size, move |i| F::from_i64(x(i).abs()))
                    },
                    Column::Bits(trace.signs.as_deref().expect("activation sign witness"), 1),
                    Column::source(size, move |i| {
                        let surplus = if symmetric {
                            x(i).abs() - i64::from(address(i))
                        } else {
                            (x(i) - i64::from(self.table.lower) - i64::from(address(i))).abs()
                        };
                        F::from_i64(surplus)
                    }),
                    Column::source(size, move |i| F::from_bool(active(i))),
                    Column::equality(&q.point),
                    Column::equality(&random),
                ]
                .map(|column| column.replay(replay))
            },
            |v| v[6] * v[7] * (v[4] * magnitude(v) + v[2]) + v[8] * clamp(v),
            |p, v: &[F; 6]| {
                let (row, col) = p.split_at(shape.row_vars());
                let active = weights::<F>(row).into_iter().take(shape.rows).sum::<F>()
                    * weights::<F>(col).into_iter().take(shape.width).sum::<F>();
                active * eq(&q.point, p) * (v[4] * magnitude(v) + v[2]) + eq(&random, p) * clamp(v)
            },
        )?;
        ranges.check(&[(&point, [v[3], v[5]])], io)?;
        io.openings.push(Opening::new(
            sign,
            Claim {
                point: point.clone(),
                value: v[4],
            },
        ));
        lookups.push(
            &[Table::Slice(table)],
            &self.lookup,
            Read {
                cell: point.clone(),
                address: v[1],
                output: Box::new([v[2]]),
                witness: io.proving().then(|| Arc::clone(&trace().1.indices)),
            },
        );
        Ok(Claim { point, value: v[0] })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NormalizationKind {
    Layer,
    Rms,
}

impl NormalizationKind {
    fn stats(self) -> usize {
        if self == Self::Layer {
            9
        } else {
            6
        }
    }
}

#[derive(Clone, Copy)]
pub struct NormalizationParameters<'a> {
    pub kind: NormalizationKind,
    pub gain: &'a [i32],
    pub bias: &'a [i32],
    /// Epsilon quantized in the same units as the variance.
    pub epsilon_variance_units: i64,
}

impl<'a> NormalizationParameters<'a> {
    fn quotient_bound(self, width: usize, bits: usize) -> u128 {
        let min_dev = (self.epsilon_variance_units as u128).isqrt();
        // Each centered square is bounded by the sum of squares, below width * (dev + 1)^2.
        let width = width as u128;
        let root = width.isqrt();
        let root = root + u128::from(root * root < width);
        ((root << bits) * (min_dev + 1)).div_ceil(min_dev)
    }

    fn output_bounds(self, width: usize, bits: u8) -> impl Iterator<Item = u128> + 'a {
        let centered = self.quotient_bound(width, bits.into());
        let scale = 1u128 << bits;
        self.gain.iter().zip(self.bias).map(move |(&gain, &bias)| {
            ((centered * u128::from(gain.unsigned_abs())).div_ceil(scale)
                + u128::from(bias.unsigned_abs()))
            .min(1 << 31)
        })
    }

    pub(crate) fn output_bound(self, width: usize, bits: u8) -> Option<u128> {
        (self.epsilon_variance_units > 0)
            .then(|| self.output_bounds(width, bits).max())
            .flatten()
    }
}

pub(crate) struct Norm<'a> {
    shape: Shape,
    bits: usize,
    p: NormalizationParameters<'a>,
    rows: Ranges,
    cells: Ranges,
}

pub(crate) struct NormTrace {
    pub(crate) output: Vec<i32>,
    rows: [Vec<i128>; 9],
    quotients: Vec<i64>,
}

impl<'a> Norm<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        shape: Shape,
        bits: usize,
        p: NormalizationParameters<'a>,
    ) -> Result<Self> {
        if bits > 15
            || p.gain.len() != shape.width
            || p.bias.len() != shape.width
            || p.epsilon_variance_units <= 0
            || shape.width > 1 << 30
        {
            return Err("invalid normalization parameters");
        }
        let u = Range::unsigned;
        let rem = u(shape.width.next_power_of_two().ilog2() as usize);
        let row = Shape::new(shape.rows, 1)?;
        let gap = if shape.width.is_power_of_two() {
            u(0)
        } else {
            rem
        };
        let ranges = [u(65), rem, gap, u(33), u(34), u(34), I32, rem, gap];
        let rows = Ranges::new(layout, row, &ranges[..p.kind.stats()])?;
        let quotient_bits = if p.kind == NormalizationKind::Rms {
            p.quotient_bound(shape.width, bits).ilog2() as usize + 2
        } else {
            49
        };
        let cells = Ranges::new(
            layout,
            shape,
            [Range::signed(quotient_bits), I32, u(33), u(33), u(bits)],
        )?;
        Ok(Self {
            shape,
            bits,
            p,
            rows,
            cells,
        })
    }

    pub(crate) fn output_bounds(&self) -> Vec<u128> {
        self.p
            .output_bounds(self.shape.width, self.bits as u8)
            .collect()
    }

    pub(crate) fn bind(&self, t: &mut impl Transcript) {
        t.append(&(self.p.epsilon_variance_units as u64));
        t.append(&u64::from(self.p.kind == NormalizationKind::Rms));
        bind_i32(t, self.p.gain);
        bind_i32(t, self.p.bias);
    }

    pub(crate) fn execute(&self, input: &[i32], advice: &mut Advice) -> Result<NormTrace> {
        let shape = self.shape;
        shape.check(input)?;
        let width = shape.width as i128;
        let scale = 1i128 << self.bits;
        let mut rows: [Vec<i128>; 9] = std::array::from_fn(|_| Vec::new());
        let mut cells: [Vec<i128>; 5] = std::array::from_fn(|_| Vec::new());
        let mut output = Vec::with_capacity(input.len());
        let layer = self.p.kind == NormalizationKind::Layer;
        for row in input.chunks_exact(shape.width) {
            let (mean, mrem) = if layer {
                let sum: i128 = row.iter().copied().map(i128::from).sum();
                (sum.div_euclid(width), sum.rem_euclid(width))
            } else {
                (0, 0)
            };
            let squares: i128 = row.iter().map(|&x| (i128::from(x) - mean).pow(2)).sum();
            let var = squares.div_euclid(width) + i128::from(self.p.epsilon_variance_units);
            let vrem = squares.rem_euclid(width);
            let dev = var.isqrt();
            let root = var - dev * dev;
            let gap = |r| {
                if shape.width.is_power_of_two() {
                    0
                } else {
                    width - 1 - r
                }
            };
            let stats = [
                var,
                vrem,
                gap(vrem),
                dev,
                root,
                2 * dev - root,
                mean,
                mrem,
                gap(mrem),
            ];
            for (column, &v) in rows.iter_mut().zip(&stats[..self.p.kind.stats()]) {
                column.push(v);
            }
            for (i, &x) in row.iter().enumerate() {
                let x = (i128::from(x) - mean) * scale;
                let z = x.div_euclid(dev);
                let rem = x - z * dev;
                let affine = z * i128::from(self.p.gain[i]);
                let y = affine.div_euclid(scale) + i128::from(self.p.bias[i]);
                output.push(i32::try_from(y).map_err(|_| "normalization overflow")?);
                for (c, v) in cells
                    .iter_mut()
                    .zip([z, y, rem, dev - 1 - rem, affine & (scale - 1)])
                {
                    c.push(v);
                }
            }
        }
        self.rows.record(
            advice,
            &rows.each_ref().map(Vec::as_slice)[..self.p.kind.stats()],
        )?;
        self.cells
            .record(advice, &cells.each_ref().map(Vec::as_slice))?;
        Ok(NormTrace {
            output,
            rows,
            // The centered i32 input is scaled by at most 2^15 and divided by a positive deviation.
            quotients: cells[0].iter().map(|&z| z as i64).collect(),
        })
    }

    pub(crate) fn reverse<'w, F: ProofField>(
        &self,
        claims: &[Claim<F>],
        trace: impl Fn() -> (&'w [i32], &'w NormTrace),
        io: &mut Session<'_, '_, F>,
    ) -> Result<Claim<F>> {
        let shape = self.shape;
        let (row_vars, vars) = (shape.row_vars(), shape.vars());
        let (rows, width) = (1 << row_vars, shape.width.next_power_of_two());
        let scale = F::from_u64(1 << self.bits);
        let w = F::from_u64(shape.width as u64);
        let epsilon = F::from_i64(self.p.epsilon_variance_units);
        let stats = self.p.kind.stats();
        let checks = stats - 2;
        io.pow(row_vars + vars + checks + 3 + claims.len())?;
        let mix = io.challenge(checks + 3)?;
        let row_query = io.challenge(row_vars)?;
        let query = io.challenge(vars)?;
        let batch = io.challenge(claims.len())?;
        let active_col: Vec<_> = (0..width).map(|i| F::from_bool(i < shape.width)).collect();
        let first: Vec<_> = (0..width).map(|i| F::from_bool(i == 0)).collect();
        let row_mask: Vec<_> = (0..rows).map(|i| F::from_bool(i < shape.rows)).collect();
        let mask: Vec<_> = weights(&row_query)
            .into_iter()
            .zip(&row_mask)
            .map(|(q, &r)| q * r)
            .collect();
        let padded = |values| {
            Shape {
                rows: 1,
                width: shape.width,
            }
            .padded(values, F::zero(), F::from_i32)
        };
        let (gain, bias) = (padded(self.p.gain), padded(self.p.bias));
        // Batch the zero-sum moment constraints and affine relation with independent challenges.
        // Both reductions then use the same input and row-statistic evaluations.
        let expression = |values: &[F]| {
            let [x, a, first, m, gain, bias, active, output, eq, z, y, rem, gap, arem]: [F; 14] =
                values[..14].try_into().expect("normalization columns");
            let r = &values[14..];
            let [var, vr, vg, dev, rt, rg]: [F; 6] =
                r[..6].try_into().expect("normalization statistics");
            let (centered, mean_check) = if stats == 9 {
                let [mean, rem, gap]: [F; 3] = r[6..9].try_into().expect("mean statistics");
                (
                    x - mean,
                    mix[4] * (a * x - first * (w * mean + rem))
                        + if shape.width.is_power_of_two() {
                            F::zero()
                        } else {
                            mix[5] * first * (rem + gap - w + F::one())
                        },
                )
            } else {
                (x, F::zero())
            };
            let moments = [
                a * centered * centered - first * (w * (var - epsilon) + vr),
                if shape.width.is_power_of_two() {
                    F::zero()
                } else {
                    first * (vr + vg - w + F::one())
                },
                first * (var - dev * dev - rt),
                first * (dev + dev - rt - rg),
            ];
            let affine = [
                scale * centered - dev * z - rem,
                dev - F::one() - rem - gap,
                gain * z - scale * (y - bias) - arem,
            ];
            m * (mean_check + dot(moments, mix.iter().copied()))
                + active * (output * y + eq * dot(affine, mix[checks..].iter().copied()))
        };
        let sum = dot(claims.iter().map(|q| q.value), batch.iter().copied());
        let (end, [x, cs @ ..]) = io.fold(
            instance(sum, vars, 4),
            || {
                let (input, trace) = trace();
                let col = |v| Column::repeat(v, rows);
                let row = |v| Column::rows(v, width);
                let mut cols = vec![
                    Column::integer(input, shape),
                    col(active_col.as_slice().into()),
                    col(first.as_slice().into()),
                    row(mask.as_slice().into()),
                    col(gain.as_slice().into()),
                    col(bias.as_slice().into()),
                    Column::owned(
                        row_mask
                            .iter()
                            .flat_map(|&r| active_col.iter().map(move |&c| r * c))
                            .collect(),
                    ),
                    Column::owned(query_column(claims, &batch)),
                    Column::equality(&query),
                ];
                let mut cells: [Vec<F>; 4] = std::array::from_fn(|_| vec![F::zero(); 1 << vars]);
                let scale = 1i128 << self.bits;
                for (r, (inputs, quotients)) in input
                    .chunks_exact(shape.width)
                    .zip(trace.quotients.chunks_exact(shape.width))
                    .enumerate()
                {
                    let mean = if stats == 9 { trace.rows[6][r] } else { 0 };
                    let dev = trace.rows[3][r];
                    for (col, ((&x, &z), &gain)) in
                        inputs.iter().zip(quotients).zip(self.p.gain).enumerate()
                    {
                        let z = i128::from(z);
                        let rem = (i128::from(x) - mean) * scale - z * dev;
                        let arem = (i128::from(gain) * z) & (scale - 1);
                        for (column, value) in cells.iter_mut().zip([z, rem, dev - 1 - rem, arem]) {
                            column[r * width + col] = F::from_i128(value);
                        }
                    }
                }
                let [z, rem, gap, arem] = cells.map(Column::owned);
                cols.extend([z, Column::integer(&trace.output, shape), rem, gap, arem]);
                cols.extend(trace.rows[..stats].iter().map(|values| {
                    row(self
                        .rows
                        .shape
                        .padded(values, F::zero(), F::from_i128)
                        .into())
                }));
                cols
            },
            [0, 9, 10, 11, 12, 13],
            expression,
        )?;
        let p = &end.challenges;
        let (r, c) = p.split_at(row_vars);
        let rs: [F; 9] = io.values_prefix(stats, || {
            let row_weights = weights::<F>(r);
            trace().1.rows.each_ref().map(|column| {
                dot(
                    column.iter().copied().map(F::from_i128),
                    row_weights.iter().copied(),
                )
            })
        })?;
        io.check_value(&end, || {
            let a = active_col.as_slice().evaluate(c);
            let mut values = [
                x,
                a,
                first.as_slice().evaluate(c),
                mask.as_slice().evaluate(r),
                gain.as_slice().evaluate(c),
                bias.as_slice().evaluate(c),
                row_mask.as_slice().evaluate(r) * a,
                query_weight(claims, &batch, p),
                eq(&query, p),
            ]
            .to_vec();
            values.extend(cs);
            values.extend(&rs[..stats]);
            expression(&values)
        })?;
        self.rows.check(&[(r, &rs[..stats])], io)?;
        self.cells.check(&[(p, cs)], io)?;
        Ok(Claim {
            point: end.challenges,
            value: x,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_activation_preserves_clipping_and_signed_inputs() {
        let input = [i32::MIN, -4, -3, -2, -1, 0, 1, 2, 3, 4, i32::MAX];
        let correction = [0, 0, -1, -1, 0, -1, -1, 0];
        let table = Activation {
            lower: -4,
            correction: &correction,
        };
        let mut layout = Layout::default();
        let activation = Activate::new(
            &mut layout,
            Shape::new(1, input.len()).unwrap(),
            table,
            None,
        )
        .unwrap();
        assert!(matches!(
            activation.proof,
            ActivationProof::Clamp {
                symmetric: true,
                ..
            }
        ));
        let trace = activation.execute(&input, &mut layout.witness()).unwrap();
        let expected = input.map(|x| x.max(0) + correction[(x.clamp(-4, 3) + 4) as usize]);
        assert_eq!(trace.output, expected);
    }

    #[test]
    fn normalization_uses_epsilon_in_variance_units() {
        let shape = Shape::new(1, 2).unwrap();
        for kind in [NormalizationKind::Layer, NormalizationKind::Rms] {
            let mut layout = Layout::default();
            let norm = Norm::new(
                &mut layout,
                shape,
                14,
                NormalizationParameters {
                    kind,
                    gain: &[1 << 14; 2],
                    bias: &[0; 2],
                    epsilon_variance_units: 268,
                },
            )
            .unwrap();
            let trace = norm.execute(&[1, -1], &mut layout.witness()).unwrap();
            // The integer deviation is floor(sqrt(1 + 268)) = 16.
            assert_eq!(trace.output, [1024, -1024]);
        }
    }
}
