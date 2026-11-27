//! A normalization proof also reduces any outstanding evaluation of its input.

use crate::fold::{agreement, basis, basis_scaled, Column, Fold};
use crate::{
    advice::{Advice, Requests, Slot},
    channel::Channel,
    field::{Mixing, F as Scalar},
    layers::NormalizationParameters,
    tensor::Shape,
};
use jolt_field::Field;
use jolt_sumcheck::single::{SingleFormat, SingleInstance};
use jolt_transcript::Encoding;

type Result<T> = std::result::Result<T, &'static str>;
const PRIVATE: usize = 15;
const TOTAL: usize = 24;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim<F> {
    pub point: Vec<F>,
    pub value: F,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Width {
    Signed(u8),
    Unsigned(u8),
}

impl Width {
    pub(crate) fn bits(self) -> usize {
        match self {
            Self::Signed(n) | Self::Unsigned(n) => n.into(),
        }
    }

    pub(crate) fn offset(self) -> i128 {
        match self {
            Self::Signed(n) => 1 << (n - 1),
            Self::Unsigned(_) => 0,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Normalized {
    pub(crate) trace: Option<Witness>,
    slots: Vec<Slot>,
}

impl Normalized {
    pub(crate) fn forward(
        advice: &mut Advice,
        shape: Shape,
        bits: u32,
        params: NormalizationParameters<'_>,
        input: Option<&[i32]>,
    ) -> Result<Self> {
        if params.gain.len() != shape.cols {
            return Err("invalid layer normalization width");
        }
        let plan = params.at(shape.rows, bits)?;
        let trace = input.map(|v| plan.witness(v)).transpose()?;
        let slots = plan.record(advice, input, trace.as_ref())?;
        Ok(Self { trace, slots })
    }
    pub(crate) fn values(&self) -> Option<&[i32]> {
        self.trace.as_ref().map(|w| w.result.as_slice())
    }
    pub(crate) fn reverse(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        shape: Shape,
        bits: u32,
        params: NormalizationParameters<'_>,
        input: Option<&[i32]>,
        output: &[Claim<Scalar>],
        carry: &[Claim<Scalar>],
    ) -> Result<Claim<Scalar>> {
        params.at(shape.rows, bits)?.reduce_model(
            io,
            input,
            self.trace.as_ref(),
            output,
            carry,
            &self.slots,
            requests,
        )
    }
}

pub struct Norm<'a> {
    pub rows: usize,
    pub bits: u32,
    pub epsilon: u64,
    pub center: bool,
    pub gain: &'a [i32],
    pub bias: &'a [i32],
}

#[derive(Clone, Copy)]
struct Domain {
    height: usize,
    stride: usize,
}

impl Domain {
    fn rounds(self) -> usize {
        (self.height * self.stride).ilog2() as usize
    }
    fn row_bits(self) -> usize {
        self.height.ilog2() as usize
    }
}

struct Weights<F> {
    points: Vec<(F, Vec<F>)>,
    split: usize,
}

impl<F: Field + 'static> Weights<F> {
    fn new(claims: &[Claim<F>], mixing: &[F], split: usize) -> Self {
        let points = mixing
            .iter()
            .zip(claims)
            .map(|(&a, q)| (a, q.point.clone()))
            .collect();
        Self { points, split }
    }

    fn column(&self, len: usize, stride: usize) -> Column<'static, F> {
        use rayon::prelude::*;
        if self.points.is_empty() {
            return Column::table(vec![F::zero()], len, len);
        }
        let column = &self.points[0].1[self.split..];
        if self.points.iter().all(|(_, p)| &p[self.split..] == column) {
            let col = basis(column);
            let mut row = vec![F::zero(); len / stride];
            for (a, p) in &self.points {
                for (acc, w) in row.iter_mut().zip(basis_scaled(&p[..self.split], *a)) {
                    *acc += w;
                }
            }
            return Column::source(len, move |i| row[i / stride] * col[i % stride]);
        }
        let terms: Vec<_> = self
            .points
            .iter()
            .map(|(a, point)| {
                let (row, col) = point.split_at(self.split);
                (basis_scaled(row, *a), basis(col))
            })
            .collect();
        let mut out = vec![F::zero(); len];
        out.par_chunks_mut(stride)
            .with_min_len(8)
            .enumerate()
            .for_each(|(r, out)| {
                for (row, col) in &terms {
                    let m = row[r];
                    for (y, &c) in out.iter_mut().zip(col) {
                        *y += m * c;
                    }
                }
            });
        Column::table(out, len, 1)
    }

    fn evaluate(&self, point: &[F]) -> F {
        self.points
            .iter()
            .map(|(a, p)| *a * agreement(p, point))
            .sum()
    }
}

struct Check<F> {
    claims: SingleInstance<F>,
    point: Vec<F>,
    rows: Vec<F>,
    carry: Weights<F>,
    output: Weights<F>,
    coefficients: [F; 9],
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Witness {
    pub(crate) result: Vec<i32>,
    quotient: Vec<i64>,
    stats: Vec<[i128; 9]>,
}

impl Norm<'_> {
    fn domain(&self) -> Result<Domain> {
        if self.rows == 0
            || self.rows > 1 << 20
            || self.gain.is_empty()
            || self.gain.len() > 1 << 20
            || self.gain.len() != self.bias.len()
            || self.epsilon == 0
            || self.bits > 30
        {
            return Err("invalid normalization parameters");
        }
        let height = self.rows.next_power_of_two();
        let stride = self.gain.len().next_power_of_two();
        height
            .checked_mul(stride)
            .ok_or("normalization domain too large")?;
        Ok(Domain { height, stride })
    }

    pub(crate) fn witness(&self, input: &[i32]) -> Result<Witness> {
        self.domain()?;
        let width = self.gain.len();
        if input.len() != self.rows * width {
            return Err("invalid normalization input");
        }
        let mut result = Vec::with_capacity(input.len());
        let mut quotient = Vec::with_capacity(input.len());
        let mut stats = Vec::with_capacity(self.rows);
        let width = width as i128;
        for cells in input.chunks_exact(width as usize) {
            let sum: i128 = if self.center {
                cells.iter().copied().map(i128::from).sum()
            } else {
                0
            };
            let (mean, mr) = (sum.div_euclid(width), sum.rem_euclid(width));
            let squares: i128 = cells.iter().map(|&x| (i128::from(x) - mean).pow(2)).sum();
            let (var, vr) = (squares / width + i128::from(self.epsilon), squares % width);
            let root = var.isqrt();
            let err = var - root * root;
            stats.push([
                var,
                vr,
                width - 1 - vr,
                root,
                err,
                2 * root - err,
                mean,
                mr,
                width - 1 - mr,
            ]);
            for (col, &cell) in cells.iter().enumerate() {
                let q = ((i128::from(cell) - mean) << self.bits).div_euclid(root);
                let y =
                    ((q * i128::from(self.gain[col])) >> self.bits) + i128::from(self.bias[col]);
                quotient.push(i64::try_from(q).map_err(|_| "invalid normalization quotient")?);
                result.push(i32::try_from(y).map_err(|_| "normalization result exceeds i32")?);
            }
        }
        Ok(Witness {
            result,
            quotient,
            stats,
        })
    }

    fn check<F: Field + Encoding + 'static>(
        &self,
        domain: Domain,
        channel: &mut impl Mixing<F>,
        output: &[Claim<F>],
        carry: &[Claim<F>],
    ) -> Result<Check<F>> {
        let count = output
            .len()
            .checked_add(carry.len())
            .ok_or("too many evaluations")?;
        if count == 0
            || output
                .iter()
                .chain(carry)
                .any(|q| q.point.len() != domain.rounds())
        {
            return Err("invalid normalization evaluation");
        }
        channel.append(&(output.len() as u64));
        channel.append(&(carry.len() as u64));
        for claim in output.iter().chain(carry) {
            channel.append_values(&claim.point);
            channel.append_values(&[claim.value]);
        }
        // The caller binds model parameters and witness commitments before this reduction.
        let size = 9 + domain.rounds() + domain.row_bits() + count;
        let random = channel.mix(size, size)?;
        let (coefficients, random) = random
            .split_first_chunk::<9>()
            .ok_or("missing coefficients")?;
        let (point, random) = random.split_at(domain.rounds());
        let (rows, mix) = random.split_at(domain.row_bits());
        let (out_mix, in_mix) = mix.split_at(output.len());
        let input_claim = output
            .iter()
            .chain(carry)
            .zip(mix)
            .map(|(q, &a)| q.value * a)
            .sum();
        Ok(Check {
            claims: SingleInstance {
                input_claim,
                rounds: domain.rounds(),
                degree: if domain.height == self.rows && domain.stride == self.gain.len() {
                    3
                } else {
                    4
                },
                format: SingleFormat::Compressed,
            },
            point: point.to_vec(),
            rows: rows.to_vec(),
            coefficients: *coefficients,
            carry: Weights::new(carry, in_mix, domain.row_bits()),
            output: Weights::new(output, out_mix, domain.row_bits()),
        })
    }

    pub(crate) fn record(
        &self,
        advice: &mut Advice,
        input: Option<&[i32]>,
        witness: Option<&Witness>,
    ) -> Result<Vec<Slot>> {
        let shape = Shape::new(self.rows, self.gain.len())?;
        let (cells, stats) = self.widths();
        let mut slots = Vec::with_capacity(14);
        for (i, width) in cells.into_iter().enumerate() {
            slots.push(advice.column(shape, width, |j| {
                self.cell(
                    input.unwrap(),
                    witness.unwrap(),
                    j / shape.cols,
                    j % shape.cols,
                    i + 1,
                )
            })?);
        }
        let shape = Shape::new(self.rows, 1)?;
        for (i, width) in stats.into_iter().enumerate() {
            slots.push(advice.column(shape, width, |j| witness.unwrap().stats[j][i])?);
        }
        Ok(slots)
    }

    pub(crate) fn reduce_model(
        &self,
        io: &mut Channel<'_, '_>,
        input: Option<&[i32]>,
        witness: Option<&Witness>,
        output: &[Claim<Scalar>],
        carry: &[Claim<Scalar>],
        slots: &[Slot],
        requests: &mut Requests,
    ) -> Result<Claim<Scalar>> {
        let domain = self.domain()?;
        let check = match io {
            Channel::Send(t, ..) => self.check(domain, *t, output, carry)?,
            Channel::Check(t, ..) => self.check(domain, *t, output, carry)?,
        };
        let witness = || Fold {
            columns: self.columns(domain, input.unwrap(), witness.unwrap(), &check),
            degree: check.claims.degree,
            relation: |v: &[Scalar]| self.polynomial(&check.coefficients, v.try_into().unwrap()),
        };
        let (end, terminal) = io.sumcheck(&check.claims, PRIVATE, witness, |w| {
            w.terminal().into_iter().take(PRIVATE).collect()
        })?;
        let input = self.reduce(domain, &check, end, terminal.as_slice().try_into().unwrap())?;
        for (i, (&slot, &value)) in slots.iter().zip(&terminal[1..]).enumerate() {
            let point = if i < 5 {
                &input.point
            } else {
                &input.point[..domain.row_bits()]
            };
            requests.push(slot, point, value);
        }
        Ok(input)
    }

    fn polynomial<F: Field>(&self, a: &[F; 9], v: &[F; TOTAL]) -> F {
        let &[x, q, y, rem, gap, aff, var, vr, vg, root, err, root_gap, mean, mr, mg, gain, bias, lane, active, first, output, input, position, row] =
            v;
        let width = F::from_u64(self.gain.len() as u64);
        let scale = F::from_u64(1 << self.bits);
        let one = F::one();
        let zero = F::zero();
        let x0 = x - mean;
        let full_width = self.gain.len().is_power_of_two();
        let constraints = [
            lane * x0 * x0 - first * ((var - F::from_u64(self.epsilon)) * width + vr),
            if full_width {
                zero
            } else {
                first * (vr + vg - width + one)
            },
            first * (var - root * root - err),
            first * (root + root - err - root_gap),
            if self.center {
                lane * x - first * (mean * width + mr)
            } else {
                lane * mean
            },
            if self.center && !full_width {
                first * (mr + mg - width + one)
            } else {
                zero
            },
            scale * x0 - root * q - rem,
            root - one - rem - gap,
            q * gain - scale * (y - bias) - aff,
        ];
        let fold = |start, end| {
            (start..end)
                .map(|i: usize| a[i] * constraints[i])
                .sum::<F>()
        };
        row * fold(0, 6) + active * (output * y + position * fold(6, 9)) + input * x
    }

    fn cell(&self, input: &[i32], witness: &Witness, r: usize, c: usize, column: usize) -> i128 {
        let width = self.gain.len();
        if r >= self.rows || c >= width {
            return 0;
        }
        let index = r * width + c;
        let x = i128::from(input[index]);
        let q = i128::from(witness.quotient[index]);
        let root = witness.stats[r][3];
        let mean = witness.stats[r][6];
        let remainder = || ((x - mean) << self.bits) - root * q;
        match column {
            0 => x,
            1 => q,
            2 => i128::from(witness.result[index]),
            3 => remainder(),
            4 => root - 1 - remainder(),
            _ => (q * i128::from(self.gain[c])).rem_euclid(1 << self.bits),
        }
    }

    fn widths(&self) -> ([Width; 5], [Width; 9]) {
        use Width::{Signed, Unsigned};
        let r = self.gain.len().next_power_of_two().ilog2() as u8;
        let cells = [
            Signed(63),
            Signed(32),
            Unsigned(33),
            Unsigned(33),
            Unsigned(self.bits as u8),
        ];
        let stats = [
            Unsigned(65),
            Unsigned(r),
            Unsigned(r),
            Unsigned(33),
            Unsigned(34),
            Unsigned(34),
            Signed(32),
            Unsigned(r),
            Unsigned(r),
        ];
        (cells, stats)
    }

    fn columns<'a, F: Field + 'static>(
        &'a self,
        d: Domain,
        input: &'a [i32],
        witness: &'a Witness,
        check: &Check<F>,
    ) -> Vec<Column<'a, F>> {
        let width = self.gain.len();
        let size = d.height * d.stride;
        let (r, c) = check.point.split_at(d.row_bits());
        let mut rows = basis(&check.rows);
        rows[self.rows..].fill(F::zero());
        let public = |v: &[i32]| {
            let mut values: Vec<_> = v.iter().map(|&v| F::from_i32(v)).collect();
            values.resize(d.stride, F::zero());
            Column::table(values, size, 1)
        };
        let mut columns: Vec<_> = (0..6)
            .map(|column| {
                Column::source(size, move |i| {
                    F::from_i128(self.cell(input, witness, i / d.stride, i % d.stride, column))
                })
            })
            .collect();
        columns.extend((0..9).map(|i| {
            let mut values: Vec<_> = witness.stats.iter().map(|s| F::from_i128(s[i])).collect();
            values.resize(d.height, F::zero());
            Column::table(values, size, d.stride)
        }));
        let mask = |rows: usize| {
            if d.stride == width && rows == d.height {
                Column::table(vec![F::one()], size, size)
            } else {
                Column::source(size, move |i| {
                    F::from_bool(i / d.stride < rows && i % d.stride < width)
                })
            }
        };
        columns.extend([
            public(self.gain),
            public(self.bias),
            mask(d.height),
            mask(self.rows),
            Column::table(
                (0..d.stride).map(|i| F::from_bool(i == 0)).collect(),
                size,
                1,
            ),
            check.output.column(size, d.stride),
            check.carry.column(size, d.stride),
            Column::Product(basis(r), basis(c)),
            Column::table(rows, size, d.stride),
        ]);
        columns
    }

    fn reduce<F: Field + 'static>(
        &self,
        domain: Domain,
        check: &Check<F>,
        end: jolt_sumcheck::single::SingleReduction<F>,
        terminal: [F; PRIVATE],
    ) -> Result<Claim<F>> {
        let (r, c) = end.challenges.split_at(domain.row_bits());
        let row = basis(r);
        let col = basis(c);
        let dot = |values: &[i32]| {
            col.iter()
                .zip(values)
                .map(|(&w, &v)| w * F::from_i32(v))
                .sum()
        };
        let lane: F = col.iter().take(self.gain.len()).copied().sum();
        let active_rows: F = row.iter().take(self.rows).copied().sum();
        let row_check = row
            .iter()
            .zip(basis(&check.rows))
            .take(self.rows)
            .map(|(&a, b)| a * b)
            .sum();
        let public = [
            dot(self.gain),
            dot(self.bias),
            lane,
            lane * active_rows,
            col[0],
            check.output.evaluate(&end.challenges),
            check.carry.evaluate(&end.challenges),
            agreement(&check.point, &end.challenges),
            row_check,
        ];
        let values = std::array::from_fn(|i| {
            if i < PRIVATE {
                terminal[i]
            } else {
                public[i - PRIVATE]
            }
        });
        if self.polynomial(&check.coefficients, &values) != end.output_claim {
            return Err("invalid normalization terminal evaluation");
        }
        Ok(Claim {
            point: end.challenges,
            value: terminal[0],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{channel::Messages, field::F};
    use jolt_field::{One, Ring};
    use jolt_transcript::{prover_transcript, verifier_transcript, Keccak, Transcript};

    #[test]
    fn normalization_reduces_carry_and_outputs_at_distinct_points() {
        let input = [21, -12, 4, -16, 13, 11, 4, -14, 3];
        let shape = Shape::new(3, 3).unwrap();
        for center in [false, true] {
            let norm = Norm {
                rows: 3,
                bits: 4,
                epsilon: 5,
                center,
                gain: &[16; 3],
                bias: &[0; 3],
            };
            let witness = norm.witness(&input).unwrap();
            let claim = |values: &[i32], start| {
                let point = (start..start + 4).map(F::from_u64).collect::<Vec<_>>();
                Claim {
                    value: shape.eval(values, &point),
                    point,
                }
            };
            let carry = [claim(&input, 2)];
            let outputs = [claim(&witness.result, 6), claim(&witness.result, 10)];
            let tag = [0x45; 32];
            let mut transcript = prover_transcript::<Keccak>(&tag, b"");
            let mut advice = Advice::new(true).unwrap();
            advice.cut();
            let slots = norm
                .record(&mut advice, Some(&input), Some(&witness))
                .unwrap();
            advice.cut();
            let widths = advice.widths();
            transcript.append_bytes(&widths);
            advice.seal().unwrap();
            let plan = advice.windows(1).unwrap();
            let roots = plan[0].commit(&advice, &mut transcript).unwrap();
            let mut requests = Requests(Vec::new());
            let mut messages = Messages::default();
            let actual = norm
                .reduce_model(
                    &mut Channel::Send(&mut transcript, &mut messages, &advice),
                    Some(&input),
                    Some(&witness),
                    &outputs,
                    &carry,
                    &slots,
                    &mut requests,
                )
                .unwrap();
            assert_eq!(actual.value, shape.eval(&input, &actual.point));
            let mut proofs = Vec::new();
            plan[0]
                .prove(&advice, &roots, requests, &mut transcript, |item| {
                    proofs.push(item);
                    Ok(())
                })
                .unwrap();

            let verify = |carry: &[Claim<F>]| -> Result<Claim<F>> {
                let mut t = verifier_transcript::<Keccak>(&tag, b"", transcript.narg_string());
                let mut advice = Advice::new(false)?;
                advice.cut();
                let slots = norm.record(&mut advice, None, None)?;
                advice.cut();
                advice.bind_widths(&widths)?;
                t.append_bytes(&widths);
                let plan = advice.windows(1)?;
                plan[0].bind(&roots, &mut t)?;
                let mut io = Channel::checking(&mut t, &messages);
                let mut requests = Requests(Vec::new());
                let result =
                    norm.reduce_model(&mut io, None, None, &outputs, carry, &slots, &mut requests)?;
                io.finish()?;
                let mut groups = proofs.iter().cloned();
                plan[0].verify(&advice, requests, &roots, &mut t, || {
                    groups.next().ok_or("missing BitZ group")
                })?;
                t.check_eof().map_err(|_| "extra transcript bytes")?;
                Ok(result)
            };
            assert_eq!(verify(&carry).unwrap(), actual);
            let mut invalid = carry;
            invalid[0].value += F::one();
            assert!(verify(&invalid).is_err());
        }
    }
}
