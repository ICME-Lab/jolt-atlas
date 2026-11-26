//! A normalization proof also reduces any outstanding evaluation of its input.

use crate::fold::{agreement, basis, basis_scaled, Column, Fold};
use crate::{
    advice::{Advice, Requests, Slot},
    channel::Channel,
    field::{Mixing, F as Scalar},
    tensor::Shape,
};
use jolt_field::Field;
use jolt_sumcheck::single::{
    prove_single, verify_single, SingleFormat, SingleInstance, SingleSumcheckProof,
};
use jolt_transcript::{Encoding, FieldTranscript, ProverTranscript};

#[cfg(test)]
use crate::bitz;
#[cfg(test)]
use jolt_transcript::{ProverState, Transcript, VerifierState};

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

#[must_use = "discharge the input claim and all range evaluations"]
pub struct Reduction<F> {
    pub input: Claim<F>,
    pub cells: Vec<(Width, F)>,
    pub statistics: Vec<(Width, F)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Proof<F> {
    rounds: SingleSumcheckProof<F>,
    terminal: [F; PRIVATE],
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

    pub fn execute(&self, input: &[i32]) -> Result<Vec<i32>> {
        Ok(self.witness(input)?.result)
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
        let reduction = match io {
            Channel::Send(t, data) => {
                let (proof, reduction) =
                    self.prove_with(domain, input.unwrap(), witness.unwrap(), output, carry, *t)?;
                data.norms.push(proof);
                reduction
            }
            Channel::Check(t, _, _, norms, _) => {
                let (proof, rest) = norms.split_first().ok_or("missing normalization proof")?;
                *norms = rest;
                self.verify(output, carry, proof, *t)?
            }
        };
        let point = &reduction.input.point;
        for (&slot, (_, value)) in slots.iter().zip(reduction.cells) {
            requests.push(slot, point, value);
        }
        for (&slot, (_, value)) in slots.iter().skip(5).zip(reduction.statistics) {
            requests.push(slot, &point[..domain.row_bits()], value);
        }
        Ok(reduction.input)
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
    ) -> [Column<'a, F>; TOTAL] {
        let width = self.gain.len();
        let size = d.height * d.stride;
        let p_row = basis(&check.point[..d.row_bits()]);
        let p_col = basis(&check.point[d.row_bits()..]);
        let mut out = Some(check.output.column(size, d.stride));
        let mut carry = Some(check.carry.column(size, d.stride));
        let mut position = Some((p_row, p_col));
        let mut rows = basis(&check.rows);
        rows[self.rows..].fill(F::zero());
        let public = |v: &[i32]| {
            let mut values: Vec<_> = v.iter().map(|&v| F::from_i32(v)).collect();
            values.resize(d.stride, F::zero());
            Column::table(values, size, 1)
        };
        let at = |column| {
            Column::source(size, move |i| {
                F::from_i128(self.cell(input, witness, i / d.stride, i % d.stride, column))
            })
        };
        std::array::from_fn(|i| match i {
            0..=5 => at(i),
            6..=14 => {
                let mut values: Vec<_> = witness
                    .stats
                    .iter()
                    .map(|s| F::from_i128(s[i - 6]))
                    .collect();
                values.resize(d.height, F::zero());
                Column::table(values, size, d.stride)
            }
            15 => public(self.gain),
            16 => public(self.bias),
            17 if d.stride == width => Column::table(vec![F::one()], size, size),
            18 if d.stride == width && d.height == self.rows => {
                Column::table(vec![F::one()], size, size)
            }
            17 => Column::source(size, move |i| F::from_bool(i % d.stride < width)),
            18 => Column::source(size, move |i| {
                F::from_bool(i / d.stride < self.rows && i % d.stride < width)
            }),
            19 => Column::table(
                (0..d.stride).map(|i| F::from_bool(i == 0)).collect(),
                size,
                1,
            ),
            20 => out.take().unwrap(),
            21 => carry.take().unwrap(),
            22 => Column::source(size, {
                let (row, col) = position.take().unwrap();
                move |i| row[i / d.stride] * col[i % d.stride]
            }),
            _ => Column::table(std::mem::take(&mut rows), size, d.stride),
        })
    }

    #[cfg(test)]
    pub(crate) fn prove<F: Field + Encoding + 'static>(
        &self,
        input: &[i32],
        output: &[Claim<F>],
        carry: &[Claim<F>],
        channel: &mut (impl Mixing<F> + ProverTranscript),
    ) -> Result<(Proof<F>, Reduction<F>)> {
        let domain = self.domain()?;
        let witness = self.witness(input)?;
        self.prove_with(domain, input, &witness, output, carry, channel)
    }

    fn prove_with<F: Field + Encoding + 'static>(
        &self,
        domain: Domain,
        input: &[i32],
        witness: &Witness,
        output: &[Claim<F>],
        carry: &[Claim<F>],
        channel: &mut (impl Mixing<F> + ProverTranscript),
    ) -> Result<(Proof<F>, Reduction<F>)> {
        let check = self.check(domain, channel, output, carry)?;
        let mut fold = Fold {
            columns: Vec::from(self.columns(domain, input, witness, &check)),
            degree: check.claims.degree,
            relation: |v: &[F]| self.polynomial(&check.coefficients, v.try_into().unwrap()),
        };
        let (rounds, end) = prove_single(&mut fold, &check.claims, channel)
            .map_err(|_: jolt_sumcheck::SumcheckError<F>| "normalization Sumcheck failed")?;
        let values = fold.terminal();
        let terminal = std::array::from_fn(|i| values[i]);
        let proof = Proof { rounds, terminal };
        let reduction = self.reduce(domain, &check, end, terminal, channel)?;
        Ok((proof, reduction))
    }

    pub(crate) fn verify<F: Field + Encoding + 'static>(
        &self,
        output: &[Claim<F>],
        carry: &[Claim<F>],
        proof: &Proof<F>,
        channel: &mut impl Mixing<F>,
    ) -> Result<Reduction<F>> {
        let domain = self.domain()?;
        let check = self.check(domain, channel, output, carry)?;
        let end = verify_single(&check.claims, &proof.rounds, channel)
            .map_err(|_| "invalid normalization Sumcheck")?;
        self.reduce(domain, &check, end, proof.terminal, channel)
    }

    fn reduce<F: Field + Encoding + 'static>(
        &self,
        domain: Domain,
        check: &Check<F>,
        end: jolt_sumcheck::single::SingleReduction<F>,
        terminal: [F; PRIVATE],
        channel: &mut impl FieldTranscript<F>,
    ) -> Result<Reduction<F>> {
        channel.append_values(&terminal);
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
        let (cells, stats) = self.widths();
        let cells = cells
            .into_iter()
            .zip(terminal[1..6].iter().copied())
            .collect();
        let statistics = stats
            .into_iter()
            .zip(terminal[6..].iter().copied())
            .collect();
        Ok(Reduction {
            input: Claim {
                point: end.challenges,
                value: terminal[0],
            },
            cells,
            statistics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::F;
    use jolt_field::{One, Ring};
    use jolt_transcript::{prover_transcript, verifier_transcript, Keccak};

    impl Norm<'_> {
        fn group(&self, domain: Domain) -> Result<bitz::Group> {
            let (cells, stats) = self.widths();
            let columns: Vec<_> = cells
                .into_iter()
                .map(|width| bitz::RangeColumn {
                    vars: domain.rounds(),
                    width,
                })
                .chain(stats.into_iter().map(|width| bitz::RangeColumn {
                    vars: domain.row_bits(),
                    width,
                }))
                .collect();
            bitz::Group::new(&columns)
        }

        /// Binds the integer columns and returns the one input claim for the previous operation.
        fn prove_committed(
            &self,
            input: &[i32],
            output: &[Claim<Scalar>],
            carry: &[Claim<Scalar>],
            channel: &mut ProverState<Keccak>,
        ) -> Result<(CommittedProof, Claim<Scalar>)> {
            let domain = self.domain()?;
            let witness = self.witness(input)?;
            let group = self.group(domain)?;
            let source = |slot: usize, i: usize| {
                if slot < 5 {
                    self.cell(
                        input,
                        &witness,
                        i / domain.stride,
                        i % domain.stride,
                        slot + 1,
                    )
                } else {
                    witness.stats.get(i).map_or(0, |r| r[slot - 5])
                }
            };
            let committed = group.commit_detached(source)?;
            channel.append_bytes(&committed.root);
            let root = committed.root;
            let (relation, reduction) =
                self.prove_with(domain, input, &witness, output, carry, channel)?;
            let evaluations = group
                .reduce(
                    committed,
                    &reduction.queries(domain.row_bits()),
                    channel,
                    source,
                )?
                .open(&mut std::collections::HashMap::new())?;
            Ok((
                CommittedProof {
                    root,
                    relation,
                    evaluations,
                },
                reduction.input,
            ))
        }

        fn verify_committed(
            &self,
            output: &[Claim<Scalar>],
            carry: &[Claim<Scalar>],
            proof: &CommittedProof,
            channel: &mut VerifierState<'_, Keccak>,
        ) -> Result<Claim<Scalar>> {
            let domain = self.domain()?;
            let group = self.group(domain)?;
            group.bind_root(proof.root, channel);
            let reduction = self.verify(output, carry, &proof.relation, channel)?;
            group.verify(
                proof.root,
                &reduction.queries(domain.row_bits()),
                &proof.evaluations,
                channel,
            )?;
            Ok(reduction.input)
        }
    }

    impl Reduction<Scalar> {
        fn queries(&self, row_bits: usize) -> Vec<bitz::Query> {
            let query = |(slot, (_, value)), point: &[Scalar]| bitz::Query {
                slot,
                planes: None,
                claim: Claim {
                    point: point.to_vec(),
                    value,
                },
            };
            self.cells
                .iter()
                .copied()
                .enumerate()
                .map(|item| query(item, &self.input.point))
                .chain(
                    self.statistics
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(i, item)| {
                            query((i + self.cells.len(), item), &self.input.point[..row_bits])
                        }),
                )
                .collect()
        }
    }

    struct CommittedProof {
        root: [u8; 32],
        relation: Proof<Scalar>,
        evaluations: bitz::Proof,
    }

    fn evaluate(values: &[i32], point: &[F]) -> F {
        let width = 3;
        basis(point)
            .into_iter()
            .enumerate()
            .filter_map(|(i, w)| {
                let (row, col) = (i / 4, i % 4);
                (row < 3 && col < width).then(|| F::from_i32(values[row * width + col]) * w)
            })
            .sum()
    }

    #[test]
    fn residual_skip_reduces_with_norm_and_bit_z() {
        let input = [21, -12, 4, -16, 13, 11, 4, -14, 3];
        let norm = Norm {
            rows: 3,
            bits: 4,
            epsilon: 5,
            center: false,
            gain: &[16, 16, 16],
            bias: &[0, 0, 0],
        };
        let update = norm.execute(&input).unwrap();
        let sum = crate::residual::add(&input, &update).unwrap();
        let point = (2..6).map(F::from_u64).collect::<Vec<_>>();
        let claim = Claim {
            value: evaluate(&sum, &point),
            point,
        };
        let left_value = evaluate(&input, &claim.point);
        let tag = [0x42; 32];
        let mut prover = prover_transcript::<Keccak>(&tag, b"");
        let (carry, output) = crate::residual::split(claim.clone(), left_value, &mut prover);
        let (proof, terminal) = norm
            .prove_committed(&input, &[output], &[carry], &mut prover)
            .unwrap();

        let mut verifier = verifier_transcript::<Keccak>(&tag, b"", prover.narg_string());
        let (carry, output) = crate::residual::split(claim, left_value, &mut verifier);
        let checked = norm
            .verify_committed(&[output], &[carry], &proof, &mut verifier)
            .unwrap();
        assert_eq!(checked, terminal);
        assert_eq!(checked.value, evaluate(&input, &checked.point));
        verifier.check_eof().unwrap();
    }

    #[test]
    fn normalization_reduces_carry_and_outputs_at_distinct_points() {
        let input = [21, -12, 4, -16, 13, 11, 4, -14, 3];
        let norm = Norm {
            rows: 3,
            bits: 4,
            epsilon: 5,
            center: false,
            gain: &[16, 16, 16],
            bias: &[0, 0, 0],
        };
        for center in [false, true] {
            let norm = Norm { center, ..norm };
            let actual = norm.execute(&input).unwrap();
            let point = |start| (start..start + 4).map(F::from_u64).collect::<Vec<_>>();
            let claim = |values, start| {
                let point = point(start);
                Claim {
                    value: evaluate(values, &point),
                    point,
                }
            };
            let carry = [claim(&input, 2)];
            let output = [claim(&actual, 6), claim(&actual, 10)];
            let tag = [0x45; 32];
            let mut prover = prover_transcript::<Keccak>(&tag, b"");
            let (proof, reduction) = norm.prove(&input, &output, &carry, &mut prover).unwrap();
            assert_eq!(
                evaluate(&input, &reduction.input.point),
                reduction.input.value
            );

            let mut verifier = verifier_transcript::<Keccak>(&tag, b"", prover.narg_string());
            let checked = norm.verify(&output, &carry, &proof, &mut verifier).unwrap();
            assert_eq!(reduction.input, checked.input);
            assert_eq!(reduction.cells, checked.cells);
            assert_eq!(reduction.statistics, checked.statistics);

            let mut false_carry = carry.clone();
            false_carry[0].value += F::one();
            let mut verifier = verifier_transcript::<Keccak>(&tag, b"", prover.narg_string());
            assert!(norm
                .verify(&output, &false_carry, &proof, &mut verifier)
                .is_err());
        }
    }
}
