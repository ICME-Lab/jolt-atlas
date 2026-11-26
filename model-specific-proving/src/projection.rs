//! Joint reduction for fixed-point matrix projections with the same input tensor.

use crate::channel::Channel;
use crate::field::{Mixing, F};
use crate::fold::{basis, InnerProducts};
use crate::norm::Claim;
use jolt_field::{One, Ring, Zero};
use jolt_sumcheck::single::{SingleFormat, SingleInstance, SingleReduction};
use rayon::prelude::*;

type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Copy)]
pub struct Projection<'a> {
    pub weights: &'a [i32],
    pub bias: &'a [i32],
}

#[derive(Clone, Copy)]
pub struct Input {
    pub rows: usize,
    pub width: usize,
    pub bits: u32,
}

/// The claimed output and its remainder have the same evaluation point.
/// The caller authenticates both according to the integer bounds of this projection.
pub struct Query<'a, F> {
    pub projection: Projection<'a>,
    pub output: &'a Claim<F>,
    pub remainder: F,
}

struct Group<F> {
    point: Vec<F>,
    public: Vec<F>,
}

struct Batch<F> {
    instance: SingleInstance<F>,
    groups: Vec<Group<F>>,
}

impl Input {
    fn dimensions(self) -> Result<(usize, usize)> {
        if self.rows == 0
            || self.rows > 1 << 20
            || self.width == 0
            || self.width > 1 << 20
            || self.bits > 30
        {
            return Err("invalid projection input shape");
        }
        Ok((
            self.rows.next_power_of_two().ilog2() as usize,
            self.width.next_power_of_two(),
        ))
    }

    pub fn execute(
        self,
        input: &[i32],
        projection: Projection<'_>,
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        self.dimensions()?;
        let output = projection.bias.len();
        if input.len() != self.rows * self.width || !projection.valid(self.width) {
            return Err("invalid projection dimensions");
        }
        let mut values = vec![0; self.rows * output];
        let mut remainders = vec![0; values.len()];
        let weight_bound: Vec<_> = projection
            .weights
            .chunks_exact(self.width)
            .map(|r| r.iter().map(|v| u128::from(v.unsigned_abs())).sum::<u128>())
            .collect();
        values
            .par_chunks_mut(output)
            .zip(remainders.par_chunks_mut(output))
            .zip(input.par_chunks(self.width))
            .try_for_each(|((values, remainders), row)| {
                let bound = row
                    .iter()
                    .map(|x| u128::from(x.unsigned_abs()))
                    .max()
                    .unwrap_or(0);
                for (j, (((value, rem), weight), &bias)) in values
                    .iter_mut()
                    .zip(remainders)
                    .zip(projection.weights.chunks_exact(self.width))
                    .zip(projection.bias)
                    .enumerate()
                {
                    let adapt = crate::tensor::dot(
                        row,
                        weight,
                        bound * weight_bound[j] <= i64::MAX as u128,
                    );
                    let raw = adapt + (i128::from(bias) << self.bits);
                    *value = i32::try_from(raw >> self.bits)
                        .map_err(|_| "projection output exceeds i32")?;
                    *rem = raw.rem_euclid(1 << self.bits) as u32;
                }
                Ok(())
            })?;
        Ok((values, remainders))
    }

    fn batch(self, queries: &[Query<'_, F>], channel: &mut impl Mixing<F>) -> Result<Batch<F>> {
        let (row_bits, width) = self.dimensions()?;
        if queries.is_empty()
            || queries.iter().any(|q| {
                !q.projection.valid(self.width)
                    || q.output.point.len()
                        != row_bits + q.projection.bias.len().next_power_of_two().ilog2() as usize
            })
        {
            return Err("invalid projection queries");
        }
        channel.append(&(queries.len() as u64));
        for q in queries {
            channel.append_values(&q.output.point);
            channel.append_values(&[q.output.value, q.remainder]);
        }
        let mix = channel.mix(queries.len(), queries.len())?;
        let mut groups: Vec<Group<F>> = Vec::new();
        let mut shared: Vec<(&[i32], &[F], Vec<F>)> = Vec::new();
        let mut claim = F::zero();
        let scale = F::from_u64(1 << self.bits);
        for (index, (q, a)) in queries.iter().zip(mix).enumerate() {
            let (row, col) = q.output.point.split_at(row_bits);
            let rows: F = basis(row).into_iter().take(self.rows).sum();
            let weights = basis(col);
            let bias: F = weights
                .iter()
                .zip(q.projection.bias)
                .map(|(&w, &b)| w * F::from_i32(b))
                .sum();
            claim += a * (scale * (q.output.value - rows * bias) + q.remainder);
            let group = match groups.iter().position(|g| g.point == row) {
                Some(i) => &mut groups[i],
                None => {
                    groups.push(Group {
                        point: row.to_vec(),
                        public: vec![F::zero(); width],
                    });
                    groups.last_mut().unwrap()
                }
            };
            let same = |w: &[i32], c: &[F]| std::ptr::eq(w, q.projection.weights) && c == col;
            if let Some((_, _, reduced)) = shared.iter().find(|(w, c, _)| same(w, c)) {
                for (out, &value) in group.public.iter_mut().zip(reduced) {
                    *out += a * value;
                }
                continue;
            }
            let reuse = queries[index + 1..]
                .iter()
                .any(|next| same(next.projection.weights, &next.output.point[row_bits..]));
            let project = |out: &mut [F], scale: F| {
                crate::tensor::add_rows(
                    out,
                    q.projection.weights,
                    self.width,
                    weights.iter().map(|&w| scale * w),
                );
            };
            if reuse {
                let mut reduced = vec![F::zero(); width];
                project(&mut reduced, F::one());
                for (out, &value) in group.public.iter_mut().zip(&reduced) {
                    *out += a * value;
                }
                shared.push((q.projection.weights, col, reduced));
            } else {
                project(&mut group.public, a);
            }
        }
        Ok(Batch {
            instance: SingleInstance {
                input_claim: claim,
                rounds: width.ilog2() as usize,
                degree: 2,
                format: SingleFormat::Compressed,
            },
            groups,
        })
    }

    pub(crate) fn reduce(
        self,
        io: &mut Channel<'_, '_>,
        input: Option<&[i32]>,
        queries: &[Query<'_, F>],
    ) -> Result<Vec<Claim<F>>> {
        let (_, width) = self.dimensions()?;
        if input.is_some_and(|v| v.len() != self.rows * self.width) {
            return Err("invalid projection input");
        }
        let batch = match io {
            Channel::Send(t, ..) => self.batch(queries, *t)?,
            Channel::Check(t, ..) => self.batch(queries, *t)?,
        };
        let witness = || {
            let input = input.unwrap();
            let pairs = batch
                .groups
                .iter()
                .map(|group| {
                    let mut restricted = vec![F::zero(); width];
                    crate::tensor::add_rows(
                        &mut restricted,
                        input,
                        self.width,
                        basis(&group.point),
                    );
                    (restricted, group.public.clone())
                })
                .collect();
            InnerProducts {
                pairs,
                rounds: batch.instance.rounds,
            }
        };
        let (end, values) = io.sumcheck(&batch.instance, batch.groups.len(), witness, |w| {
            w.pairs.into_iter().map(|part| part.0[0]).collect()
        })?;
        batch.finish(end, &values)
    }
}

impl Projection<'_> {
    fn valid(self, input: usize) -> bool {
        !self.bias.is_empty()
            && self.bias.len() <= 1 << 20
            && self.weights.len() == self.bias.len() * input
    }
}

impl Batch<F> {
    fn finish(self, end: SingleReduction<F>, values: &[F]) -> Result<Vec<Claim<F>>> {
        let weights = basis(&end.challenges);
        let expected: F = self
            .groups
            .iter()
            .zip(values)
            .map(|(group, &value)| {
                let public: F = weights
                    .iter()
                    .zip(&group.public)
                    .map(|(&a, &b)| a * b)
                    .sum();
                value * public
            })
            .sum();
        if expected != end.output_claim {
            return Err("invalid projection terminal evaluation");
        }
        Ok(self
            .groups
            .into_iter()
            .zip(values)
            .map(|(mut group, &value)| {
                group.point.extend_from_slice(&end.challenges);
                Claim {
                    point: group.point,
                    value,
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::F;
    use jolt_field::Ring;
    use jolt_transcript::{prover_transcript, verifier_transcript, Keccak};

    #[test]
    fn shares_input_evaluations_when_output_claims_use_the_same_row() {
        let shape = Input {
            rows: 2,
            width: 2,
            bits: 3,
        };
        let input = [5, 3, 2, -4];
        let projections = [
            Projection {
                weights: &[4, 7, -5, 9],
                bias: &[-1, 0],
            },
            Projection {
                weights: &[2, -5, 7, 1],
                bias: &[3, 0],
            },
        ];
        let calculate = |projection, point: [u64; 2]| {
            let point = point.map(F::from_u64);
            let (out, rest) = shape.execute(&input, projection).unwrap();
            let weights = basis(&point);
            let value = weights
                .iter()
                .zip(out)
                .map(|(&w, x)| w * F::from_i32(x))
                .sum();
            let remainder = weights
                .iter()
                .zip(rest)
                .map(|(&w, x)| w * F::from_u32(x))
                .sum();
            (
                Claim {
                    point: point.to_vec(),
                    value,
                },
                remainder,
            )
        };
        let (left, l_rem) = calculate(projections[0], [3, 7]);
        let (right, r_rem) = calculate(projections[1], [3, 11]);
        let (other, o_rem) = calculate(projections[1], [13, 11]);
        let queries = [
            Query {
                projection: projections[0],
                output: &left,
                remainder: l_rem,
            },
            Query {
                projection: projections[1],
                output: &right,
                remainder: r_rem,
            },
            Query {
                projection: projections[1],
                output: &other,
                remainder: o_rem,
            },
        ];
        let tag = [0x58; 32];
        let mut prover = prover_transcript::<Keccak>(&tag, b"");
        let advice = crate::advice::Advice::new(false).unwrap();
        let mut proof = crate::channel::Messages::default();
        let mut send = Channel::Send(&mut prover, &mut proof, &advice);
        let claims = shape.reduce(&mut send, Some(&input), &queries).unwrap();
        assert_eq!(claims.len(), 2);
        for claim in &claims {
            let expected = basis(&claim.point)
                .into_iter()
                .zip(input)
                .map(|(weight, x)| weight * F::from_i32(x))
                .sum();
            assert_eq!(claim.value, expected);
        }
        let mut verifier = verifier_transcript::<Keccak>(&tag, b"", prover.narg_string());
        let mut check = Channel::checking(&mut verifier, &proof);
        assert_eq!(shape.reduce(&mut check, None, &queries).unwrap(), claims);
        check.finish().unwrap();
    }
}
