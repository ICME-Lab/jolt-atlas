use jolt_field::{Field, One, Zero};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, MultilinearPoly, UnivariatePoly};
use jolt_sumcheck::{
    prove_single, round_pow_bits, verify_single, ProveRounds, SingleFormat, SingleInstance,
    SingleReduction, SingleSumcheckProof, SumcheckError,
};
use jolt_transcript::{
    FieldTranscript, Keccak, ProverState, ProverTranscript, Transcript, VerifierState,
    VerifierTranscript,
};
use rayon::prelude::*;

use super::{
    field::F,
    tensor::{equality, inner, weights, Claim},
    Result,
};

#[derive(Default)]
pub(crate) struct Messages {
    fields: Vec<F>,
    checks: Vec<SingleSumcheckProof<F>>,
}

pub(crate) enum Session<'a, 'proof> {
    Prover(&'a mut ProverState<Keccak>, &'a mut Messages),
    Verifier(
        &'a mut VerifierState<'proof, Keccak>,
        &'proof [F],
        &'proof [SingleSumcheckProof<F>],
    ),
}

impl Messages {
    pub(crate) fn verifier<'a, 'p>(
        &'p self,
        transcript: &'a mut VerifierState<'p, Keccak>,
    ) -> Session<'a, 'p> {
        Session::Verifier(transcript, &self.fields, &self.checks)
    }
}

impl Session<'_, '_> {
    pub(crate) fn proving(&self) -> bool {
        matches!(self, Self::Prover(..))
    }

    pub(crate) fn finish(self) -> Result<()> {
        match self {
            Self::Verifier(_, fields, checks) if !fields.is_empty() || !checks.is_empty() => {
                Err("unused proof messages")
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn random(&mut self, count: usize, loss: usize) -> Result<Vec<F>> {
        let bits = round_pow_bits::<F, ProverState<Keccak>>(loss)
            .map_err(|_| "unsupported challenge size")?;
        match self {
            Self::Prover(transcript, _) => {
                transcript.grind(bits).map_err(|_| "proof of work failed")?;
                transcript.challenge_vector(count)
            }
            Self::Verifier(transcript, _, _) => {
                transcript
                    .check_grind(bits)
                    .map_err(|_| "invalid proof of work")?;
                transcript.challenge_vector(count)
            }
        }
        .map_err(|_| "invalid challenges")
    }

    pub(crate) fn values<const N: usize>(
        &mut self,
        witness: impl FnOnce() -> [F; N],
    ) -> Result<[F; N]> {
        match self {
            Self::Prover(transcript, proof) => {
                let values = witness();
                proof.fields.extend(values);
                transcript.append_values(&values);
                Ok(values)
            }
            Self::Verifier(transcript, fields, _) => {
                let (values, remaining) = fields
                    .split_first_chunk::<N>()
                    .ok_or("missing evaluations")?;
                *fields = remaining;
                transcript.append_values(values);
                Ok(*values)
            }
        }
    }

    fn fields(&mut self, count: usize, witness: impl FnOnce() -> Vec<F>) -> Result<Vec<F>> {
        match self {
            Self::Prover(transcript, proof) => {
                let values = witness();
                proof.fields.extend(&values);
                transcript.append_values(&values);
                Ok(values)
            }
            Self::Verifier(transcript, fields, _) => {
                let (values, remaining) = fields
                    .split_at_checked(count)
                    .ok_or("missing evaluations")?;
                *fields = remaining;
                transcript.append_values(values);
                Ok(values.to_vec())
            }
        }
    }

    pub(crate) fn fold<'w, const N: usize>(
        &mut self,
        input: F,
        rounds: usize,
        degree: usize,
        columns: impl FnOnce() -> Vec<Column<'w>>,
        endpoints: [usize; N],
        formula: impl Fn(&[F]) -> F + Sync,
    ) -> Result<(SingleReduction<F>, [F; N])> {
        let (end, values) = self.fold_many(input, rounds, degree, columns, &endpoints, formula)?;
        Ok((
            end,
            values
                .try_into()
                .map_err(|_| "invalid Sumcheck evaluation count")?,
        ))
    }

    pub(crate) fn fold_many<'w>(
        &mut self,
        input: F,
        rounds: usize,
        degree: usize,
        columns: impl FnOnce() -> Vec<Column<'w>>,
        endpoints: &[usize],
        formula: impl Fn(&[F]) -> F + Sync,
    ) -> Result<(SingleReduction<F>, Vec<F>)> {
        let statement = SingleInstance {
            input_claim: input,
            rounds,
            degree,
            format: SingleFormat::Compressed,
        };
        let mut last = Vec::new();
        let result = match self {
            Self::Prover(transcript, proof) => {
                let mut fold = Fold {
                    columns: columns(),
                    rounds,
                    degree,
                    formula,
                };
                let (proof_data, result) = prove_single(&mut fold, &statement, *transcript)
                    .map_err(|_| "could not prove Sumcheck")?;
                last = endpoints
                    .iter()
                    .map(|&index| fold.columns[index].get(0))
                    .collect();
                proof.checks.push(proof_data);
                result
            }
            Self::Verifier(transcript, _, checks) => {
                let (check, remaining) = checks.split_first().ok_or("missing Sumcheck")?;
                *checks = remaining;
                verify_single(&statement, check, *transcript).map_err(|_| "invalid Sumcheck")?
            }
        };
        Ok((result, self.fields(endpoints.len(), || last)?))
    }

    pub(crate) fn dot(
        &mut self,
        value: F,
        public: &[F],
        witness: impl FnOnce() -> Vec<F>,
    ) -> Result<Claim> {
        let (end, [value]) = self.fold(
            value,
            public.len().ilog2() as usize,
            2,
            || vec![Column::Dense(witness()), Column::Public(public)],
            [0],
            |v| v[0] * v[1],
        )?;
        self.expect(&end, || public.evaluate(&end.challenges) * value)?;
        Ok(Claim::new(end.challenges, value))
    }

    pub(crate) fn combine(
        &mut self,
        claims: &[Claim],
        witness: impl FnOnce() -> Vec<F>,
    ) -> Result<Claim> {
        let first = claims.first().ok_or("missing evaluation claims")?;
        if claims.len() == 1 {
            return Ok(first.clone());
        }
        let mix = self.random(claims.len(), claims.len())?;
        let values = || {
            let mut result = vec![F::zero(); 1 << first.point.len()];
            for (claim, &factor) in claims.iter().zip(&mix) {
                for (out, weight) in result.iter_mut().zip(weights(&claim.point)) {
                    *out += factor * weight;
                }
            }
            result
        };
        let sum = inner(claims.iter().map(|q| q.value), mix.iter().copied());
        let (end, [value]) = self.fold(
            sum,
            first.point.len(),
            2,
            || vec![Column::Dense(witness()), Column::Dense(values())],
            [0],
            |v| v[0] * v[1],
        )?;
        self.expect(&end, || {
            value
                * inner(
                    mix,
                    claims
                        .iter()
                        .map(|claim| equality(&end.challenges, &claim.point)),
                )
        })?;
        Ok(Claim::new(end.challenges, value))
    }

    pub(crate) fn expect(
        &self,
        end: &SingleReduction<F>,
        evaluate: impl FnOnce() -> F,
    ) -> Result<()> {
        if !self.proving() && end.output_claim != evaluate() {
            return Err("invalid terminal evaluation");
        }
        Ok(())
    }
}

pub(crate) enum Column<'a> {
    Dense(Vec<F>),
    Public(&'a [F]),
    Source(usize, Box<dyn Fn(usize) -> F + Send + Sync + 'a>),
    Lookup {
        len: usize,
        read: Box<dyn Fn(usize) -> usize + Send + Sync + 'a>,
        width: usize,
        tables: Vec<F>,
    },
    Equality(Box<SplitEquality>),
}

pub(crate) struct SplitEquality {
    polynomial: GruenSplitEqPolynomial<F>,
    len: usize,
    high: Vec<F>,
}

impl SplitEquality {
    fn refresh(&mut self) {
        self.high.clear();
        if self.len == 1 {
            self.high.push(self.polynomial.current_scalar());
            return;
        }
        let (low, high) = self.polynomial.current_linear_evals();
        let inner = self.polynomial.e_in_current();
        for scale in [low, high] {
            self.high.extend(inner.iter().map(|&value| scale * value));
        }
    }

    fn get(&self, i: usize) -> F {
        let low = self.polynomial.e_out_current();
        self.high[i >> low.len().trailing_zeros()] * low[i & (low.len() - 1)]
    }
}

impl<'a> Column<'a> {
    pub(crate) fn source(n: usize, read: impl Fn(usize) -> F + Send + Sync + 'a) -> Self {
        Self::Source(n, Box::new(read))
    }

    pub(crate) fn lookup(
        len: usize,
        table: &[F],
        read: impl Fn(usize) -> usize + Send + Sync + 'a,
    ) -> Self {
        Self::Lookup {
            len,
            read: Box::new(read),
            width: table.len(),
            tables: table.to_vec(),
        }
    }

    pub(crate) fn split_equality(point: &[F]) -> Self {
        let mut column = Box::new(SplitEquality {
            polynomial: GruenSplitEqPolynomial::new(point, BindingOrder::HighToLow),
            len: 1 << point.len(),
            high: Vec::new(),
        });
        column.refresh();
        Self::Equality(column)
    }

    fn len(&self) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::Public(v) => v.len(),
            Self::Source(n, _) | Self::Lookup { len: n, .. } => *n,
            Self::Equality(column) => column.len,
        }
    }

    pub(crate) fn get(&self, i: usize) -> F {
        match self {
            Self::Dense(v) => v[i],
            Self::Public(v) => v[i],
            Self::Source(_, read) => read(i),
            Self::Lookup {
                len,
                read,
                width,
                tables,
            } => tables
                .chunks_exact(*width)
                .enumerate()
                .map(|(prefix, table)| table[read(prefix * len + i)])
                .sum(),
            Self::Equality(column) => column.get(i),
        }
    }

    fn bind(&mut self, r: F) {
        let half = self.len() / 2;
        match self {
            Self::Equality(column) => {
                column.polynomial.bind(r);
                column.len = half;
                column.refresh();
                return;
            }
            Self::Lookup {
                len, width, tables, ..
            } if tables.len() < 4 * *width => {
                // Bind the first two coordinates in the digit tables.
                let mut next = Vec::with_capacity(2 * tables.len());
                let low = F::one() - r;
                for table in tables.chunks_exact(*width) {
                    next.extend(table.iter().map(|&value| low * value));
                    next.extend(table.iter().map(|&value| r * value));
                }
                *tables = next;
                *len = half;
                return;
            }
            _ => {}
        }
        if let Self::Dense(v) = self {
            let (lo, hi) = v.split_at_mut(half);
            if half >= 2048 {
                lo.par_iter_mut()
                    .with_min_len(512)
                    .zip(hi)
                    .for_each(|(x, &mut y)| *x = r.mul_add(y - *x, *x));
            } else {
                for (x, y) in lo.iter_mut().zip(hi) {
                    *x = r.mul_add(*y - *x, *x);
                }
            }
            v.truncate(half);
            return;
        }
        let evaluate = |i| {
            let low = self.get(i);
            r.mul_add(self.get(i + half) - low, low)
        };
        let values = if half >= 2048 {
            (0..half)
                .into_par_iter()
                .with_min_len(512)
                .map(evaluate)
                .collect()
        } else {
            (0..half).map(evaluate).collect()
        };
        *self = Self::Dense(values);
    }
}

struct Fold<'a, E> {
    columns: Vec<Column<'a>>,
    rounds: usize,
    degree: usize,
    formula: E,
}

impl<E: Fn(&[F]) -> F + Sync> ProveRounds<F> for Fold<'_, E> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        previous: Option<F>,
        _: usize,
        sum: F,
    ) -> std::result::Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = previous {
            for column in &mut self.columns {
                column.bind(r);
            }
        }
        let half = self.columns[0].len() / 2;
        let run = |start: usize, end: usize| {
            let mut low = vec![F::zero(); self.columns.len()];
            let mut delta = low.clone();
            let mut sums = vec![F::zero(); self.degree + 1];
            for index in start..end {
                for (j, column) in self.columns.iter().enumerate() {
                    low[j] = column.get(index);
                    delta[j] = column.get(index + half) - low[j];
                }
                sums[0] += (self.formula)(&low);
                for (x, d) in low.iter_mut().zip(&delta) {
                    *x += *d;
                }
                for total in &mut sums[2..] {
                    for (x, d) in low.iter_mut().zip(&delta) {
                        *x += *d;
                    }
                    *total += (self.formula)(&low);
                }
            }
            sums
        };
        let mut sums = if half >= 2048 {
            (0..half.div_ceil(512))
                .into_par_iter()
                .map(|block| run(block * 512, ((block + 1) * 512).min(half)))
                .reduce_with(|mut a, b| {
                    for (lhs, rhs) in a.iter_mut().zip(b) {
                        *lhs += rhs;
                    }
                    a
                })
                .expect("nonempty Sumcheck round")
        } else {
            run(0, half)
        };
        sums[1] = sum - sums[0];
        Ok(UnivariatePoly::from_evals(&sums))
    }

    fn finish_rounds(&mut self, last: F) -> std::result::Result<(), SumcheckError<F>> {
        if self.rounds != 0 {
            for column in &mut self.columns {
                column.bind(last);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::Ring;

    use super::*;

    #[test]
    fn compact_lookup_columns_fold_like_dense_columns() {
        let values: Vec<_> = (1..=8).map(F::from_u64).collect();
        let point = &values[..5];
        let digit = |i: usize| (i ^ (i >> 3)) & 7;
        let mut lookup = Column::lookup(32, &values, digit);
        let mut dense_lookup = Column::Dense((0..32).map(|i| values[digit(i)]).collect());
        let mut equality = Column::split_equality(point);
        let mut dense_equality = Column::Dense(weights(point));
        for round in 0..=point.len() {
            if round > 0 {
                let challenge = F::from_u64(10 + round as u64);
                for column in [
                    &mut lookup,
                    &mut dense_lookup,
                    &mut equality,
                    &mut dense_equality,
                ] {
                    column.bind(challenge);
                }
            }
            for i in 0..lookup.len() {
                assert_eq!(lookup.get(i), dense_lookup.get(i));
                assert_eq!(equality.get(i), dense_equality.get(i));
            }
        }
    }
}
