use jolt_field::{Field, Zero};
use jolt_poly::{MultilinearPoly, UnivariatePoly};
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
}

impl<'a> Column<'a> {
    pub(crate) fn source(n: usize, read: impl Fn(usize) -> F + Send + Sync + 'a) -> Self {
        Self::Source(n, Box::new(read))
    }

    fn len(&self) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::Public(v) => v.len(),
            Self::Source(n, _) => *n,
        }
    }

    pub(crate) fn get(&self, i: usize) -> F {
        match self {
            Self::Dense(v) => v[i],
            Self::Public(v) => v[i],
            Self::Source(_, read) => read(i),
        }
    }

    fn bind(&mut self, r: F) {
        let half = self.len() / 2;
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
