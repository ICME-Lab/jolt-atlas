//! Shared transcript and Sumcheck execution, with one statement per relation.

use jolt_field::JoltField;
use jolt_poly::{MultilinearPoly, UnivariatePoly};
use jolt_sumcheck::{
    prove_single, round_pow_bits, verify_single, ProveRounds, SingleFormat, SingleInstance,
    SingleReduction, SingleSumcheckProof, SumcheckError,
};
use jolt_transcript::{
    Encoding, FieldTranscript, Keccak, ProverState, ProverTranscript, Transcript, VerifierState,
    VerifierTranscript,
};
use rayon::prelude::*;
use std::borrow::Cow;

use super::{
    pcs::Opening,
    tensor::{eq, weights},
    Result,
};

pub(super) const PARALLEL_ROUND_SIZE: usize = 1 << 11;
pub(super) const ROUND_CHUNK_SIZE: usize = 1 << 9;

pub trait ProofField: JoltField + Encoding {
    type Prover: FieldTranscript<Self> + ProverTranscript;
    type Verifier<'p>: FieldTranscript<Self> + VerifierTranscript;
    fn prover(t: &mut ProverState<Keccak>) -> &mut Self::Prover;
    fn verifier<'a, 'p>(t: &'a mut VerifierState<'p, Keccak>) -> &'a mut Self::Verifier<'p>;
}

impl<F: JoltField + Encoding> ProofField for F
where
    ProverState<Keccak>: FieldTranscript<F>,
    for<'p> VerifierState<'p, Keccak>: FieldTranscript<F>,
{
    type Prover = ProverState<Keccak>;
    type Verifier<'p> = VerifierState<'p, Keccak>;

    fn prover(t: &mut Self::Prover) -> &mut Self::Prover {
        t
    }

    fn verifier<'a, 'p>(t: &'a mut VerifierState<'p, Keccak>) -> &'a mut Self::Verifier<'p> {
        t
    }
}

#[derive(Clone)]
pub(crate) struct Claim<F> {
    pub(crate) point: Vec<F>,
    pub(crate) value: F,
}

#[derive(Default)]
pub struct Messages<F> {
    values: Vec<F>,
    checks: Vec<SingleSumcheckProof<F>>,
}

pub(crate) struct Session<'a, 'proof, F: ProofField> {
    mode: Mode<'a, 'proof, F>,
    pub(crate) openings: Vec<Opening<F>>,
}

enum Mode<'a, 'proof, F: ProofField> {
    Prove(&'a mut F::Prover, &'a mut Messages<F>),
    Verify(
        &'a mut F::Verifier<'proof>,
        &'a [F],
        std::slice::Iter<'a, SingleSumcheckProof<F>>,
    ),
}

impl<F: ProofField> Messages<F> {
    pub(crate) fn read<'a, 'p>(
        &'a self,
        transcript: &'a mut VerifierState<'p, Keccak>,
    ) -> Session<'a, 'p, F> {
        Session {
            mode: Mode::Verify(F::verifier(transcript), &self.values, self.checks.iter()),
            openings: Vec::new(),
        }
    }
}

impl<'a, F: ProofField> Session<'a, '_, F> {
    pub(crate) fn prove(t: &'a mut F::Prover, messages: &'a mut Messages<F>) -> Self {
        Self {
            mode: Mode::Prove(t, messages),
            openings: Vec::new(),
        }
    }

    pub(crate) fn proving(&self) -> bool {
        matches!(self.mode, Mode::Prove(..))
    }

    pub(crate) fn finish(self) -> Result<Vec<Opening<F>>> {
        if let Mode::Verify(_, values, checks) = self.mode {
            if !values.is_empty() || checks.len() != 0 {
                return Err("unused proof messages");
            }
        }
        Ok(self.openings)
    }

    pub(crate) fn pow(&mut self, loss: usize) -> Result<()> {
        let bits =
            round_pow_bits::<F, F::Prover>(loss).map_err(|_| "unsupported PoW difficulty")?;
        match &mut self.mode {
            Mode::Prove(t, ..) => t.grind(bits),
            Mode::Verify(t, ..) => t.check_grind(bits),
        }
        .map_err(|_| "invalid proof of work")
    }

    pub(crate) fn challenge(&mut self, count: usize) -> Result<Vec<F>> {
        match &mut self.mode {
            Mode::Prove(t, ..) => t.challenge_vector(count),
            Mode::Verify(t, ..) => t.challenge_vector(count),
        }
        .map_err(|_| "invalid challenge")
    }

    pub(crate) fn point(&mut self, vars: usize) -> Result<Vec<F>> {
        self.pow(vars)?;
        self.challenge(vars)
    }

    pub(crate) fn values<const N: usize>(
        &mut self,
        produce: impl FnOnce() -> [F; N],
    ) -> Result<[F; N]> {
        self.values_prefix(N, produce)
    }

    pub(crate) fn values_prefix<const N: usize>(
        &mut self,
        count: usize,
        produce: impl FnOnce() -> [F; N],
    ) -> Result<[F; N]> {
        match &mut self.mode {
            Mode::Prove(t, messages, ..) => {
                let values = produce();
                messages.values.extend_from_slice(&values[..count]);
                t.append_values(&values[..count]);
                Ok(values)
            }
            Mode::Verify(t, unread, ..) => {
                let (values, rest) = unread.split_at_checked(count).ok_or("missing evaluation")?;
                *unread = rest;
                t.append_values(values);
                let mut padded = [F::zero(); N];
                padded[..count].copy_from_slice(values);
                Ok(padded)
            }
        }
    }

    pub(crate) fn reduce(
        &mut self,
        statement: &SingleInstance<F>,
        prove: impl FnOnce(&mut F::Prover) -> Result<(SingleSumcheckProof<F>, SingleReduction<F>)>,
    ) -> Result<SingleReduction<F>> {
        match &mut self.mode {
            Mode::Prove(t, messages, ..) => {
                let (proof, end) = prove(t)?;
                messages.checks.push(proof);
                Ok(end)
            }
            Mode::Verify(t, _, messages) => {
                let proof = messages.next().ok_or("missing Sumcheck")?;
                verify_single(statement, proof, *t).map_err(|_| "invalid Sumcheck")
            }
        }
    }

    pub(crate) fn fold<'w, const N: usize, C: AsMut<[Column<'w, F>]>>(
        &mut self,
        stmt: SingleInstance<F>,
        columns: impl FnOnce() -> C,
        endpoints: [usize; N],
        expression: impl Fn(&[F]) -> F + Sync,
    ) -> Result<(SingleReduction<F>, [F; N])> {
        let mut values = [F::zero(); N];
        let end = self.reduce(&stmt, |t| {
            let mut fold = Fold::new(columns(), stmt.rounds, stmt.degree, expression);
            let reduced = prove_single(&mut fold, &stmt, t).map_err(|_| "Sumcheck failed")?;
            values = endpoints.map(|i| fold.columns.as_mut()[i].at(0));
            Ok(reduced)
        })?;
        Ok((end, self.values(|| values)?))
    }

    pub(crate) fn fold_weighted<'w, const N: usize, C: AsMut<[Column<'w, F>]> + Send>(
        &mut self,
        stmt: SingleInstance<F>,
        mix: &[F],
        columns: impl FnOnce() -> Vec<C>,
        endpoints: [usize; N],
        expression: impl Fn(&[F]) -> F + Sync,
    ) -> Result<(SingleReduction<F>, Vec<[F; N]>)> {
        let mut values = Vec::new();
        let end = self.reduce(&stmt, |t| {
            let mut batch = BatchFold(
                columns()
                    .into_iter()
                    .zip(mix)
                    .map(|(columns, &w)| {
                        (w, Fold::new(columns, stmt.rounds, stmt.degree, &expression))
                    })
                    .collect(),
            );
            let reduced =
                prove_single(&mut batch, &stmt, t).map_err(|_| "batched Sumcheck failed")?;
            if N != 0 {
                values = batch
                    .0
                    .iter_mut()
                    .map(|(_, f)| endpoints.map(|i| f.columns.as_mut()[i].at(0)))
                    .collect();
            }
            Ok(reduced)
        })?;
        let values = if N == 0 {
            Vec::new()
        } else {
            (0..mix.len())
                .map(|i| self.values(|| values[i]))
                .collect::<Result<_>>()?
        };
        Ok((end, values))
    }

    pub(crate) fn public_dot(
        &mut self,
        sum: F,
        public: &[F],
        witness: impl FnOnce() -> Vec<F>,
    ) -> Result<Claim<F>> {
        let (point, [value]) = self.relation(
            instance(sum, public.len().ilog2() as usize, 2),
            || [Column::owned(witness()), Column::borrowed(public)],
            |v| v[0] * v[1],
            |p, &[v]| v * public.evaluate(p),
        )?;
        Ok(Claim { point, value })
    }

    pub(crate) fn relation<'w, const C: usize, const N: usize>(
        &mut self,
        stmt: SingleInstance<F>,
        columns: impl FnOnce() -> [Column<'w, F>; C],
        expression: impl Fn(&[F]) -> F + Sync,
        oracle: impl FnOnce(&[F], &[F; N]) -> F,
    ) -> Result<(Vec<F>, [F; N])> {
        let (end, values) = self.fold(stmt, columns, std::array::from_fn(|i| i), expression)?;
        self.check_value(&end, || oracle(&end.challenges, &values))?;
        Ok((end.challenges, values))
    }

    pub(crate) fn check_value(
        &self,
        end: &SingleReduction<F>,
        value: impl FnOnce() -> F,
    ) -> Result<()> {
        if !self.proving() && end.output_claim != value() {
            return Err("invalid terminal evaluation");
        }
        Ok(())
    }

    pub(crate) fn merge(
        &mut self,
        claims: &[Claim<F>],
        polynomial: impl FnOnce() -> Vec<F>,
    ) -> Result<Claim<F>> {
        let [first, ..] = claims else {
            return Err("missing input claims");
        };
        if claims.len() == 1 {
            return Ok(first.clone());
        }
        self.pow(claims.len())?;
        let mix = self.challenge(claims.len())?;
        let sum = claims
            .iter()
            .zip(&mix)
            .map(|(claim, &scale)| claim.value * scale)
            .sum();
        let (point, [value]) = self.relation(
            instance(sum, first.point.len(), 2),
            || {
                [
                    Column::owned(polynomial()),
                    Column::owned(query_column(claims, &mix)),
                ]
            },
            |v| v[0] * v[1],
            |p, &[v]| v * query_weight(claims, &mix, p),
        )?;
        Ok(Claim { point, value })
    }
}

pub(crate) fn instance<F>(input_claim: F, rounds: usize, degree: usize) -> SingleInstance<F> {
    SingleInstance {
        input_claim,
        rounds,
        degree,
        format: SingleFormat::Compressed,
    }
}

pub(crate) fn query_column<F: JoltField>(claims: &[Claim<F>], mix: &[F]) -> Vec<F> {
    let mut column = vec![F::zero(); 1 << claims[0].point.len()];
    for (claim, &mix) in claims.iter().zip(mix) {
        let fixed = claim
            .point
            .iter()
            .take_while(|&&r| r == F::zero() || r == F::one())
            .count();
        let prefix = claim.point[..fixed]
            .iter()
            .fold(0, |i, &r| (i << 1) | usize::from(r == F::one()));
        let free = &claim.point[fixed..];
        let start = prefix << free.len();
        for (x, w) in column[start..start + (1 << free.len())]
            .iter_mut()
            .zip(weights(free))
        {
            *x += mix * w;
        }
    }
    column
}

pub(crate) fn query_weight<F: JoltField>(claims: &[Claim<F>], mix: &[F], point: &[F]) -> F {
    claims
        .iter()
        .zip(mix)
        .map(|(claim, &mix)| mix * eq(&claim.point, point))
        .sum()
}

pub(crate) enum Column<'a, F: Clone> {
    Field(Cow<'a, [F]>, usize, usize),
    Bits(&'a [u8], usize),
}

impl<'a, F: JoltField> Column<'a, F> {
    pub(crate) fn borrowed(values: &'a [F]) -> Self {
        Self::Field(Cow::Borrowed(values), 1, 1)
    }
    pub(crate) fn owned(values: Vec<F>) -> Self {
        Self::Field(Cow::Owned(values), 1, 1)
    }
    pub(crate) fn rows(values: Cow<'a, [F]>, width: usize) -> Self {
        Self::Field(values, width, 1)
    }
    pub(crate) fn repeat(values: Cow<'a, [F]>, copies: usize) -> Self {
        Self::Field(values, 1, copies)
    }

    fn len(&self) -> usize {
        match self {
            Self::Field(v, width, copies) => v.len() * width * copies,
            Self::Bits(v, copies) => v.len() * copies,
        }
    }

    pub(super) fn at(&self, i: usize) -> F {
        match self {
            Self::Field(v, 1, 1) => v[i],
            Self::Field(v, width, 1) => v[i >> width.trailing_zeros()],
            Self::Field(v, _, _) => v[i & (v.len() - 1)],
            Self::Bits(v, 1) => F::from_bool(v[i] != 0),
            Self::Bits(v, _) => F::from_bool(v[i & (v.len() - 1)] != 0),
        }
    }

    fn bind(&mut self, r: F) {
        match self {
            Self::Field(_, _, copies) | Self::Bits(_, copies) if *copies > 1 => *copies /= 2,
            Self::Field(values, width, _) if values.len() == 1 => *width /= 2,
            Self::Field(values, _, _) => bind_halves(values.to_mut(), r),
            bits => {
                let half = bits.len() / 2;
                let bind = |i| {
                    let a = bits.at(i);
                    a + r * (bits.at(i + half) - a)
                };
                *bits = Self::owned(if half >= PARALLEL_ROUND_SIZE {
                    (0..half)
                        .into_par_iter()
                        .with_min_len(ROUND_CHUNK_SIZE)
                        .map(bind)
                        .collect()
                } else {
                    (0..half).map(bind).collect()
                });
            }
        }
    }
}

pub(crate) fn bind_halves<F: JoltField>(v: &mut Vec<F>, r: F) {
    let half = v.len() / 2;
    let (low, high) = v.split_at_mut(half);
    let bind = |(left, &right): (&mut F, &F)| *left += r * (right - *left);
    if half >= PARALLEL_ROUND_SIZE {
        low.par_iter_mut()
            .with_min_len(ROUND_CHUNK_SIZE)
            .zip(high.par_iter())
            .for_each(bind);
    } else {
        low.iter_mut().zip(high.iter()).for_each(bind);
    }
    v.truncate(half);
}

pub(crate) struct Fold<C, E> {
    pub(super) columns: C,
    expression: E,
    rounds: usize,
    degree: usize,
}

impl<C, E> Fold<C, E> {
    pub(crate) fn new(columns: C, rounds: usize, degree: usize, expression: E) -> Self {
        Self {
            columns,
            rounds,
            degree,
            expression,
        }
    }
}

impl<'a, F: JoltField + 'a, E: Fn(&[F]) -> F + Sync, C: AsMut<[Column<'a, F>]>> ProveRounds<F>
    for Fold<C, E>
{
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        previous: Option<F>,
        _: usize,
        _: F,
    ) -> std::result::Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = previous {
            self.finish_rounds(r)?;
        }
        let columns = &*self.columns.as_mut();
        let half = columns[0].len() / 2;
        let zeros = || vec![F::zero(); self.degree + 1];
        let sum = |start: usize, end: usize| {
            let mut deltas = vec![F::zero(); columns.len()];
            let mut values = vec![F::zero(); columns.len()];
            let mut samples = zeros();
            for i in start..end {
                for ((v, delta), col) in values.iter_mut().zip(&mut deltas).zip(columns) {
                    *v = col.at(i);
                    *delta = col.at(i + half) - *v;
                }
                samples[0] += (self.expression)(&values);
                for sample in &mut samples[1..] {
                    for (v, &delta) in values.iter_mut().zip(&deltas) {
                        *v += delta;
                    }
                    *sample += (self.expression)(&values);
                }
            }
            samples
        };
        let samples = if half >= PARALLEL_ROUND_SIZE {
            let chunk = half
                .div_ceil(rayon::current_num_threads())
                .max(ROUND_CHUNK_SIZE);
            (0..half)
                .into_par_iter()
                .step_by(chunk)
                .map(|start| sum(start, (start + chunk).min(half)))
                .reduce(zeros, |mut total, other| {
                    for (value, partial) in total.iter_mut().zip(other) {
                        *value += partial;
                    }
                    total
                })
        } else {
            sum(0, half)
        };
        Ok(UnivariatePoly::from_evals(&samples))
    }

    fn finish_rounds(&mut self, last: F) -> std::result::Result<(), SumcheckError<F>> {
        if self.rounds > 0 {
            for column in self.columns.as_mut() {
                column.bind(last);
            }
        }
        Ok(())
    }
}

struct BatchFold<F, C, E>(Vec<(F, Fold<C, E>)>);

impl<'a, F: JoltField + 'a, C: AsMut<[Column<'a, F>]> + Send, E: Fn(&[F]) -> F + Send + Sync>
    ProveRounds<F> for BatchFold<F, C, E>
{
    fn num_rounds(&self) -> usize {
        self.0[0].1.rounds
    }

    fn prove_round(
        &mut self,
        previous: Option<F>,
        round: usize,
        _: F,
    ) -> core::result::Result<UnivariatePoly<F>, SumcheckError<F>> {
        self.0
            .par_iter_mut()
            .map(|(w, fold)| fold.prove_round(previous, round, F::zero()).map(|p| p * *w))
            .try_reduce_with(|a, b| Ok(a + b))
            .expect("nonempty Sumcheck batch")
    }

    fn finish_rounds(&mut self, last: F) -> core::result::Result<(), SumcheckError<F>> {
        self.0
            .par_iter_mut()
            .try_for_each(|(_, fold)| fold.finish_rounds(last))
    }
}
