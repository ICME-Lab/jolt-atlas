//! Shared transcript and Sumcheck execution, with one statement per relation.

use jolt_field::JoltField;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, MultilinearPoly, UnivariatePoly};
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
    tensor::{eq, weights, Shape},
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

    #[cfg(feature = "akita")]
    pub(crate) fn bind_values(&mut self, values: &[F]) {
        match &mut self.mode {
            Mode::Prove(t, ..) => t.append_values(values),
            Mode::Verify(t, ..) => t.append_values(values),
        }
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
        endpoint_count: usize,
        expression: impl Fn(&[F]) -> F + Sync,
    ) -> Result<(SingleReduction<F>, Vec<[F; N]>)> {
        self.fold_weighted_masked(
            stmt,
            mix,
            None,
            columns,
            endpoints,
            endpoint_count,
            expression,
        )
    }

    pub(crate) fn fold_weighted_masked<'w, const N: usize, C: AsMut<[Column<'w, F>]> + Send>(
        &mut self,
        stmt: SingleInstance<F>,
        mix: &[F],
        mask: Option<[usize; 2]>,
        columns: impl FnOnce() -> Vec<C>,
        endpoints: [usize; N],
        endpoint_count: usize,
        expression: impl Fn(&[F]) -> F + Sync,
    ) -> Result<(SingleReduction<F>, Vec<[F; N]>)> {
        let mut values = Vec::new();
        let end = self.reduce(&stmt, |t| {
            let mut batch = BatchFold(
                columns()
                    .into_iter()
                    .zip(mix)
                    .map(|(columns, &w)| {
                        let mut fold = Fold::new(columns, stmt.rounds, stmt.degree, &expression);
                        fold.mask = mask;
                        (w, fold)
                    })
                    .collect(),
            );
            let reduced =
                prove_single(&mut batch, &stmt, t).map_err(|_| "batched Sumcheck failed")?;
            if endpoint_count != 0 {
                values = batch
                    .0
                    .iter_mut()
                    .map(|(_, f)| endpoints.map(|i| f.columns.as_mut()[i].at(0)))
                    .collect();
            }
            Ok(reduced)
        })?;
        let values = if endpoint_count == 0 {
            Vec::new()
        } else {
            (0..mix.len())
                .map(|i| self.values_prefix(endpoint_count, || values[i]))
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

/// Cached factors for a high-to-low equality column. The current coordinate
/// is multiplied into the high table once per round, outside the row scan.
pub(crate) struct Equality<F: JoltField> {
    split: GruenSplitEqPolynomial<F>,
    high: Vec<F>,
    len: usize,
}

impl<F: JoltField> Equality<F> {
    pub(crate) fn new(point: &[F]) -> Self {
        let mut result = Self {
            split: GruenSplitEqPolynomial::new(point, BindingOrder::HighToLow),
            high: Vec::new(),
            len: 1 << point.len(),
        };
        result.refresh();
        result
    }

    fn refresh(&mut self) {
        self.high.clear();
        if self.len == 1 {
            self.high.push(self.split.current_scalar());
        } else {
            let (lo, hi) = self.split.current_linear_evals();
            for scale in [lo, hi] {
                self.high
                    .extend(self.split.e_in_current().iter().map(|&v| scale * v));
            }
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn bind(&mut self, challenge: F) {
        self.split.bind(challenge);
        self.len /= 2;
        self.refresh();
    }

    pub(crate) fn factors(&self) -> (&[F], &[F]) {
        (&self.high, self.split.e_out_current())
    }

    fn at(&self, i: usize) -> F {
        let (high, low) = self.factors();
        high[i >> low.len().trailing_zeros()] * low[i & (low.len() - 1)]
    }
}

pub(crate) enum Column<'a, F: JoltField> {
    Field(Cow<'a, [F]>, usize, usize),
    Bits(&'a [u8], usize),
    Source(Box<dyn Fn(usize) -> F + Send + Sync + 'a>, usize, Option<F>),
    Lookup {
        addresses: &'a super::lookup::Addresses,
        shift: usize,
        width: usize,
        len: usize,
        weights: Vec<F>,
    },
    Equality(Equality<F>),
    Integer(&'a [i32], Shape),
    Prefix(Vec<F>, usize),
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

    pub(crate) fn source(len: usize, read: impl Fn(usize) -> F + Send + Sync + 'a) -> Self {
        Self::Source(Box::new(read), len, None)
    }

    pub(crate) fn lookup(
        addresses: &'a super::lookup::Addresses,
        shift: usize,
        weights: Vec<F>,
    ) -> Self {
        Self::Lookup {
            addresses,
            shift,
            width: weights.len(),
            len: addresses.len(),
            weights,
        }
    }

    pub(crate) fn equality(point: &[F]) -> Self {
        Self::Equality(Equality::new(point))
    }

    pub(crate) fn integer(values: &'a [i32], shape: Shape) -> Self {
        Self::Integer(values, shape)
    }

    fn len(&self) -> usize {
        match self {
            Self::Field(v, width, copies) => v.len() * width * copies,
            Self::Bits(v, copies) => v.len() * copies,
            Self::Source(_, len, _) | Self::Lookup { len, .. } => *len,
            Self::Equality(weights) => weights.len(),
            Self::Integer(_, shape) => 1 << shape.vars(),
            Self::Prefix(_, len) => *len,
        }
    }

    pub(super) fn at(&self, i: usize) -> F {
        match self {
            Self::Field(v, 1, 1) => v[i],
            Self::Field(v, width, 1) => v[i >> width.trailing_zeros()],
            Self::Field(v, _, _) => v[i & (v.len() - 1)],
            Self::Bits(v, 1) => F::from_bool(v[i] != 0),
            Self::Bits(v, _) => F::from_bool(v[i & (v.len() - 1)] != 0),
            Self::Lookup {
                addresses,
                shift,
                width,
                len,
                weights,
            } => weights
                .chunks_exact(*width)
                .enumerate()
                .map(|(prefix, table)| {
                    let digit =
                        usize::from(addresses.digit(prefix * len + i, *shift, (*width - 1) as u32));
                    table[digit]
                })
                .sum(),
            Self::Equality(weights) => weights.at(i),
            Self::Integer(values, shape) => {
                let width = shape.width.next_power_of_two();
                let (row, col) = (i / width, i % width);
                if row < shape.rows && col < shape.width {
                    F::from_i32(values[row * shape.width + col])
                } else {
                    F::zero()
                }
            }
            Self::Prefix(values, _) => values.get(i).copied().unwrap_or_else(F::zero),
            Self::Source(read, _, None) => read(i),
            Self::Source(read, half, Some(r)) => {
                let low = read(i);
                low + *r * (read(i + half) - low)
            }
        }
    }

    fn bind(&mut self, r: F) {
        match self {
            Self::Field(_, _, copies) | Self::Bits(_, copies) if *copies > 1 => *copies /= 2,
            Self::Field(values, width, _) if values.len() == 1 => *width /= 2,
            Self::Field(values, _, _) => bind_halves(values.to_mut(), r),
            Self::Lookup {
                width,
                len,
                weights,
                ..
            } if weights.len() < 4 * *width => {
                // Keep two bound coordinates in the digit weights before allocating a column.
                let mut next = Vec::with_capacity(weights.len() * 2);
                for table in weights.chunks_exact(*width) {
                    next.extend(table.iter().map(|&w| w * (F::one() - r)));
                    next.extend(table.iter().map(|&w| w * r));
                }
                *weights = next;
                *len /= 2;
            }
            Self::Equality(weights) => weights.bind(r),
            Self::Integer(_, shape) => {
                let half = 1 << (shape.vars() - 1);
                let end = (shape.rows - 1) * shape.width.next_power_of_two() + shape.width;
                let bind = |i| {
                    let low = self.at(i);
                    r.mul_add(self.at(i + half) - low, low)
                };
                // The untouched suffix is zero under high-to-low binding.
                let prefix = half.min(end);
                let values = if prefix >= PARALLEL_ROUND_SIZE {
                    (0..prefix)
                        .into_par_iter()
                        .with_min_len(ROUND_CHUNK_SIZE)
                        .map(bind)
                        .collect()
                } else {
                    (0..prefix).map(bind).collect()
                };
                *self = Self::Prefix(values, half);
            }
            Self::Prefix(values, len) => {
                *len /= 2;
                if values.len() == *len * 2 {
                    bind_halves(values, r);
                } else {
                    let split = (*len).min(values.len());
                    let (lo, hi) = values.split_at_mut(split);
                    let bind = |(i, value): (usize, &mut F)| {
                        let other = hi.get(i).copied().unwrap_or_else(F::zero);
                        *value = r.mul_add(other - *value, *value);
                    };
                    if lo.len() >= PARALLEL_ROUND_SIZE {
                        lo.par_iter_mut()
                            .with_min_len(ROUND_CHUNK_SIZE)
                            .enumerate()
                            .for_each(bind);
                    } else {
                        lo.iter_mut().enumerate().for_each(bind);
                    }
                    values.truncate(*len);
                }
            }
            Self::Source(_, len, first @ None) => {
                *first = Some(r);
                *len /= 2;
            }
            column => {
                let half = column.len() / 2;
                let bind = |i| {
                    let a = column.at(i);
                    a + r * (column.at(i + half) - a)
                };
                *column = Self::owned(if half >= PARALLEL_ROUND_SIZE {
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
    mask: Option<[usize; 2]>,
}

impl<C, E> Fold<C, E> {
    pub(crate) fn new(columns: C, rounds: usize, degree: usize, expression: E) -> Self {
        Self {
            columns,
            rounds,
            degree,
            expression,
            mask: None,
        }
    }
}

impl<C, E> Fold<C, E> {
    fn samples<'a, F>(&mut self, previous: Option<F>) -> Vec<F>
    where
        F: JoltField + 'a,
        C: AsMut<[Column<'a, F>]>,
        E: Fn(&[F]) -> F + Sync,
    {
        if let Some(r) = previous {
            for column in self.columns.as_mut() {
                column.bind(r);
            }
        }
        let columns = &*self.columns.as_mut();
        let half = columns[0].len() / 2;
        let masks = if columns[0].len() >= 1 << self.rounds.saturating_sub(1) {
            self.mask.map(|indices| indices.map(|i| &columns[i]))
        } else {
            None
        };
        let zeros = || vec![F::zero(); self.degree + 1];
        let sum = |start: usize, end: usize| {
            let mut deltas = vec![F::zero(); columns.len()];
            let mut values = vec![F::zero(); columns.len()];
            let mut samples = zeros();
            for i in start..end {
                if masks.is_some_and(|masks| {
                    masks
                        .iter()
                        .all(|m| m.at(i).is_zero() && m.at(i + half).is_zero())
                }) {
                    continue;
                }
                for ((v, delta), col) in values.iter_mut().zip(&mut deltas).zip(columns) {
                    *v = col.at(i);
                    *delta = col.at(i + half) - *v;
                }
                samples[0] += (self.expression)(&values);
                for (v, &delta) in values.iter_mut().zip(&deltas) {
                    *v += delta;
                }
                for sample in &mut samples[2..] {
                    for (v, &delta) in values.iter_mut().zip(&deltas) {
                        *v += delta;
                    }
                    *sample += (self.expression)(&values);
                }
            }
            samples
        };
        if half >= PARALLEL_ROUND_SIZE {
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
        claim: F,
    ) -> std::result::Result<UnivariatePoly<F>, SumcheckError<F>> {
        let mut samples = self.samples(previous);
        samples[1] = claim - samples[0];
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
        _: usize,
        claim: F,
    ) -> core::result::Result<UnivariatePoly<F>, SumcheckError<F>> {
        let mut samples = self
            .0
            .par_iter_mut()
            .map(|(w, fold)| {
                let mut samples = fold.samples(previous);
                for value in &mut samples {
                    *value *= *w;
                }
                samples
            })
            .reduce_with(|mut a, b| {
                for (value, next) in a.iter_mut().zip(b) {
                    *value += next;
                }
                a
            })
            .expect("nonempty Sumcheck batch");
        samples[1] = claim - samples[0];
        Ok(UnivariatePoly::from_evals(&samples))
    }

    fn finish_rounds(&mut self, last: F) -> core::result::Result<(), SumcheckError<F>> {
        self.0
            .par_iter_mut()
            .try_for_each(|(_, fold)| fold.finish_rounds(last))
    }
}

#[cfg(all(test, feature = "akita"))]
#[test]
fn lookup_and_equality_columns_match_dense_folding() {
    use jolt_akita::AkitaField as F;
    use jolt_field::{Ring, Zero};

    let addresses: Vec<_> = (0..128).map(|i| (i * 17) % 251).collect();
    let digits = [3, 1, 8, 5].map(F::from_u64);
    let point = [7, 1, 11, 0, 4, 2, 3].map(F::from_u64);
    let packed = super::lookup::Addresses::Words(addresses.clone());
    let mut lookup = Column::lookup(&packed, 2, digits.to_vec());
    let mut equality = Column::equality(&point);
    let mut dense_digits: Vec<_> = addresses
        .iter()
        .map(|&i| digits[((i >> 2) & 3) as usize])
        .collect();
    let mut dense_eq = weights(&point);
    let shape = Shape::new(5, 11).unwrap();
    let integers: Vec<_> = (-35..20).collect();
    let mut integer = Column::integer(&integers, shape);
    let mut dense_integer = shape.padded(&integers, F::zero(), F::from_i32);
    for round in 0..=point.len() {
        for i in 0..dense_eq.len() {
            assert_eq!(lookup.at(i), dense_digits[i]);
            assert_eq!(equality.at(i), dense_eq[i]);
            assert_eq!(integer.at(i), dense_integer[i]);
        }
        if round < point.len() {
            let r = F::from_u64(if round == 3 { 0 } else { round as u64 + 13 });
            lookup.bind(r);
            equality.bind(r);
            integer.bind(r);
            bind_halves(&mut dense_integer, r);
            bind_halves(&mut dense_digits, r);
            bind_halves(&mut dense_eq, r);
        }
    }
}
