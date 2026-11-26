use jolt_field::{Field, One, Ring, Zero};
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

#[cfg(all(feature = "metal", target_os = "macos"))]
use super::metal::Round;
#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub(crate) enum Round {}

#[derive(Default, serde::Serialize, serde::Deserialize)]
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

    pub(crate) fn fields(
        &mut self,
        count: usize,
        witness: impl FnOnce() -> Vec<F>,
    ) -> Result<Vec<F>> {
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
        self.fold_accelerated(input, rounds, degree, columns, endpoints, formula, || None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fold_accelerated<'w>(
        &mut self,
        input: F,
        rounds: usize,
        degree: usize,
        columns: impl FnOnce() -> Vec<Column<'w>>,
        endpoints: &[usize],
        formula: impl Fn(&[F]) -> F + Sync,
        prepare: impl FnOnce() -> Option<Round>,
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
                let _acceleration = prepare();
                let mut fold = Fold {
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    acceleration: _acceleration,
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
    Mask(RowMask),
    Axis {
        len: usize,
        repeat: usize,
        values: Vec<F>,
    },
}

pub(crate) struct RowMask {
    rows: usize,
    cols: usize,
    stride: usize,
    remaining: usize,
    causal: bool,
    prefix: Vec<F>,
}

impl RowMask {
    fn get(&self, i: usize) -> F {
        let (row, col) = (i / self.stride, i % self.stride);
        if col >= self.cols {
            return F::zero();
        }
        if self.remaining == self.stride {
            return F::from_bool(row < self.rows && (!self.causal || col <= row));
        }
        // A valid original row lies in [col, rows) for the causal mask.
        let lower = if self.causal {
            col.saturating_sub(row).div_ceil(self.remaining)
        } else {
            0
        };
        let upper = self.rows.saturating_sub(row).div_ceil(self.remaining);
        if lower < upper {
            self.prefix[upper] - self.prefix[lower]
        } else {
            F::zero()
        }
    }

    fn bind_row(&mut self, r: F) {
        let mut prefix = Vec::with_capacity(2 * self.prefix.len() - 1);
        prefix.push(F::zero());
        for pair in self.prefix.windows(2) {
            let high = r * (pair[1] - pair[0]);
            prefix.extend([pair[1] - high, pair[1]]);
        }
        self.prefix = prefix;
        self.remaining /= 2;
    }
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

    pub(crate) fn row_mask(rows: usize, cols: usize, causal: bool) -> Self {
        let stride = rows.next_power_of_two();
        debug_assert!(cols <= stride);
        Self::Mask(RowMask {
            rows,
            cols,
            stride,
            remaining: stride,
            causal,
            prefix: vec![F::zero(), F::one()],
        })
    }

    // Each value spans 2^repeat consecutive cells; the axis can repeat in the outer dimensions.
    pub(crate) fn axis(len: usize, repeat: usize, values: Vec<F>) -> Self {
        debug_assert!(values.len().is_power_of_two() && values.len() << repeat <= len);
        Self::Axis {
            len,
            repeat,
            values,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::Public(v) => v.len(),
            Self::Source(n, _) | Self::Lookup { len: n, .. } => *n,
            Self::Equality(column) => column.len,
            Self::Mask(mask) => mask.remaining * mask.stride,
            Self::Axis { len, .. } => *len,
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
            Self::Mask(mask) => mask.get(i),
            Self::Axis { repeat, values, .. } => values[(i >> repeat) & (values.len() - 1)],
        }
    }

    fn bind(&mut self, r: F) {
        let half = self.len() / 2;
        match self {
            Self::Axis {
                len,
                repeat,
                values,
            } => {
                if half < values.len() << *repeat {
                    if values.len() == 1 {
                        *repeat -= 1;
                    } else {
                        let half = values.len() / 2;
                        let (lo, hi) = values.split_at_mut(half);
                        for (x, y) in lo.iter_mut().zip(hi) {
                            *x = r.mul_add(*y - *x, *x);
                        }
                        values.truncate(half);
                    }
                }
                *len = half;
                return;
            }
            Self::Mask(mask) => {
                mask.bind_row(r);
                if mask.remaining == 1 {
                    *self = Self::Dense((0..mask.stride).map(|i| mask.get(i)).collect());
                }
                return;
            }
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
    #[cfg(all(feature = "metal", target_os = "macos"))]
    acceleration: Option<Round>,
    columns: Vec<Column<'a>>,
    rounds: usize,
    degree: usize,
    formula: E,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl<E> Fold<'_, E> {
    fn restore_cpu(&mut self) {
        let Some(round) = self.acceleration.take() else {
            return;
        };
        let mut dense = round.snapshot().into_iter().peekable();
        for (index, column) in self.columns.iter_mut().enumerate() {
            if dense.peek().is_some_and(|(i, _)| *i == index) {
                *column = Column::Dense(dense.next().unwrap().1);
            } else {
                for &r in &round.challenges {
                    column.bind(r);
                }
            }
        }
    }
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
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Some(round) = &mut self.acceleration {
            if let Some(mut sums) = round.step(previous, self.degree) {
                sums[1] = sum - sums[0];
                return Ok(UnivariatePoly::from_evals(&sums));
            }
            self.restore_cpu();
        }
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
        #[cfg(all(feature = "metal", target_os = "macos"))]
        self.restore_cpu();
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
    use super::*;

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_lookup_matches_cpu_through_the_handoff() {
        use super::super::{metal, storage::Codes};

        let rounds = 15;
        let cells = 1 << rounds;
        let points = [
            (2..17).map(F::from_u64).collect::<Vec<_>>(),
            (31..46).map(F::from_u64).collect::<Vec<_>>(),
        ];
        let scales = [F::from_u64(3), F::from_u64(7)];
        let mut addresses = std::array::from_fn::<_, 2, _>(|_| Codes::new(cells, 4));
        for (read, addresses) in addresses.iter_mut().enumerate() {
            for i in 0..cells {
                addresses.set(i, ((i >> 3) ^ i ^ (read * 3)) & 15);
            }
        }
        let digits = [(2usize, 2usize), (0, 2)];
        let selectors = [
            weights(&[F::from_u64(47), F::from_u64(53)]),
            weights(&[F::from_u64(59), F::from_u64(61)]),
        ];
        let reads = addresses
            .iter()
            .zip(&points)
            .zip(scales)
            .map(|((code, point), scale)| (code, point.as_slice(), scale));
        let Some(round) = metal::lookup(rounds, reads, digits.into_iter(), &selectors) else {
            return;
        };
        let columns = || {
            let mut columns = Vec::new();
            for (codes, point) in addresses.iter().zip(&points) {
                for (&(shift, width), selector) in digits.iter().zip(&selectors) {
                    columns.push(Column::lookup(cells, selector, move |i| {
                        (codes.at(i) >> shift) & ((1 << width) - 1)
                    }));
                }
                columns.push(Column::split_equality(point));
            }
            columns
        };
        let formula = |values: &[F]| {
            inner(
                scales,
                values
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|part| part.iter().copied().product()),
            )
        };
        let mut cpu = Fold {
            acceleration: None,
            columns: columns(),
            rounds,
            degree: 3,
            formula,
        };
        let mut gpu = Fold {
            acceleration: Some(round),
            columns: columns(),
            rounds,
            degree: 3,
            formula,
        };
        let mut previous = None;
        for r in 0..rounds {
            let sum = F::from_u64((r * r + 17) as u64);
            let expected = cpu.prove_round(previous, r, sum).unwrap();
            let actual = gpu.prove_round(previous, r, sum).unwrap();
            assert_eq!(expected.coefficients(), actual.coefficients(), "round {r}");
            if r == 0 {
                assert!(
                    gpu.acceleration.is_some(),
                    "Metal did not submit the first round"
                );
            }
            previous = Some(F::from_u64((r + 71) as u64));
        }
        cpu.finish_rounds(previous.unwrap()).unwrap();
        gpu.finish_rounds(previous.unwrap()).unwrap();
        assert!(gpu.acceleration.is_none());
        for (left, right) in cpu.columns.iter().zip(&gpu.columns) {
            assert_eq!(left.get(0), right.get(0));
        }
    }

    #[test]
    fn compact_masks_fold_like_dense_masks() {
        for rows in [1usize, 5, 8] {
            let stride = rows.next_power_of_two();
            for (cols, causal) in [(1, false), (rows, false), (rows, true)] {
                let mut mask = Column::row_mask(rows, cols, causal);
                let mut dense = Column::Dense(
                    (0..stride * stride)
                        .map(|i| {
                            let (row, col) = (i / stride, i % stride);
                            F::from_bool(row < rows && col < cols && (!causal || col <= row))
                        })
                        .collect(),
                );
                for round in 0..=2 * stride.ilog2() {
                    assert_eq!(mask.len(), dense.len());
                    for i in 0..mask.len() {
                        assert_eq!(mask.get(i), dense.get(i));
                    }
                    if mask.len() > 1 {
                        let r = F::from_u64(u64::from(round) + 2);
                        mask.bind(r);
                        dense.bind(r);
                    }
                }
            }
        }
    }

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
