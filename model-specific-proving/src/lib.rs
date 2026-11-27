//! Fixed-point Transformer execution and Sumcheck proofs.
use bincode::Options;
use jolt_field::{CanonicalBytes, CanonicalEncoding, Field, One, Ring, Zero};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    single::{
        prove_single, verify_single, SingleFormat, SingleInstance, SingleReduction,
        SingleSumcheckProof,
    },
    ProveRounds, SumcheckError,
};
use jolt_transcript::{
    protocol_tag, prover_transcript, verifier_transcript, ChallengeSetSize, Encoding,
    FieldTranscript, Keccak, ProverState, ProverTranscript, Transcript, VerificationError,
    VerifierState, VerifierTranscript, MAX_CHALLENGE_BYTES, MAX_POW_BITS,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    collections::HashMap,
    fs::File,
    io::{BufWriter, Read, Seek, SeekFrom, Take, Write},
    ops::Range,
    sync::{mpsc, Arc, OnceLock},
};
use world_common::{shape::MIN_LOG_BITS, BitZParams, LinearClaim, Root, SecurityLevel};
use world_field::{gf128::smallest_generator, F128};

type Result<T, E = &'static str> = std::result::Result<T, E>;
type Writer = ProverState<Keccak>;
type Reader<'a> = VerifierState<'a, Keccak>;
type Digest = [u8; 32];
type Rounds = SingleSumcheckProof<F>;
type Evaluation = Claim<F>;

const MODULUS: u128 = (1 << 119) - 69;
const MASK: u128 = (1 << 119) - 1;
const WINDOW: usize = 4;
type Scalar = world_field::Fq<MODULUS>;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct F(u128);

#[inline]
fn canonical(x: u128) -> u128 {
    if x >= MODULUS {
        x - MODULUS
    } else {
        x
    }
}

#[inline]
fn reduce(x: u128) -> u128 {
    canonical((x & MASK) + 69 * (x >> 119))
}

jolt_field::impl_ring_ops!(impl[] F {
    add(a,b): Self(canonical(a.0 + b.0)),
    sub(a,b): Self(canonical(a.0 + MODULUS - b.0)),
    mul(a,b): a.product(b),
    neg(a): Self(canonical(MODULUS - a.0)),
    zero: Self(0),
    one: Self(1),
});
jolt_field::impl_serde_bytes!(impl[] F, 16);

impl std::fmt::Display for F {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(out)
    }
}

impl F {
    #[inline]
    fn product(self, b: Self) -> Self {
        let [a0, a1] = [(self.0 as u64) as u128, self.0 >> 64];
        let [b0, b1] = [(b.0 as u64) as u128, b.0 >> 64];
        let lo = a0 * b0;
        let mid = a0 * b1 + a1 * b0 + (lo >> 64);
        let hi = a1 * b1 + (mid >> 64);
        let low = ((mid & ((1 << 55) - 1)) << 64) | (lo as u64) as u128;
        let high = (hi << 9) | ((mid as u64 >> 55) as u128);
        Self(reduce(low + 69 * high))
    }

    #[inline]
    fn times(self, x: i128) -> Self {
        let Ok(abs) = u64::try_from(x.unsigned_abs()) else {
            return self * Self::from_i128(x);
        };
        let abs = u128::from(abs);
        let lo = u128::from(self.0 as u64) * abs;
        let hi = (self.0 >> 64) * abs + (lo >> 64);
        let value = Self(reduce(
            (lo as u64 as u128 | ((hi & ((1 << 55) - 1)) << 64)) + 69 * (hi >> 55),
        ));
        if x < 0 {
            -value
        } else {
            value
        }
    }

    fn scalar(self) -> Scalar {
        Scalar::from(self.0)
    }
}

impl Ring for F {
    fn from_u64(x: u64) -> Self {
        Self(x.into())
    }
    fn from_i64(x: i64) -> Self {
        let a = Self(u128::from(x.unsigned_abs()));
        if x < 0 {
            -a
        } else {
            a
        }
    }
    fn from_u128(x: u128) -> Self {
        Self(reduce(x))
    }
    fn from_i128(x: i128) -> Self {
        let a = Self(reduce(x.unsigned_abs()));
        if x < 0 {
            -a
        } else {
            a
        }
    }
}

impl Field for F {
    fn inverse(&self) -> Option<Self> {
        (self.0 != 0).then(|| Self(Scalar::META.pow(self.0, MODULUS - 2)))
    }
    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        loop {
            let mut bytes = [0u8; 16];
            rng.fill_bytes(&mut bytes);
            if let Some(x) = challenge(bytes) {
                return x;
            }
        }
    }
}

impl CanonicalBytes for F {
    const NUM_BYTES: usize = 16;
    fn to_bytes_le(&self, out: &mut [u8]) {
        out.copy_from_slice(&self.0.to_le_bytes());
    }
}

impl CanonicalEncoding for F {
    const MODULUS_BITS: u32 = 119;
    const CHALLENGE_SET_LIMBS: [u64; 4] = [MODULUS as u64, (MODULUS >> 64) as u64, 0, 0];
    fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
        Self::from_u128_checked(u128::from_le_bytes(bytes.try_into().ok()?))
    }
    fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
        bytes
            .iter()
            .rev()
            .fold(Self::zero(), |sum, &b| sum.times(256) + Self(u128::from(b)))
    }
    fn num_bits(&self) -> u32 {
        128 - self.0.leading_zeros()
    }
    fn to_u128_checked(&self) -> Option<u128> {
        Some(self.0)
    }
    fn from_u128_checked(x: u128) -> Option<Self> {
        (x < MODULUS).then_some(Self(x))
    }
    fn from_u128_reduced(x: u128) -> Self {
        Self(reduce(x))
    }
}

impl Encoding for F {
    fn encode(&self) -> impl AsRef<[u8]> {
        self.0.to_le_bytes()
    }
}

fn challenge(bytes: [u8; 16]) -> Option<F> {
    F::from_u128_checked(u128::from_le_bytes(bytes) & MASK)
}

macro_rules! field_channel {
    ($stream:ty) => {
        impl FieldTranscript<F> for $stream {
            const CHALLENGE_SET: ChallengeSetSize =
                ChallengeSetSize::from_limbs(F::CHALLENGE_SET_LIMBS);
            fn challenge_vector(
                &mut self,
                n: usize,
            ) -> std::result::Result<Vec<F>, VerificationError> {
                if n == 0 {
                    return Ok(Vec::new());
                }
                let bytes = n
                    .checked_add(1)
                    .and_then(|i| i.checked_mul(16))
                    .filter(|&i| i <= MAX_CHALLENGE_BYTES)
                    .ok_or(VerificationError)?;
                self.verifier_message_as(bytes, |raw| {
                    let out: Vec<_> = raw
                        .as_chunks::<16>()
                        .0
                        .iter()
                        .copied()
                        .filter_map(challenge)
                        .take(n)
                        .collect();
                    (out.len() == n).then_some(out).ok_or(VerificationError)
                })
            }
        }
    };
}
field_channel!(Writer);
field_channel!(Reader<'_>);

trait Mix: FieldTranscript<F> + Sized {
    fn work(&mut self, bits: u32) -> std::result::Result<(), VerificationError>;
    fn draw(&mut self, count: usize, loss: usize) -> Result<Vec<F>> {
        let bits = jolt_sumcheck::round_pow_bits::<F, Self>(loss)
            .map_err(|_| "invalid challenge count")?;
        self.work(bits).map_err(|_| "invalid proof of work")?;
        self.challenge_vector(count)
            .map_err(|_| "invalid field challenges")
    }
}

impl Mix for Reader<'_> {
    fn work(&mut self, bits: u32) -> std::result::Result<(), VerificationError> {
        self.check_grind(bits)
    }
}

impl Mix for Writer {
    fn work(&mut self, bits: u32) -> std::result::Result<(), VerificationError> {
        let threads = rayon::current_num_threads();
        if bits < 15 || bits > MAX_POW_BITS || threads == 1 {
            return self.grind(bits);
        }
        let oracle = self.verifier_message_oracle();
        let mask = (1u64 << bits) - 1;
        let passes = |x: &[u8]| u64::from_le_bytes(x.try_into().unwrap()) & mask == 0;
        let chunks = threads.next_power_of_two() as u64;
        let end = u64::MAX >> (MAX_POW_BITS - bits);
        let find = |first: u64| {
            let mut hash = [0; 8];
            (first..=first.saturating_add(2047).min(end)).find(|n| {
                oracle(&n.to_le_bytes(), &mut hash);
                passes(&hash)
            })
        };
        let nonce = (0..=end)
            .step_by(2048 * chunks as usize)
            .find_map(|i| {
                (0..chunks.min(1 + (end - i) / 2048))
                    .into_par_iter()
                    .find_map_first(|j| find(i + j * 2048))
            })
            .ok_or(VerificationError)?;
        self.prover_message_with(
            &nonce,
            |n| n.to_le_bytes(),
            |&n, out| {
                let mut n = n;
                while n >= 128 {
                    out.push(n as u8 | 128);
                    n >>= 7;
                }
                out.push(n as u8);
            },
        );
        self.verifier_message_as(8, |x| passes(x).then_some(()).ok_or(VerificationError))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Shape {
    rows: usize,
    cols: usize,
}

impl Shape {
    fn new(rows: usize, cols: usize) -> Result<Self> {
        let limit = 1..=1 << 20;
        if limit.contains(&rows) && limit.contains(&cols) {
            Ok(Self { rows, cols })
        } else {
            Err("invalid tensor dimensions")
        }
    }
    fn rows(self) -> usize {
        self.rows.next_power_of_two()
    }
    fn stride(self) -> usize {
        self.cols.next_power_of_two()
    }
    fn row_bits(self) -> usize {
        self.rows().ilog2() as usize
    }
    fn len(self) -> usize {
        self.rows() * self.stride()
    }
    fn vars(self) -> usize {
        self.len().ilog2() as usize
    }
    fn cells(self) -> usize {
        self.rows * self.cols
    }
    fn index(self, i: usize) -> Option<usize> {
        let (r, c) = (i / self.stride(), i % self.stride());
        (r < self.rows && c < self.cols).then_some(r * self.cols + c)
    }
    fn value(self, x: &[i32], i: usize) -> F {
        self.index(i).map_or(F::zero(), |j| F::from_i32(x[j]))
    }
    fn point(self, p: &[F]) -> (&[F], &[F]) {
        p.split_at(self.row_bits())
    }
    fn active(self, p: &[F]) -> F {
        let (r, c) = self.point(p);
        prefix(r, self.rows) * prefix(c, self.cols)
    }
    fn eval(self, x: &[i32], p: &[F]) -> F {
        self.eval_by(p, |r, c| i128::from(x[r * self.cols + c]))
    }
    fn eval_by(self, p: &[F], x: impl Fn(usize, usize) -> i128) -> F {
        let (r, c) = self.point(p);
        let col = basis(c);
        let mut value = F::zero();
        for (i, w) in basis(r).into_iter().take(self.rows).enumerate() {
            let sum: F = col[..self.cols]
                .iter()
                .enumerate()
                .map(|(j, &c)| c.times(x(i, j)))
                .sum();
            value += w * sum;
        }
        value
    }
}

fn basis(p: &[F]) -> Vec<F> {
    scaled_basis(p, F::one())
}

fn scaled_basis(p: &[F], scale: F) -> Vec<F> {
    let mut out = Vec::with_capacity(1 << p.len());
    out.push(scale);
    for &r in p.iter().rev() {
        let split = out.len();
        for i in 0..split {
            let high = out[i] * r;
            out[i] -= high;
            out.push(high);
        }
    }
    out
}

fn prefix(p: &[F], live: usize) -> F {
    basis(p).into_iter().take(live).sum()
}

fn same_point(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x * y + (F::one() - x) * (F::one() - y))
        .product()
}

fn inner(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}
fn affine_sum(a: &[F], b: &[i32]) -> F {
    a.iter().zip(b).map(|(&w, &x)| w.times(i128::from(x))).sum()
}

fn merge(a: &mut [F], b: &[F]) {
    a.iter_mut().zip(b).for_each(|(x, &y)| *x += y);
}

fn project_rows(out: &mut [F], matrix: &[i32], width: usize, weights: &[F]) {
    let accumulate = |out: &mut [F], m: &[i32], w: &[F]| {
        for (r, &c) in m.chunks_exact(width).zip(w) {
            out.iter_mut()
                .zip(r)
                .for_each(|(y, &x)| *y += c.times(i128::from(x)));
        }
    };
    let rows = weights.len().min(matrix.len() / width);
    let batch = (32768 / width).max(16);
    if rows <= batch {
        accumulate(out, matrix, weights);
        return;
    }
    let value = matrix[..rows * width]
        .par_chunks(batch * width)
        .zip(weights[..rows].par_chunks(batch))
        .fold(
            || vec![F::zero(); out.len()],
            |mut total, (matrix, weight)| {
                accumulate(&mut total, matrix, weight);
                total
            },
        )
        .reduce_with(|mut a, b| {
            merge(&mut a, &b);
            a
        })
        .unwrap();
    merge(out, &value);
}

fn magnitude(x: &[i32]) -> u128 {
    u128::from(x.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0))
}

fn dot_i32(a: &[i32], b: &[i32], narrow: bool) -> i128 {
    if !narrow {
        return a
            .iter()
            .zip(b)
            .map(|(&x, &y)| i128::from(x) * i128::from(y))
            .sum();
    }
    let a = a.as_chunks::<4>();
    let b = b.as_chunks::<4>();
    let mut sum = [0i64; 4];
    for (x, y) in a.0.iter().zip(b.0) {
        for i in 0..4 {
            sum[i] += i64::from(x[i]) * i64::from(y[i]);
        }
    }
    let tail: i64 =
        a.1.iter()
            .zip(b.1)
            .map(|(&x, &y)| i64::from(x) * i64::from(y))
            .sum();
    i128::from(sum.into_iter().sum::<i64>() + tail)
}

fn dot_two(a: &[i32], b: &[i32], weight: &[i32]) -> [i128; 2] {
    let [a, b, w] = [a, b, weight].map(|v| v.as_chunks::<4>());
    let mut sums = [[0i64; 4]; 2];
    for ((x, y), z) in a.0.iter().zip(b.0).zip(w.0) {
        for i in 0..4 {
            let z = i64::from(z[i]);
            sums[0][i] += i64::from(x[i]) * z;
            sums[1][i] += i64::from(y[i]) * z;
        }
    }
    std::array::from_fn(|i| {
        let tail: i64 = [a.1, b.1][i]
            .iter()
            .zip(w.1)
            .map(|(&x, &y)| i64::from(x) * i64::from(y))
            .sum();
        i128::from(sums[i].into_iter().sum::<i64>() + tail)
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claim<T> {
    pub point: Vec<T>,
    pub value: T,
}

fn at(point: &[F], value: F) -> Evaluation {
    Evaluation {
        point: point.to_vec(),
        value,
    }
}

type ReadCell<'a, T> = Box<dyn Fn(usize) -> T + Send + Sync + 'a>;
enum Column<'a> {
    Read(usize, ReadCell<'a, F>),
    Repeat(usize, usize, Vec<F>),
    Tensor(Vec<F>, Vec<F>),
    Digit(usize, usize, Vec<F>, ReadCell<'a, usize>),
}

fn bind(values: &mut Vec<F>, r: F, parallel: bool) {
    let n = values.len() / 2;
    let (a, b) = values.split_at_mut(n);
    let fold = |(x, y): (&mut F, &F)| *x += r * (*y - *x);
    if parallel && n >= 4096 {
        a.par_iter_mut().zip(b.par_iter()).for_each(fold);
    } else {
        a.iter_mut().zip(b.iter()).for_each(fold);
    }
    values.truncate(n);
}

impl<'a> Column<'a> {
    fn read(n: usize, f: impl Fn(usize) -> F + Send + Sync + 'a) -> Self {
        Self::Read(n, Box::new(f))
    }
    fn repeat(n: usize, stride: usize, values: Vec<F>) -> Self {
        Self::Repeat(n, stride.ilog2() as usize, values)
    }
    fn eq(p: &[F]) -> Self {
        let (a, b) = p.split_at(p.len() / 2);
        Self::Tensor(basis(a), basis(b))
    }
    fn digit(n: usize, table: &[F], f: impl Fn(usize) -> usize + Send + Sync + 'a) -> Self {
        Self::Digit(n, table.len(), table.to_vec(), Box::new(f))
    }
    fn len(&self) -> usize {
        match self {
            Self::Read(n, _) | Self::Repeat(n, ..) | Self::Digit(n, ..) => *n,
            Self::Tensor(a, b) => a.len() * b.len(),
        }
    }
    fn get(&self, i: usize) -> F {
        match self {
            Self::Read(_, f) => f(i),
            Self::Repeat(_, stride, v) => v[(i >> stride) & (v.len() - 1)],
            Self::Tensor(a, b) => a[i / b.len()] * b[i % b.len()],
            Self::Digit(n, radix, v, f) => {
                let mut sum = v[f(i)];
                if v.len() != *radix {
                    sum += v[radix + f(i + n)];
                }
                sum
            }
        }
    }
    fn fold(&mut self, r: F) {
        let len = self.len() / 2;
        match self {
            Self::Tensor(a, b) => bind(if a.len() > 1 { a } else { b }, r, false),
            Self::Repeat(n, stride, v) => {
                *n = len;
                if len < v.len() << *stride {
                    if v.len() == 1 {
                        *stride = len.ilog2() as usize;
                    } else {
                        bind(v, r, true);
                    }
                }
            }
            Self::Digit(n, radix, v, _) if v.len() == *radix => {
                for i in 0..*radix {
                    let x = r * v[i];
                    v[i] -= x;
                    v.push(x);
                }
                *n = len;
            }
            _ => {
                let eval = |i| {
                    let a = self.get(i);
                    a + r * (self.get(i + len) - a)
                };
                let data = if len >= 4096 {
                    (0..len)
                        .into_par_iter()
                        .with_min_len(1024)
                        .map(eval)
                        .collect()
                } else {
                    (0..len).map(eval).collect()
                };
                *self = Self::repeat(len, 1, data);
            }
        }
    }
}

struct Terms<'a, T> {
    columns: Vec<Column<'a>>,
    degree: usize,
    expression: T,
}
struct Products(Vec<(Vec<F>, Vec<F>)>, usize);

impl<T: Fn(&[F]) -> F + Sync> ProveRounds<F> for Terms<'_, T> {
    fn num_rounds(&self) -> usize {
        self.columns[0].len().ilog2() as usize
    }
    fn finish_rounds(&mut self, r: F) -> std::result::Result<(), SumcheckError<F>> {
        let n = self.columns.len();
        let size = self.columns[0].len() / 2;
        if n >= 4 && size >= 256 && size * n >= 8192 {
            self.columns.par_iter_mut().for_each(|x| x.fold(r));
        } else {
            self.columns.iter_mut().for_each(|x| x.fold(r));
        }
        Ok(())
    }
    fn prove_round(
        &mut self,
        last: Option<F>,
        _: usize,
        sum: F,
    ) -> std::result::Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = last {
            self.finish_rounds(r)?;
        }
        let pairs = self.columns[0].len() / 2;
        let n = self.columns.len();
        let evaluate = |from, until| {
            let mut values = vec![F::zero(); self.degree + 1];
            let mut x = vec![F::zero(); n];
            let mut delta = x.clone();
            for i in from..until {
                for (j, col) in self.columns.iter().enumerate() {
                    x[j] = col.get(i);
                    delta[j] = col.get(i + pairs) - x[j];
                }
                values[0] += (self.expression)(&x);
                merge(&mut x, &delta);
                for value in &mut values[2..] {
                    merge(&mut x, &delta);
                    *value += (self.expression)(&x);
                }
            }
            values
        };
        let split = pairs >= 128 && pairs * n >= 32768 && rayon::current_num_threads() > 1;
        let step = if split {
            (16384 / n).clamp(128, 1024)
        } else {
            1024
        };
        let mut values = if split || pairs >= 4096 {
            (0..pairs.div_ceil(step))
                .into_par_iter()
                .map(|i| evaluate(i * step, ((i + 1) * step).min(pairs)))
                .reduce_with(|mut a, b| {
                    merge(&mut a, &b);
                    a
                })
                .unwrap()
        } else {
            evaluate(0, pairs)
        };
        values[1] = sum - values[0];
        Ok(UnivariatePoly::from_evals(&values))
    }
}

impl ProveRounds<F> for Products {
    fn num_rounds(&self) -> usize {
        self.1
    }
    fn finish_rounds(&mut self, r: F) -> std::result::Result<(), SumcheckError<F>> {
        for (a, b) in &mut self.0 {
            bind(a, r, false);
            bind(b, r, false);
        }
        Ok(())
    }
    fn prove_round(
        &mut self,
        last: Option<F>,
        _: usize,
        sum: F,
    ) -> std::result::Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(r) = last {
            self.finish_rounds(r)?;
        }
        let (mut zero, mut two) = (F::zero(), F::zero());
        for (a, b) in &self.0 {
            let split = a.len() / 2;
            for i in 0..split {
                let (x, y) = (a[i + split], b[i + split]);
                zero += a[i] * b[i];
                two += (x + x - a[i]) * (y + y - b[i]);
            }
        }
        Ok(UnivariatePoly::from_evals(&[zero, sum - zero, two]))
    }
}

fn statement(sum: F, rounds: usize, degree: usize) -> SingleInstance<F> {
    SingleInstance {
        input_claim: sum,
        rounds,
        degree,
        format: SingleFormat::Compressed,
    }
}

#[derive(Default, Serialize, Deserialize)]
struct Tape(Vec<F>, Vec<Rounds>);
#[derive(Serialize, Deserialize)]
struct Fork(Tape, Vec<u8>);
#[derive(Default, Serialize, Deserialize)]
struct Messages(Tape, Vec<Fork>, Vec<WindowProof>);
#[derive(Serialize, Deserialize)]
struct WindowProof(Tape, Vec<Fork>, Vec<u8>, Vec<u8>);

type Cursor<'a> = (&'a [F], &'a [Rounds], &'a [Fork], &'a [WindowProof]);

fn take<'a, T>(stream: &mut &'a [T], n: usize) -> Result<&'a [T]> {
    let (head, tail) = stream.split_at_checked(n).ok_or("missing proof values")?;
    *stream = tail;
    Ok(head)
}
fn digest(bytes: &[u8]) -> Digest {
    bytes.try_into().unwrap()
}

enum Channel<'a, 'p> {
    Send(&'a mut Writer, &'a mut Messages, &'a Advice),
    Check(&'a mut Reader<'p>, Cursor<'p>),
}

impl Tape {
    fn cursor<'a>(&'a self, forks: &'a [Fork], windows: &'a [WindowProof]) -> Cursor<'a> {
        (&self.0, &self.1, forks, windows)
    }
}

impl<'a, 'p> Channel<'a, 'p> {
    fn append(&mut self, v: &impl Encoding) {
        match self {
            Self::Send(t, ..) => t.append(v),
            Self::Check(t, ..) => t.append(v),
        }
    }
    fn append_fields(&mut self, v: &[F]) {
        match self {
            Self::Send(t, ..) => t.append_values(v),
            Self::Check(t, ..) => t.append_values(v),
        }
    }
    fn key(&mut self) -> Digest {
        match self {
            Self::Send(t, ..) => t.verifier_message_as(32, digest),
            Self::Check(t, ..) => t.verifier_message_as(32, digest),
        }
    }
    fn draw(&mut self, n: usize) -> Result<Vec<F>> {
        match self {
            Self::Send(t, ..) => t.draw(n, n),
            Self::Check(t, ..) => t.draw(n, n),
        }
    }
    fn advice(&self) -> Option<&'a Advice> {
        if let Self::Send(_, _, x) = self {
            Some(*x)
        } else {
            None
        }
    }
    fn fields(&mut self, n: usize, builder: impl FnOnce() -> Vec<F>) -> Result<Vec<F>> {
        match self {
            Self::Send(t, m, _) => {
                let values = builder();
                t.append_values(&values);
                m.0 .0.extend_from_slice(&values);
                Ok(values)
            }
            Self::Check(t, (f, ..)) => {
                let values = take(f, n)?;
                t.append_values(values);
                Ok(values.to_vec())
            }
        }
    }
    fn values<const N: usize>(&mut self, builder: impl FnOnce() -> [F; N]) -> Result<[F; N]> {
        Ok(self.fields(N, || builder().into())?.try_into().unwrap())
    }
    fn sumcheck<W: ProveRounds<F>>(
        &mut self,
        s: SingleInstance<F>,
        n: usize,
        build: impl FnOnce() -> W,
        values: impl FnOnce(&W) -> Vec<F>,
    ) -> Result<(SingleReduction<F>, Vec<F>)> {
        let (result, aux) = match self {
            Self::Send(t, m, _) => {
                let mut w = build();
                let (rounds, result) =
                    prove_single(&mut w, &s, *t).map_err(|_| "prove Sumcheck")?;
                m.0 .1.push(rounds);
                (result, Some(values(&w)))
            }
            Self::Check(t, (_, rounds, ..)) => {
                let result =
                    verify_single(&s, &take(rounds, 1)?[0], *t).map_err(|_| "verify Sumcheck")?;
                (result, None)
            }
        };
        Ok((result, self.fields(n, || aux.unwrap())?))
    }
    fn fold<'b>(
        &mut self,
        s: SingleInstance<F>,
        wanted: &[usize],
        build: impl FnOnce() -> Vec<Column<'b>>,
        g: impl Fn(&[F]) -> F + Sync,
    ) -> Result<(SingleReduction<F>, Vec<F>)> {
        let degree = s.degree;
        self.sumcheck(
            s,
            wanted.len(),
            || Terms {
                columns: build(),
                degree,
                expression: g,
            },
            |w| wanted.iter().map(|&i| w.columns[i].get(0)).collect(),
        )
    }
    fn dot(
        &mut self,
        sum: F,
        weights: Vec<F>,
        build: impl FnOnce() -> Vec<F>,
    ) -> Result<Evaluation> {
        let n = weights.len();
        let (end, value) = self.fold(
            statement(sum, n.ilog2() as usize, 2),
            &[0],
            || {
                vec![
                    Column::repeat(n, 1, build()),
                    Column::repeat(n, 1, weights.clone()),
                ]
            },
            |v| v[0] * v[1],
        )?;
        if end.output_claim != value[0] * inner(&basis(&end.challenges), &weights) {
            return Err("invalid terminal dot product");
        }
        Ok(Evaluation {
            point: end.challenges,
            value: value[0],
        })
    }
    fn used(&self) -> Result<()> {
        match self {
            Self::Send(..) | Self::Check(_, ([], [], [], [])) => Ok(()),
            _ => Err("unexpected proof values"),
        }
    }
    fn fork<T: Send>(
        &mut self,
        label: &[u8],
        n: usize,
        work: impl Fn(usize, &mut Channel<'_, '_>) -> Result<T> + Sync,
    ) -> Result<Vec<T>> {
        if n == 1 {
            return Ok(vec![work(0, self)?]);
        }
        let seed = self.key();
        let tag = protocol_tag(label);
        match self {
            Self::Send(t, stream, advice) => {
                let a = *advice;
                let parts = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let mut local = prover_transcript::<Keccak>(&tag, &seed);
                        local.append(&[n as u64, i as u64]);
                        let mut m = Messages::default();
                        let value = work(i, &mut Channel::Send(&mut local, &mut m, a))?;
                        if !m.1.is_empty() || !m.2.is_empty() {
                            return Err("nested transcript branch");
                        }
                        Ok((
                            value,
                            local.verifier_message_as(32, digest),
                            Fork(m.0, local.narg_string().to_vec()),
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(parts
                    .into_iter()
                    .map(|(v, root, f)| {
                        t.append(&root);
                        stream.1.push(f);
                        v
                    })
                    .collect())
            }
            Self::Check(t, (_, _, parts, _)) => take(parts, n)?
                .iter()
                .enumerate()
                .map(|(i, Fork(tape, bytes))| {
                    let mut local = verifier_transcript::<Keccak>(&tag, &seed, bytes);
                    local.append(&[n as u64, i as u64]);
                    let mut io = Channel::Check(&mut local, tape.cursor(&[], &[]));
                    let v = work(i, &mut io)?;
                    io.used()?;
                    t.append(&local.verifier_message_as(32, digest));
                    local.check_eof().map_err(|_| "extra branch transcript")?;
                    Ok(v)
                })
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Width {
    Unsigned(u8),
    Signed(u8),
}

impl Width {
    fn bits(self) -> usize {
        match self {
            Self::Unsigned(n) | Self::Signed(n) => n.into(),
        }
    }
    fn signed(self) -> bool {
        matches!(self, Self::Signed(_))
    }
    fn offset(self) -> i128 {
        if self.signed() {
            1 << (self.bits() - 1)
        } else {
            0
        }
    }
    fn with_bits(self, n: usize) -> Self {
        if self.signed() {
            Self::Signed(n as u8)
        } else {
            Self::Unsigned(n as u8)
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct Slot(usize);
#[derive(Clone, Copy, Serialize, Deserialize)]
struct Packed(usize, u8, u32);
#[derive(Clone)]
struct Storage {
    shape: Shape,
    width: Width,
    data: Packed,
    digit: Option<(usize, usize)>,
}
struct Advice {
    pending: Option<File>,
    maps: Vec<Option<Arc<memmap2::Mmap>>>,
    shard: u32,
    offset: usize,
    slots: Vec<Storage>,
    layers: Vec<usize>,
}

impl Advice {
    fn new(proving: bool) -> Result<Self> {
        let pending = proving
            .then(tempfile::tempfile)
            .transpose()
            .map_err(|_| "create witnesses")?;
        Ok(Self {
            pending,
            maps: vec![None],
            shard: 0,
            offset: 0,
            slots: Vec::new(),
            layers: Vec::new(),
        })
    }
    fn seal(&mut self) -> Result<()> {
        let mut f = self.pending.take().ok_or("missing witness file")?;
        f.write_all(&[0; 16]).map_err(|_| "finish witnesses")?;
        // After this mapping the private file has no writable handle.
        let map = unsafe { memmap2::Mmap::map(&f) }.map_err(|_| "map witnesses")?;
        self.maps[self.shard as usize] = Some(Arc::new(map));
        Ok(())
    }
    fn cut(&mut self) {
        self.layers.push(self.slots.len());
    }
    fn pack(&mut self, s: Shape, bits: usize, f: impl Fn(usize) -> i128 + Sync) -> Result<Packed> {
        let width = u8::try_from(bits).map_err(|_| "invalid packed width")?;
        let at = self.offset;
        if let Some(file) = self.pending.as_mut().filter(|_| bits != 0) {
            let size = s.len();
            let mut raw = vec![0; (size * bits).div_ceil(8)];
            raw.par_chunks_mut(bits * 128)
                .enumerate()
                .try_for_each(|(block, out)| {
                    let limit = ((block + 1) * 1024).min(size);
                    let mut bytes = out.chunks_mut(8);
                    let (mut pending, mut buffered) = (0u128, 0);
                    for i in block * 1024..limit {
                        let x = f(i) as u128;
                        if x >= 1 << bits {
                            return Err("witness outside integer range");
                        }
                        pending |= x << buffered;
                        buffered += bits;
                        while buffered >= 64 {
                            bytes
                                .next()
                                .unwrap()
                                .copy_from_slice(&(pending as u64).to_le_bytes());
                            buffered -= 64;
                            pending = x >> (bits - buffered);
                        }
                    }
                    if let Some(tail) = bytes.next() {
                        tail.copy_from_slice(&pending.to_le_bytes()[..buffered.div_ceil(8)]);
                    }
                    Ok(())
                })?;
            file.write_all(&raw).map_err(|_| "write witnesses")?;
            self.offset += raw.len();
        }
        Ok(Packed(at, width, self.shard))
    }
    fn add(&mut self, entry: Storage) -> Slot {
        let id = Slot(self.slots.len());
        self.slots.push(entry);
        id
    }
    fn column(&mut self, s: Shape, limit: Width, f: impl Fn(usize) -> i128 + Sync) -> Result<Slot> {
        let mut width = limit;
        if self.pending.is_some() {
            let signed = limit.signed();
            let max = (0..s.cells())
                .into_par_iter()
                .with_min_len(8192)
                .map(|i| {
                    let x = f(i);
                    (if signed && x < 0 { !x } else { x }) as u128
                })
                .max()
                .unwrap_or(0);
            let bits = 128 - max.leading_zeros() as usize + usize::from(signed);
            if bits > limit.bits() {
                return Err("witness outside integer range");
            }
            width = limit.with_bits(bits);
        }
        let offset = width.offset();
        let data = self.pack(s, width.bits(), |i| s.index(i).map_or(0, &f) + offset)?;
        Ok(self.add(Storage {
            shape: s,
            width,
            data,
            digit: None,
        }))
    }
    fn digit(&mut self, s: Shape, data: Packed, shift: usize, bits: usize) -> Slot {
        self.add(Storage {
            shape: s,
            width: Width::Unsigned(1 << bits),
            data,
            digit: Some((shift, (1 << bits) - 1)),
        })
    }
    fn widths(&self) -> Vec<u8> {
        self.slots.iter().map(|x| x.width.bits() as u8).collect()
    }
    fn check_widths(&mut self, widths: &[u8]) -> Result<()> {
        if widths.len() != self.slots.len() {
            return Err("wrong number of witness widths");
        }
        for (slot, &n) in self.slots.iter_mut().zip(widths) {
            let n = usize::from(n);
            let allowed = slot.width.bits();
            if n > allowed
                || (n == 0 && slot.width.signed())
                || (slot.digit.is_some() && n != allowed)
            {
                return Err("invalid witness width");
            }
            slot.width = slot.width.with_bits(n);
        }
        Ok(())
    }
    fn get(&self, Packed(base, width, shard): Packed, i: usize) -> u128 {
        let width = usize::from(width);
        if width == 0 {
            return 0;
        }
        let bit = i * width;
        let bytes = &self.maps[shard as usize].as_ref().unwrap()[base + bit / 8..][..16];
        (u128::from_le_bytes(bytes.try_into().unwrap()) >> (bit % 8)) & ((1 << width) - 1)
    }
    fn read(&self, Slot(id): Slot, i: usize) -> u128 {
        let s = &self.slots[id];
        let v = self.get(s.data, i);
        match s.digit {
            Some((shift, mask)) => 1 << ((v as usize >> shift) & mask),
            None => v,
        }
    }
}

type CommitRoots = HashMap<usize, Vec<Digest>>;
type CommitJob = (usize, Advice);

struct EarlyCommit {
    first: Option<(File, usize)>,
    phase: Arc<std::sync::atomic::AtomicU8>,
    send: Option<mpsc::Sender<CommitJob>>,
    worker: Option<std::thread::JoinHandle<Result<CommitRoots>>>,
}

impl EarlyCommit {
    const TRACING: u8 = 0;
    const FINISHING: u8 = 1;
    const CANCELLED: u8 = 2;

    fn new(threads: usize) -> Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|_| "create initial commitment pool")?;
        let (send, jobs) = mpsc::channel::<CommitJob>();
        let phase = Arc::new(std::sync::atomic::AtomicU8::new(Self::TRACING));
        let state = phase.clone();
        let worker = std::thread::Builder::new()
            .spawn(move || {
                let mut roots = HashMap::new();
                for (first, advice) in jobs {
                    let mut window = Vec::new();
                    for ids in bins(&advice.slots, (0..advice.slots.len()).collect()) {
                        let group = Group::new(&advice, ids)?;
                        let root = match state.load(std::sync::atomic::Ordering::Relaxed) {
                            Self::TRACING => pool.install(|| group.commit(&advice))?.root,
                            Self::FINISHING => group.commit(&advice)?.root,
                            _ => return Ok(roots),
                        };
                        window.push(root);
                    }
                    roots.insert(first, window);
                }
                Ok(roots)
            })
            .map_err(|_| "start initial commitment worker")?;
        Ok(Self {
            first: None,
            phase,
            send: Some(send),
            worker: Some(worker),
        })
    }

    fn boundary(&mut self, advice: &mut Advice, end: usize, total: usize) -> Result<()> {
        if end % WINDOW != 0 && end != total {
            return Ok(());
        }
        if end == WINDOW {
            // Final normalization and Top-K belong to the first window.
            self.first = Some((advice.pending.take().unwrap(), advice.offset));
        } else {
            let first = ((end - 1) / WINDOW) * WINDOW;
            advice.seal()?;
            let view = Advice {
                maps: advice.maps.clone(),
                slots: advice.slots[advice.layers[first]..advice.layers[end]].to_vec(),
                ..Advice::new(false)?
            };
            self.send
                .as_ref()
                .unwrap()
                .send((first, view))
                .map_err(|_| "initial commitment worker stopped")?;
        }
        if end == total {
            let (file, offset) = self.first.take().unwrap();
            advice.pending = Some(file);
            advice.offset = offset;
            advice.shard = 0;
        } else {
            advice.pending = Some(tempfile::tempfile().map_err(|_| "create witness shard")?);
            advice.offset = 0;
            advice.shard =
                u32::try_from(advice.maps.len()).map_err(|_| "too many witness shards")?;
            advice.maps.push(None);
        }
        Ok(())
    }

    fn finish(mut self) -> Result<CommitRoots> {
        self.phase
            .store(Self::FINISHING, std::sync::atomic::Ordering::Relaxed);
        self.send.take();
        self.worker
            .take()
            .unwrap()
            .join()
            .map_err(|_| "initial commitment worker panicked")?
    }
}

impl Drop for EarlyCommit {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            self.phase
                .store(Self::CANCELLED, std::sync::atomic::Ordering::Relaxed);
            self.send.take();
            let _ = worker.join();
        }
    }
}

struct Request {
    id: Slot,
    evaluation: Evaluation,
    weights: Option<Vec<F>>,
}
type Requests = Vec<Request>;

fn ask(queries: &mut Requests, id: Slot, point: &[F], value: F) {
    queries.push(Request {
        id,
        evaluation: at(point, value),
        weights: None,
    });
}

const ROW_BITS: usize = 9;
const PCS_ROWS: usize = 1 << ROW_BITS;
const MAX_GROUP: usize = 32 << 23;
const PCS_DOMAIN: &[u8] = b"atlas/model/bitz/ranges/v2";

struct Layout {
    id: Slot,
    vars: usize,
    width: Width,
    start: usize,
    cols: usize,
}
struct Group {
    columns: Vec<Layout>,
    params: BitZParams<Scalar>,
    pcs: world_pcs::Pcs,
}
struct Window {
    layers: Range<usize>,
    groups: Vec<Group>,
    roots: Range<usize>,
}
struct Plan {
    windows: Vec<Window>,
    owners: Vec<(usize, usize, usize)>,
    roots: usize,
}

fn bins(slots: &[Storage], mut indices: Vec<usize>) -> Vec<Vec<usize>> {
    let size = |i: usize| {
        let slot = &slots[i];
        (1usize << slot.shape.vars().max(ROW_BITS)) * slot.width.bits()
    };
    indices.sort_unstable_by_key(|&i| (std::cmp::Reverse(size(i)), i));
    let mut bins = Vec::<(usize, Vec<usize>)>::new();
    for i in indices {
        let cost = size(i);
        if let Some((total, ids)) = bins.iter_mut().find(|(n, _)| *n + cost <= MAX_GROUP) {
            *total += cost;
            ids.push(i);
        } else {
            bins.push((cost, vec![i]));
        }
    }
    bins.into_iter().map(|(_, ids)| ids).collect()
}

impl Plan {
    fn new(a: &Advice) -> Result<Self> {
        let mut plan = Self {
            windows: Vec::new(),
            owners: vec![(0, 0, 0); a.slots.len()],
            roots: 0,
        };
        let layers = a.layers.len() - 1;
        for first in (0..layers).step_by(WINDOW).rev() {
            let last = (first + WINDOW).min(layers);
            let lo = if first == 0 { 0 } else { a.layers[first] };
            let mut indices: Vec<_> = (lo..a.layers[last]).collect();
            if first == 0 {
                indices.extend(a.layers[layers]..a.slots.len());
            }
            let bins = bins(&a.slots, indices);
            let mut groups = Vec::new();
            for ids in bins {
                for (j, &id) in ids.iter().enumerate() {
                    plan.owners[id] = (plan.windows.len(), groups.len(), j);
                }
                groups.push(Group::new(a, ids)?);
            }
            let roots = plan.roots..plan.roots + groups.len();
            plan.roots = roots.end;
            plan.windows.push(Window {
                layers: first..last,
                groups,
                roots,
            });
        }
        Ok(plan)
    }
    fn split(&self, window: usize, requests: &mut Requests) -> Requests {
        let (here, later) = std::mem::take(requests)
            .into_iter()
            .partition(|r| self.owners[r.id.0].0 == window);
        *requests = later;
        here
    }
    fn group(&self, w: usize, requests: Requests) -> Result<Vec<Requests>> {
        let mut groups: Vec<Requests> = self.windows[w].groups.iter().map(|_| Vec::new()).collect();
        for mut q in requests {
            let &(window, group, local) = self.owners.get(q.id.0).ok_or("invalid witness slot")?;
            if window != w {
                return Err("wrong witness window");
            }
            q.id = Slot(local);
            groups[group].push(q);
        }
        if groups.iter().any(Vec::is_empty) {
            return Err("unconstrained commitment");
        }
        Ok(groups)
    }
}

#[derive(Serialize, Deserialize)]
struct Proof {
    row: Rounds,
    #[serde(with = "Binary")]
    binary: world_transcript::Proof,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "world_transcript::Proof")]
struct Binary {
    #[serde(with = "serde_bytes")]
    narg_string: Vec<u8>,
    #[serde(with = "serde_bytes")]
    hints: Vec<u8>,
}

struct Commit {
    root: Digest,
    data: world_pcs::ProverData,
    packed: Vec<F128>,
    t: world_transcript::ProverState,
}
struct Opening<'a> {
    group: &'a Group,
    commitment: Commit,
    row: Rounds,
    claim: LinearClaim<Scalar>,
}
struct Query<'a> {
    layout: &'a Layout,
    scale: F,
    row: Vec<F>,
    cols: Vec<F>,
    bits: Vec<F>,
    weighted: bool,
}

fn bitplanes(
    out: &mut [&mut [F128]],
    from: usize,
    words: usize,
    read: &(impl Fn(usize) -> u128 + Sync),
) {
    if words >= 128 {
        let half = words / 2;
        let (mut a, mut b): (Vec<_>, Vec<_>) = out.iter_mut().map(|x| x.split_at_mut(half)).unzip();
        rayon::join(
            || bitplanes(&mut a, from, half, read),
            || bitplanes(&mut b, from + half, words - half, read),
        );
        return;
    }
    for i in 0..words {
        let mut packed = [0u128; 65];
        for r in 0..128 {
            let mut bits = read((from + i) * 128 + r);
            while bits != 0 {
                packed[bits.trailing_zeros() as usize] |= 1 << r;
                bits &= bits - 1;
            }
        }
        for (plane, v) in out.iter_mut().zip(packed) {
            plane[i] = F128::from(v);
        }
    }
}

impl Group {
    fn new(a: &Advice, slots: Vec<usize>) -> Result<Self> {
        let mut used = 0usize;
        let mut columns = Vec::with_capacity(slots.len());
        for id in slots {
            let s = &a.slots[id];
            let vars = s.shape.vars();
            if vars > 28 || s.width.bits() > 65 {
                return Err("invalid BitZ layout");
            }
            let cols = 1 << vars.saturating_sub(ROW_BITS);
            columns.push(Layout {
                id: Slot(id),
                vars,
                width: s.width,
                start: used,
                cols,
            });
            used += cols * s.width.bits();
        }
        let bits = (used.next_power_of_two().ilog2() as usize).max(MIN_LOG_BITS - ROW_BITS);
        let shape =
            world_common::Shape::new(ROW_BITS, bits).map_err(|_| "unsupported BitZ shape")?;
        static BASE: OnceLock<F128> = OnceLock::new();
        let params = BitZParams::new(shape, *BASE.get_or_init(smallest_generator))
            .map_err(|_| "invalid BitZ params")?;
        let pcs =
            world_pcs::Pcs::new(&shape, SecurityLevel::Bits128).map_err(|_| "invalid BitZ PCS")?;
        Ok(Self {
            columns,
            params,
            pcs,
        })
    }
    fn commit(&self, a: &Advice) -> Result<Commit> {
        let mut packed = vec![F128::from(0u128); self.pcs.packed_len()];
        let mut rest = packed.as_mut_slice();
        let jobs: Vec<_> = self
            .columns
            .iter()
            .map(|col| {
                let words = col.cols * (PCS_ROWS / 128);
                let (dst, tail) = std::mem::take(&mut rest).split_at_mut(words * col.width.bits());
                rest = tail;
                (col, words, dst)
            })
            .collect();
        jobs.into_par_iter().for_each(|(col, words, dst)| {
            let mut planes: Vec<_> = dst.chunks_exact_mut(words).collect();
            bitplanes(&mut planes, 0, words, &|i| {
                if i < 1 << col.vars {
                    a.read(col.id, i)
                } else {
                    0
                }
            });
        });
        let mut t = world_transcript::build_prover(PCS_DOMAIN, b"");
        let (root, data) = self
            .pcs
            .commit(&packed, &mut t)
            .map_err(|_| "commit witnesses")?;
        Ok(Commit {
            root: root.0,
            data,
            packed,
            t,
        })
    }
    fn mix(&self, requests: &Requests, t: &mut impl Mix) -> Result<(F, Vec<Query<'_>>)> {
        if requests.is_empty() {
            return Err("empty opening");
        }
        t.append(&(requests.len() as u64));
        for q in requests {
            let col = self.columns.get(q.id.0).ok_or("invalid opening slot")?;
            if q.evaluation.point.len() != col.vars {
                return Err("invalid opening point");
            }
            if let Some(w) = &q.weights {
                if w.len() != col.width.bits() || col.width.signed() {
                    return Err("invalid opening weights");
                }
                t.append_values(w);
            }
            t.append(&(q.id.0 as u64));
            t.append_values(&q.evaluation.point);
            t.append_values(&[q.evaluation.value]);
        }
        let mut sum = F::zero();
        let mix = t.draw(requests.len(), requests.len())?;
        let queries = requests
            .iter()
            .zip(mix)
            .map(|(q, scale)| {
                let col = &self.columns[q.id.0];
                let (c, r) = q
                    .evaluation
                    .point
                    .split_at(col.vars.saturating_sub(ROW_BITS));
                let mut row = vec![F::zero(); ROW_BITS - r.len()];
                row.extend_from_slice(r);
                sum += scale * (q.evaluation.value + F::from_i128(col.width.offset()));
                let bits = q.weights.clone().unwrap_or_else(|| {
                    (0..col.width.bits())
                        .map(|i| F::from_u128(1 << i))
                        .collect()
                });
                Query {
                    layout: col,
                    scale,
                    row,
                    cols: basis(c),
                    bits,
                    weighted: q.weights.is_some(),
                }
            })
            .collect();
        Ok((sum, queries))
    }
    fn linear(&self, terms: &[Query<'_>], end: SingleReduction<F>) -> Result<LinearClaim<Scalar>> {
        let mut col = vec![F::zero(); self.params.shape().columns()];
        for q in terms {
            let scale = q.scale * same_point(&q.row, &end.challenges);
            for (bit, &w) in q.bits.iter().enumerate() {
                let start = q.layout.start + bit * q.layout.cols;
                for (v, &a) in col[start..][..q.layout.cols].iter_mut().zip(&q.cols) {
                    *v += a * (scale * w);
                }
            }
        }
        LinearClaim::new(
            &self.params,
            basis(&end.challenges).into_iter().map(F::scalar).collect(),
            col.into_iter().map(F::scalar).collect(),
            end.output_claim.scalar(),
        )
        .map_err(|_| "BitZ linear claim")
    }
    fn prove<'a>(
        &'a self,
        a: &Advice,
        root: Digest,
        q: Requests,
        t: &mut Writer,
    ) -> Result<Opening<'a>> {
        let commitment = self.commit(a)?;
        if root != commitment.root {
            return Err("witness commitment changed");
        }
        let (sum, terms) = self.mix(&q, t)?;
        let pairs = terms
            .par_iter()
            .map(|x| x.project(a, terms.len()))
            .collect();
        let s = statement(sum, ROW_BITS, 2);
        let (row, end) = prove_single(&mut Products(pairs, ROW_BITS), &s, t)
            .map_err(|_| "prove BitZ row Sumcheck")?;
        Ok(Opening {
            group: self,
            commitment,
            row,
            claim: self.linear(&terms, end)?,
        })
    }
    fn verify(&self, root: Digest, q: Requests, p: Proof, t: &mut Reader<'_>) -> Result<()> {
        let (sum, terms) = self.mix(&q, t)?;
        let end = verify_single(&statement(sum, ROW_BITS, 2), &p.row, t)
            .map_err(|_| "verify BitZ row Sumcheck")?;
        let linear = self.linear(&terms, end)?;
        let mut reader = world_transcript::build_verifier(PCS_DOMAIN, b"", &p.binary);
        let data = self
            .pcs
            .receive_commitment(Root(root), &mut reader)
            .map_err(|_| "read BitZ commitment")?;
        world_verifier::BitZVerifier::new(self.params, 8)
            .verify(&linear, &self.pcs, &data, reader)
            .map_err(|_| "verify BitZ opening")?;
        Ok(())
    }
}

impl Query<'_> {
    fn project(&self, a: &Advice, batch: usize) -> (Vec<F>, Vec<F>) {
        let mut result = vec![F::zero(); PCS_ROWS];
        let n = (1 << self.layout.vars).min(PCS_ROWS);
        if self.weighted && self.bits.iter().all(|&w| w == F::one()) {
            result[..n].fill(self.scale);
        } else {
            let compute = |start, out: &mut [F]| {
                let mut weights = [F::zero(); 65];
                for (i, &w) in self.cols.iter().enumerate() {
                    let scale = w * self.scale;
                    if self.weighted {
                        weights
                            .iter_mut()
                            .zip(&self.bits)
                            .for_each(|(c, &w)| *c = scale * w);
                    }
                    for (r, value) in out.iter_mut().enumerate() {
                        let mut x = a.read(self.layout.id, i * PCS_ROWS + start + r);
                        if self.weighted {
                            while x != 0 {
                                *value += weights[x.trailing_zeros() as usize];
                                x &= x - 1;
                            }
                        } else {
                            *value += scale.times(x as i128);
                        }
                    }
                }
            };
            if batch < rayon::current_num_threads() && self.cols.len() * n >= 32768 {
                result[..n]
                    .par_chunks_mut(128)
                    .enumerate()
                    .for_each(|(i, x)| compute(i * 128, x));
            } else {
                compute(0, &mut result[..n]);
            }
        }
        (basis(&self.row), result)
    }
}

type Cache = HashMap<usize, world_prover::BitZProver<Scalar>>;

impl Opening<'_> {
    fn finish(self, cache: &mut Cache) -> Result<Proof> {
        let g = self.group;
        let Commit {
            data,
            packed,
            mut t,
            ..
        } = self.commitment;
        let n = g.params.shape().columns();
        cache
            .entry(n)
            .or_insert_with(|| world_prover::BitZProver::new(g.params, 8))
            .prove(&self.claim, &g.pcs, &data, packed, &mut t)
            .map_err(|_| "prove BitZ opening")?;
        Ok(Proof {
            row: self.row,
            binary: t.finish(),
        })
    }
}

fn can_overlap() -> bool {
    rayon::current_thread_index().is_none() && rayon::current_num_threads() > 1
}

impl Window {
    fn prove(
        &self,
        a: &Advice,
        roots: &[Digest],
        groups: Vec<Requests>,
        t: &mut Writer,
        mut write: impl FnMut(Proof) -> Result<()>,
    ) -> Result<()> {
        let jobs = self.groups.iter().zip(roots.iter().copied()).zip(groups);
        if self.groups.len() == 1 || !can_overlap() {
            let mut cache = Cache::new();
            for ((g, r), q) in jobs {
                write(g.prove(a, r, q, t)?.finish(&mut cache)?)?;
            }
            return Ok(());
        }
        std::thread::scope(|s| {
            let (send, open) = mpsc::sync_channel::<Opening<'_>>(0);
            let (completed, read) = mpsc::sync_channel(1);
            s.spawn(move || {
                let mut cache = Cache::new();
                for p in open {
                    if completed.send(p.finish(&mut cache)).is_err() {
                        break;
                    }
                }
            });
            let next = || read.recv().map_err(|_| "BitZ worker stopped")?;
            for (i, ((g, r), q)) in jobs.enumerate() {
                let pending = g.prove(a, r, q, t)?;
                if i > 0 {
                    write(next()?)?;
                }
                send.send(pending).map_err(|_| "BitZ worker stopped")?;
            }
            drop(send);
            write(next()?)
        })
    }
}

#[derive(Clone, Copy)]
enum Table<'a> {
    Activation(&'a [i32]),
    Soft(SoftmaxTables<'a>, u128),
}

impl Table<'_> {
    fn len(self) -> usize {
        match self {
            Self::Activation(a) => a.len(),
            Self::Soft(t, _) => t.sums.len(),
        }
    }
    fn same(self, b: Self) -> bool {
        match (self, b) {
            (Self::Activation(a), Self::Activation(b)) => std::ptr::eq(a, b),
            (Self::Soft(a, r), Self::Soft(b, s)) => {
                r == s
                    && a.limit == b.limit
                    && std::ptr::eq(a.probabilities, b.probabilities)
                    && std::ptr::eq(a.sums, b.sums)
            }
            _ => false,
        }
    }
    fn get(self, i: usize) -> [F; 2] {
        match self {
            Self::Activation(a) => [F::from_i32(a[i]), F::zero()],
            Self::Soft(t, base) => {
                let previous = i.checked_sub(1).map_or(t.limit + 1, |j| t.sums[j]);
                let second = i128::from(t.sums[i]) - i128::from(t.sums[0])
                    + (i128::from(previous) - i128::from(t.limit) - 1) * base as i128;
                [
                    F::from_i64(i64::from(t.probabilities[i]) - i64::from(t.probabilities[0])),
                    F::from_i128(second),
                ]
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Digit {
    id: Slot,
    shift: usize,
    bits: usize,
}
#[derive(Clone, Serialize, Deserialize)]
struct Lookup {
    packed: Packed,
    digits: Vec<Digit>,
    vars: usize,
}

impl Lookup {
    fn record(
        a: &mut Advice,
        s: Shape,
        size: usize,
        code: impl Fn(usize) -> u32 + Sync,
    ) -> Result<Self> {
        if size < 2 || !size.is_power_of_two() {
            return Err("invalid lookup size");
        }
        let bits = size.ilog2() as usize;
        let chunk = if (17..=24).contains(&bits) { 3 } else { 4 };
        let shape = Shape::new(s.rows(), s.stride())?;
        let packed = a.pack(shape, bits, |i| i128::from(code(i)))?;
        let digits = (0..bits)
            .step_by(chunk)
            .map(|shift| {
                let bits = (bits - shift).min(chunk);
                Digit {
                    id: a.digit(shape, packed, shift, bits),
                    shift,
                    bits,
                }
            })
            .collect();
        Ok(Self {
            packed,
            digits,
            vars: shape.vars(),
        })
    }
    fn code(&self, a: &Advice, i: usize) -> usize {
        a.get(self.packed, i) as usize
    }
}

struct TableRead<'a> {
    table: Table<'a>,
    key: Lookup,
    point: Vec<F>,
    values: [F; 3],
}
type Reads<'a> = Vec<TableRead<'a>>;

fn finish_reads(io: &mut Channel<'_, '_>, reads: Reads<'_>, requests: &mut Requests) -> Result<()> {
    let mut batches: Vec<Reads<'_>> = Vec::new();
    for r in reads {
        let slot = batches
            .iter_mut()
            .find(|b| b[0].key.vars == r.key.vars && b[0].table.same(r.table));
        if let Some(b) = slot {
            b.push(r);
        } else {
            batches.push(vec![r]);
        }
    }
    for b in batches {
        lookup_batch(io, &b, requests)?;
    }
    Ok(())
}

fn lookup_batch(
    io: &mut Channel<'_, '_>,
    batch: &[TableRead<'_>],
    requests: &mut Requests,
) -> Result<()> {
    let first = &batch[0];
    let len = first.table.len();
    let bits = len.ilog2() as usize;
    let n = 1 << first.key.vars;
    let step = ((64 << 20) / (n * first.key.digits.len() * 4)).clamp(1, 16);
    let draw = io.draw(batch.len() + 3)?;
    let (mix, coeff) = draw.split_at(3);
    let value = |i| {
        let [a, b] = first.table.get(i);
        mix[0] * F::from_u64(i as u64) + mix[1] * a + mix[2] * b
    };
    let start = batch
        .iter()
        .zip(coeff)
        .map(|(r, &c)| c * inner(&r.values, mix))
        .sum();
    let advice = io.advice();
    let (end, claims) = io.fold(
        statement(start, bits, 2),
        &[0],
        || {
            let span = batch.len().div_ceil(rayon::current_num_threads().min(4));
            let list = batch
                .par_chunks(span)
                .zip(coeff.par_chunks(span))
                .map(|(r, w)| {
                    let mut hist = vec![F::zero(); len];
                    for (read, &c) in r.iter().zip(w) {
                        for (i, m) in scaled_basis(&read.point, c).into_iter().enumerate() {
                            hist[read.key.code(advice.unwrap(), i)] += m;
                        }
                    }
                    hist
                })
                .reduce_with(|mut a, b| {
                    merge(&mut a, &b);
                    a
                })
                .unwrap();
            vec![Column::repeat(len, 1, list), Column::read(len, value)]
        },
        |v| v[0] * v[1],
    )?;
    let histogram = claims[0];
    let weights = basis(&end.challenges);
    let public: F = weights.iter().enumerate().map(|(i, &w)| w * value(i)).sum();
    if end.output_claim != histogram * public {
        return Err("invalid lookup table evaluation");
    }
    let parts = if batch.len() <= step {
        vec![histogram]
    } else {
        let parts = io.fields(batch.len().div_ceil(step), || {
            batch
                .par_chunks(step)
                .zip(coeff.par_chunks(step))
                .map(|(reads, mix)| {
                    reads
                        .iter()
                        .zip(mix)
                        .map(|(r, &m)| {
                            basis(&r.point)
                                .into_iter()
                                .enumerate()
                                .map(|(i, w)| w * weights[r.key.code(advice.unwrap(), i)])
                                .sum::<F>()
                                * m
                        })
                        .sum()
                })
                .collect()
        })?;
        if parts.iter().copied().sum::<F>() != histogram {
            return Err("invalid lookup batch total");
        }
        parts
    };
    let planes: Vec<_> = first
        .key
        .digits
        .iter()
        .map(|d| basis(&end.challenges[bits - d.shift - d.bits..bits - d.shift]))
        .collect();
    let jobs: Vec<_> = batch
        .chunks(step)
        .zip(coeff.chunks(step))
        .zip(parts)
        .collect();
    for pair in jobs.chunks(2) {
        for group in io.fork(b"atlas/lookup-links/v1", pair.len(), |i, io| {
            let ((reads, mix), sum) = pair[i];
            lookup_digits(io, reads, mix, &planes, sum)
        })? {
            requests.extend(group);
        }
    }
    Ok(())
}

fn lookup_digits(
    io: &mut Channel<'_, '_>,
    reads: &[TableRead<'_>],
    mix: &[F],
    planes: &[Vec<F>],
    sum: F,
) -> Result<Requests> {
    let vars = reads[0].key.vars;
    let n = 1 << vars;
    let degree = planes.len() + 1;
    let indices: Vec<_> = (0..reads.len() * degree)
        .filter(|i| i % degree != degree - 1)
        .collect();
    let advice = io.advice();
    let evaluate = |v: &[F]| {
        v.chunks_exact(degree)
            .zip(mix)
            .map(|(v, &m)| m * v.iter().copied().product::<F>())
            .sum()
    };
    let (end, outputs) = io.fold(
        statement(sum, vars, degree),
        &indices,
        || {
            let mut columns = Vec::new();
            for read in reads {
                let codes: Arc<[u32]> = (0..n)
                    .map(|i| read.key.code(advice.unwrap(), i) as u32)
                    .collect::<Vec<_>>()
                    .into();
                for (d, table) in read.key.digits.iter().zip(planes) {
                    let codes = Arc::clone(&codes);
                    let (shift, mask) = (d.shift, (1 << d.bits) - 1);
                    columns.push(Column::digit(n, table, move |i| {
                        (codes[i] as usize >> shift) & mask
                    }));
                }
                columns.push(Column::eq(&read.point));
            }
            columns
        },
        evaluate,
    )?;
    let mut terminal = Vec::new();
    let mut requests = Vec::new();
    for (r, q) in reads.iter().zip(outputs.chunks_exact(planes.len())) {
        terminal.extend_from_slice(q);
        terminal.push(same_point(&r.point, &end.challenges));
        for ((d, w), &v) in r.key.digits.iter().zip(planes).zip(q) {
            let id = d.id;
            requests.push(Request {
                id,
                evaluation: at(&end.challenges, v),
                weights: Some(w.clone()),
            });
            requests.push(Request {
                id,
                evaluation: at(&end.challenges, F::one()),
                weights: Some(vec![F::one(); 1 << d.bits]),
            });
        }
    }
    if end.output_claim != evaluate(&terminal) {
        return Err("invalid terminal lookup linkage");
    }
    Ok(requests)
}

#[derive(Clone, Copy)]
pub struct Projection<'a> {
    pub weights: &'a [i32],
    pub bias: &'a [i32],
}

fn projection_shape(s: Shape, p: Projection<'_>) -> Result<Shape> {
    let output = Shape::new(s.rows, p.bias.len())?;
    if p.weights.len() != s.cols * output.cols {
        return Err("invalid projection weights");
    }
    Ok(output)
}

fn project(x: &[i32], s: Shape, bits: u32, p: Projection<'_>) -> Result<(Vec<i32>, Vec<u32>)> {
    let shape = projection_shape(s, p)?;
    if x.len() != s.cells() {
        return Err("invalid projection input");
    }
    let n = shape.cols;
    let rows = p.weights.chunks_exact(s.cols);
    let bound: Vec<u128> = rows
        .clone()
        .map(|w| w.iter().map(|x| u128::from(x.unsigned_abs())).sum())
        .collect();
    let mut result = vec![0; shape.cells()];
    let mut rem = vec![0; result.len()];
    result
        .par_chunks_mut(2 * n)
        .zip(rem.par_chunks_mut(2 * n))
        .zip(x.par_chunks(2 * s.cols))
        .try_for_each(|((value, rem), x)| -> Result<()> {
            let (a, b) = x.split_at(s.cols);
            let (ma, mb) = (magnitude(a), magnitude(b));
            let limit = i64::MAX as u128;
            let mut write = |i, y: i128, bias: i32| -> Result<()> {
                let raw = y + (i128::from(bias) << bits);
                value[i] = i32::try_from(raw >> bits).map_err(|_| "projection exceeds i32")?;
                rem[i] = (raw & ((1 << bits) - 1)) as u32;
                Ok(())
            };
            for (i, ((w, &bnd), &bias)) in rows.clone().zip(&bound).zip(p.bias).enumerate() {
                let narrow = ma * bnd <= limit;
                let second = mb * bnd <= limit;
                if !b.is_empty() && narrow && second {
                    let [x, y] = dot_two(a, b, w);
                    write(i, x, bias)?;
                    write(n + i, y, bias)?;
                } else {
                    write(i, dot_i32(a, w, narrow), bias)?;
                    if !b.is_empty() {
                        write(n + i, dot_i32(b, w, second), bias)?;
                    }
                }
            }
            Ok(())
        })?;
    Ok((result, rem))
}

#[derive(Serialize, Deserialize)]
struct Rounded {
    shape: Shape,
    #[serde(with = "trace_array")]
    output: Option<Vec<i32>>,
    range: Option<Slot>,
    rem: Slot,
    bits: u32,
}

impl Rounded {
    fn rounded(
        shape: Shape,
        bits: u32,
        f: impl Fn(usize, usize) -> i128 + Sync,
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        let values = (0..shape.cells())
            .into_par_iter()
            .with_min_len(512)
            .map(|i| {
                let raw = f(i / shape.cols, i % shape.cols);
                let integer = i32::try_from(raw >> bits).map_err(|_| "fixed-point overflow")?;
                Ok((integer, (raw & ((1 << bits) - 1)) as u32))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(values.into_iter().unzip())
    }
    fn record(
        a: &mut Advice,
        shape: Shape,
        bits: u32,
        check: bool,
        values: Option<(Vec<i32>, Vec<u32>)>,
    ) -> Result<Self> {
        let (output, rem) = values.unzip();
        let range = check
            .then(|| {
                a.column(shape, Width::Signed(32), |i| {
                    i128::from(output.as_ref().unwrap()[i])
                })
            })
            .transpose()?;
        let rem = a.column(shape, Width::Unsigned(bits as u8), |i| {
            i128::from(rem.as_ref().unwrap()[i])
        })?;
        Ok(Self {
            shape,
            output,
            range,
            rem,
            bits,
        })
    }
    fn apply(
        a: &mut Advice,
        s: Shape,
        bits: u32,
        p: Projection<'_>,
        x: Option<&[i32]>,
        check: bool,
    ) -> Result<Self> {
        let shape = projection_shape(s, p)?;
        let witness = x.map(|x| project(x, s, bits, p)).transpose()?;
        Self::record(a, shape, bits, check, witness)
    }
    fn values(&self) -> Option<&[i32]> {
        self.output.as_deref()
    }
    fn remainder(
        &self,
        io: &mut Channel<'_, '_>,
        r: &mut Requests,
        claim: &Evaluation,
    ) -> Result<F> {
        let advice = io.advice();
        let [v] = io.values(|| {
            [self.shape.eval_by(&claim.point, |i, j| {
                advice.unwrap().read(self.rem, i * self.shape.stride() + j) as i128
            })]
        })?;
        if let Some(slot) = self.range {
            ask(r, slot, &claim.point, claim.value);
        }
        ask(r, self.rem, &claim.point, v);
        Ok(v)
    }
    fn raw(&self, io: &mut Channel<'_, '_>, r: &mut Requests, q: &Evaluation) -> Result<F> {
        Ok(q.value * F::from_u64(1 << self.bits) + self.remainder(io, r, q)?)
    }
}

type ProjectionClaim<'a> = (&'a Rounded, Projection<'a>, Evaluation);

fn projections(
    io: &mut Channel<'_, '_>,
    requests: &mut Requests,
    s: Shape,
    bits: u32,
    claims: &[ProjectionClaim<'_>],
    input: Option<&[i32]>,
) -> Result<Vec<Evaluation>> {
    if claims.is_empty() {
        return Err("empty projection batch");
    }
    let rem = claims
        .iter()
        .map(|(w, _, q)| w.remainder(io, requests, q))
        .collect::<Result<Vec<_>>>()?;
    io.append(&(claims.len() as u64));
    for ((_, p, q), &r) in claims.iter().zip(&rem) {
        if q.point.len() != projection_shape(s, *p)?.vars() {
            return Err("wrong projection point");
        }
        io.append_fields(&q.point);
        io.append_fields(&[q.value, r]);
    }
    let mix = io.draw(claims.len())?;
    let scale = F::from_u64(1 << bits);
    let width = s.stride();
    let mut groups = Vec::<(&[F], Vec<F>)>::new();
    let mut saved = Vec::<(&[i32], &[F], Vec<F>)>::new();
    let mut sum = F::zero();
    for (i, (((_, p, q), r), m)) in claims.iter().zip(rem).zip(mix).enumerate() {
        let (rows, cols) = q.point.split_at(s.row_bits());
        let c = basis(cols);
        let bias = affine_sum(&c, p.bias);
        sum += m * (scale * (q.value - prefix(rows, s.rows) * bias) + r);
        let j = groups
            .iter()
            .position(|(r, _)| *r == rows)
            .unwrap_or_else(|| {
                groups.push((rows, vec![F::zero(); width]));
                groups.len() - 1
            });
        let dest = &mut groups[j].1;
        let same = |w, c: &[F]| std::ptr::eq(p.weights, w) && cols == c;
        if let Some((_, _, v)) = saved.iter().find(|(w, c, _)| same(*w, c)) {
            for (x, &v) in dest.iter_mut().zip(v) {
                *x += m * v;
            }
        } else if claims[i + 1..]
            .iter()
            .any(|(_, p, q)| same(p.weights, &q.point[s.row_bits()..]))
        {
            let mut v = vec![F::zero(); width];
            project_rows(&mut v, p.weights, s.cols, &c);
            for (x, &v) in dest.iter_mut().zip(&v) {
                *x += m * v;
            }
            saved.push((p.weights, cols, v));
        } else {
            let c: Vec<_> = c.into_iter().map(|v| m * v).collect();
            project_rows(dest, p.weights, s.cols, &c);
        }
    }
    let rounds = width.ilog2() as usize;
    let (end, values) = io.sumcheck(
        statement(sum, rounds, 2),
        groups.len(),
        || {
            let pairs = groups
                .iter()
                .map(|(r, c)| {
                    let mut x = vec![F::zero(); width];
                    project_rows(&mut x, input.unwrap(), s.cols, &basis(r));
                    (x, c.clone())
                })
                .collect();
            Products(pairs, rounds)
        },
        |w| w.0.iter().map(|(a, _)| a[0]).collect(),
    )?;
    let weight = basis(&end.challenges);
    let check: F = groups
        .iter()
        .zip(&values)
        .map(|((_, c), &v)| v * inner(c, &weight))
        .sum();
    if check != end.output_claim {
        return Err("invalid terminal projection");
    }
    Ok(groups
        .into_iter()
        .zip(values)
        .map(|((r, _), value)| Evaluation {
            point: [r, &end.challenges].concat(),
            value,
        })
        .collect())
}

#[derive(Copy, Clone)]
pub struct Rotary<'a> {
    pub cos: &'a [i32],
    pub sin: &'a [i32],
}

impl Rotary<'_> {
    fn record(self, a: &mut Advice, s: Shape, bits: u32, x: Option<&[i32]>) -> Result<Rounded> {
        let half = s.cols / 2;
        if half == 0
            || !s.cols.is_power_of_two()
            || self.cos.len() != s.rows * half
            || self.sin.len() != self.cos.len()
        {
            return Err("invalid RoPE table");
        }
        let witness = x
            .map(|x| {
                Rounded::rounded(s, bits, |r, c| {
                    let i = r * half + c % half;
                    let sign = if c < half { -1 } else { 1 };
                    let a = i128::from(self.cos[i]) * i128::from(x[r * s.cols + c]);
                    let b = i128::from(self.sin[i]) * i128::from(x[r * s.cols + (c ^ half)]);
                    a + sign * b
                })
            })
            .transpose()?;
        Rounded::record(a, s, bits, true, witness)
    }
    fn reverse(
        self,
        io: &mut Channel<'_, '_>,
        r: &mut Requests,
        w: &Rounded,
        input: Option<&[i32]>,
        claims: &[Evaluation],
    ) -> Result<Evaluation> {
        let s = w.shape;
        let half = s.cols / 2;
        let mut weights = vec![F::zero(); s.len()];
        let mut sum = F::zero();
        for (q, m) in claims.iter().zip(io.draw(claims.len())?) {
            sum += m * w.raw(io, r, q)?;
            merge(&mut weights, &scaled_basis(&q.point, m));
        }
        for (r, row) in weights.chunks_exact_mut(s.cols).take(s.rows).enumerate() {
            let (left, right) = row.split_at_mut(half);
            for (i, (x, y)) in left.iter_mut().zip(right).enumerate() {
                let c = F::from_i32(self.cos[r * half + i]);
                let t = F::from_i32(self.sin[r * half + i]);
                (*x, *y) = (*x * c + *y * t, *y * c - *x * t);
            }
        }
        io.dot(sum, weights, || {
            (0..s.len()).map(|i| s.value(input.unwrap(), i)).collect()
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NormalizationKind {
    Layer,
    Rms,
}
#[derive(Clone, Copy)]
pub struct NormalizationParameters<'a> {
    pub kind: NormalizationKind,
    pub gain: &'a [i32],
    pub bias: &'a [i32],
    pub epsilon_variance_units: i64,
}

#[derive(Serialize, Deserialize)]
struct Normalized {
    shape: Shape,
    bits: u32,
    #[serde(with = "trace_array")]
    values: Option<Vec<i32>>,
    #[serde(with = "trace_array")]
    quotients: Option<Vec<i64>>,
    #[serde(with = "trace_array")]
    rows: Option<Vec<[i128; 9]>>,
    slots: Vec<Slot>,
}

impl NormalizationParameters<'_> {
    fn centered(self) -> bool {
        self.kind == NormalizationKind::Layer
    }
    fn narrowed(self, bits: u32) -> Option<u128> {
        let root = u128::try_from(self.epsilon_variance_units).ok()?.isqrt();
        (root != 0).then(|| {
            ceil_sqrt(self.gain.len() as u128) * (((1u128 << bits) * (root + 1)).div_ceil(root) + 1)
        })
    }
}

fn ceil_sqrt(n: u128) -> u128 {
    let r = n.isqrt();
    r + u128::from(r * r < n)
}

impl Normalized {
    fn record(
        a: &mut Advice,
        s: Shape,
        bits: u32,
        p: NormalizationParameters<'_>,
        x: Option<&[i32]>,
    ) -> Result<Self> {
        if p.gain.len() != s.cols
            || p.bias.len() != s.cols
            || p.epsilon_variance_units <= 0
            || x.is_some_and(|x| x.len() != s.cells())
        {
            return Err("invalid normalization parameters");
        }
        let mut w = Self {
            shape: s,
            bits,
            values: None,
            quotients: None,
            rows: None,
            slots: Vec::new(),
        };
        if let Some(x) = x {
            let mut values = Vec::with_capacity(x.len());
            let mut qs = Vec::with_capacity(x.len());
            let mut stats = Vec::with_capacity(s.rows);
            let n = s.cols as i128;
            for row in x.chunks_exact(s.cols) {
                let sum = if p.centered() {
                    row.iter().map(|&v| i128::from(v)).sum::<i128>()
                } else {
                    0
                };
                let (mean, mr) = (sum.div_euclid(n), sum.rem_euclid(n));
                let square: i128 = row.iter().map(|&v| (i128::from(v) - mean).pow(2)).sum();
                let var = square / n + i128::from(p.epsilon_variance_units);
                let vr = square % n;
                let root = var.isqrt();
                let err = var - root * root;
                stats.push([
                    var,
                    vr,
                    n - 1 - vr,
                    root,
                    err,
                    2 * root - err,
                    mean,
                    mr,
                    n - 1 - mr,
                ]);
                for ((&value, &gain), &bias) in row.iter().zip(p.gain).zip(p.bias) {
                    let q = ((i128::from(value) - mean) << bits).div_euclid(root);
                    let y = ((q * i128::from(gain)) >> bits) + i128::from(bias);
                    values.push(i32::try_from(y).map_err(|_| "normalization exceeds i32")?);
                    qs.push(i64::try_from(q).map_err(|_| "normalization quotient exceeds i64")?);
                }
            }
            (w.values, w.quotients, w.rows) = (Some(values), Some(qs), Some(stats));
        }
        use Width::*;
        let r = s.stride().ilog2() as u8;
        for (i, width) in [
            Signed(63),
            Signed(32),
            Unsigned(33),
            Unsigned(33),
            Unsigned(bits as u8),
        ]
        .into_iter()
        .enumerate()
        {
            let slot = a.column(s, width, |j| w.cell(p, x.unwrap(), j, i + 1))?;
            w.slots.push(slot);
        }
        let rows = Shape::new(s.rows, 1)?;
        let widths = [
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
        for (i, width) in widths.into_iter().enumerate() {
            w.slots
                .push(a.column(rows, width, |j| w.rows.as_ref().unwrap()[j][i])?);
        }
        Ok(w)
    }
    fn values(&self) -> Option<&[i32]> {
        self.values.as_deref()
    }
    fn cell(&self, p: NormalizationParameters<'_>, x: &[i32], i: usize, j: usize) -> i128 {
        let q = i128::from(self.quotients.as_ref().unwrap()[i]);
        let s = self.rows.as_ref().unwrap()[i / self.shape.cols];
        let x = i128::from(x[i]);
        let remainder = || ((x - s[6]) << self.bits) - s[3] * q;
        match j {
            0 => x,
            1 => q,
            2 => i128::from(self.values.as_ref().unwrap()[i]),
            3 => remainder(),
            4 => s[3] - 1 - remainder(),
            _ => (q * i128::from(p.gain[i % self.shape.cols])) & ((1 << self.bits) - 1),
        }
    }
    fn reverse(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        p: NormalizationParameters<'_>,
        x: Option<&[i32]>,
        out: &[Evaluation],
        carry: &[Evaluation],
    ) -> Result<Evaluation> {
        let s = self.shape;
        let n = s.len();
        let stride = s.stride();
        let count = out.len() + carry.len();
        if count == 0 || out.iter().chain(carry).any(|q| q.point.len() != s.vars()) {
            return Err("invalid normalization evaluation");
        }
        io.append(&(out.len() as u64));
        io.append(&(carry.len() as u64));
        for q in out.iter().chain(carry) {
            io.append_fields(&q.point);
            io.append_fields(&[q.value]);
        }
        let draw = io.draw(count + 9 + s.vars() + s.row_bits())?;
        let (a, rest) = draw.split_at(9);
        let (spot, rest) = rest.split_at(s.vars());
        let (row, mix) = rest.split_at(s.row_bits());
        let (mix_out, mix_in) = mix.split_at(out.len());
        let value = out
            .iter()
            .chain(carry)
            .zip(mix)
            .map(|(q, &a)| q.value * a)
            .sum();
        let real_width = F::from_u64(s.cols as u64);
        let one = F::one();
        let divisor = F::from_u64(1 << self.bits);
        let epsilon = F::from_i64(p.epsilon_variance_units);
        let relation = |v: &[F]| {
            let &[x, q, y, rem, gap, aff, var, vr, vg, root, err, rg, mean, mr, mg, gain, bias, lane, active, first, out, input, spot, row] =
                v
            else {
                unreachable!()
            };
            let centered = x - mean;
            let mean_check = if p.centered() {
                lane * x - first * (mean * real_width + mr)
            } else {
                lane * mean
            };
            let variance = lane * centered * centered - first * ((var - epsilon) * real_width + vr);
            let mut sums = a[0] * variance
                + a[2] * first * (var - root * root - err)
                + a[3] * first * (root + root - err - rg)
                + a[4] * mean_check;
            if !s.cols.is_power_of_two() {
                sums += a[1] * first * (vr + vg - real_width + one);
                if p.centered() {
                    sums += a[5] * first * (mr + mg - real_width + one);
                }
            }
            let quot = a[6] * (divisor * centered - root * q - rem)
                + a[7] * (root - one - rem - gap)
                + a[8] * (q * gain - divisor * (y - bias) - aff);
            row * sums + active * (out * y + spot * quot) + input * x
        };
        let mut row_weights = basis(row);
        row_weights[s.rows..].fill(F::zero());
        let degree = if n == s.cells() { 3 } else { 4 };
        let indices: Vec<_> = (0..15).collect();
        let (end, mut terminal) = io.fold(
            statement(value, s.vars(), degree),
            &indices,
            || {
                let x = x.unwrap();
                let mut cols: Vec<_> = (0..6)
                    .map(|j| {
                        Column::read(n, move |i| {
                            s.index(i)
                                .map_or(F::zero(), |i| F::from_i128(self.cell(p, x, i, j)))
                        })
                    })
                    .collect();
                cols.extend((0..9).map(|j| {
                    let mut values: Vec<_> = self
                        .rows
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|r| F::from_i128(r[j]))
                        .collect();
                    values.resize(s.rows(), F::zero());
                    Column::repeat(n, stride, values)
                }));
                let public = |v: &[i32]| {
                    let mut v: Vec<_> = v.iter().map(|&x| F::from_i32(x)).collect();
                    v.resize(stride, F::zero());
                    Column::repeat(n, 1, v)
                };
                let mask = |h| {
                    if s.cols == stride && h == s.rows() {
                        Column::repeat(n, n, vec![one])
                    } else {
                        Column::read(n, move |i| {
                            F::from_bool(i / stride < h && i % stride < s.cols)
                        })
                    }
                };
                let (r, c) = s.point(spot);
                cols.extend([
                    public(p.gain),
                    public(p.bias),
                    mask(s.rows()),
                    mask(s.rows),
                    Column::repeat(n, 1, (0..stride).map(|i| F::from_bool(i == 0)).collect()),
                    claim_weight(s, out, mix_out),
                    claim_weight(s, carry, mix_in),
                    Column::Tensor(basis(r), basis(c)),
                    Column::repeat(n, stride, row_weights.clone()),
                ]);
                cols
            },
            relation,
        )?;
        let (r, c) = s.point(&end.challenges);
        let row = basis(r);
        let col = basis(c);
        let mix_eval = |qs: &[Evaluation], coeff: &[F]| {
            qs.iter()
                .zip(coeff)
                .map(|(q, &m)| m * same_point(&q.point, &end.challenges))
                .sum()
        };
        let lanes: F = col[..s.cols].iter().copied().sum();
        let above: F = row[..s.rows].iter().copied().sum();
        terminal.extend([
            affine_sum(&col, p.gain),
            affine_sum(&col, p.bias),
            lanes,
            lanes * above,
            col[0],
            mix_eval(out, mix_out),
            mix_eval(carry, mix_in),
            same_point(spot, &end.challenges),
            inner(&row, &row_weights),
        ]);
        if end.output_claim != relation(&terminal) {
            return Err("invalid terminal normalization");
        }
        for (i, (&slot, &v)) in self.slots.iter().zip(&terminal[1..15]).enumerate() {
            ask(requests, slot, if i < 5 { &end.challenges } else { r }, v);
        }
        Ok(Evaluation {
            point: end.challenges,
            value: terminal[0],
        })
    }
}

fn claim_weight(s: Shape, qs: &[Evaluation], mix: &[F]) -> Column<'static> {
    let n = s.len();
    let Some(q) = qs.first() else {
        return Column::repeat(n, n, vec![F::zero()]);
    };
    let (_, c) = s.point(&q.point);
    if qs.iter().all(|q| s.point(&q.point).1 == c) {
        let mut row = vec![F::zero(); s.rows()];
        for (q, &m) in qs.iter().zip(mix) {
            merge(&mut row, &scaled_basis(s.point(&q.point).0, m));
        }
        return Column::Tensor(row, basis(c));
    }
    let terms: Vec<_> = qs
        .iter()
        .zip(mix)
        .map(|(q, &a)| {
            let (r, c) = s.point(&q.point);
            (scaled_basis(r, a), basis(c))
        })
        .collect();
    let mut values = vec![F::zero(); n];
    values
        .par_chunks_mut(s.stride())
        .with_min_len(8)
        .enumerate()
        .for_each(|(r, row)| {
            for (a, b) in &terms {
                row.iter_mut().zip(b).for_each(|(v, &w)| *v += a[r] * w);
            }
        });
    Column::repeat(n, 1, values)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxMask {
    Full,
    Causal,
}

#[derive(Clone, Copy)]
pub struct SoftmaxTables<'a> {
    pub probabilities: &'a [i32],
    pub sums: &'a [i32],
    pub limit: i32,
}

#[derive(Clone, Copy)]
struct Tile {
    row: usize,
    col: usize,
    size: usize,
}

impl Tile {
    fn visit(self, n: usize, mask: SoftmaxMask, dst: &mut Vec<Self>) {
        let Self { row, col, size } = self;
        let causal = mask == SoftmaxMask::Causal;
        if row >= n || col >= n || causal && col >= row + size {
            return;
        }
        if size <= 64 || size <= 512 && (!causal || col + size - 1 <= row) {
            dst.push(self);
        } else {
            let size = size / 2;
            for row in [row, row + size] {
                for col in [col, col + size] {
                    Self { row, col, size }.visit(n, mask, dst);
                }
            }
        }
    }
    fn at(self, i: usize) -> (usize, usize) {
        (self.row + i / self.size, self.col + i % self.size)
    }
    fn local(self, r: &[F], c: &[F]) -> (F, Vec<F>) {
        let at = r.len() - self.size.ilog2() as usize;
        let (p, r) = r.split_at(at);
        let (q, c) = c.split_at(at);
        let w = basis(p)[self.row / self.size] * basis(q)[self.col / self.size];
        (w, [r, c].concat())
    }
}

struct Softmax<'a> {
    shape: Shape,
    mask: SoftmaxMask,
    tables: SoftmaxTables<'a>,
    radix: u128,
    slack: u8,
    tiles: Vec<Tile>,
}
struct SoftValues {
    score: Vec<i32>,
    rem: Vec<u32>,
    rows: Vec<[i64; 3]>,
    excess: Option<Vec<i64>>,
    p: Vec<i32>,
    address: Vec<u32>,
}
#[derive(Serialize, Deserialize)]
struct TileTrace {
    key: Lookup,
    score: Option<Slot>,
    excess: Slot,
    rem: Slot,
}
#[derive(Serialize, Deserialize)]
struct SoftTrace {
    tiles: Vec<TileTrace>,
    #[serde(with = "trace_array")]
    rows: Option<Vec<[i64; 3]>>,
    #[serde(with = "trace_array")]
    excess: Option<Vec<i64>>,
    stats: [Slot; 3],
}

impl<'a> Softmax<'a> {
    fn new(rows: usize, mask: SoftmaxMask, t: SoftmaxTables<'a>) -> Result<Self> {
        let shape = Shape::new(rows, rows)?;
        let size = t.sums.len();
        if size < 2
            || !size.is_power_of_two()
            || size != t.probabilities.len()
            || !(1..i32::MAX).contains(&t.limit)
            || t.sums.iter().any(|&v| v < 0)
            || t.sums.windows(2).any(|p| p[0] < p[1])
            || rows as u128 * t.sums[size - 1] as u128 > t.limit as u128
        {
            return Err("invalid Softmax table");
        }
        let width = rows as u128 * (t.limit + 1).max(t.sums[0]) as u128;
        let slack = (width.ilog2() + 2) as u8;
        if slack > 55 {
            return Err("Softmax range exceeds field");
        }
        let radix = (width + (1 << slack) + t.limit as u128).next_power_of_two();
        let mut tiles = Vec::new();
        Tile {
            row: 0,
            col: 0,
            size: shape.stride(),
        }
        .visit(rows, mask, &mut tiles);
        Ok(Self {
            shape,
            mask,
            tables: t,
            radix,
            slack,
            tiles,
        })
    }
    fn columns(&self, i: usize) -> usize {
        if i >= self.shape.rows {
            0
        } else if self.mask == SoftmaxMask::Causal {
            i + 1
        } else {
            self.shape.cols
        }
    }
    fn eval(&self, bits: u32, dot: impl Fn(usize, usize) -> i128 + Sync) -> Result<SoftValues> {
        let n = self.shape.rows;
        let mut score = vec![0; n * n];
        let mut rem = vec![0; n * n];
        score
            .par_chunks_mut(n)
            .zip(rem.par_chunks_mut(n))
            .enumerate()
            .try_for_each(|(i, (row, rem))| {
                for j in 0..self.columns(i) {
                    let x = dot(i, j);
                    row[j] = i32::try_from(x >> bits).map_err(|_| "QK score exceeds i32")?;
                    rem[j] = (x & ((1 << bits) - 1)) as u32;
                }
                Ok(())
            })?;
        let limit = i64::from(self.tables.limit);
        let last = (self.tables.sums.len() - 1) as i64;
        let entry = |i: i64| {
            if i < 0 {
                limit + 1
            } else {
                i64::from(self.tables.sums[i as usize])
            }
        };
        let rows: Vec<_> = score
            .par_chunks(n)
            .enumerate()
            .map(|(i, row)| {
                let row = &row[..self.columns(i)];
                let max = i64::from(*row.iter().max().unwrap());
                let (mut lo, mut hi) = (max, max + last);
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let good = row
                        .iter()
                        .try_fold(0, |sum, &v| {
                            // The search starts at the maximum score in this row.
                            let address = (mid - i64::from(v)).min(last) as usize;
                            let sum = sum + i64::from(self.tables.sums[address]);
                            (sum <= limit).then_some(sum)
                        })
                        .is_some();
                    if good {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
                let sum = |earlier: i64| {
                    row.iter()
                        .map(|&v| entry((lo - i64::from(v)).min(last) - earlier))
                        .sum::<i64>()
                };
                [lo, limit - sum(0), sum(1) - limit - 1]
            })
            .collect();
        let mut p = vec![0; n * n];
        let mut extra = vec![0; n * n];
        let mut address = vec![0; self.shape.len()];
        p.par_chunks_mut(n)
            .zip(extra.par_chunks_mut(n))
            .zip(address.par_chunks_mut(self.shape.stride()))
            .zip(score.par_chunks(n))
            .zip(rows.par_iter())
            .enumerate()
            .for_each(|(i, ((((p, excess), address), score), row))| {
                for j in 0..self.columns(i) {
                    let difference = row[0] - i64::from(score[j]);
                    let value = difference.min(last);
                    address[j] = value as u32;
                    excess[j] = difference - value;
                    p[j] = self.tables.probabilities[value as usize];
                }
            });
        let excess = extra.iter().any(|&x| x != 0).then_some(extra);
        Ok(SoftValues {
            score,
            rem,
            rows,
            excess,
            p,
            address,
        })
    }
    fn record(
        &self,
        a: &mut Advice,
        bits: u32,
        check: bool,
        w: Option<SoftValues>,
    ) -> Result<SoftTrace> {
        let n = self.shape.rows;
        let stride = self.shape.stride();
        let value = || w.as_ref().unwrap();
        let mut tiles = Vec::with_capacity(self.tiles.len());
        for &tile in &self.tiles {
            let shape = Shape::new(tile.size, tile.size)?;
            let live = |j| {
                let (r, c) = tile.at(j);
                (r < n && c < n).then_some(r * n + c)
            };
            let key = Lookup::record(a, shape, self.tables.sums.len(), |i| {
                let (r, c) = tile.at(i);
                value().address[r * stride + c]
            })?;
            let excess = a.column(shape, Width::Unsigned(34), |i| {
                i128::from(
                    live(i)
                        .and_then(|j| value().excess.as_ref().map(|x| x[j]))
                        .unwrap_or(0),
                )
            })?;
            let rem = a.column(shape, Width::Unsigned(bits as u8), |i| {
                i128::from(live(i).map_or(0, |j| value().rem[j]))
            })?;
            let score = check
                .then(|| {
                    a.column(shape, Width::Signed(32), |i| {
                        i128::from(live(i).map_or(0, |j| value().score[j]))
                    })
                })
                .transpose()?;
            tiles.push(TileTrace {
                key,
                score,
                excess,
                rem,
            });
        }
        let shape = Shape::new(n, 1)?;
        let mut col = |i: usize, w| a.column(shape, w, |j| i128::from(value().rows[j][i]));
        let stats = [
            col(0, Width::Signed(34))?,
            col(1, Width::Unsigned(self.slack))?,
            col(2, Width::Unsigned(self.slack))?,
        ];
        let (rows, excess) = w.map_or((None, None), |w| (Some(w.rows), w.excess));
        Ok(SoftTrace {
            tiles,
            rows,
            excess,
            stats,
        })
    }
    fn reverse(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        reads: &mut Reads<'a>,
        w: &SoftTrace,
        claim: F,
        row: &[F],
        value: Option<&[F]>,
        bits: u32,
    ) -> Result<(Evaluation, Evaluation)> {
        let s = self.shape;
        let n = s.len();
        let stride = s.stride();
        let advice = io.advice();
        let table = Table::Soft(self.tables, self.radix);
        let codes = advice.map(|a| {
            let mut codes = vec![0u32; n];
            for (&tile, w) in self.tiles.iter().zip(&w.tiles) {
                for i in 0..tile.size * tile.size {
                    let (r, c) = tile.at(i);
                    codes[r * stride + c] = w.key.code(a, i) as u32;
                }
            }
            codes
        });
        let eval = |r: usize, c: usize, code: u32| -> (F, F) {
            if c >= self.columns(r) {
                return (F::zero(), F::zero());
            }
            let overflow = w.excess.as_ref().map_or(0, |x| x[r * s.rows + c]);
            let score = w.rows.as_ref().unwrap()[r][0] - i64::from(code) - overflow;
            (F::from_i64(score), F::from_i64(overflow))
        };
        let draw = io.draw(s.vars() + s.row_bits() + 3)?;
        let (spot, extra) = draw.split_at(s.vars());
        let (stat, a) = extra.split_at(s.row_bits());
        let limit = F::from_i32(self.tables.limit);
        let one = F::one();
        let radix = F::from_u128(self.radix);
        let last = F::from_u64(table.len() as u64 - 1);
        let zero_p = F::from_i32(self.tables.probabilities[0]);
        let zero_t = F::from_i32(self.tables.sums[0]) + radix * (limit + one);
        let relation = |v: &[F]| {
            let &[i, p, t, overflow, below, above, mask, first, pv, spot, stat] = v else {
                unreachable!()
            };
            let clamp = a[0] * overflow * (last - i) + (one - mask) * (a[1] * i + a[2] * overflow);
            let sums = first * (below - limit - radix * (above + limit + one)) + t + mask * zero_t;
            pv * (p + mask * zero_p) + spot * clamp + stat * sums
        };
        let mut stats = basis(stat);
        stats[s.rows..].fill(F::zero());
        let stat_column = |i: usize| {
            let mut col: Vec<_> = w
                .rows
                .as_ref()
                .unwrap()
                .iter()
                .map(|r| F::from_i64(r[i]))
                .collect();
            col.resize(stride, F::zero());
            col
        };
        let (end, columns) = io.fold(
            statement(claim, s.vars(), 3),
            &[0, 1, 2, 3, 4, 5],
            || {
                let c = codes.as_ref().unwrap();
                vec![
                    Column::read(n, |i| F::from_u32(c[i])),
                    Column::read(n, |i| table.get(c[i] as usize)[0]),
                    Column::read(n, |i| table.get(c[i] as usize)[1]),
                    Column::read(n, |i| eval(i / stride, i % stride, c[i]).1),
                    Column::repeat(n, stride, stat_column(1)),
                    Column::repeat(n, stride, stat_column(2)),
                    Column::read(n, |i| F::from_bool(i % stride < self.columns(i / stride))),
                    Column::repeat(n, 1, (0..stride).map(|i| F::from_bool(i == 0)).collect()),
                    Column::Tensor(basis(row), value.unwrap().to_vec()),
                    Column::repeat(n, 1, basis(spot)),
                    Column::repeat(n, stride, stats.clone()),
                ]
            },
            relation,
        )?;
        let point = end.challenges;
        let (r, c) = s.point(&point);
        let cw = basis(c);
        let [v] = io.values(|| [inner(&cw, value.unwrap())])?;
        let mut prefix = vec![F::zero()];
        for x in cw {
            prefix.push(*prefix.last().unwrap() + x);
        }
        let rw = basis(r);
        let row_weights: Vec<_> = rw
            .iter()
            .enumerate()
            .map(|(i, &x)| x * prefix[self.columns(i)])
            .collect();
        let [index, p, t, overflow, below, above]: [F; 6] = columns.try_into().unwrap();
        let public = [
            index,
            p,
            t,
            overflow,
            below,
            above,
            row_weights.iter().copied().sum(),
            prefix[1],
            same_point(row, r) * v,
            same_point(spot, &point),
            inner(&rw, &stats),
        ];
        if end.output_claim != relation(&public) {
            return Err("invalid terminal Softmax");
        }
        ask(requests, w.stats[1], r, below);
        ask(requests, w.stats[2], r, above);
        let points: Vec<_> = self.tiles.iter().map(|tile| tile.local(r, c)).collect();
        let values = advice.map(|a| {
            self.tiles
                .par_iter()
                .zip(&w.tiles)
                .zip(&points)
                .map(|((&tile, t), (_, p))| {
                    let (rr, cc) = p.split_at(tile.size.ilog2() as usize);
                    let c = basis(cc);
                    let mut sums = [F::zero(); 6];
                    let codes = codes.as_ref().unwrap();
                    for (x, w_r) in basis(rr).into_iter().enumerate() {
                        let mut row = [F::zero(); 6];
                        for (y, &w_c) in c.iter().enumerate() {
                            let (r, c) = tile.at(x * tile.size + y);
                            let code = codes[r * stride + c];
                            let [p, u] = table.get(code as usize);
                            let (score, over) = eval(r, c, code);
                            let rem = a.read(t.rem, x * tile.size + y) as u32;
                            for (s, v) in row.iter_mut().zip([
                                score,
                                F::from_u32(code),
                                p,
                                u,
                                over,
                                F::from_u32(rem),
                            ]) {
                                *s += w_c * v;
                            }
                        }
                        for (s, v) in sums.iter_mut().zip(row) {
                            *s += w_r * v;
                        }
                    }
                    sums
                })
                .collect::<Vec<_>>()
        });
        let mut sum = [F::zero(); 6];
        for (i, (tile, (factor, p))) in w.tiles.iter().zip(points).enumerate() {
            let values = io.values(|| values.as_ref().unwrap()[i])?;
            for (s, v) in sum.iter_mut().zip(values) {
                *s += factor * v;
            }
            let [score, index, pv, table_value, overflow, rem] = values;
            if let Some(slot) = tile.score {
                ask(requests, slot, &p, score);
            }
            ask(requests, tile.excess, &p, overflow);
            ask(requests, tile.rem, &p, rem);
            reads.push(TableRead {
                table,
                key: tile.key.clone(),
                point: p,
                values: [index, pv, table_value],
            });
        }
        if sum[1..5] != [index, p, t, overflow] {
            return Err("invalid Softmax tile values");
        }
        let shift = io.dot(sum[0] + index + overflow, row_weights, || stat_column(0))?;
        ask(requests, w.stats[0], &shift.point, shift.value);
        let column = at(c, v);
        Ok((
            Evaluation {
                point,
                value: sum[0] * F::from_u64(1 << bits) + sum[5],
            },
            column,
        ))
    }
}

#[derive(Clone, Copy)]
pub struct Activation<'a> {
    pub lower: i32,
    pub correction: &'a [i32],
}
#[derive(Clone, Copy)]
pub struct Mlp<'a> {
    pub up: Projection<'a>,
    pub gate: Option<Projection<'a>>,
    pub down: Projection<'a>,
    pub activation: Activation<'a>,
}

struct ActivationTable<'a> {
    values: &'a [i32],
    low: i64,
    symmetric: bool,
}
#[derive(Serialize, Deserialize)]
struct Feed {
    up: Rounded,
    gate: Option<Rounded>,
    product: Option<Rounded>,
    #[serde(with = "trace_array")]
    activated: Option<Vec<i32>>,
    key: Lookup,
    slots: Vec<Slot>,
    output: Rounded,
}

impl<'a> ActivationTable<'a> {
    fn new(p: Activation<'a>) -> Result<Self> {
        let half = p.correction.len() / 2;
        let low = i64::from(p.lower);
        let symmetric = half > 1
            && low == -(half as i64)
            && p.correction.first() == p.correction.last()
            && (1..half).all(|i| p.correction[half - i] == p.correction[half + i]);
        let values = if symmetric {
            &p.correction[half..]
        } else {
            p.correction
        };
        if !values.len().is_power_of_two() {
            return Err("invalid activation table");
        }
        Ok(Self {
            values,
            low,
            symmetric,
        })
    }
    fn code(&self, x: i32) -> (usize, [i64; 4]) {
        let (x, sign) = (i64::from(x), i64::from(x >= 0));
        let last = self.values.len() as i64 - 1;
        if self.symmetric {
            let i = x.abs().min(last);
            (i as usize, [sign, 0, 0, x.abs() - i])
        } else {
            let address = x - self.low;
            let i = address.clamp(0, last);
            (
                i as usize,
                [sign, x.abs(), (i - address).max(0), (address - i).max(0)],
            )
        }
    }
}

fn bounded_gate(p: NormalizationParameters<'_>, gate: Projection<'_>, bits: u32) -> bool {
    let width = p.gain.len();
    if p.centered()
        || !(1..=1 << 20).contains(&width)
        || p.bias.len() != width
        || gate.weights.len() != width.saturating_mul(gate.bias.len())
    {
        return false;
    }
    let Some(q) = p.narrowed(bits) else {
        return false;
    };
    let scale = 1u128 << bits;
    gate.weights
        .chunks_exact(width)
        .zip(gate.bias)
        .all(|(row, &b)| {
            let moments = row.iter().zip(p.gain).zip(p.bias).try_fold(
                (0u128, 0u128),
                |(sq, error), ((&w, &g), &c)| {
                    let w = u128::from(w.unsigned_abs());
                    let wg = (w * u128::from(g.unsigned_abs())).div_ceil(scale);
                    Some((
                        sq.checked_add(wg * wg)?,
                        error + w * (u128::from(c.unsigned_abs()) + 1),
                    ))
                },
            );
            moments.is_some_and(|(sq, e)| {
                (q * ceil_sqrt(sq) + e).div_ceil(scale) + u128::from(b.unsigned_abs())
                    <= i32::MAX as u128
            })
        })
}

impl Feed {
    fn record(
        a: &mut Advice,
        s: Shape,
        bits: u32,
        m: Mlp<'_>,
        x: Option<&[i32]>,
        bounded: bool,
    ) -> Result<Self> {
        let up = Rounded::apply(a, s, bits, m.up, x, true)?;
        let gate = m
            .gate
            .map(|p| Rounded::apply(a, s, bits, p, x, !bounded))
            .transpose()?;
        let s = up.shape;
        let input = gate.as_ref().unwrap_or(&up);
        if input.shape != s {
            return Err("invalid gated MLP size");
        }
        let input = input.values();
        let table = ActivationTable::new(m.activation)?;
        let code = input.map(|x| {
            (0..s.len())
                .map(|i| s.index(i).map_or(0, |j| table.code(x[j]).0 as u32))
                .collect::<Vec<_>>()
        });
        let key = Lookup::record(a, s, table.values.len(), |i| code.as_ref().unwrap()[i])?;
        let n = u8::from(!table.symmetric);
        let slots = [1, 32 * n, 33 * n, 33]
            .into_iter()
            .enumerate()
            .map(|(i, w)| {
                a.column(s, Width::Unsigned(w), |j| {
                    i128::from(table.code(input.unwrap()[j]).1[i])
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let activated = input
            .map(|x| {
                x.iter()
                    .map(|&v| {
                        v.max(0)
                            .checked_add(table.values[table.code(v).0])
                            .ok_or("activation exceeds i32")
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let (product, activated) = if gate.is_some() {
            let witness = activated
                .as_ref()
                .zip(up.output.as_ref())
                .map(|(g, x)| {
                    Rounded::rounded(s, bits, |r, c| {
                        let i = r * s.cols + c;
                        i128::from(g[i]) * i128::from(x[i])
                    })
                })
                .transpose()?;
            (Some(Rounded::record(a, s, bits, true, witness)?), None)
        } else {
            (None, activated)
        };
        let inputs = product
            .as_ref()
            .map_or(activated.as_deref(), Rounded::values);
        let output = Rounded::apply(a, s, bits, m.down, inputs, true)?;
        Ok(Self {
            up,
            gate,
            product,
            activated,
            key,
            slots,
            output,
        })
    }

    fn reverse<'a>(
        &self,
        io: &mut Channel<'_, '_>,
        r: &mut Requests,
        reads: &mut Reads<'a>,
        m: Mlp<'a>,
        s: Shape,
        x: Option<&[i32]>,
        out: Evaluation,
    ) -> Result<Vec<Evaluation>> {
        let shape = self.up.shape;
        let bits = self.up.bits;
        let y = self
            .product
            .as_ref()
            .map_or(self.activated.as_deref(), Rounded::values);
        let mut q = projections(io, r, shape, bits, &[(&self.output, m.down, out)], y)?.remove(0);
        if let Some(w) = &self.product {
            q.value = w.raw(io, r, &q)?;
        }
        let a = ActivationTable::new(m.activation)?;
        let n = shape.len();
        let gated = self.gate.is_some();
        let draw = io.draw(shape.vars() + 4)?;
        let (spot, mix) = draw.split_at(shape.vars());
        let one = F::one();
        let last = F::from_u64((a.values.len() - 1) as u64);
        let expression = |v: &[F]| {
            let &[x, index, t, sign, abs, low, high, mask, output, spot, up] = v else {
                unreachable!()
            };
            let magnitude = if a.symmetric { index + high } else { abs };
            let signed = x - (sign + sign - one) * magnitude;
            let upper = high * (last - index);
            let clamp = if a.symmetric {
                mix[1] * upper
            } else {
                mix[1] * (x - F::from_i64(a.low) - index + low - high)
                    + mix[2] * low * index
                    + mix[3] * upper
            };
            let factor = if gated { up } else { one };
            mask * (output * (sign * magnitude + t) * factor + spot * (mix[0] * signed + clamp))
        };
        let advice = io.advice();
        let (end, values) = io.fold(
            statement(q.value, shape.vars(), if gated { 5 } else { 4 }),
            &[0, 1, 2, 3, 4, 5, 6, 10],
            || {
                let x = self.gate.as_ref().unwrap_or(&self.up).values().unwrap();
                let code = |i| self.key.code(advice.unwrap(), i);
                let table = &a;
                let mut columns = vec![
                    Column::read(n, move |i| shape.value(x, i)),
                    Column::read(n, move |i| F::from_u64(code(i) as u64)),
                    Column::read(n, move |i| F::from_i32(table.values[code(i)])),
                ];
                columns.extend((0..4).map(|j| {
                    Column::read(n, move |i| {
                        shape
                            .index(i)
                            .map_or(F::zero(), |i| F::from_i64(table.code(x[i]).1[j]))
                    })
                }));
                columns.extend([
                    Column::read(n, |i| F::from_bool(shape.index(i).is_some())),
                    Column::repeat(n, 1, basis(&q.point)),
                    Column::repeat(n, 1, basis(spot)),
                    Column::read(n, |i| shape.value(self.up.values().unwrap(), i)),
                ]);
                columns
            },
            expression,
        )?;
        let [g, index, t, sign, abs, low, high, up]: [F; 8] = values.try_into().unwrap();
        let point = end.challenges;
        let public = [
            g,
            index,
            t,
            sign,
            abs,
            low,
            high,
            shape.active(&point),
            same_point(&q.point, &point),
            same_point(spot, &point),
            up,
        ];
        if expression(&public) != end.output_claim {
            return Err("invalid terminal activation");
        }
        for (&slot, x) in self.slots.iter().zip([sign, abs, low, high]) {
            ask(r, slot, &point, x);
        }
        reads.push(TableRead {
            table: Table::Activation(a.values),
            key: self.key.clone(),
            point: point.clone(),
            values: [index, t, F::zero()],
        });
        if let Some((w, p)) = self.gate.as_ref().zip(m.gate) {
            projections(
                io,
                r,
                s,
                bits,
                &[
                    (w, p, at(&point, g)),
                    (&self.up, m.up, Evaluation { point, value: up }),
                ],
                x,
            )
        } else {
            projections(
                io,
                r,
                s,
                bits,
                &[(&self.up, m.up, Evaluation { point, value: g })],
                x,
            )
        }
    }
}

pub struct Attention<'a> {
    pub queries: Vec<Projection<'a>>,
    pub groups: Vec<[Projection<'a>; 2]>,
    pub output: Projection<'a>,
    pub rotary: Option<Rotary<'a>>,
    pub mask: SoftmaxMask,
    pub tables: SoftmaxTables<'a>,
    pub factor: i32,
}

#[derive(Serialize, Deserialize)]
struct HeadInput {
    projection: Rounded,
    norm: Option<Normalized>,
    rotary: Option<Rounded>,
}
#[derive(Serialize, Deserialize)]
struct Head {
    softmax: SoftTrace,
    product: Rounded,
}
#[derive(Serialize, Deserialize)]
struct AttentionTrace {
    queries: Vec<HeadInput>,
    keys: Vec<HeadInput>,
    values: Vec<Rounded>,
    heads: Vec<Head>,
    output: Rounded,
}

fn join_heads(shape: Shape, heads: &[Head], channels: usize) -> Option<Vec<i32>> {
    heads.first()?.product.values()?;
    let mut joined = Vec::with_capacity(shape.cells());
    for row in 0..shape.rows {
        for head in heads {
            joined.extend_from_slice(&head.product.values().unwrap()[row * channels..][..channels]);
        }
    }
    Some(joined)
}

impl HeadInput {
    fn normal(&self) -> Option<&[i32]> {
        self.norm
            .as_ref()
            .map_or(self.projection.values(), Normalized::values)
    }
    fn values(&self) -> Option<&[i32]> {
        self.rotary
            .as_ref()
            .map_or_else(|| self.normal(), Rounded::values)
    }
    fn reverse(
        &self,
        io: &mut Channel<'_, '_>,
        r: &mut Requests,
        p: &Attention<'_>,
        norm: Option<NormalizationParameters<'_>>,
        mut q: Vec<Evaluation>,
    ) -> Result<Vec<Evaluation>> {
        if let Some((op, w)) = p.rotary.zip(self.rotary.as_ref()) {
            q = vec![op.reverse(io, r, w, self.normal(), &q)?];
        }
        if let Some((op, w)) = norm.zip(self.norm.as_ref()) {
            q = vec![w.reverse(io, r, op, self.projection.values(), &q, &[])?];
        }
        Ok(q)
    }
}

#[derive(Clone, Copy)]
struct AttentionPlan<'a> {
    p: &'a Attention<'a>,
    shape: Shape,
    bits: u32,
    norm: Option<[NormalizationParameters<'a>; 2]>,
}

impl<'a> AttentionPlan<'a> {
    fn factors(self) -> (i32, u32) {
        let shift = self.p.factor.trailing_zeros().min(self.bits * 2);
        (self.p.factor >> shift, self.bits * 2 - shift)
    }
    fn norm(self, i: usize) -> Option<NormalizationParameters<'a>> {
        self.norm.map(|n| n[i])
    }
    fn scores_fit(self) -> bool {
        let scale = 1u128 << self.bits;
        let rotation = self.p.rotary.map(|r| {
            r.cos
                .iter()
                .zip(r.sin)
                .map(|(&a, &b)| {
                    ceil_sqrt(
                        u128::from(a.unsigned_abs()).pow(2) + u128::from(b.unsigned_abs()).pow(2),
                    )
                })
                .max()
                .unwrap_or(0)
        });
        let bound = |p: NormalizationParameters<'_>| {
            let q = p.narrowed(self.bits)?;
            let w = ceil_sqrt(p.gain.len() as u128);
            let gain = u128::from(p.gain.iter().map(|x| x.unsigned_abs()).max()?);
            let bias = u128::from(p.bias.iter().map(|x| x.unsigned_abs()).max()?);
            let n = (q * gain).div_ceil(scale) + w * (bias + 1);
            Some(rotation.map_or(n, |rot| (rot * n).div_ceil(scale) + w))
        };
        self.norm(0)
            .and_then(bound)
            .zip(self.norm(1).and_then(bound))
            .and_then(|(a, b)| a.checked_mul(b))
            .and_then(|p| p.checked_mul(self.p.factor as u128))
            .is_some_and(|x| x.div_ceil(scale * scale) <= i32::MAX as u128)
    }
    fn record(self, a: &mut Advice, x: Option<&[i32]>) -> Result<AttentionTrace> {
        let p = self.p;
        if p.groups.is_empty()
            || p.queries.is_empty()
            || p.queries.len() % p.groups.len() != 0
            || p.factor <= 0
        {
            return Err("invalid Attention dimensions");
        }
        let d = p.queries[0].bias.len();
        let channels = p.groups[0][1].bias.len();
        if !d.is_power_of_two()
            || !channels.is_power_of_two()
            || p.queries.iter().any(|p| p.bias.len() != d)
            || p.groups
                .iter()
                .any(|[k, v]| k.bias.len() != d || v.bias.len() != channels)
        {
            return Err("invalid Attention widths");
        }
        let mut head = |proj, norm: Option<NormalizationParameters<'a>>| {
            let projection = Rounded::apply(a, self.shape, self.bits, proj, x, true)?;
            let y = projection.values();
            let norm = norm
                .map(|p| Normalized::record(a, projection.shape, self.bits, p, y))
                .transpose()?;
            let y = norm.as_ref().map_or(y, Normalized::values);
            let rotary = p
                .rotary
                .map(|op| op.record(a, projection.shape, self.bits, y))
                .transpose()?;
            Ok(HeadInput {
                projection,
                norm,
                rotary,
            })
        };
        let queries = p
            .queries
            .iter()
            .map(|&q| head(q, self.norm(0)))
            .collect::<Result<Vec<_>>>()?;
        let keys = p
            .groups
            .iter()
            .map(|[k, _]| head(*k, self.norm(1)))
            .collect::<Result<Vec<_>>>()?;
        let values = p
            .groups
            .iter()
            .map(|[_, v]| Rounded::apply(a, self.shape, self.bits, *v, x, true))
            .collect::<Result<Vec<_>>>()?;
        let transpose: Vec<_> = values
            .iter()
            .map(|v| {
                v.values().map(|v| {
                    (0..channels)
                        .flat_map(|c| v[c..].iter().step_by(channels).copied())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let n = self.shape.rows;
        let (factor, bits) = self.factors();
        let soft = Softmax::new(n, p.mask, p.tables)?;
        let per = queries.len() / keys.len();
        let check = !self.scores_fit();
        let calc = |i: usize| -> Result<_> {
            let group = i / per;
            let (Some(q), Some(k), Some(v)) = (
                queries[i].values(),
                keys[group].values(),
                transpose[group].as_deref(),
            ) else {
                return Ok(None);
            };
            let narrow = magnitude(q) * magnitude(k) * d as u128 <= i64::MAX as u128;
            let score =
                |r, c| i128::from(factor) * dot_i32(&q[r * d..][..d], &k[c * d..][..d], narrow);
            let witness = soft.eval(bits, score)?;
            let narrow = magnitude(&witness.p) * magnitude(v) * n as u128 <= i64::MAX as u128;
            let product = Rounded::rounded(values[group].shape, self.bits, |r, c| {
                let len = soft.columns(r);
                dot_i32(&witness.p[r * n..][..len], &v[c * n..][..len], narrow)
            })?;
            Ok(Some((witness, product)))
        };
        let mut heads = Vec::with_capacity(queries.len());
        let mut save = |i: usize, w: Option<(_, _)>| -> Result<()> {
            let (w, product) = w.unzip();
            heads.push(Head {
                softmax: soft.record(a, bits, check, w)?,
                product: Rounded::record(a, values[i / per].shape, self.bits, true, product)?,
            });
            Ok(())
        };
        let batch = if x.is_some() && n * n * 20 + soft.shape.len() * 4 <= 32 << 20 {
            2
        } else {
            1
        };
        for i in (0..queries.len()).step_by(batch) {
            if batch == 2 && i + 1 < queries.len() {
                let (left, right) = rayon::join(|| calc(i), || calc(i + 1));
                save(i, left?)?;
                save(i + 1, right?)?;
            } else {
                save(i, calc(i)?)?;
            }
        }
        let shape = Shape::new(n, channels * heads.len())?;
        let joined = join_heads(shape, &heads, channels);
        let output = Rounded::apply(a, shape, self.bits, p.output, joined.as_deref(), true)?;
        Ok(AttentionTrace {
            queries,
            keys,
            values,
            heads,
            output,
        })
    }
    fn reverse(
        self,
        io: &mut Channel<'_, '_>,
        r: &mut Requests,
        reads: &mut Reads<'a>,
        w: &AttentionTrace,
        x: Option<&[i32]>,
        out: Evaluation,
    ) -> Result<Vec<Evaluation>> {
        let p = self.p;
        let c = w.values[0].shape.cols;
        let shape = Shape::new(self.shape.rows, c * w.heads.len())?;
        let per = w.heads.len() / w.keys.len();
        let soft = Softmax::new(shape.rows, p.mask, p.tables)?;
        let (factor, bits) = self.factors();
        let joined = join_heads(shape, &w.heads, c);
        let q = projections(
            io,
            r,
            shape,
            self.bits,
            &[(&w.output, p.output, out)],
            joined.as_deref(),
        )?
        .remove(0);
        drop(joined);
        let (row, col) = shape.point(&q.point);
        let (head, col) = col.split_at(col.len() - c.ilog2() as usize);
        let point = [row, col].concat();
        let head_weights = basis(head);
        let mut total = F::zero();
        let mut requests = Vec::new();
        for (group, (key, value)) in w.keys.iter().zip(&w.values).enumerate() {
            let weights = basis(col);
            let v = value.values().map(|x| {
                let mut v: Vec<_> = x.chunks_exact(c).map(|x| affine_sum(&weights, x)).collect();
                v.resize(shape.rows(), F::zero());
                v
            });
            let mut key_claims = Vec::new();
            for start in (group * per..(group + 1) * per).step_by(2) {
                let count = ((group + 1) * per - start).min(2);
                let y = io.fields(count, || {
                    w.heads[start..start + count]
                        .iter()
                        .map(|h| h.product.shape.eval(h.product.values().unwrap(), &point))
                        .collect()
                })?;
                let terms = io.fork(b"atlas/attention-heads/v1", count, |j, io| {
                    let i = start + j;
                    let h = &w.heads[i];
                    let q = &w.queries[i];
                    let mut local = Vec::new();
                    let mut lookup = Vec::new();
                    let sum = h.product.raw(io, &mut local, &at(&point, y[j]))?;
                    let (score, v) = soft.reverse(
                        io,
                        &mut local,
                        &mut lookup,
                        &h.softmax,
                        sum,
                        row,
                        v.as_deref(),
                        bits,
                    )?;
                    let [q_claim, k_claim] = qk(
                        io,
                        &soft,
                        q.projection.shape,
                        factor,
                        score,
                        q.values().zip(key.values()),
                    )?;
                    let q_claims = q.reverse(io, &mut local, p, self.norm(0), vec![q_claim])?;
                    let v = Evaluation {
                        point: [v.point.as_slice(), col].concat(),
                        value: v.value,
                    };
                    Ok((q_claims, k_claim, v, local, lookup))
                })?;
                for (j, (query, key, v, local, lookup)) in terms.into_iter().enumerate() {
                    let i = start + j;
                    total += head_weights[i] * y[j];
                    r.extend(local);
                    reads.extend(lookup);
                    requests.extend(
                        query
                            .into_iter()
                            .map(|q| (&w.queries[i].projection, p.queries[i], q)),
                    );
                    key_claims.push(key);
                    requests.push((value, p.groups[group][1], v));
                }
            }
            let k = key.reverse(io, r, p, self.norm(1), key_claims)?;
            requests.extend(
                k.into_iter()
                    .map(|q| (&key.projection, p.groups[group][0], q)),
            );
        }
        if total != q.value {
            return Err("invalid Attention concatenation");
        }
        projections(io, r, self.shape, self.bits, &requests, x)
    }
}

fn qk(
    io: &mut Channel<'_, '_>,
    soft: &Softmax<'_>,
    s: Shape,
    factor: i32,
    q: Evaluation,
    input: Option<(&[i32], &[i32])>,
) -> Result<[Evaluation; 2]> {
    let (row, col) = s.point(&q.point);
    let a = basis(row);
    let b = basis(col);
    let f = F::from_i32(factor);
    let n = s.len();
    let d = s.cols;
    let (end, v) = io.fold(
        statement(q.value, s.vars(), 3),
        &[0, 1],
        || {
            let (query, key) = input.unwrap();
            let mut prefixes = vec![F::zero(); n];
            let mut sum = vec![F::zero(); d];
            let mut last = 0;
            for i in 0..s.rows {
                let width = soft.columns(i);
                for j in last..width {
                    for (out, &x) in sum.iter_mut().zip(&key[j * d..][..d]) {
                        *out += b[j] * F::from_i32(x);
                    }
                }
                last = width;
                prefixes[i * d..][..d].copy_from_slice(&sum);
            }
            vec![
                Column::read(n, move |i| s.value(query, i)),
                Column::repeat(n, 1, prefixes),
                Column::repeat(n, d, a),
            ]
        },
        |v| f * v[0] * v[1] * v[2],
    )?;
    let (r, c) = s.point(&end.challenges);
    if end.output_claim != f * v[0] * v[1] * same_point(row, r) {
        return Err("invalid terminal QK");
    }
    let mut row = basis(r);
    row[s.rows..].fill(F::zero());
    if soft.mask == SoftmaxMask::Causal {
        for i in (0..s.rows - 1).rev() {
            let x = row[i + 1];
            row[i] += x;
        }
    } else {
        let sum = row.iter().copied().sum();
        row[..s.rows].fill(sum);
    }
    for (r, w) in row.iter_mut().zip(b) {
        *r *= w;
    }
    let c_weights = basis(c);
    let k = io.dot(v[1], row, || {
        let key = input.unwrap().1;
        let mut out: Vec<_> = key
            .chunks_exact(d)
            .map(|x| affine_sum(&c_weights, x))
            .collect();
        out.resize(s.rows(), F::zero());
        out
    })?;
    let k = Evaluation {
        point: [k.point.as_slice(), c].concat(),
        value: k.value,
    };
    Ok([
        Evaluation {
            point: end.challenges,
            value: v[0],
        },
        k,
    ])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: usize,
    pub logit: i32,
}
#[derive(Clone)]
pub struct TopKConfig {
    pub positions: Range<usize>,
    pub k: usize,
    pub rows_per_block: usize,
    pub vocab_per_block: usize,
}

struct Top<'a> {
    cfg: &'a TopKConfig,
    input: Shape,
    weights: &'a [i32],
    bits: u32,
}
struct TopBounds {
    row: usize,
    col: usize,
    shape: Shape,
    lo: Slot,
    hi: Slot,
}

fn rank(x: &Candidate) -> (i64, usize) {
    (-i64::from(x.logit), x.id)
}

fn aligned(mut range: Range<usize>, max: usize) -> impl Iterator<Item = (usize, usize)> {
    std::iter::from_fn(move || {
        if range.is_empty() {
            return None;
        }
        let first = range.start;
        let mut size = (1 << range.len().ilog2()).min(max);
        if first != 0 {
            size = size.min(1 << first.trailing_zeros());
        }
        range.start += size;
        Some((first, size))
    })
}

impl<'a> Top<'a> {
    fn new(cfg: &'a TopKConfig, input: Shape, weights: &'a [i32], bits: u32) -> Result<Self> {
        if cfg.positions.is_empty()
            || cfg.positions.end > input.rows
            || !(1..=weights.len() / input.cols).contains(&cfg.k)
            || !cfg.rows_per_block.is_power_of_two()
            || !cfg.vocab_per_block.is_power_of_two()
        {
            return Err("invalid Top-K configuration");
        }
        Ok(Self {
            cfg,
            input,
            weights,
            bits,
        })
    }
    fn select(&self, x: &[i32]) -> Result<(Vec<Vec<Candidate>>, Vec<i128>)> {
        let mut candidates = Vec::new();
        let mut raw = Vec::new();
        let cols = self.input.cols;
        for r in self.cfg.positions.clone() {
            let row = &x[r * cols..][..cols];
            let sums: Vec<_> = self
                .weights
                .par_chunks_exact(cols)
                .with_min_len(128)
                .map(|w| dot_i32(w, row, false))
                .collect();
            let mut top = sums
                .iter()
                .enumerate()
                .map(|(id, &v)| {
                    Ok(Candidate {
                        id,
                        logit: i32::try_from(v >> self.bits).map_err(|_| "logit exceeds i32")?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if self.cfg.k < top.len() {
                top.select_nth_unstable_by_key(self.cfg.k, rank);
                top.truncate(self.cfg.k);
            }
            top.sort_unstable_by_key(rank);
            candidates.push(top);
            raw.extend(sums);
        }
        Ok((candidates, raw))
    }
    fn limits(&self, row: &[Candidate], i: usize) -> [i128; 2] {
        let scale = 1i128 << self.bits;
        if let Some(c) = row.iter().find(|c| c.id == i) {
            let x = scale * i128::from(c.logit);
            return [x, x + scale - 1];
        }
        let c = row.last().unwrap();
        [
            scale * i128::from(i32::MIN),
            scale * (i128::from(c.logit) + i128::from(i > c.id)) - 1,
        ]
    }
    fn evaluate(&self, row: &[Candidate], first: usize, c: &[F]) -> [F; 2] {
        let scale = 1i128 << self.bits;
        let last = row.last().unwrap();
        let lo = scale * i128::from(i32::MIN);
        let hi = scale * i128::from(last.logit) - 1;
        let suffix: F = c
            .iter()
            .skip((last.id + 1).saturating_sub(first))
            .copied()
            .sum();
        let mut result = [F::from_i128(lo), F::from_i128(hi) + suffix.times(scale)];
        for cand in row {
            if let Some(&w) = cand.id.checked_sub(first).and_then(|i| c.get(i)) {
                let x = scale * i128::from(cand.logit);
                result[0] += w.times(x - lo);
                result[1] += w.times(x + scale - 1 - hi - scale * i128::from(cand.id > last.id));
            }
        }
        result
    }
    fn record(
        &self,
        a: &mut Advice,
        top: &[Vec<Candidate>],
        raw: Option<&[i128]>,
    ) -> Result<Vec<TopBounds>> {
        let vocab = self.weights.len() / self.input.cols;
        if top.len() != self.cfg.positions.len()
            || top.iter().any(|row| {
                row.len() != self.cfg.k
                    || row.windows(2).any(|v| rank(&v[0]) >= rank(&v[1]))
                    || row
                        .iter()
                        .enumerate()
                        .any(|(i, c)| c.id >= vocab || row[..i].iter().any(|x| x.id == c.id))
            })
        {
            return Err("invalid Top-K candidates");
        }
        let mut shapes = Vec::new();
        let max = self.cfg.vocab_per_block.min(1 << 17);
        for (row, n) in aligned(self.cfg.positions.clone(), self.cfg.rows_per_block) {
            for (col, m) in aligned(0..vocab, max) {
                let shape = Shape::new(n, m)?;
                let gap = |i: usize| {
                    let r = row + i / m - self.cfg.positions.start;
                    let c = col + i % m;
                    let [lo, hi] = self.limits(top[r].as_slice(), c);
                    let x = raw.unwrap()[r * vocab + c];
                    [x - lo, hi - x]
                };
                let w = Width::Unsigned(self.bits as u8 + 32);
                let lo = a.column(shape, w, |i| gap(i)[0])?;
                let hi = a.column(shape, w, |i| gap(i)[1])?;
                shapes.push(TopBounds {
                    row,
                    col,
                    shape,
                    lo,
                    hi,
                });
            }
        }
        Ok(shapes)
    }
    fn reverse(
        &self,
        io: &mut Channel<'_, '_>,
        r: &mut Requests,
        rects: &[TopBounds],
        top: &[Vec<Candidate>],
        values: Option<&[i32]>,
    ) -> Result<Vec<Evaluation>> {
        let mut claims = Vec::new();
        let cols = self.input.cols;
        let project = |x: &[i32], coeff: &[F]| {
            let mut t = vec![F::zero(); self.input.stride()];
            project_rows(&mut t, x, cols, coeff);
            t
        };
        let draw = |io: &mut Channel<'_, '_>, n| if n == 0 { Ok(Vec::new()) } else { io.draw(n) };
        for group in rects.chunk_by(|x, y| x.row == y.row) {
            let start = group[0].row;
            let vars = group[0].shape.row_bits();
            let row = draw(io, vars)?;
            let r_weight = basis(&row);
            let witness = values.map(|x| project(&x[start * cols..], &r_weight));
            let mut terms = Vec::with_capacity(group.len());
            for b in group {
                let c = draw(io, b.shape.vars() - vars)?;
                let point = [row.as_slice(), &c].concat();
                let weight = basis(&c);
                let matrix = project(&self.weights[b.col * cols..], &weight);
                let mut bounds = [F::zero(); 2];
                for (i, &x) in r_weight.iter().enumerate() {
                    let y = &top[b.row + i - self.cfg.positions.start];
                    for (b, v) in bounds.iter_mut().zip(self.evaluate(y, b.col, &weight)) {
                        *b += x * v;
                    }
                }
                let [value] = io.values(|| [inner(witness.as_ref().unwrap(), &matrix)])?;
                ask(r, b.lo, &point, value - bounds[0]);
                ask(r, b.hi, &point, bounds[1] - value);
                terms.push((value, matrix));
            }
            let mut weight = vec![F::zero(); self.input.stride()];
            let mut sum = F::zero();
            for ((v, c), a) in terms.into_iter().zip(io.draw(group.len())?) {
                sum += v * a;
                for (x, v) in weight.iter_mut().zip(c) {
                    *x += a * v;
                }
            }
            let term = io.dot(sum, weight, || witness.unwrap())?;
            let prefix = self.input.row_bits() - vars;
            let index = start / group[0].shape.rows;
            let point = (0..prefix)
                .rev()
                .map(|i| F::from_bool(index & (1 << i) != 0))
                .chain(row)
                .chain(term.point)
                .collect();
            claims.push(Evaluation {
                point,
                value: term.value,
            });
        }
        Ok(claims)
    }
}

const MAGIC: [u8; 8] = *b"ATLASBW1";
const WIRE: &str = "invalid proof record";
const FILE_IO: &str = "proof file operation failed";
const TRANSFORMER: &[u8] = b"atlas/window-bitz/v1";
const LOOKUP: &[u8] = b"atlas/lookup-window/v1";
const OPENING: &[u8] = b"atlas/bitz-window/v1";

fn codec() -> impl Options {
    bincode::DefaultOptions::new().with_fixint_encoding()
}

#[derive(Serialize, Deserialize)]
struct Header {
    widths: Vec<u8>,
    messages: Messages,
    #[serde(with = "serde_bytes")]
    transcript: Vec<u8>,
    roots: Vec<Digest>,
}

struct Archive<W: Write>(BufWriter<W>);

impl<W: Write + Seek> Archive<W> {
    fn new(mut file: W) -> Result<Self> {
        if file.seek(SeekFrom::End(0)).map_err(|_| FILE_IO)? != 0 {
            return Err("proof destination is not empty");
        }
        let mut file = BufWriter::with_capacity(256 << 10, file);
        file.write_all(&[0; 16]).map_err(|_| FILE_IO)?;
        Ok(Self(file))
    }
    fn write(&mut self, p: Proof) -> Result<()> {
        codec().serialize_into(&mut self.0, &p).map_err(|_| FILE_IO)
    }
    fn end(mut self, footer: &Header) -> Result<()> {
        let out = &mut self.0;
        let pos = out.stream_position().map_err(|_| FILE_IO)?;
        codec()
            .serialize_into(&mut *out, footer)
            .map_err(|_| FILE_IO)?;
        out.rewind()
            .and_then(|_| out.write_all(&MAGIC))
            .and_then(|_| out.write_all(&pos.to_le_bytes()))
            .and_then(|_| out.flush())
            .map_err(|_| FILE_IO)
    }
}

fn decode<T: serde::de::DeserializeOwned>(r: &mut Take<impl Read>) -> Result<T> {
    codec()
        .with_limit(r.limit())
        .deserialize_from(r)
        .map_err(|_| WIRE)
}

fn footer(mut r: impl Read + Seek) -> Result<(Header, Take<impl Read>)> {
    let size = r.seek(SeekFrom::End(0)).map_err(|_| FILE_IO)?;
    r.rewind().map_err(|_| FILE_IO)?;
    let mut header = [0; 16];
    r.read_exact(&mut header).map_err(|_| WIRE)?;
    let n = u64::from_le_bytes(header[8..].try_into().unwrap());
    if header[..8] != MAGIC || !(16..=size).contains(&n) {
        return Err(WIRE);
    }
    r.seek(SeekFrom::Start(n)).map_err(|_| FILE_IO)?;
    let mut metadata = (&mut r).take(size - n);
    let header = decode(&mut metadata)?;
    if metadata.limit() != 0 {
        return Err("extra proof metadata");
    }
    r.seek(SeekFrom::Start(16)).map_err(|_| FILE_IO)?;
    Ok((header, r.take(n - 16)))
}

pub struct PublicEmbeddings<'a> {
    pub width: usize,
    pub tokens: &'a [i32],
    pub positions: Option<&'a [i32]>,
    pub segments: Option<&'a [i32]>,
}
pub struct Layer<'a> {
    pub attention: Attention<'a>,
    pub mlp: Mlp<'a>,
    pub attention_norm: NormalizationParameters<'a>,
    pub mlp_norm: NormalizationParameters<'a>,
}
pub enum Architecture<'a> {
    Encoder {
        embedding_norm: NormalizationParameters<'a>,
    },
    Decoder {
        final_norm: NormalizationParameters<'a>,
    },
}

pub struct Model<'a> {
    shape: Shape,
    bits: u32,
    embedding: PublicEmbeddings<'a>,
    layers: Vec<Layer<'a>>,
    head_norms: Vec<Option<[NormalizationParameters<'a>; 2]>>,
    gates: Vec<bool>,
    decoder: bool,
    norm: NormalizationParameters<'a>,
    key: Digest,
}

#[derive(Serialize, Deserialize)]
struct Residual {
    norm: Normalized,
    #[serde(with = "trace_array")]
    y: Option<Vec<i32>>,
    range: Slot,
}
#[derive(Serialize, Deserialize)]
struct LayerTrace {
    attention: AttentionTrace,
    mlp: Feed,
    after_attention: Residual,
    after_mlp: Residual,
}

impl Residual {
    fn record(
        m: &Model<'_>,
        a: &mut Advice,
        p: NormalizationParameters<'_>,
        before: Option<Normalized>,
        x: Option<&[i32]>,
        update: &Rounded,
    ) -> Result<Self> {
        if m.shape != update.shape {
            return Err("invalid residual dimensions");
        }
        let y = x
            .zip(update.values())
            .map(|(a, b)| residual_add(a, b))
            .transpose()?;
        let range = a.column(m.shape, Width::Signed(32), |i| {
            i128::from(y.as_ref().unwrap()[i])
        })?;
        let norm = before.map_or_else(|| m.normalize(a, p, y.as_deref()), Ok)?;
        Ok(Self { norm, y, range })
    }
    fn values(&self, decoder: bool) -> Option<&[i32]> {
        if decoder {
            self.y.as_deref()
        } else {
            self.norm.values()
        }
    }
    fn take(&mut self, decoder: bool) -> Vec<i32> {
        let values = if decoder {
            &mut self.y
        } else {
            &mut self.norm.values
        };
        std::mem::take(values.as_mut().unwrap())
    }
    fn reverse(
        &self,
        io: &mut Channel<'_, '_>,
        req: &mut Requests,
        step: BlockInput<'_>,
        mut claims: Vec<Evaluation>,
        f: impl FnOnce(
            &mut Channel<'_, '_>,
            &mut Requests,
            Option<&[i32]>,
            Evaluation,
        ) -> Result<Vec<Evaluation>>,
    ) -> Result<Vec<Evaluation>> {
        let BlockInput { decoder, p, x } = step;
        let q = if decoder {
            claims.pop().unwrap()
        } else {
            self.norm
                .reverse(io, req, p, self.y.as_deref(), &claims, &[])?
        };
        ask(req, self.range, &q.point, q.value);
        let [skip] = io.values(|| [self.norm.shape.eval(x.unwrap(), &q.point)])?;
        let carry = at(&q.point, skip);
        let update = Evaluation {
            point: q.point,
            value: q.value - skip,
        };
        let input = if decoder { self.norm.values() } else { x };
        let mut q = f(io, req, input, update)?;
        if decoder {
            Ok(vec![self.norm.reverse(io, req, p, x, &q, &[carry])?])
        } else {
            q.push(carry);
            Ok(q)
        }
    }
}

fn residual_add(a: &[i32], b: &[i32]) -> Result<Vec<i32>> {
    if a.len() != b.len() {
        return Err("invalid residual lengths");
    }
    a.iter()
        .zip(b)
        .map(|(&a, &b)| a.checked_add(b).ok_or("residual exceeds i32"))
        .collect()
}

struct BlockInput<'a> {
    decoder: bool,
    p: NormalizationParameters<'a>,
    x: Option<&'a [i32]>,
}
enum Saved {
    InMemory(Box<LayerTrace>),
    OnDisk(usize, usize),
}
struct Run {
    advice: Advice,
    early: Option<EarlyCommit>,
    input: Vec<i32>,
    norm: Normalized,
    layers: Vec<Saved>,
    map: Option<memmap2::Mmap>,
}

mod trace_array {
    use bytemuck::{allocation::pod_collect_to_vec, cast_slice, Pod};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use serde_bytes::Bytes;

    // Layer traces are temporary files read by the same process.
    pub(super) fn bytes<T: Pod>(values: &[T]) -> &Bytes {
        Bytes::new(cast_slice(values))
    }

    pub(super) fn values<T: Pod>(bytes: &Bytes) -> Option<Vec<T>> {
        bytes
            .len()
            .is_multiple_of(size_of::<T>())
            .then(|| pod_collect_to_vec(bytes.as_ref()))
    }

    pub(super) fn serialize<T: Pod, S: Serializer>(
        values: &Option<Vec<T>>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        values.as_deref().map(bytes).serialize(s)
    }

    pub(super) fn deserialize<'de, T: Pod, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Vec<T>>, D::Error> {
        Option::<&Bytes>::deserialize(d)?
            .map(|bytes| {
                values(bytes).ok_or_else(|| serde::de::Error::custom("invalid trace array length"))
            })
            .transpose()
    }
}

impl Run {
    fn read<'a, T: Deserialize<'a>>(&'a self, offset: usize) -> Result<T> {
        codec()
            .allow_trailing_bytes()
            .deserialize(&self.map.as_ref().unwrap()[offset..])
            .map_err(|_| "read layer trace")
    }
    fn output(&self, i: usize, decoder: bool) -> Result<Option<Cow<'_, [i32]>>> {
        match &self.layers[i] {
            Saved::InMemory(v) => Ok(v.after_mlp.values(decoder).map(Cow::Borrowed)),
            Saved::OnDisk(at, _) => trace_array::values(self.read(*at)?)
                .map(Cow::Owned)
                .map(Some)
                .ok_or("invalid layer output length"),
        }
    }
}

fn save_layer(file: &mut File, hidden: &[i32], layer: &LayerTrace) -> Result<Saved> {
    let _span = tracing::info_span!("atlas trace layer write").entered();
    let mut out = BufWriter::new(file);
    let output = out.stream_position().map_err(|_| "seek layer file")? as usize;
    codec()
        .serialize_into(&mut out, trace_array::bytes(hidden))
        .map_err(|_| "write layer output")?;
    let record = out.stream_position().map_err(|_| "seek layer file")? as usize;
    codec()
        .serialize_into(&mut out, layer)
        .map_err(|_| "write layer trace")?;
    out.flush().map_err(|_| "flush layer trace")?;
    Ok(Saved::OnDisk(output, record))
}

#[derive(Clone, Copy)]
enum Target<'a> {
    Hidden(&'a [i32]),
    Top(&'a TopKConfig, &'a [Vec<Candidate>]),
}
struct Statement<'a> {
    ids: &'a [usize],
    segments: Option<&'a [usize]>,
    target: Target<'a>,
}

fn integers(t: &mut impl Transcript, x: &[i32]) {
    t.append(&(x.len() as u64));
    let mut bytes = [0u8; 4096];
    for chunk in x.chunks(1024) {
        let bytes = &mut bytes[..4 * chunk.len()];
        for (dest, &v) in bytes.chunks_exact_mut(4).zip(chunk) {
            dest.copy_from_slice(&v.to_le_bytes());
        }
        t.append_bytes(bytes);
    }
}

impl Statement<'_> {
    fn append(&self, t: &mut impl Transcript, key: &Digest, widths: &[u8]) {
        t.append_bytes(key);
        for &i in self.ids {
            t.append(&(i as u64));
        }
        t.append(&(self.segments.is_some() as u64));
        for &i in self.segments.unwrap_or_default() {
            t.append(&(i as u64));
        }
        match self.target {
            Target::Hidden(v) => {
                t.append(&0u64);
                integers(t, v);
            }
            Target::Top(cfg, candidates) => {
                t.append(&[
                    1,
                    cfg.positions.start as u64,
                    cfg.positions.end as u64,
                    cfg.k as u64,
                    cfg.rows_per_block as u64,
                    cfg.vocab_per_block as u64,
                ]);
                for c in candidates.iter().flatten() {
                    t.append(&[c.id as u64, c.logit as u64]);
                }
            }
        }
        t.append_bytes(widths);
        t.append(&(WINDOW as u64));
    }
}

impl<'a> Model<'a> {
    pub fn new(
        rows: usize,
        bits: u8,
        embedding: PublicEmbeddings<'a>,
        layers: Vec<Layer<'a>>,
        arch: Architecture<'a>,
    ) -> Result<Self> {
        Self::with_heads(rows, bits, embedding, layers, arch, Vec::new())
    }
    fn with_heads(
        rows: usize,
        bits: u8,
        embedding: PublicEmbeddings<'a>,
        layers: Vec<Layer<'a>>,
        arch: Architecture<'a>,
        head_norms: Vec<Option<[NormalizationParameters<'a>; 2]>>,
    ) -> Result<Self> {
        let shape = Shape::new(rows, embedding.width)?;
        if bits > 15
            || layers.is_empty()
            || embedding.tokens.is_empty()
            || embedding.tokens.len() % shape.cols != 0
            || head_norms.len() > layers.len()
        {
            return Err("invalid model dimensions");
        }
        let (decoder, norm) = match arch {
            Architecture::Encoder { embedding_norm } => (false, embedding_norm),
            Architecture::Decoder { final_norm } if embedding.segments.is_none() => {
                (true, final_norm)
            }
            _ => return Err("decoder does not use segment embeddings"),
        };
        let bits = u32::from(bits);
        let gates = layers
            .iter()
            .map(|l| {
                decoder
                    && l.mlp
                        .gate
                        .is_some_and(|g| bounded_gate(l.mlp_norm, g, bits))
            })
            .collect();
        let mut model = Self {
            shape,
            bits,
            embedding,
            layers,
            head_norms,
            gates,
            decoder,
            norm,
            key: [0; 32],
        };
        model.key = model.hash();
        Ok(model)
    }
    fn hash(&self) -> Digest {
        let tag = protocol_tag(b"atlas/transformer/model/v2");
        let mut t = prover_transcript::<Keccak>(&tag, b"");
        let norm = |t: &mut Writer, p: NormalizationParameters<'_>| {
            t.append(&[p.kind as u64, p.epsilon_variance_units as u64]);
            integers(t, p.gain);
            integers(t, p.bias);
        };
        let tensor = |t: &mut Writer, p: Projection<'_>| {
            integers(t, p.weights);
            integers(t, p.bias);
        };
        let [rows, cols] = [self.shape.rows, self.shape.cols].map(|n| n as u64);
        t.append(&[
            rows,
            cols,
            self.bits as u64,
            self.decoder as u64,
            self.layers.len() as u64,
        ]);
        let e = &self.embedding;
        for i in [Some(e.tokens), e.positions, e.segments] {
            integers(&mut t, i.unwrap_or_default());
        }
        norm(&mut t, self.norm);
        for (i, l) in self.layers.iter().enumerate() {
            norm(&mut t, l.attention_norm);
            norm(&mut t, l.mlp_norm);
            let a = &l.attention;
            t.append(&[
                a.mask as u64,
                a.factor as u64,
                a.tables.limit as u64,
                a.queries.len() as u64,
                a.groups.len() as u64,
            ]);
            for &p in a
                .queries
                .iter()
                .chain(a.groups.iter().flatten())
                .chain([&a.output])
            {
                tensor(&mut t, p);
            }
            integers(&mut t, a.tables.probabilities);
            integers(&mut t, a.tables.sums);
            t.append(&(a.rotary.is_some() as u64));
            if let Some(r) = a.rotary {
                integers(&mut t, r.cos);
                integers(&mut t, r.sin);
            }
            let head = self.head_norms.get(i).copied().flatten();
            t.append(&(head.is_some() as u64));
            if let Some([q, k]) = head {
                norm(&mut t, q);
                norm(&mut t, k);
            }
            let m = l.mlp;
            tensor(&mut t, m.up);
            tensor(&mut t, m.down);
            t.append(&(m.gate.is_some() as u64));
            if let Some(g) = m.gate {
                tensor(&mut t, g);
            }
            t.append(&(m.activation.lower as u64));
            integers(&mut t, m.activation.correction);
        }
        t.verifier_message_as(32, digest)
    }
    pub fn is_decoder(&self) -> bool {
        self.decoder
    }
    pub fn cls<'b>(&self, output: &'b [i32]) -> Result<&'b [i32]> {
        if self.decoder || output.len() != self.shape.cells() {
            return Err("invalid encoder output");
        }
        Ok(&output[..self.shape.cols])
    }
    fn input(&self, tokens: &[usize], segments: Option<&[usize]>) -> Result<Vec<i32>> {
        let e = &self.embedding;
        if tokens.len() != self.shape.rows
            || segments.is_some_and(|s| s.len() != tokens.len() || e.segments.is_none())
        {
            return Err("invalid token sequence");
        }
        let cols = self.shape.cols;
        let row = |table: &'a [i32], i| {
            table
                .chunks_exact(cols)
                .nth(i)
                .ok_or("invalid embedding index")
        };
        let mut output = Vec::with_capacity(self.shape.cells());
        for (i, &id) in tokens.iter().enumerate() {
            let token = row(e.tokens, id)?;
            let position = e.positions.map(|p| row(p, i)).transpose()?;
            let segment = e
                .segments
                .map(|p| row(p, segments.map_or(0, |s| s[i])))
                .transpose()?;
            for (c, &v) in token.iter().enumerate() {
                let x = i64::from(v)
                    + position.map_or(0, |v| i64::from(v[c]))
                    + segment.map_or(0, |v| i64::from(v[c]));
                output.push(i32::try_from(x).map_err(|_| "embedding exceeds i32")?);
            }
        }
        Ok(output)
    }
    fn attention(&self, i: usize) -> AttentionPlan<'_> {
        AttentionPlan {
            p: &self.layers[i].attention,
            shape: self.shape,
            bits: self.bits,
            norm: self.head_norms.get(i).copied().flatten(),
        }
    }
    fn normalize(
        &self,
        a: &mut Advice,
        p: NormalizationParameters<'_>,
        x: Option<&[i32]>,
    ) -> Result<Normalized> {
        Normalized::record(a, self.shape, self.bits, p, x)
    }
    fn forward_layer(&self, i: usize, a: &mut Advice, x: Option<&[i32]>) -> Result<LayerTrace> {
        let p = &self.layers[i];
        let pre = |a: &mut Advice, n, x: Option<&[i32]>| {
            self.decoder.then(|| self.normalize(a, n, x)).transpose()
        };
        let norm = pre(a, p.attention_norm, x)?;
        let attention = self
            .attention(i)
            .record(a, norm.as_ref().map_or(x, Normalized::values))?;
        let after_attention =
            Residual::record(self, a, p.attention_norm, norm, x, &attention.output)?;
        let x = after_attention.values(self.decoder);
        let norm = pre(a, p.mlp_norm, x)?;
        let y = norm.as_ref().map_or(x, Normalized::values);
        let mlp = Feed::record(a, self.shape, self.bits, p.mlp, y, self.gates[i])?;
        let after_mlp = Residual::record(self, a, p.mlp_norm, norm, x, &mlp.output)?;
        Ok(LayerTrace {
            attention,
            mlp,
            after_attention,
            after_mlp,
        })
    }
    fn forward(&self, ids: &[usize], segments: Option<&[usize]>, prove: bool) -> Result<Run> {
        let input = self.input(ids, segments)?;
        let _span = tracing::info_span!("atlas forward").entered();
        let mut advice = Advice::new(prove)?;
        let x = prove.then_some(input.as_slice());
        let mut norm = (!self.decoder)
            .then(|| self.normalize(&mut advice, self.norm, x))
            .transpose()?;
        let mut file = (prove && self.layers.len() > 1 && self.shape.rows >= 128)
            .then(tempfile::tempfile)
            .transpose()
            .map_err(|_| "create layer file")?;
        let mut early = (file.is_some() && self.layers.len() > WINDOW && can_overlap())
            .then(|| EarlyCommit::new((rayon::current_num_threads() - 1).min(3)))
            .transpose()?;
        let mut records = Vec::with_capacity(self.layers.len());
        let mut hidden = Vec::new();
        advice.cut();
        for i in 0..self.layers.len() {
            let x = match records.last() {
                None => norm.as_ref().map_or(x, Normalized::values),
                Some(Saved::OnDisk(..)) => Some(hidden.as_slice()),
                Some(Saved::InMemory(layer)) => layer.after_mlp.values(self.decoder),
            };
            let mut layer = self.forward_layer(i, &mut advice, x)?;
            let saved = if let Some(file) = file.as_mut().filter(|_| i + 1 < self.layers.len()) {
                hidden = layer.after_mlp.take(self.decoder);
                save_layer(file, &hidden, &layer)?
            } else {
                Saved::InMemory(Box::new(layer))
            };
            records.push(saved);
            advice.cut();
            if let Some(early) = &mut early {
                early.boundary(&mut advice, i + 1, self.layers.len())?;
            }
        }
        let norm = if let Some(norm) = norm.take() {
            norm
        } else {
            let Saved::InMemory(layer) = records.last().unwrap() else {
                unreachable!()
            };
            self.normalize(&mut advice, self.norm, layer.after_mlp.values(true))?
        };
        // The private file is complete and no writable handle is retained.
        let map = file
            .map(|f| unsafe { memmap2::Mmap::map(&f) })
            .transpose()
            .map_err(|_| "map layer file")?;
        Ok(Run {
            advice,
            early,
            input,
            norm,
            layers: records,
            map,
        })
    }
    fn top<'b>(&'b self, c: &'b TopKConfig) -> Result<Top<'b>> {
        Top::new(c, self.shape, self.embedding.tokens, self.bits)
    }
    pub fn prove_to(
        &self,
        ids: &[usize],
        segments: Option<&[usize]>,
        file: impl Write + Seek + Send,
    ) -> Result<Vec<i32>> {
        if self.decoder {
            return Err("decoder requires Top-K");
        }
        let mut run = self.forward(ids, segments, true)?;
        let output = run
            .output(run.layers.len() - 1, false)?
            .unwrap()
            .into_owned();
        let stmt = Statement {
            ids,
            segments,
            target: Target::Hidden(&output),
        };
        self.write(stmt, &mut run, &[], Archive::new(file)?)?;
        Ok(output)
    }
    pub fn prove_top_k_to(
        &self,
        ids: &[usize],
        cfg: &TopKConfig,
        file: impl Write + Seek + Send,
    ) -> Result<Vec<Vec<Candidate>>> {
        if !self.decoder {
            return Err("Top-K requires decoder");
        }
        let file = Archive::new(file)?;
        let top = self.top(cfg)?;
        let mut run = self.forward(ids, None, true)?;
        let (result, raw) = top.select(run.norm.values().unwrap())?;
        let bounds = top.record(&mut run.advice, &result, Some(&raw))?;
        drop(raw);
        let stmt = Statement {
            ids,
            segments: None,
            target: Target::Top(cfg, &result),
        };
        self.write(stmt, &mut run, &bounds, file)?;
        Ok(result)
    }
    pub fn verify_from(
        &self,
        ids: &[usize],
        segments: Option<&[usize]>,
        output: &[i32],
        file: impl Read + Seek,
    ) -> Result<()> {
        self.cls(output)?;
        self.verify(
            Statement {
                ids,
                segments,
                target: Target::Hidden(output),
            },
            file,
        )
    }
    pub fn verify_top_k_from(
        &self,
        ids: &[usize],
        cfg: &TopKConfig,
        output: &[Vec<Candidate>],
        file: impl Read + Seek,
    ) -> Result<()> {
        if !self.decoder {
            return Err("Top-K requires decoder");
        }
        self.verify(
            Statement {
                ids,
                segments: None,
                target: Target::Top(cfg, output),
            },
            file,
        )
    }
}

struct Reverse<'a> {
    m: &'a Model<'a>,
    run: &'a Run,
    claims: Vec<Evaluation>,
    requests: Requests,
    next: Option<LayerTrace>,
    output: Option<Cow<'a, [i32]>>,
}
type Job<'a> = (usize, Digest, Reads<'a>, Requests);
type Prepared = (usize, Digest, WindowProof, Vec<Requests>);

impl<'a> Reverse<'a> {
    fn new(
        m: &'a Model<'a>,
        run: &'a Run,
        target: Target<'_>,
        bounds: &[TopBounds],
        io: &mut Channel<'_, '_>,
    ) -> Result<Self> {
        let mut requests = Vec::new();
        let claims = match target {
            Target::Hidden(x) => {
                let point = io.draw(m.shape.vars())?;
                vec![Evaluation {
                    value: m.shape.eval(x, &point),
                    point,
                }]
            }
            Target::Top(cfg, top) => {
                m.top(cfg)?
                    .reverse(io, &mut requests, bounds, top, run.norm.values())?
            }
        };
        let last = run.output(run.layers.len() - 1, m.decoder)?;
        let mut walk = Self {
            m,
            run,
            claims,
            requests,
            next: None,
            output: None,
        };
        if m.decoder {
            walk.norm(io, last.as_deref())?;
        } else {
            walk.output = last;
        }
        Ok(walk)
    }
    fn norm(&mut self, io: &mut Channel<'_, '_>, x: Option<&[i32]>) -> Result<()> {
        self.claims = vec![self.run.norm.reverse(
            io,
            &mut self.requests,
            self.m.norm,
            x,
            &self.claims,
            &[],
        )?];
        Ok(())
    }
    fn window(&mut self, io: &mut Channel<'_, '_>, plan: &Plan, i: usize) -> Result<Job<'a>> {
        let span = plan.windows[i].layers.clone();
        let mut reads = Vec::new();
        for layer in span.clone().rev() {
            self.layer(io, layer, &mut reads)?;
        }
        if span.start == 0 {
            if !self.m.decoder {
                self.norm(io, Some(&self.run.input))?;
            }
            if self
                .claims
                .iter()
                .any(|q| self.m.shape.eval(&self.run.input, &q.point) != q.value)
            {
                return Err("invalid public embedding evaluation");
            }
            tracing::info_span!("atlas chain end").in_scope(|| ());
        }
        io.append(&[span.start as u64, span.end as u64]);
        let key = io.key();
        let q = plan.split(i, &mut self.requests);
        if span.start == 0 && !self.requests.is_empty() {
            return Err("unused witness requests");
        }
        Ok((i, key, reads, q))
    }
    fn layer(&mut self, io: &mut Channel<'_, '_>, i: usize, reads: &mut Reads<'a>) -> Result<()> {
        let m = self.m;
        let run = self.run;
        let mut owned;
        let layer = match &run.layers[i] {
            Saved::InMemory(layer) => layer.as_ref(),
            Saved::OnDisk(_, offset) => {
                owned = self.next.take().map_or_else(|| run.read(*offset), Ok)?;
                if let Some(x) = self.output.take() {
                    owned.after_mlp.norm.values = Some(x.into_owned());
                }
                &owned
            }
        };
        self.next = std::thread::scope(|s| -> Result<_> {
            let prev = i.checked_sub(1);
            let next = match prev.map(|i| &run.layers[i]) {
                Some(Saved::OnDisk(_, at)) => Some(s.spawn(|| run.read(*at))),
                _ => None,
            };
            let previous = prev
                .map(|i| run.output(i, m.decoder))
                .transpose()?
                .flatten();
            let input = if i != 0 {
                previous.as_deref()
            } else if m.decoder {
                Some(run.input.as_slice())
            } else {
                run.norm.values()
            };
            let config = &m.layers[i];
            let gate = BlockInput {
                decoder: m.decoder,
                p: config.mlp_norm,
                x: layer.after_attention.values(m.decoder),
            };
            let claims = std::mem::take(&mut self.claims);
            let forward = |io: &mut Channel<'_, '_>, r: &mut Requests, x: Option<&[i32]>, q| {
                tracing::info_span!("atlas mlp reverse")
                    .in_scope(|| layer.mlp.reverse(io, r, reads, config.mlp, m.shape, x, q))
            };
            let claims = layer
                .after_mlp
                .reverse(io, &mut self.requests, gate, claims, forward)?;
            let gate = BlockInput {
                decoder: m.decoder,
                p: config.attention_norm,
                x: input,
            };
            self.claims = layer.after_attention.reverse(
                io,
                &mut self.requests,
                gate,
                claims,
                |io, r, x, q| {
                    tracing::info_span!("atlas attention reverse")
                        .in_scope(|| m.attention(i).reverse(io, r, reads, &layer.attention, x, q))
                },
            )?;
            if !m.decoder {
                self.output = previous;
            }
            next.map(|job| job.join().map_err(|_| "layer prefetch failed")?)
                .transpose()
        })?;
        Ok(())
    }
}

fn lookup_window(
    plan: &Plan,
    a: &Advice,
    (i, key, reads, mut requests): Job<'_>,
) -> Result<Prepared> {
    let _span = tracing::info_span!("atlas lookup").entered();
    let tag = protocol_tag(LOOKUP);
    let mut t = prover_transcript::<Keccak>(&tag, &key);
    let mut m = Messages::default();
    finish_reads(&mut Channel::Send(&mut t, &mut m, a), reads, &mut requests)?;
    let key = t.verifier_message_as(32, digest);
    let window = WindowProof(m.0, m.1, t.narg_string().to_vec(), Vec::new());
    Ok((i, key, window, plan.group(i, requests)?))
}

fn open_window<W: Write + Seek>(
    plan: &Plan,
    a: &Advice,
    roots: &[Digest],
    file: &mut Archive<W>,
    (i, key, mut proof, requests): Prepared,
) -> Result<(WindowProof, Digest)> {
    let _span = tracing::info_span!("atlas open").entered();
    let tag = protocol_tag(OPENING);
    let mut t = prover_transcript::<Keccak>(&tag, &key);
    let window = &plan.windows[i];
    window.prove(a, &roots[window.roots.clone()], requests, &mut t, |proof| {
        tracing::info_span!("atlas group complete").in_scope(|| ());
        tracing::info_span!("atlas bitz write").in_scope(|| file.write(proof))
    })?;
    proof.3 = t.narg_string().to_vec();
    Ok((proof, t.verifier_message_as(32, digest)))
}

impl Model<'_> {
    fn write(
        &self,
        stmt: Statement<'_>,
        run: &mut Run,
        bounds: &[TopBounds],
        mut file: Archive<impl Write + Seek + Send>,
    ) -> Result<()> {
        let tag = protocol_tag(TRANSFORMER);
        let mut t = prover_transcript::<Keccak>(&tag, b"");
        let widths = run.advice.widths();
        let plan = Plan::new(&run.advice)?;
        stmt.append(&mut t, &self.key, &widths);
        let mut roots = Vec::with_capacity(plan.roots);
        tracing::info_span!("atlas commit").in_scope(|| {
            run.advice.seal()?;
            let mut early = run
                .early
                .take()
                .map(EarlyCommit::finish)
                .transpose()?
                .unwrap_or_default();
            for w in &plan.windows {
                let window = if let Some(roots) = early.remove(&w.layers.start) {
                    if roots.len() != w.groups.len() {
                        return Err("wrong number of initial commitments");
                    }
                    roots
                } else {
                    w.groups
                        .iter()
                        .map(|g| g.commit(&run.advice).map(|c| c.root))
                        .collect::<Result<Vec<_>>>()?
                };
                for root in window {
                    t.append_bytes(&root);
                    roots.push(root);
                }
            }
            Ok(())
        })?;
        let advice = &run.advice;
        let mut m = Messages::default();
        let mut io = Channel::Send(&mut t, &mut m, advice);
        let _span = tracing::info_span!("atlas reduce").entered();
        let mut reverse = Reverse::new(self, run, stmt.target, bounds, &mut io)?;
        let mut roots_after = Vec::with_capacity(plan.windows.len());
        let mut append = |io: &mut Channel<'_, '_>, (proof, root)| {
            if let Channel::Send(_, m, _) = io {
                m.2.push(proof);
            }
            roots_after.push(root);
        };
        let plan = &plan;
        let roots_ref = &roots;
        let lookup = |j| lookup_window(plan, advice, j);
        let mut open = |j| open_window(plan, advice, roots_ref, &mut file, j);
        if plan.windows.len() > 1 && can_overlap() {
            std::thread::scope(|s| -> Result<()> {
                let (submit, jobs) = mpsc::sync_channel(1);
                let (provide, ready) = mpsc::sync_channel::<Result<Prepared>>(0);
                let (send, done) = mpsc::channel();
                s.spawn(move || {
                    for j in jobs {
                        if provide.send(lookup(j)).is_err() {
                            break;
                        }
                    }
                });
                s.spawn(move || {
                    for p in ready {
                        let result = p.and_then(&mut open);
                        let fail = result.is_err();
                        if send.send(result).is_err() || fail {
                            break;
                        }
                    }
                });
                for i in 0..plan.windows.len() {
                    let job = reverse.window(&mut io, plan, i)?;
                    for result in done.try_iter() {
                        append(&mut io, result?);
                    }
                    if tracing::info_span!("atlas pipeline backpressure")
                        .in_scope(|| submit.send(job))
                        .is_err()
                    {
                        return Err(done
                            .recv()
                            .ok()
                            .and_then(Result::err)
                            .unwrap_or("Lookup worker stopped"));
                    }
                }
                drop(submit);
                for result in done {
                    append(&mut io, result?);
                }
                Ok(())
            })?;
        } else {
            for i in 0..plan.windows.len() {
                let job = reverse.window(&mut io, plan, i)?;
                append(&mut io, open(lookup(job)?)?);
            }
        }
        drop(_span);
        if roots_after.len() != plan.windows.len() {
            return Err("incomplete opening pipeline");
        }
        for root in roots_after {
            io.append(&root);
        }
        file.end(&Header {
            widths,
            messages: m,
            transcript: t.narg_string().to_vec(),
            roots,
        })
    }
    fn verify(&self, stmt: Statement<'_>, file: impl Read + Seek) -> Result<()> {
        let (header, mut proofs) = footer(file)?;
        let mut run = self.forward(stmt.ids, stmt.segments, false)?;
        let bounds = match stmt.target {
            Target::Top(cfg, out) => self.top(cfg)?.record(&mut run.advice, out, None)?,
            _ => Vec::new(),
        };
        run.advice.check_widths(&header.widths)?;
        let plan = Plan::new(&run.advice)?;
        if header.roots.len() != plan.roots {
            return Err("invalid number of commitments");
        }
        let tag = protocol_tag(TRANSFORMER);
        let mut t = verifier_transcript::<Keccak>(&tag, b"", &header.transcript);
        stmt.append(&mut t, &self.key, &header.widths);
        for root in &header.roots {
            t.append_bytes(root);
        }
        let messages = &header.messages;
        let mut io = Channel::Check(&mut t, messages.0.cursor(&messages.1, &messages.2));
        let mut reverse = Reverse::new(self, &run, stmt.target, &bounds, &mut io)?;
        let mut after = Vec::with_capacity(plan.windows.len());
        let lookup = protocol_tag(LOOKUP);
        let pcs = protocol_tag(OPENING);
        for (i, window) in plan.windows.iter().enumerate() {
            let (_, key, reads, mut requests) = reverse.window(&mut io, &plan, i)?;
            let Channel::Check(_, (_, _, _, ws)) = &mut io else {
                unreachable!()
            };
            let WindowProof(tape, forks, bytes, bitz) = &take(ws, 1)?[0];
            let mut l = verifier_transcript::<Keccak>(&lookup, &key, bytes);
            let mut child = Channel::Check(&mut l, tape.cursor(forks, &[]));
            finish_reads(&mut child, reads, &mut requests)?;
            child.used()?;
            let key = l.verifier_message_as(32, digest);
            l.check_eof().map_err(|_| "extra Lookup transcript")?;
            let mut p = verifier_transcript::<Keccak>(&pcs, &key, bitz);
            let roots = &header.roots[window.roots.clone()];
            for ((g, &root), q) in window
                .groups
                .iter()
                .zip(roots)
                .zip(plan.group(i, requests)?)
            {
                g.verify(root, q, decode(&mut proofs)?, &mut p)?;
            }
            after.push(p.verifier_message_as(32, digest));
            p.check_eof().map_err(|_| "extra BitZ transcript")?;
        }
        for r in after {
            io.append(&r);
        }
        io.used()?;
        if proofs.limit() != 0 {
            return Err("extra opening proofs");
        }
        t.check_eof().map_err(|_| "extra proof transcript")
    }
}

pub mod models {
    pub use super::{Architecture, Layer, Model, PublicEmbeddings};
}

pub mod layers {
    pub use super::{
        Activation, Attention, Mlp, NormalizationKind, NormalizationParameters, Projection, Rotary,
        SoftmaxMask, SoftmaxTables,
    };
}

pub mod protocol {
    pub use super::{Candidate, TopKConfig};
    pub type Result<T> = super::Result<T>;
}

pub mod norm {
    pub use super::{Claim, Width};
}

pub mod field {
    pub use super::F;
}

pub mod projection {
    pub use super::Projection;

    #[derive(Clone, Copy)]
    pub struct Input {
        pub rows: usize,
        pub width: usize,
        pub bits: u32,
    }

    impl Input {
        pub fn execute(self, x: &[i32], p: Projection<'_>) -> super::Result<(Vec<i32>, Vec<u32>)> {
            if self.bits > 30 {
                return Err("invalid projection precision");
            }
            super::project(x, super::Shape::new(self.rows, self.width)?, self.bits, p)
        }
    }
}

pub mod residual {
    pub fn add(a: &[i32], b: &[i32]) -> super::Result<Vec<i32>> {
        super::residual_add(a, b)
    }
}

#[cfg(feature = "safetensors-weights")]
pub mod weights {
    use super::{
        Activation, Architecture, Attention, Layer, Mlp, Model, NormalizationKind,
        NormalizationParameters, Projection, PublicEmbeddings, Rotary, SoftmaxMask, SoftmaxTables,
    };
    use atlas_trace::{
        semantics::{
            softmax,
            tables::{table, IndexRule, TableKind},
        },
        Scale, SemVersion,
    };
    use safetensors::{Dtype, SafeTensors};
    use serde_json::Value;
    use std::{fs, path::Path, sync::OnceLock};

    type Load<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    type Affine = (Vec<i32>, Vec<i32>);

    #[derive(Copy, Clone, PartialEq, Eq)]
    enum Family {
        Gpt,
        Bert,
        Qwen2,
        Qwen3,
    }
    use Family::*;

    struct Names {
        prefix: &'static str,
        embed: &'static str,
        position: &'static str,
        segment: &'static str,
        head: &'static str,
        blocks: &'static str,
        qkv: [&'static str; 3],
        output: &'static str,
        feed: [&'static str; 2],
        norm: [&'static str; 2],
    }

    const GPT: Names = Names {
        prefix: "transformer.",
        embed: "wte",
        position: "wpe",
        segment: "",
        head: "ln_f",
        blocks: "h",
        qkv: ["attn.c_attn", "", ""],
        output: "attn.c_proj",
        feed: ["mlp.c_fc", "mlp.c_proj"],
        norm: ["ln_1", "ln_2"],
    };
    const BERT: Names = Names {
        prefix: "bert.",
        embed: "embeddings.word_embeddings",
        position: "embeddings.position_embeddings",
        segment: "embeddings.token_type_embeddings",
        head: "embeddings.LayerNorm",
        blocks: "encoder.layer",
        qkv: [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
        ],
        output: "attention.output.dense",
        feed: ["intermediate.dense", "output.dense"],
        norm: ["attention.output.LayerNorm", "output.LayerNorm"],
    };
    const QWEN: Names = Names {
        prefix: "model.",
        embed: "embed_tokens",
        position: "",
        segment: "",
        head: "norm",
        blocks: "layers",
        qkv: ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        output: "self_attn.o_proj",
        feed: ["mlp.up_proj", "mlp.down_proj"],
        norm: ["input_layernorm", "post_attention_layernorm"],
    };

    struct LayerData {
        attention: [Affine; 4],
        mlp: [Affine; 2],
        norm: [Affine; 2],
        qk: Option<[Affine; 2]>,
        gate: Option<Affine>,
    }
    pub struct ModelWeights {
        family: Family,
        bits: u8,
        width: usize,
        head: usize,
        limit: usize,
        epsilon: i64,
        theta: Option<f64>,
        tokens: Vec<i32>,
        positions: Option<Vec<i32>>,
        segments: Option<Vec<i32>>,
        layers: Vec<LayerData>,
        norm: Affine,
        activation: (i32, Vec<i32>),
        soft: softmax::SoftmaxTables,
        rotary: Vec<OnceLock<Affine>>,
    }

    struct Weights<'a> {
        content: SafeTensors<'a>,
        scale: f64,
    }

    impl Weights<'_> {
        fn read(&self, key: &str, shape: &[usize]) -> Load<Vec<i32>> {
            let tensor = self.content.tensor(key)?;
            if tensor.shape() != shape {
                return Err(format!("wrong tensor shape: {key}").into());
            }
            let (n, convert): (usize, fn(&[u8]) -> f64) = match tensor.dtype() {
                Dtype::BF16 => (2, |b| {
                    half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f64()
                }),
                Dtype::F16 => (2, |b| {
                    half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f64()
                }),
                Dtype::F32 => (4, |b| f32::from_le_bytes(b.try_into().unwrap()) as f64),
                Dtype::F64 => (8, |b| f64::from_le_bytes(b.try_into().unwrap())),
                _ => return Err(format!("unsupported weight encoding: {key}").into()),
            };
            tensor
                .data()
                .chunks_exact(n)
                .map(|b| {
                    let value = (convert(b) * self.scale).round();
                    if !(i32::MIN as f64..=i32::MAX as f64).contains(&value) {
                        return Err(format!("weight exceeds i32: {key}").into());
                    }
                    Ok(value as i32)
                })
                .collect()
        }
        fn affine(&self, key: &str, shape: &[usize], bias: bool, transpose: bool) -> Load<Affine> {
            let n = shape[0];
            let w = if transpose {
                let src = self.read(&format!("{key}.weight"), &[shape[1], n])?;
                (0..n)
                    .flat_map(|i| src[i..].iter().step_by(n).copied())
                    .collect()
            } else {
                self.read(&format!("{key}.weight"), shape)?
            };
            let b = if bias {
                self.read(&format!("{key}.bias"), &[n])?
            } else {
                vec![0; n]
            };
            Ok((w, b))
        }
    }

    impl ModelWeights {
        pub fn from_safetensors(dir: &Path, bits: u8) -> Load<Self> {
            if bits > 15 {
                return Err("unsupported fractional precision".into());
            }
            let config: Value = serde_json::from_slice(&fs::read(dir.join("config.json"))?)?;
            let (family, names) = match config["model_type"].as_str() {
                Some("gpt2") => (Gpt, GPT),
                Some("bert") => (Bert, BERT),
                Some("qwen2") => (Qwen2, QWEN),
                Some("qwen3") => (Qwen3, QWEN),
                _ => return Err("unsupported model type".into()),
            };
            let gpt = family == Gpt;
            let qwen = matches!(family, Qwen2 | Qwen3);
            let key = |gpt_name, other| if gpt { gpt_name } else { other };
            let dim = |k: &str| -> Load<usize> {
                config[k]
                    .as_u64()
                    .filter(|x| (1..=1 << 20).contains(x))
                    .map(|x| x as usize)
                    .ok_or_else(|| format!("invalid model dimension: {k}").into())
            };
            let width = dim(key("n_embd", "hidden_size"))?;
            let heads = dim(key("n_head", "num_attention_heads"))?;
            let head = if qwen {
                dim("head_dim").unwrap_or(width / heads)
            } else {
                width / heads
            };
            let kv = if qwen {
                dim("num_key_value_heads").unwrap_or(heads)
            } else {
                heads
            };
            let limit = dim(key("n_positions", "max_position_embeddings"))?;
            let inner = dim(key("n_inner", "intermediate_size")).unwrap_or(width * 4);
            let layers = dim(key("n_layer", "num_hidden_layers"))?;
            let vocab = dim("vocab_size")?;
            let on = |name| config[name].as_bool() == Some(true);
            let valid = match family {
                Gpt => {
                    !on("scale_attn_by_inverse_layer_idx")
                        && config["scale_attn_weights"].as_bool() != Some(false)
                }
                Bert => config["position_embedding_type"]
                    .as_str()
                    .is_none_or(|x| x == "absolute"),
                _ => config["rope_scaling"].is_null() && !on("use_sliding_window"),
            };
            if !valid
                || !head.is_power_of_two()
                || heads % kv != 0
                || (!qwen && head * heads != width)
            {
                return Err("unsupported transformer configuration".into());
            }
            let (act, eps) = match family {
                Gpt => ("gelu_new", "layer_norm_epsilon"),
                Bert => ("gelu", "layer_norm_eps"),
                _ => ("silu", "rms_norm_eps"),
            };
            if config[key("activation_function", "hidden_act")]
                .as_str()
                .is_some_and(|x| x != act)
            {
                return Err("unsupported activation".into());
            }
            let scale = Scale::new(bits)?;
            let factor = f64::from(scale.denom_i32());
            let eps = config[eps]
                .as_f64()
                .ok_or("missing normalization epsilon")?;
            let epsilon = (eps * factor * factor).round().max(1.0);
            if !eps.is_finite() || eps < 0.0 || epsilon >= i64::MAX as f64 {
                return Err("invalid normalization epsilon".into());
            }
            let theta = qwen
                .then(|| {
                    config["rope_theta"]
                        .as_f64()
                        .filter(|x| x.is_finite() && *x > 0.0)
                        .ok_or("invalid RoPE base")
                })
                .transpose()?;
            let bytes = fs::read(dir.join("model.safetensors"))?;
            let f = Weights {
                content: SafeTensors::deserialize(&bytes)?,
                scale: factor,
            };
            let has = |name: &str| f.content.tensor(name).is_ok();
            let prefix = if qwen || has(&format!("{}{}.weight", names.prefix, names.embed)) {
                names.prefix
            } else {
                ""
            };
            let embedding = |name, rows| f.read(&format!("{prefix}{name}.weight"), &[rows, width]);
            let tokens = embedding(names.embed, vocab)?;
            if family != Bert
                && has("lm_head.weight")
                && f.read("lm_head.weight", &[vocab, width])? != tokens
            {
                return Err("decoder requires tied embeddings".into());
            }
            let positions = (!qwen)
                .then(|| embedding(names.position, limit))
                .transpose()?;
            let segments = (family == Bert)
                .then(|| embedding(names.segment, dim("type_vocab_size")?))
                .transpose()?;
            let norm = f.affine(&format!("{prefix}{}", names.head), &[width], !qwen, false)?;
            let bias = family != Qwen3 || on("attention_bias");
            let mut data = Vec::with_capacity(layers);
            for layer in 0..layers {
                let stem = format!("{prefix}{}.{layer}", names.blocks);
                let name = |k| format!("{stem}.{k}");
                let mat = |k, n, m, b| f.affine(&name(k), &[n, m], b, gpt);
                let norm = |k, n| f.affine(&name(k), &[n], !qwen, false);
                let [q, k, v] = names.qkv;
                let [q, k, v] = if gpt {
                    let (w, b) = mat(q, width * 3, width, true)?;
                    std::array::from_fn(|i| {
                        let at = i * width;
                        (
                            w[at * width..][..width * width].to_vec(),
                            b[at..][..width].to_vec(),
                        )
                    })
                } else {
                    [
                        mat(q, heads * head, width, bias)?,
                        mat(k, kv * head, width, bias)?,
                        mat(v, kv * head, width, bias)?,
                    ]
                };
                let out = mat(
                    names.output,
                    width,
                    heads * head,
                    !qwen || family == Qwen3 && bias,
                )?;
                let [up, down] = names.feed;
                let [a, b] = names.norm;
                let qk = if family == Qwen3 {
                    Some([
                        norm("self_attn.q_norm", head)?,
                        norm("self_attn.k_norm", head)?,
                    ])
                } else {
                    None
                };
                data.push(LayerData {
                    attention: [q, k, v, out],
                    mlp: [
                        mat(up, inner, width, !qwen)?,
                        mat(down, width, inner, !qwen)?,
                    ],
                    norm: [norm(a, width)?, norm(b, width)?],
                    qk,
                    gate: qwen
                        .then(|| mat("mlp.gate_proj", inner, width, false))
                        .transpose()?,
                });
            }
            let activation = activation(scale, family)?;
            let soft = softmax::softmax_tables(scale, SemVersion::V3);
            let rotary = (0..=limit.next_power_of_two().ilog2())
                .map(|_| OnceLock::new())
                .collect();
            Ok(Self {
                family,
                bits,
                width,
                head,
                limit,
                epsilon: epsilon as i64,
                theta,
                tokens,
                positions,
                segments,
                layers: data,
                norm,
                activation,
                soft,
                rotary,
            })
        }
        fn rotation(&self, rows: usize) -> Option<Rotary<'_>> {
            let theta = self.theta?;
            let n = rows.next_power_of_two();
            let i = n.ilog2() as usize;
            let half = self.head / 2;
            let (c, s) = self.rotary[i..]
                .iter()
                .find_map(OnceLock::get)
                .unwrap_or_else(|| {
                    self.rotary[i].get_or_init(|| {
                        let scale = (1u64 << self.bits) as f64;
                        let waves: Vec<_> = (0..half)
                            .map(|i| libm::pow(theta, -((2 * i) as f64) / self.head as f64))
                            .collect();
                        (0..n.min(self.limit))
                            .flat_map(|r| {
                                waves.iter().map(move |&w| {
                                    let (s, c) = libm::sincos(r as f64 * w);
                                    ((c * scale).round() as i32, (s * scale).round() as i32)
                                })
                            })
                            .unzip()
                    })
                });
            Some(Rotary {
                cos: &c[..rows * half],
                sin: &s[..rows * half],
            })
        }
        pub fn model<'a>(&'a self, rows: usize) -> super::Result<Model<'a>> {
            if !(1..=self.limit).contains(&rows) {
                return Err("invalid sequence length");
            }
            let rotary = self.rotation(rows);
            let normalize = |(gain, bias): &'a Affine| NormalizationParameters {
                kind: if self.theta.is_some() {
                    NormalizationKind::Rms
                } else {
                    NormalizationKind::Layer
                },
                gain,
                bias,
                epsilon_variance_units: self.epsilon,
            };
            let encoder = self.family == Bert;
            let mask = if encoder {
                SoftmaxMask::Full
            } else {
                SoftmaxMask::Causal
            };
            let tables = SoftmaxTables {
                probabilities: &self.soft.out,
                sums: &self.soft.sum,
                limit: self.soft.sum_scale as i32,
            };
            let factor = ((1u64 << self.bits) as f64 / libm::sqrt(self.head as f64)).round() as i32;
            let activation = Activation {
                lower: self.activation.0,
                correction: &self.activation.1,
            };
            let mut norms = Vec::with_capacity(self.layers.len());
            let layers = self
                .layers
                .iter()
                .map(|l| {
                    let [q, k, v, o] = &l.attention;
                    let heads = |(w, b): &'a Affine| {
                        w.chunks_exact(self.width * self.head)
                            .zip(b.chunks_exact(self.head))
                            .map(|(weights, bias)| Projection { weights, bias })
                    };
                    norms.push(l.qk.as_ref().map(|pair| pair.each_ref().map(normalize)));
                    Layer {
                        attention: Attention {
                            queries: heads(q).collect(),
                            groups: heads(k).zip(heads(v)).map(|(k, v)| [k, v]).collect(),
                            output: projection(o),
                            rotary,
                            mask,
                            tables,
                            factor,
                        },
                        mlp: Mlp {
                            up: projection(&l.mlp[0]),
                            down: projection(&l.mlp[1]),
                            gate: l.gate.as_ref().map(projection),
                            activation,
                        },
                        attention_norm: normalize(&l.norm[0]),
                        mlp_norm: normalize(&l.norm[1]),
                    }
                })
                .collect();
            let norm = normalize(&self.norm);
            let arch = if encoder {
                Architecture::Encoder {
                    embedding_norm: norm,
                }
            } else {
                Architecture::Decoder { final_norm: norm }
            };
            let input = PublicEmbeddings {
                width: self.width,
                tokens: &self.tokens,
                positions: self.positions.as_deref(),
                segments: self.segments.as_deref(),
            };
            Model::with_heads(rows, self.bits, input, layers, arch, norms)
        }
    }

    fn activation(scale: Scale, family: Family) -> Load<(i32, Vec<i32>)> {
        if family == Gpt {
            let factor = f64::from(scale.denom_i32());
            let bound = 8 * scale.denom_i32();
            let a = libm::sqrt(2.0 / std::f64::consts::PI);
            let values = (0..2 * bound)
                .map(|i| {
                    let magnitude = f64::from((i - bound).abs());
                    let x = magnitude / factor;
                    let tail = 0.5 * magnitude * (1.0 - libm::tanh(a * (x + 0.044715 * x.powi(3))));
                    libm::floor(0.5 - tail) as i32
                })
                .collect();
            return Ok((-bound, values));
        }
        let kind = if family == Bert {
            TableKind::GeluErf
        } else {
            TableKind::Silu
        };
        let t = table(kind, scale, SemVersion::V3);
        let IndexRule::SymClamp { bound_log } = t.index else {
            return Err("unsupported activation table".into());
        };
        Ok((-(1 << bound_log), t.entries))
    }

    fn projection((weights, bias): &Affine) -> Projection<'_> {
        Projection { weights, bias }
    }
}

#[cfg(test)]
mod tests;
