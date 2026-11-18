//! The exhaustive single-op prove/verify table.
//!
//! One `#[test]` per case (nextest runs each in its own process and names
//! failures), all driven by [`prove_verify_roundtrip`] under BitZ over Fp128. Two
//! meta-tests make the table total: every [`OpKind`] must have at least one
//! declared case (an `#[ignore]`d case is a declared stance), and every case's
//! model must contain a node of the kind it claims to cover.
//!
//! Sizes: `T = 1 << 10` by default (prove cost is linear in `T`, the
//! overflow paths are value-driven, and no size-gated prover
//! branch exists above `T = 8`); `t = 1000` for the non-power-of-two
//! variants; one large smoke case (`mul_large_2p15`).
//!
//! Overflow cases are sized for the current `MODEL_SCALE` and are expected rejections: an
//! arithmetic result outside `i32` is an execution error, so they must panic at the trace. A
//! scale change that stops them overflowing fails loudly.

use std::collections::BTreeSet;

use atlas_pcs::{bitz::BitzScheme, commitment_scheme::CommitmentScheme};
use atlas_trace::{op::SmoothReluForm, Model, Tensor};
use common::consts::{ACTIVATION_BOUND, MODEL_SCALE, TRIG_PERIOD_MODULUS};
use rand::{rngs::StdRng, SeedableRng};
use strum::IntoEnumIterator;

use super::{op_kind::OpKind, prove_verify_roundtrip, ModelBuilder, RoundtripOptions};

/// One row of the table.
pub(crate) struct OpCase {
    pub name: &'static str,
    pub kind: OpKind,
    /// Builds the model and its inputs from a seeded RNG.
    pub build: fn(&mut StdRng) -> (Model, Vec<Tensor<i32>>),
}

/// Per-case seed derived from the case name (FNV-1a), so every case is
/// deterministic and distinct without a seed column.
const fn seed_of(name: &str) -> u64 {
    let bytes = name.as_bytes();
    let mut hash: u64 = 0xcbf29ce484222325;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        i += 1;
    }
    hash
}

/// Run one case under `PCS`.
pub(crate) fn run_case<PCS: CommitmentScheme>(case: &OpCase) {
    let mut rng = StdRng::seed_from_u64(seed_of(case.name));
    let (model, inputs) = (case.build)(&mut rng);
    prove_verify_roundtrip::<PCS>(model, &inputs, RoundtripOptions::default());
}

/// Run the case called `name` under `PCS`.
pub(crate) fn run_named<PCS: CommitmentScheme>(name: &str) {
    let case = CASES
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no op-surface case named {name}"));
    run_case::<PCS>(case);
}

macro_rules! op_surface {
    ($(
        $(#[$meta:meta])*
        $name:ident : $kind:ident => $build:expr ;
    )*) => {
        /// Every declared case.
        pub(crate) const CASES: &[OpCase] = &[$(
            OpCase {
                name: stringify!($name),
                kind: OpKind::$kind,
                build: $build,
            },
        )*];
        $(
            #[test]
            $(#[$meta])*
            fn $name() {
                run_named::<BitzScheme>(stringify!($name));
            }
        )*
    };
}

// ── shared constants and helpers ─────────────────────────────────────────

const T: usize = 1 << 10;
const T_ODD: usize = 1000;
const S: u8 = MODEL_SCALE as u8;
const MIN_INPUT: i32 = -(1 << 16);
const MAX_INPUT: i32 = 1 << 16;

fn vec_of(v: i32, dims: &[usize]) -> Tensor<i32> {
    Tensor::from_slice(&vec![v; dims.iter().product()], dims).unwrap()
}

fn tensor(data: &[i32], dims: &[usize]) -> Tensor<i32> {
    Tensor::from_slice(data, dims).unwrap()
}

/// `op(input, random constant)` over `dims`, inputs in the i8 range.
fn binary_small(
    rng: &mut StdRng,
    dims: &[usize],
    op: fn(&mut ModelBuilder, usize, usize) -> usize,
) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::new();
    let i = b.input(dims.to_vec());
    let c = b.constant(Tensor::random_small(rng, dims));
    let o = op(&mut b, i, c);
    b.mark_output(o);
    (b.build(), vec![Tensor::random_small(rng, dims)])
}

/// `op(input, constant)` with every element fixed.
fn binary_const(
    dims: &[usize],
    input: i32,
    constant: i32,
    op: fn(&mut ModelBuilder, usize, usize) -> usize,
) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::new();
    let i = b.input(dims.to_vec());
    let c = b.constant(vec_of(constant, dims));
    let o = op(&mut b, i, c);
    b.mark_output(o);
    (b.build(), vec![vec_of(input, dims)])
}

/// `op(input)` over `dims` at scale `scale` with the given input tensor.
fn unary(
    dims: &[usize],
    scale: u8,
    input: Tensor<i32>,
    op: fn(&mut ModelBuilder, usize) -> usize,
) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::with_scale(scale);
    let i = b.input(dims.to_vec());
    let o = op(&mut b, i);
    b.mark_output(o);
    (b.build(), vec![input])
}

/// `op(m) + m` over `dims` with `m = −input`: the operand of `op` has a second consumer, so its
/// evaluation reduction folds the claim `op` appends in its own turn with another.
fn shared_operand(
    dims: &[usize],
    input: Tensor<i32>,
    op: fn(&mut ModelBuilder, usize) -> usize,
) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::with_scale(S);
    let i = b.input(dims.to_vec());
    let m = b.neg(i);
    let y = op(&mut b, m);
    let o = b.add(y, m);
    b.mark_output(o);
    (b.build(), vec![input])
}

fn unary_small(
    rng: &mut StdRng,
    dims: &[usize],
    op: fn(&mut ModelBuilder, usize) -> usize,
) -> (Model, Vec<Tensor<i32>>) {
    unary(dims, S, Tensor::random_small(rng, dims), op)
}

fn unary_range(
    rng: &mut StdRng,
    dims: &[usize],
    range: std::ops::Range<i32>,
    op: fn(&mut ModelBuilder, usize) -> usize,
) -> (Model, Vec<Tensor<i32>>) {
    unary(dims, S, Tensor::random_range(rng, dims, range), op)
}

/// Alternating `i32::MAX ± k` / `i32::MIN ∓ k` so the result overflows on both sides.
fn overflow_pair(t: usize, pos_delta: i32, neg_delta: i32) -> (Tensor<i32>, Tensor<i32>) {
    let input: Vec<i32> = (0..t)
        .map(|i| if i % 2 == 0 { i32::MAX } else { i32::MIN })
        .collect();
    let constant: Vec<i32> = (0..t)
        .map(|i| if i % 2 == 0 { pos_delta } else { neg_delta })
        .collect();
    (tensor(&input, &[t]), tensor(&constant, &[t]))
}

fn add_const_overflow(t: usize, pos: i32, neg: i32) -> (Model, Vec<Tensor<i32>>) {
    let (input, constant) = overflow_pair(t, pos, neg);
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let c = b.constant(constant);
    let o = b.add(i, c);
    b.mark_output(o);
    (b.build(), vec![input])
}

fn sub_const_overflow(t: usize, pos: i32, neg: i32) -> (Model, Vec<Tensor<i32>>) {
    let (input, constant) = overflow_pair(t, pos, neg);
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let c = b.constant(constant);
    let o = b.sub(i, c);
    b.mark_output(o);
    (b.build(), vec![input])
}

fn rsqrt_model(
    rng: &mut StdRng,
    t: usize,
    scale: u8,
    range: std::ops::Range<i32>,
) -> (Model, Vec<Tensor<i32>>) {
    unary(
        &[t],
        scale,
        Tensor::random_range(rng, &[t], range),
        |b, i| b.rsqrt(i),
    )
}

fn iff_model(rng: &mut StdRng, t: usize) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let mask = b.constant(Tensor::random_boolean(rng, &[t]));
    let other = b.constant(Tensor::random(rng, &[t]));
    let o = b.iff(mask, i, other);
    b.mark_output(o);
    (b.build(), vec![Tensor::random(rng, &[t])])
}

/// Clamped-domain boundaries of the table activations.
fn activation_edges() -> Tensor<i32> {
    let bound = 1i32 << ACTIVATION_BOUND;
    tensor(
        &[
            -bound,
            bound,
            -(bound - 1),
            bound - 1,
            0,
            -1,
            1,
            i32::MAX,
            i32::MIN,
            -(bound + 1),
            bound + 1,
            12345,
            -12345,
            7,
            -7,
            1 << 14,
        ],
        &[16],
    )
}

/// Both clamp bounds of the smooth ReLUs (`±8` for GELU, `±16` for SiLU) and the far tails,
/// where the output is the input or zero.
fn smooth_relu_edges() -> Tensor<i32> {
    let mut data = vec![0, 1, -1, 16384, -16384, i32::MAX, i32::MIN, 71_000, -71_000];
    for bound in [1i32 << ACTIVATION_BOUND, 1 << (ACTIVATION_BOUND + 1)] {
        data.extend([-bound - 1, -bound, -bound + 1, bound - 1, bound, bound + 1]);
    }
    data.extend([1 << 24, -(1 << 24), 212_000, -212_000, 7, -7, 12345, -12345]);
    data.resize(32, 0);
    tensor(&data, &[32])
}

fn trig_edges() -> Tensor<i32> {
    let m = TRIG_PERIOD_MODULUS as i32;
    tensor(&[-m - 1, -m, -1, 0, 1, m - 1, m, m + 1], &[8])
}

fn mos_model(rng: &mut StdRng, m: usize, n: usize) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![m, n]);
    let o = b.mean_of_squares(i, vec![1], vec![m, 1]);
    b.mark_output(o);
    (b.build(), vec![Tensor::random_small(rng, &[m, n])])
}

fn sum_model(rng: &mut StdRng, m: usize, n: usize, axis: usize) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![m, n]);
    let out_dims = if axis == 0 { vec![n] } else { vec![m] };
    let o = b.sum(i, vec![axis], out_dims);
    b.mark_output(o);
    (b.build(), vec![Tensor::random_small(rng, &[m, n])])
}

fn softmax_model(dims: &[usize], scale: u8, input: Tensor<i32>) -> (Model, Vec<Tensor<i32>>) {
    unary(dims, scale, input, |b, i| b.softmax_last_axis(i))
}

#[rustfmt::skip]
const SOFTMAX_ROWS: [i32; 16] = [
    10, 20, 30, 40, 50, 60, 5, 15,
    25, 35, 45, 55, 8, 18, 28, 38,
];

fn gather_model(
    rng: &mut StdRng,
    idx_dims: &[usize],
    dict_len: usize,
    word: usize,
) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::new();
    let data: Vec<i32> = (0..dict_len * word).map(|i| i as i32).collect();
    let dict = b.constant(Tensor::construct(data, vec![dict_len, word]));
    let idx = b.input(idx_dims.to_vec());
    let o = b.gather(dict, idx, 0, [idx_dims, &[word]].concat());
    b.mark_output(o);
    let input = Tensor::random_range(rng, idx_dims, 0..dict_len as i32);
    (b.build(), vec![input])
}

fn einsum_model(
    rng: &mut StdRng,
    eq: &str,
    lhs: &[usize],
    rhs: &[usize],
    out: &[usize],
) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::new();
    let i = b.input(lhs.to_vec());
    let c = b.constant(Tensor::random_small(rng, rhs));
    let o = b.einsum(eq, vec![i, c], out.to_vec());
    b.mark_output(o);
    (b.build(), vec![Tensor::random_small(rng, lhs)])
}

/// `x` contracted with constants through `products` deferred products (a multiply by a
/// second input between two contractions when there are three) and rescaled once.
fn deferred_chain(
    rng: &mut StdRng,
    rows: usize,
    width: usize,
    products: usize,
) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::new();
    let x = b.input(vec![rows, width]);
    let w = b.constant(Tensor::random_small(rng, &[width, width]));
    let mut wide = b.einsum_deferred("mk,kn->mn", vec![x, w], vec![rows, width]);
    let mut inputs = vec![Tensor::random_small(rng, &[rows, width])];
    if products > 1 {
        let gate = b.input(vec![rows, width]);
        inputs.push(Tensor::random_small(rng, &[rows, width]));
        wide = b.mul_deferred(wide, gate);
    }
    if products > 2 {
        let w = b.constant(Tensor::random_small(rng, &[width, width]));
        wide = b.einsum_deferred("mk,kn->mn", vec![wide, w], vec![rows, width]);
    }
    let out = b.rescale(wide);
    b.mark_output(out);
    (b.build(), inputs)
}

/// One deferred product of `input` by the constant `weight`, rescaled.
fn deferred_const(input: i32, weight: i32) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::new();
    let x = b.input(vec![2, 2]);
    let w = b.constant(vec_of(weight, &[2, 2]));
    let wide = b.einsum_deferred("mk,kn->mn", vec![x, w], vec![2, 2]);
    let out = b.rescale(wide);
    b.mark_output(out);
    (b.build(), vec![vec_of(input, &[2, 2])])
}

fn einsum_const(
    eq: &str,
    lhs: &[usize],
    rhs: &[usize],
    out: &[usize],
    input: i32,
    constant: i32,
) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::new();
    let i = b.input(lhs.to_vec());
    let c = b.constant(vec_of(constant, rhs));
    let o = b.einsum(eq, vec![i, c], out.to_vec());
    b.mark_output(o);
    (b.build(), vec![vec_of(input, lhs)])
}

fn structural(
    rng: &mut StdRng,
    dims: &[usize],
    op: impl FnOnce(&mut ModelBuilder, usize) -> usize,
) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::new();
    let i = b.input(dims.to_vec());
    let o = op(&mut b, i);
    b.mark_output(o);
    (b.build(), vec![Tensor::random_small(rng, dims)])
}

/// [`structural`] compiled with the padding pass, for dims that are not powers of two.
fn structural_padded(
    rng: &mut StdRng,
    dims: &[usize],
    op: impl FnOnce(&mut ModelBuilder, usize) -> usize,
) -> (Model, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::new();
    let i = b.input(dims.to_vec());
    let o = op(&mut b, i);
    b.mark_output(o);
    (b.build_padded(), vec![Tensor::random_small(rng, dims)])
}

fn concat_builder(
    rng: &mut StdRng,
    shapes: &[&[usize]],
    axis: isize,
) -> (ModelBuilder, Vec<Tensor<i32>>) {
    let mut b = ModelBuilder::new();
    let ins: Vec<usize> = shapes.iter().map(|s| b.input(s.to_vec())).collect();
    let o = b.concat(&ins, axis);
    b.mark_output(o);
    let inputs = shapes
        .iter()
        .map(|s| Tensor::random_small(rng, s))
        .collect();
    (b, inputs)
}

fn concat_model(rng: &mut StdRng, shapes: &[&[usize]], axis: isize) -> (Model, Vec<Tensor<i32>>) {
    let (b, inputs) = concat_builder(rng, shapes, axis);
    (b.build(), inputs)
}

/// [`concat_model`] compiled with the padding pass, for dims that are not powers of two.
fn concat_padded(rng: &mut StdRng, shapes: &[&[usize]], axis: isize) -> (Model, Vec<Tensor<i32>>) {
    let (b, inputs) = concat_builder(rng, shapes, axis);
    (b.build_padded(), inputs)
}

/// Magnitudes that overflow the fused rescale paths at the current scale.
/// `(2^23+1)^2 >> 14 ≈ 2^32`, `(2^20+1)^3 >> 28 ≈ 2^32`, `16·(2^22+1)^2 >> 14
/// ≈ 2^34`, `16·(2^23+1)^2 / (16·2^14) ≈ 2^32`.
const SAT_SQUARE: i32 = (1 << 23) + 1;
const SAT_CUBE: i32 = (1 << 20) + 1;
const SAT_EINSUM: i32 = (1 << 22) + 1;

// ── the table ────────────────────────────────────────────────────────────

op_surface! {
    // Input / Constant / Identity / And / IsNan
    input_passthrough_1k: Input => |rng| structural(rng, &[T], |b, i| b.identity(i));
    input_passthrough_1000: Input => |rng| structural(rng, &[T_ODD], |b, i| b.identity(i));
    identity_1k: Identity => |rng| structural(rng, &[T], |b, i| b.identity(i));
    constant_output: Constant => |rng| {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let c = b.constant(Tensor::random_small(rng, &[T]));
        let o = b.add(i, c);
        b.mark_output(o);
        b.mark_output(c);
        (b.build(), vec![Tensor::random_small(rng, &[T])])
    };
    and_boolean_1k: And => |rng| {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let c = b.constant(Tensor::random_boolean(rng, &[T]));
        let o = b.and(i, c);
        b.mark_output(o);
        (b.build(), vec![Tensor::random_boolean(rng, &[T])])
    };
    is_nan_1k: IsNan => |rng| unary(&[T], S, Tensor::random(rng, &[T]), |b, i| b.is_nan(i));

    // Add
    add_random_1k: Add => |rng| binary_small(rng, &[T], |b, a, c| b.add(a, c));
    add_non_pow2_1000: Add => |rng| binary_small(rng, &[T_ODD], |b, a, c| b.add(a, c));
    add_tiny_t1: Add => |_| binary_const(&[1], 3, 2, |b, a, c| b.add(a, c));
    add_tiny_t2: Add => |_| binary_const(&[2], 3, 2, |b, a, c| b.add(a, c));
    add_tiny_t4: Add => |_| binary_const(&[4], 3, 2, |b, a, c| b.add(a, c));
    add_tiny_t8: Add => |_| binary_const(&[8], 3, 2, |b, a, c| b.add(a, c));
    #[should_panic(expected = "WitnessOverflow")]
    add_scalar_sat_pos: Add => |_| binary_const(&[1], i32::MAX, i32::MAX, |b, a, c| b.add(a, c));
    #[should_panic(expected = "WitnessOverflow")]
    add_scalar_sat_neg: Add => |_| binary_const(&[1], i32::MIN, i32::MIN, |b, a, c| b.add(a, c));
    add_scalar_no_sat: Add => |_| binary_const(&[1], i32::MAX, -5, |b, a, c| b.add(a, c));
    /// A one-element `Add` that is not the model output: the one cell of a one-cell bucket.
    add_scalar_internal: Add => |_| {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![1]);
        let c = b.constant(vec_of(40_000, &[1]));
        let a = b.add(i, c);
        let d = b.constant(vec_of(7, &[1]));
        let o = b.sub(a, d);
        b.mark_output(o);
        (b.build(), vec![vec_of(123_456, &[1])])
    };
    #[should_panic(expected = "WitnessOverflow")]
    add_saturating_overflow_1k: Add => |_| add_const_overflow(T, 1000, -1000);

    // Sub
    sub_random_1k: Sub => |rng| binary_small(rng, &[T], |b, a, c| b.sub(a, c));
    sub_non_pow2_1000: Sub => |rng| binary_small(rng, &[T_ODD], |b, a, c| b.sub(a, c));
    sub_tiny_t1: Sub => |_| binary_const(&[1], 7, 2, |b, a, c| b.sub(a, c));
    sub_tiny_t2: Sub => |_| binary_const(&[2], 7, 2, |b, a, c| b.sub(a, c));
    sub_tiny_t4: Sub => |_| binary_const(&[4], 7, 2, |b, a, c| b.sub(a, c));
    sub_tiny_t8: Sub => |_| binary_const(&[8], 7, 2, |b, a, c| b.sub(a, c));
    #[should_panic(expected = "WitnessOverflow")]
    sub_scalar_sat_pos: Sub => |_| binary_const(&[1], i32::MAX, i32::MIN, |b, a, c| b.sub(a, c));
    #[should_panic(expected = "WitnessOverflow")]
    sub_scalar_sat_neg: Sub => |_| binary_const(&[1], i32::MIN, i32::MAX, |b, a, c| b.sub(a, c));
    sub_scalar_no_sat: Sub => |_| binary_const(&[1], 10, 3, |b, a, c| b.sub(a, c));
    #[should_panic(expected = "WitnessOverflow")]
    sub_saturating_overflow_1k: Sub => |_| sub_const_overflow(T, -1000, 1000);
    /// `y = sub(x, x)`: both operands open the same producer at the same
    /// point; the openings collapse to one entry and must still verify.
    sub_duplicate_operand: Sub => |rng| {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![256]);
        let c = b.constant(Tensor::random_small(rng, &[256]));
        let x = b.sub(i, c);
        let y = b.sub(x, x);
        b.mark_output(y);
        (b.build(), vec![Tensor::random_small(rng, &[256])])
    };

    // Neg
    neg_random_1k: Neg => |rng| unary_small(rng, &[T], |b, i| b.neg(i));
    neg_non_pow2_1000: Neg => |rng| unary_small(rng, &[T_ODD], |b, i| b.neg(i));
    /// `-i32::MIN` is not an `i32`: the execution fails, and `Neg`'s exact relation holds on
    /// every execution that exists.
    #[should_panic(expected = "WitnessOverflow")]
    neg_i32_min_rejected: Neg => |_| unary(&[4], S, tensor(&[i32::MIN, i32::MIN + 1, 0, i32::MAX], &[4]), |b, i| b.neg(i));

    // Mul (fused rescale)
    mul_random_1k: Mul => |rng| binary_small(rng, &[T], |b, a, c| b.mul(a, c));
    mul_non_pow2_1000: Mul => |rng| binary_small(rng, &[T_ODD], |b, a, c| b.mul(a, c));
    #[should_panic(expected = "WitnessOverflow")]
    mul_saturates_pos: Mul => |_| binary_const(&[16], SAT_SQUARE, SAT_SQUARE, |b, a, c| b.mul(a, c));
    #[should_panic(expected = "WitnessOverflow")]
    mul_saturates_neg: Mul => |_| binary_const(&[16], -SAT_SQUARE, SAT_SQUARE, |b, a, c| b.mul(a, c));
    #[should_panic(expected = "WitnessOverflow")]
    mul_scalar_rescale_sat: Mul => |_| binary_const(&[1], SAT_SQUARE, SAT_SQUARE, |b, a, c| b.mul(a, c));
    /// A one-element fused `Mul` that is not the model output: its output and its remainder
    /// are each one cell of a range bucket.
    mul_scalar_internal: Mul => |_| {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![1]);
        let c = b.constant(vec_of(3001, &[1]));
        let m = b.mul(i, c);
        let d = b.constant(vec_of(11, &[1]));
        let o = b.add(m, d);
        b.mark_output(o);
        (b.build(), vec![vec_of(1000, &[1])])
    };
    /// The one large-polynomial smoke case (SRS, MSM and one-hot sizes).
    mul_large_2p15: Mul => |rng| binary_small(rng, &[1 << 15], |b, a, c| b.mul(a, c));

    // Square / Cube
    square_random_1k: Square => |rng| unary_small(rng, &[T], |b, i| b.square(i));
    square_non_pow2_1000: Square => |rng| unary_small(rng, &[T_ODD], |b, i| b.square(i));
    #[should_panic(expected = "WitnessOverflow")]
    square_saturates: Square => |_| unary(&[16], S, vec_of(SAT_SQUARE, &[16]), |b, i| b.square(i));
    square_scalar_out: Square => |_| unary(&[1], S, vec_of(1000, &[1]), |b, i| b.square(i));
    cube_random_1k: Cube => |rng| unary_small(rng, &[T], |b, i| b.cube(i, S));
    cube_non_pow2_1000: Cube => |rng| unary_small(rng, &[T_ODD], |b, i| b.cube(i, S));
    #[should_panic(expected = "WitnessOverflow")]
    cube_saturates_pos: Cube => |_| unary(&[16], S, vec_of(SAT_CUBE, &[16]), |b, i| b.cube(i, S));
    #[should_panic(expected = "WitnessOverflow")]
    cube_saturates_neg: Cube => |_| unary(&[16], S, vec_of(-SAT_CUBE, &[16]), |b, i| b.cube(i, S));
    cube_raw_no_rebase: Cube => |rng| unary_small(rng, &[T], |b, i| b.cube(i, 0));
    cube_scalar_out: Cube => |_| unary(&[1], S, vec_of(1000, &[1]), |b, i| b.cube(i, S));

    // ScalarConstDiv
    #[ignore = "non-power-of-two div path not supported"]
    scalar_const_div_1k: ScalarConstDiv => |rng| unary_small(rng, &[T], |b, i| b.scalar_const_div(i, 128));
    /// A divisor that is not a power of two: the remainder is bounded from both sides.
    scalar_const_div_by_7_1k: ScalarConstDiv => |rng| unary_small(rng, &[T], |b, i| b.scalar_const_div(i, 7));
    /// The importer's divisor, `2^scale`: one range claim at that width.
    scalar_const_div_by_scale_1k: ScalarConstDiv => |rng| unary_small(rng, &[T], |b, i| b.scalar_const_div(i, 1 << S));
    scalar_const_div_one_element: ScalarConstDiv => |_| unary(&[1], S, vec_of(-12_345, &[1]), |b, i| b.scalar_const_div(i, 7));
    #[ignore = "non-power-of-two scalar_const_div path not supported"]
    scalar_const_div_non_pow2_1000: ScalarConstDiv => |rng| unary_small(rng, &[T_ODD], |b, i| b.scalar_const_div(i, 128));

    // ReLU
    relu_random_1k: ReLU => |rng| unary_small(rng, &[T], |b, i| b.relu(i));
    relu_non_pow2_1000: ReLU => |rng| unary_small(rng, &[T_ODD], |b, i| b.relu(i));
    relu_shared_operand_t16: ReLU => |rng| shared_operand(&[16], Tensor::random_small(rng, &[16]), |b, i| b.relu(i));

    // Iff
    iff_random_1k: Iff => |rng| iff_model(rng, T);
    #[ignore = "non-power-of-two iff path not supported"]
    iff_non_pow2_1000: Iff => |rng| iff_model(rng, T_ODD);

    // Rescale, and the deferred products it ends a chain of
    /// One deferred contraction, rescaled from level 2: a 14-bit remainder.
    rescale_level_2: Rescale => |rng| deferred_chain(rng, 16, 32, 1);
    /// A deferred contraction times a level-1 value, rescaled from level 3: 28 bits.
    rescale_level_3: Rescale => |rng| deferred_chain(rng, 16, 32, 2);
    /// Contraction, multiply, contraction, rescaled from level 4: 42 bits.
    rescale_level_4: Rescale => |rng| deferred_chain(rng, 16, 32, 3);
    rescale_one_row: Rescale => |rng| deferred_chain(rng, 1, 16, 3);
    #[should_panic(expected = "WitnessOverflow")]
    rescale_saturates_pos: Rescale => |_| deferred_const(i32::MAX, 1 << S);
    #[should_panic(expected = "WitnessOverflow")]
    rescale_saturates_neg: Rescale => |_| deferred_const(i32::MIN, 1 << S);

    // Rsqrt
    rsqrt_scale8_1k: Rsqrt => |rng| rsqrt_model(rng, T, 8, 1..MAX_INPUT);
    /// At scale 12 `S³ = 2³⁶`, so `⌊S³/x⌋` exceeds `i32` for every `x < 32`:
    /// the u64 quotient / i64 `out²` witness path on every element.
    rsqrt_scale12_small_inputs: Rsqrt => |rng| rsqrt_model(rng, T, 12, 1..64);
    rsqrt_model_scale_1k: Rsqrt => |rng| rsqrt_model(rng, T, S, 1..MAX_INPUT);
    rsqrt_one_element: Rsqrt => |rng| rsqrt_model(rng, 1, S, 1..MAX_INPUT);
    #[ignore = "non-power-of-two rsqrt path not supported"]
    rsqrt_non_pow2_1000: Rsqrt => |rng| rsqrt_model(rng, T_ODD, 8, 1..MAX_INPUT);

    // Table activations
    // The smooth ReLUs (GELU, SiLU): ReLU plus a correction table over a clamped domain.
    gelu_erf_random_1k: SmoothRelu => |rng| unary_range(rng, &[T], -(1 << 19)..(1 << 19), |b, i| b.smooth_relu(i, SmoothReluForm::GeluErf));
    gelu_erf_domain_edges: SmoothRelu => |_| unary(&[32], S, smooth_relu_edges(), |b, i| b.smooth_relu(i, SmoothReluForm::GeluErf));
    gelu_erf_shared_operand_t16: SmoothRelu => |rng| shared_operand(&[16], Tensor::random_range(rng, &[16], MIN_INPUT..MAX_INPUT), |b, i| b.smooth_relu(i, SmoothReluForm::GeluErf));
    gelu_erf_non_pow2_1000: SmoothRelu => |rng| {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T_ODD]);
        let o = b.smooth_relu(i, SmoothReluForm::GeluErf);
        b.mark_output(o);
        (b.build_padded(), vec![Tensor::random_range(rng, &[T_ODD], -(1 << 19)..(1 << 19))])
    };
    gelu_tanh_random_1k: SmoothRelu => |rng| unary_range(rng, &[T], -(1 << 19)..(1 << 19), |b, i| b.smooth_relu(i, SmoothReluForm::GeluTanh));
    gelu_tanh_domain_edges: SmoothRelu => |_| unary(&[32], S, smooth_relu_edges(), |b, i| b.smooth_relu(i, SmoothReluForm::GeluTanh));
    gelu_tanh_shared_operand_t16: SmoothRelu => |rng| shared_operand(&[16], Tensor::random_range(rng, &[16], MIN_INPUT..MAX_INPUT), |b, i| b.smooth_relu(i, SmoothReluForm::GeluTanh));
    gelu_tanh_non_pow2_1000: SmoothRelu => |rng| {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T_ODD]);
        let o = b.smooth_relu(i, SmoothReluForm::GeluTanh);
        b.mark_output(o);
        (b.build_padded(), vec![Tensor::random_range(rng, &[T_ODD], -(1 << 19)..(1 << 19))])
    };
    silu_random_1k: SmoothRelu => |rng| unary_range(rng, &[T], -(1 << 19)..(1 << 19), |b, i| b.smooth_relu(i, SmoothReluForm::Silu));
    silu_domain_edges: SmoothRelu => |_| unary(&[32], S, smooth_relu_edges(), |b, i| b.smooth_relu(i, SmoothReluForm::Silu));
    silu_shared_operand_t16: SmoothRelu => |rng| shared_operand(&[16], Tensor::random_range(rng, &[16], MIN_INPUT..MAX_INPUT), |b, i| b.smooth_relu(i, SmoothReluForm::Silu));
    silu_non_pow2_1000: SmoothRelu => |rng| {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T_ODD]);
        let o = b.smooth_relu(i, SmoothReluForm::Silu);
        b.mark_output(o);
        (b.build_padded(), vec![Tensor::random_range(rng, &[T_ODD], -(1 << 19)..(1 << 19))])
    };
    tanh_random_1k: Tanh => |rng| unary_range(rng, &[T], MIN_INPUT..MAX_INPUT, |b, i| b.tanh(i));
    tanh_domain_edges: Tanh => |_| unary(&[16], S, activation_edges(), |b, i| b.tanh(i));
    tanh_shared_operand_t16: Tanh => |rng| shared_operand(&[16], Tensor::random_range(rng, &[16], MIN_INPUT..MAX_INPUT), |b, i| b.tanh(i));
    #[ignore = "non-power-of-two tanh path not supported"]
    tanh_non_pow2_1000: Tanh => |rng| unary_range(rng, &[T_ODD], MIN_INPUT..MAX_INPUT, |b, i| b.tanh(i));
    erf_random_1k: Erf => |rng| unary_range(rng, &[T], MIN_INPUT..MAX_INPUT, |b, i| b.erf(i));
    erf_domain_edges: Erf => |_| unary(&[16], S, activation_edges(), |b, i| b.erf(i));
    #[ignore = "non-power-of-two erf path not supported"]
    erf_non_pow2_1000: Erf => |rng| unary_range(rng, &[T_ODD], MIN_INPUT..MAX_INPUT, |b, i| b.erf(i));
    sigmoid_random_1k: Sigmoid => |rng| unary_range(rng, &[T], MIN_INPUT..MAX_INPUT, |b, i| b.sigmoid(i));
    sigmoid_domain_edges: Sigmoid => |_| unary(&[16], S, activation_edges(), |b, i| b.sigmoid(i));
    #[ignore = "non-power-of-two sigmoid path not supported"]
    sigmoid_non_pow2_1000: Sigmoid => |rng| unary_range(rng, &[T_ODD], MIN_INPUT..MAX_INPUT, |b, i| b.sigmoid(i));

    // Trig
    sin_random_1k: Sin => |rng| unary_range(rng, &[T], -50000..50000, |b, i| b.sin(i));
    sin_one_element: Sin => |rng| unary_range(rng, &[1], -50000..50000, |b, i| b.sin(i));
    sin_periodic_boundary: Sin => |_| unary(&[8], S, trig_edges(), |b, i| b.sin(i));
    sin_shared_operand_t8: Sin => |_| shared_operand(&[8], trig_edges(), |b, i| b.sin(i));
    #[ignore = "non-power-of-two sin path not supported"]
    sin_non_pow2_1000: Sin => |rng| unary_range(rng, &[T_ODD], -50000..50000, |b, i| b.sin(i));
    cos_random_1k: Cos => |rng| unary_range(rng, &[T], -50000..50000, |b, i| b.cos(i));
    cos_periodic_boundary: Cos => |_| unary(&[8], S, trig_edges(), |b, i| b.cos(i));
    cos_shared_operand_t8: Cos => |_| shared_operand(&[8], trig_edges(), |b, i| b.cos(i));
    #[ignore = "non-power-of-two cos path not supported"]
    cos_non_pow2_1000: Cos => |rng| unary_range(rng, &[T_ODD], -50000..50000, |b, i| b.cos(i));

    // MeanOfSquares
    mean_of_squares_16x128: MeanOfSquares => |rng| mos_model(rng, 16, 128);
    mean_of_squares_scalar_out: MeanOfSquares => |rng| mos_model(rng, 1, 128);
    /// `count = 96` vs `padded_count = 128` (GPT-2's 768-wide LayerNorm shape class).
    mean_of_squares_non_pow2_count_96: MeanOfSquares => |rng| mos_model(rng, 8, 96);
    #[should_panic(expected = "WitnessOverflow")]
    mean_of_squares_saturates: MeanOfSquares => |_| {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![2, 16]);
        let o = b.mean_of_squares(i, vec![1], vec![2, 1]);
        b.mark_output(o);
        (b.build(), vec![vec_of(SAT_SQUARE, &[2, 16])])
    };

    // Sum
    sum_axis0_1d: Sum => |rng| sum_model(rng, 32, 1, 0);
    sum_axis0_2d: Sum => |rng| sum_model(rng, 16, 32, 0);
    sum_axis1_2d: Sum => |rng| sum_model(rng, 16, 32, 1);
    sum_axis1_1d: Sum => |rng| sum_model(rng, 16, 1, 1);
    #[should_panic(expected = "WitnessOverflow")]
    sum_saturating_rows: Sum => |_| {
        let (m, n) = (4, 8);
        let data: Vec<i32> = (0..m * n)
            .map(|i| if (i / n) % 2 == 0 { i32::MAX / 2 } else { i32::MIN / 2 })
            .collect();
        let mut b = ModelBuilder::new();
        let i = b.input(vec![m, n]);
        let o = b.sum(i, vec![1], vec![m]);
        b.mark_output(o);
        (b.build(), vec![tensor(&data, &[m, n])])
    };
    #[should_panic(expected = "WitnessOverflow")]
    sum_scalar_saturating: Sum => |_| {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![64, 1]);
        let o = b.sum(i, vec![0], vec![1]);
        b.mark_output(o);
        (b.build(), vec![vec_of(i32::MAX / 2, &[64, 1])])
    };
    /// A one-element `Sum` that is not the model output and shares its range bucket with a
    /// four-element `Add`: the bucket grows from four cells to eight.
    sum_scalar_internal_shared_bucket: Sum => |rng| {
        let mut b = ModelBuilder::new();
        let u = b.input(vec![1, 8]);
        let cu = b.constant(Tensor::random_small(rng, &[1, 8]));
        let p = b.add(u, cu);
        let v = b.input(vec![4, 1]);
        let cv = b.constant(Tensor::random_small(rng, &[4, 1]));
        let q = b.add(v, cv);
        let s = b.sum(q, vec![0], vec![1]);
        let wide = b.broadcast(s, vec![1, 8]);
        let r = b.add(wide, p);
        let w = b.constant(Tensor::random_small(rng, &[1, 8]));
        let o = b.sub(r, w);
        b.mark_output(o);
        (
            b.build(),
            vec![
                Tensor::random_small(rng, &[1, 8]),
                Tensor::random_small(rng, &[4, 1]),
            ],
        )
    };
    #[ignore = "non-power-of-two sum(axis) path not supported"]
    sum_non_pow2: Sum => |rng| sum_model(rng, 33, 65, 1);

    // Softmax
    softmax_model_scale_2x8: SoftmaxLastAxis => |_| softmax_model(&[2, 8], S, tensor(&SOFTMAX_ROWS, &[2, 8]));
    softmax_one_element: SoftmaxLastAxis => |_| softmax_model(&[1, 1], S, vec_of(1234, &[1, 1]));
    softmax_random_4x16: SoftmaxLastAxis => |rng| softmax_model(&[4, 16], S, Tensor::random_range(rng, &[4, 16], -(1 << 15)..(1 << 15)));
    /// Five live lanes of eight, with values left in the three padding lanes: they must not
    /// reach the output or the row sums.
    softmax_masked_5_of_8: SoftmaxLastAxis => |_| unary(&[2, 8], S, tensor(&SOFTMAX_ROWS, &[2, 8]), |b, i| b.softmax_last_axis_masked(i, 5));
    softmax_masked_1_of_4: SoftmaxLastAxis => |rng| unary(&[4, 4], S, Tensor::random_range(rng, &[4, 4], -(1 << 15)..(1 << 15)), |b, i| b.softmax_last_axis_masked(i, 1));
    /// A width that is not a power of two, compiled with padding as the importer does.
    softmax_padded_3x5: SoftmaxLastAxis => |rng| {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![3, 5]);
        let o = b.softmax_last_axis(i);
        b.mark_output(o);
        (b.build_padded(), vec![Tensor::random_range(rng, &[3, 5], -(1 << 16)..(1 << 16))])
    };
    /// A row whose spread exceeds the tables' last address: the far lanes read zero.
    softmax_spread_beyond_the_table: SoftmaxLastAxis => |_| softmax_model(&[1, 4], S, tensor(&[0, -300_000, 5, -(1 << 22)], &[1, 4]));
    softmax_one_row_64: SoftmaxLastAxis => |rng| softmax_model(&[1, 64], S, Tensor::random_range(rng, &[1, 64], -(1 << 16)..(1 << 16)));

    // Gather
    gather_small_dict32: GatherSmall => |rng| gather_model(rng, &[4, 8], 32, 4);
    /// A two-row dictionary (one address variable): the shape `repeat_to_gather`
    /// produces for Qwen2's 2 KV heads.
    gather_small_dict2: GatherSmall => |rng| gather_model(rng, &[8, 16], 2, 8);
    /// A one-row dictionary has zero address variables, so its one-hot
    /// opening carries no claims and the prover asserts. This is the
    /// shape `repeat_to_gather` produces for a single-KV-head (MQA) model.
    #[ignore = "known gap: GatherSmall with dict_len == 1 panics in the prover \
                (opening_reduction::new_one_hot: polynomials.len() == claims.len(), 1 vs 0)"]
    known_gap_gather_small_dict1: GatherSmall => |rng| gather_model(rng, &[8, 16], 1, 8);
    #[ignore = "non-power-of-two gather path not supported"]
    gather_non_pow2: GatherSmall => |rng| gather_model(rng, &[5, 7], 33, 5);
    /// Just past `GATHER_SMALL_LIMIT`: the large-dictionary path (Qwen2's
    /// 151936-row embedding is the only zoo user).
    gather_large_dict65537: GatherLarge => |rng| gather_model(rng, &[1, 2], 65_537, 2);

    // Einsum: the four importer-canonical equations plus the shared `k,nk->n` layout.
    einsum_mk_kn_mn_16x32x64: Einsum => |rng| einsum_model(rng, "mk,kn->mn", &[16, 32], &[32, 64], &[16, 64]);
    einsum_mk_kn_mn_scalar_out: Einsum => |rng| einsum_model(rng, "mk,kn->mn", &[1, 16], &[16, 1], &[1, 1]);
    #[ignore = "non-power-of-two einsum path not supported"]
    einsum_mk_kn_mn_non_pow2: Einsum => |rng| einsum_model(rng, "mk,kn->mn", &[33, 65], &[65, 17], &[33, 17]);
    #[should_panic(expected = "WitnessOverflow")]
    einsum_mk_kn_mn_saturates_pos: Einsum => |_| einsum_const("mk,kn->mn", &[2, 16], &[16, 2], &[2, 2], SAT_EINSUM, SAT_EINSUM);
    #[should_panic(expected = "WitnessOverflow")]
    einsum_mk_kn_mn_saturates_neg: Einsum => |_| einsum_const("mk,kn->mn", &[2, 16], &[16, 2], &[2, 2], -SAT_EINSUM, SAT_EINSUM);
    einsum_k_nk_n_128x256: Einsum => |rng| einsum_model(rng, "k,nk->n", &[128], &[256, 128], &[256]);
    #[ignore = "non-power-of-two einsum path not supported"]
    einsum_k_nk_n_non_pow2: Einsum => |rng| einsum_model(rng, "k,nk->n", &[65], &[33, 65], &[33]);
    /// The importer-canonical transposed-weight form, provable for `m == 1`.
    einsum_mk_nk_mn_m1: Einsum => |rng| einsum_model(rng, "mk,nk->mn", &[1, 128], &[256, 128], &[1, 256]);
    #[should_panic(expected = "only provable with m == 1")]
    einsum_mk_nk_mn_m2_rejected: Einsum => |rng| einsum_model(rng, "mk,nk->mn", &[2, 128], &[256, 128], &[2, 256]);
    einsum_abmk_abnk_abmn: Einsum => |rng| einsum_model(rng, "abmk,abnk->abmn", &[2, 2, 4, 4], &[2, 2, 8, 4], &[2, 2, 4, 8]);
    einsum_amk_ank_amn: Einsum => |rng| einsum_model(rng, "amk,ank->amn", &[4, 4, 8], &[4, 8, 8], &[4, 4, 8]);
    /// Qwen2's GQA outer product (`k = 1`, no eq-pack variables).
    einsum_amk_ank_amn_qwen_outer_product: Einsum => |rng| einsum_model(rng, "amk,ank->amn", &[1, 32, 1], &[1, 16, 1], &[1, 32, 16]);

    // Structural ops
    broadcast_4_to_8x4: Broadcast => |rng| structural(rng, &[4], |b, i| b.broadcast(i, vec![8, 4]));
    broadcast_1x4_to_4x4: Broadcast => |rng| structural(rng, &[1, 4], |b, i| b.broadcast(i, vec![4, 4]));
    broadcast_4x1_to_4x8: Broadcast => |rng| structural(rng, &[4, 1], |b, i| b.broadcast(i, vec![4, 8]));
    broadcast_1x1x4_to_2x4x4: Broadcast => |rng| structural(rng, &[1, 1, 4], |b, i| b.broadcast(i, vec![2, 4, 4]));
    broadcast_1x4x1_to_2x4x8: Broadcast => |rng| structural(rng, &[1, 4, 1], |b, i| b.broadcast(i, vec![2, 4, 8]));
    #[ignore = "non-power-of-two broadcast path not supported"]
    broadcast_non_pow2: Broadcast => |rng| structural(rng, &[5], |b, i| b.broadcast(i, vec![3, 5]));
    moveaxis_4x8_0_1: MoveAxis => |rng| structural(rng, &[4, 8], |b, i| b.moveaxis(i, 0, 1));
    moveaxis_4x8_1_0: MoveAxis => |rng| structural(rng, &[4, 8], |b, i| b.moveaxis(i, 1, 0));
    moveaxis_2x4x8_0_1: MoveAxis => |rng| structural(rng, &[2, 4, 8], |b, i| b.moveaxis(i, 0, 1));
    moveaxis_2x4x8_0_2: MoveAxis => |rng| structural(rng, &[2, 4, 8], |b, i| b.moveaxis(i, 0, 2));
    moveaxis_2x4x8_1_2: MoveAxis => |rng| structural(rng, &[2, 4, 8], |b, i| b.moveaxis(i, 1, 2));
    #[ignore = "non-power-of-two moveaxis path not supported"]
    moveaxis_non_pow2: MoveAxis => |rng| structural(rng, &[5, 7], |b, i| b.moveaxis(i, 0, 1));
    reshape_16_to_4x4: Reshape => |rng| structural(rng, &[16], |b, i| b.reshape(i, vec![4, 4]));
    reshape_4x8_to_8x4: Reshape => |rng| structural(rng, &[4, 8], |b, i| b.reshape(i, vec![8, 4]));
    reshape_2x4x8_to_64: Reshape => |rng| structural(rng, &[2, 4, 8], |b, i| b.reshape(i, vec![64]));
    reshape_2x4x8_to_8x8: Reshape => |rng| structural(rng, &[2, 4, 8], |b, i| b.reshape(i, vec![8, 8]));
    reshape_padded_1x7x8_to_7x8: Reshape => |rng| structural_padded(rng, &[1, 7, 8], |b, i| b.reshape(i, vec![7, 8]));
    #[should_panic(expected = "is not a power of two")]
    reshape_non_pow2_dims_rejected: Reshape => |rng| structural(rng, &[10, 10], |b, i| b.reshape(i, vec![20, 5]));
    slice_8_ax0_0_4: Slice => |rng| structural(rng, &[8], |b, i| b.slice(i, 0, 0, 4));
    slice_8x4_ax0_6_8: Slice => |rng| structural(rng, &[8, 4], |b, i| b.slice(i, 0, 6, 8));
    slice_4x8x2_ax1_2_4: Slice => |rng| structural(rng, &[4, 8, 2], |b, i| b.slice(i, 1, 2, 4));
    slice_4x8_ax1_4_8: Slice => |rng| structural(rng, &[4, 8], |b, i| b.slice(i, 1, 4, 8));
    slice_4x8_ax1_full: Slice => |rng| structural(rng, &[4, 8], |b, i| b.slice(i, 1, 0, 8));
    slice_padded_5x3_ax0_0_2: Slice => |rng| structural_padded(rng, &[5, 3], |b, i| b.slice(i, 0, 0, 2));
    slice_padded_3x6_ax1_4_6: Slice => |rng| structural_padded(rng, &[3, 6], |b, i| b.slice(i, 1, 4, 6));
    #[should_panic(expected = "start 2 is not a multiple of the extent 4")]
    slice_unaligned_start_rejected: Slice => |rng| structural(rng, &[8], |b, i| b.slice(i, 0, 2, 6));
    concat_pow2_4x2_ax1: Concat => |rng| concat_model(rng, &[&[4, 2], &[4, 2]], 1);
    concat_pow2_2x4_ax0: Concat => |rng| concat_model(rng, &[&[2, 4], &[2, 4]], 0);
    concat_pow2_2x4x2_ax2: Concat => |rng| concat_model(rng, &[&[2, 4, 2], &[2, 4, 2]], 2);
    concat_pow2_2x2x4_axneg2: Concat => |rng| concat_model(rng, &[&[2, 2, 4], &[2, 2, 4]], -2);
    concat_pow2_2x2x2x2_ax3: Concat => |rng| concat_model(rng, &[&[2, 2, 2, 2], &[2, 2, 2, 2]], 3);
    concat_three_inputs_8_4_4_ax0: Concat => |rng| concat_model(rng, &[&[8, 2], &[4, 2], &[4, 2]], 0);
    concat_three_inputs_4_2_2_ax1: Concat => |rng| concat_model(rng, &[&[2, 4], &[2, 2], &[2, 2]], 1);
    concat_four_inputs_axneg1: Concat => |rng| concat_model(rng, &[&[2, 2, 4], &[2, 2, 4], &[2, 2, 4], &[2, 2, 4]], -1);
    concat_single_input: Concat => |rng| concat_model(rng, &[&[4, 8, 2]], 1);
    concat_padded_4_3_ax0: Concat => |rng| concat_padded(rng, &[&[4], &[3]], 0);
    concat_padded_3x4_3x3_ax1: Concat => |rng| concat_padded(rng, &[&[3, 4], &[3, 3]], 1);
    concat_padded_8_4_3_ax0: Concat => |rng| concat_padded(rng, &[&[8, 5], &[4, 5], &[3, 5]], 0);
    #[should_panic(expected = "offset 4 is not a multiple of the operand extent 8")]
    concat_unaligned_offset_rejected: Concat => |rng| concat_model(rng, &[&[4], &[8], &[4]], 0);

    // Multi-node and padded builds
    /// `Add` nodes of two different sizes in one model, packed into one range bucket of
    /// `range_bucket`; an identity is the model output, so both are members.
    range_bucket_packing_mixed_sizes: Add => |rng| {
        let mut b = ModelBuilder::new();
        let x = b.input(vec![T]);
        let c = b.constant(Tensor::random_small(rng, &[T]));
        let a = b.add(x, c);
        let r = b.relu(a);
        let s = b.slice(r, 0, 0, 256);
        let c2 = b.constant(Tensor::random_small(rng, &[256]));
        let small = b.add(s, c2);
        let o = b.identity(small);
        b.mark_output(o);
        (b.build(), vec![Tensor::random_small(rng, &[T])])
    };
    /// Production padding path (`pad_pow2 = true`): the importer's build.
    padded_mean_of_squares_96: MeanOfSquares => |rng| {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![8, 96]);
        let o = b.mean_of_squares(i, vec![1], vec![8, 1]);
        b.mark_output(o);
        (b.build_padded(), vec![Tensor::random_small(rng, &[8, 96])])
    };
    padded_add_1000: Add => |rng| {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T_ODD]);
        let c = b.constant(Tensor::random_small(rng, &[T_ODD]));
        let o = b.add(i, c);
        b.mark_output(o);
        (b.build_padded(), vec![Tensor::random_small(rng, &[T_ODD])])
    };
}

// ── coverage meta-tests ──────────────────────────────────────────────────

#[test]
fn every_op_kind_has_a_declared_case() {
    let declared: BTreeSet<OpKind> = CASES.iter().map(|c| c.kind).collect();
    let missing: Vec<OpKind> = OpKind::iter().filter(|k| !declared.contains(k)).collect();
    assert!(
        missing.is_empty(),
        "ops without an op-surface case: {missing:?}"
    );
}

#[test]
fn every_case_contains_its_declared_op() {
    let mut names = BTreeSet::new();
    for case in CASES {
        assert!(names.insert(case.name), "duplicate case name {}", case.name);
        let mut rng = StdRng::seed_from_u64(seed_of(case.name));
        let (model, inputs) = (case.build)(&mut rng);
        assert!(
            model
                .graph()
                .nodes
                .iter()
                .any(|n| OpKind::of(&n.op) == case.kind),
            "case {} declares {:?} but its model has no such node",
            case.name,
            case.kind
        );
        assert_eq!(
            inputs.len(),
            model.graph().inputs.len(),
            "case {} supplies {} inputs for {} graph inputs",
            case.name,
            inputs.len(),
            model.graph().inputs.len()
        );
    }
}
