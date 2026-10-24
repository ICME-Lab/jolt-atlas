//! Golden-output regression tests over representative graphs.
//!
//! These replace the retired differential suite against `atlas-onnx-tracer`:
//! every pinned hash below was produced by a pipeline that had been verified
//! byte-identical to the old tracer on these exact graphs and inputs (exact
//! class) or within one table ulp (table class). A hash change means the
//! traced semantics changed and must be deliberate (usually a `SemVersion`
//! bump).

use atlas_trace::{CompileOptions, GraphBuilder, Model, Scale, SemVersion, Tensor, op, op::Op};
use rand::{SeedableRng, rngs::StdRng};

const SCALE: u8 = 14;

/// FNV-1a 64-bit over a trace: every node output, in node order, plus the
/// fused quotients/remainders and softmax witness fields.
fn trace_hash(model: &Model, inputs: &[Tensor<i32>]) -> u64 {
    let trace = model.trace(inputs).expect("trace");
    let mut hash: u64 = 0xcbf29ce484222325;
    let mut absorb_i64 = |v: i64| {
        for byte in v.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    for out in &trace.node_outputs {
        for &v in out.data() {
            absorb_i64(v as i64);
        }
    }
    for id in trace.aux.keys() {
        if let Some(f) = trace.fused(*id) {
            for &q in f.quotient.data() {
                absorb_i64(q);
            }
            for &r in f.remainder.data() {
                absorb_i64(r as i64);
            }
        }
        if let Some(w) = trace.softmax(*id) {
            for &v in w.exp_q.iter().chain(&w.inv_sum).chain(&w.r).chain(&w.r_exp) {
                absorb_i64(v as i64);
            }
        }
    }
    hash
}

fn compile(b: GraphBuilder, pad: bool) -> Model {
    b.build(CompileOptions {
        scale: Scale::new(SCALE).unwrap(),
        pad_pow2: pad,
        version: SemVersion::V1,
    })
    .unwrap()
}

fn random_tensor(rng: &mut StdRng, dims: &[usize], bound: i32) -> Tensor<i32> {
    Tensor::random_range(rng, dims, -bound..bound)
}

/// Exact-class ops: elementwise arithmetic, fused rescales, reductions,
/// select/boolean ops. Pinned from the differential-validated pipeline.
#[test]
fn exact_class_golden() {
    let mut rng = StdRng::seed_from_u64(0xA71A5);
    let mut b = GraphBuilder::new();
    let dims = [4usize, 8];
    let x = b.input(&dims);
    let y = b.input(&dims);

    let divisor = b.constant(Tensor::random_range(&mut rng, &dims, 1..5000));
    let mask = b.constant(Tensor::random_range(&mut rng, &dims, 0..2));
    let boolean = b.constant(Tensor::random_range(&mut rng, &dims, 0..2));

    let add = b.push(Op::Add(op::Add), &[x, y]);
    let sub = b.push(Op::Sub(op::Sub), &[add, y]);
    let neg = b.push(Op::Neg(op::Neg), &[sub]);
    let relu = b.push(Op::ReLU(op::ReLU), &[neg]);
    let mul = b.push(Op::Mul(op::Mul { rebase: true }), &[relu, y]);
    let square = b.push(Op::Square(op::Square { rebase: true }), &[mul]);
    let cube = b.push(Op::Cube(op::Cube { rebase: true }), &[square]);
    let div = b.push(Op::Div(op::Div), &[cube, divisor]);
    let scd = b.push(
        Op::ScalarConstDiv(op::ScalarConstDiv { divisor: -3 }),
        &[div],
    );
    let clamp = b.push(Op::Clamp(op::Clamp { bound_log: 16 }), &[scd]);
    let iff = b.push(Op::Iff(op::Iff), &[mask, clamp, x]);
    let _and = b.push(Op::And(op::And), &[mask, boolean]);
    let _isnan = b.push(Op::IsNan(op::IsNan), &[iff]);
    let rsqrt = b.push(Op::Rsqrt(op::Rsqrt), &[iff]);
    let sum = b.push(Op::Sum(op::Sum { axes: vec![1] }), &[rsqrt]);
    let out = b.push(Op::Identity(op::Identity), &[sum]);
    b.output(out);

    let model = compile(b, false);
    let inputs = [
        random_tensor(&mut rng, &dims, 1 << 16),
        random_tensor(&mut rng, &dims, 1 << 16),
    ];
    assert_eq!(trace_hash(&model, &inputs), 0xbaf1350d08519bbf);
}

/// Structural ops + einsum (constant and dynamic operands) + gather.
#[test]
fn structural_and_einsum_golden() {
    let mut rng = StdRng::seed_from_u64(0x5EED);
    let mut b = GraphBuilder::new();
    let x = b.input(&[4, 8]);

    let w = b.constant(random_tensor(&mut rng, &[8, 4], 1 << 14));
    let matmul = b.push(
        Op::Einsum(op::Einsum {
            equation: "mk,kn->mn".parse().unwrap(),
        }),
        &[x, w],
    );
    let mos = b.push(
        Op::MeanOfSquares(op::MeanOfSquares {
            axes: vec![1],
            count: 4,
            padded_count: 4,
        }),
        &[matmul],
    );
    let bcast = b.push(Op::Broadcast(op::Broadcast { shape: vec![4, 4] }), &[mos]);
    let reshape = b.push(Op::Reshape(op::Reshape { shape: vec![2, 8] }), &[bcast]);
    let moveaxis = b.push(
        Op::MoveAxis(op::MoveAxis {
            source: 0,
            destination: 1,
        }),
        &[reshape],
    );
    let slice = b.push(
        Op::Slice(op::Slice {
            axis: 0,
            start: 2,
            end: 6,
        }),
        &[moveaxis],
    );
    let concat = b.push(Op::Concat(op::Concat { axis: 0 }), &[slice, slice]);

    let dict = b.constant(random_tensor(&mut rng, &[16, 4], 1 << 14));
    let indices = b.constant(Tensor::random_range(&mut rng, &[6], 0..16));
    let gather = b.push(
        Op::GatherSmall(op::GatherSmall {
            axis: 0,
            dict_len: 16,
        }),
        &[dict, indices],
    );
    let q = b.push(
        Op::Einsum(op::Einsum {
            equation: "mk,nk->mn".parse().unwrap(),
        }),
        &[gather, gather],
    );
    b.output(concat);
    b.output(q);

    let model = compile(b, false);
    let inputs = [random_tensor(&mut rng, &[4, 8], 1 << 15)];
    assert_eq!(trace_hash(&model, &inputs), 0x5d575e107a6f5d88);
}

/// Table-class ops: activations, trig, softmax — the deterministic integer
/// tables are the spec, so these pin exact values (not tolerances).
#[test]
fn table_class_golden() {
    let mut rng = StdRng::seed_from_u64(0x7AB7E);
    let mut b = GraphBuilder::new();
    let x = b.input(&[64]);
    let tanh = b.push(Op::Tanh(op::Tanh), &[x]);
    let sigmoid = b.push(Op::Sigmoid(op::Sigmoid), &[x]);
    let erf = b.push(Op::Erf(op::Erf), &[x]);
    let sin = b.push(Op::Sin(op::Sin), &[x]);
    let cos = b.push(Op::Cos(op::Cos), &[x]);
    for n in [tanh, sigmoid, erf, sin, cos] {
        b.output(n);
    }
    let model = compile(b, false);
    let inputs = [random_tensor(&mut rng, &[64], 1 << 18)];
    assert_eq!(trace_hash(&model, &inputs), 0xdd3241147bffaedb);
}

/// Softmax with and without padding masking.
#[test]
fn softmax_golden() {
    let mut rng = StdRng::seed_from_u64(0x50F7);
    let mut b = GraphBuilder::new();
    let x = b.input(&[4, 8]);
    let sm = b.push(
        Op::SoftmaxLastAxis(op::SoftmaxLastAxis { active_len: None }),
        &[x],
    );
    b.output(sm);
    let model = compile(b, false);
    let inputs = [random_tensor(&mut rng, &[4, 8], 1 << 15)];
    assert_eq!(trace_hash(&model, &inputs), 0xf01505b3d1b835d6);

    // Masked: a 3-wide row padded to 4 must reproduce the unpadded rows.
    let mut b = GraphBuilder::new();
    let x = b.input(&[2, 3]);
    let sm = b.push(
        Op::SoftmaxLastAxis(op::SoftmaxLastAxis { active_len: None }),
        &[x],
    );
    b.output(sm);
    let padded = compile(b, true);
    let live = Tensor::new(vec![100, 300, -50, 7, -1000, 12000], &[2, 3]).unwrap();
    let out = padded.forward(std::slice::from_ref(&live)).unwrap();

    let mut b = GraphBuilder::new();
    let x = b.input(&[2, 3]);
    let sm = b.push(
        Op::SoftmaxLastAxis(op::SoftmaxLastAxis { active_len: None }),
        &[x],
    );
    b.output(sm);
    let unpadded = compile(b, false);
    let expected = unpadded.forward(&[live]).unwrap();
    assert_eq!(out[0].data(), expected[0].data());
}

/// Print current hashes (run with `--ignored --nocapture` to regenerate the
/// pins after a deliberate semantics change).
#[test]
#[ignore]
fn print_hashes() {
    // Re-run each test body's model/inputs and print; simplest is to run the
    // tests and read the assertion failures, so this is a reminder stub.
    eprintln!("run the golden tests and update pins from assertion messages");
}
