//! Deferred products, wide values and `Rescale`: executions against the chain written out in
//! wide integers with one division, and the compile-time level and bound rules.

use atlas_trace::{
    CompileOptions, GraphBuilder, NodeId, Op, Tensor,
    error::{CompileError, ExecError},
    op,
};

const SCALE: u32 = 14;

fn value(seed: usize, span: i64) -> i32 {
    ((seed as i64 * 2_654_435_761) % (2 * span + 1) - span) as i32
}

fn tensor(seed: usize, dims: &[usize], span: i64) -> Tensor<i32> {
    let len: usize = dims.iter().product();
    Tensor::new((0..len).map(|i| value(i + seed, span)).collect(), dims).unwrap()
}

fn einsum(deferred: bool) -> Op {
    Op::Einsum(op::Einsum {
        equation: "mk,kn->mn".parse().unwrap(),
        deferred,
    })
}

fn mul_deferred() -> Op {
    Op::Mul(op::Mul {
        rebase: false,
        deferred: true,
    })
}

/// `down(gate · up(x))` with one rescale: the model's output is the exact chain value's
/// Euclidean quotient by `2^(3·S)`, and its remainder is the one captured.
#[test]
fn a_three_product_chain_rounds_once() {
    let (m, k, h, n) = (2usize, 4, 8, 4);
    let (w_up, w_down) = (tensor(3, &[k, h], 40_000), tensor(5, &[h, n], 40_000));
    let mut b = GraphBuilder::new();
    let x = b.input(&[m, k]);
    let gate = b.input(&[m, h]);
    let up = b.constant(w_up.clone());
    let down = b.constant(w_down.clone());
    let wide_up = b.push(einsum(true), &[x, up]);
    let gated = b.push(mul_deferred(), &[wide_up, gate]);
    let wide_down = b.push(einsum(true), &[gated, down]);
    let out = b.push(Op::Rescale(op::Rescale), &[wide_down]);
    b.output(out);
    let model = b.build(CompileOptions::default()).unwrap();
    assert_eq!(
        [wide_up, gated, wide_down, out].map(|id| model.level(id)),
        [2, 3, 4, 1]
    );

    let (x_in, gate_in) = (tensor(7, &[m, k], 3_000_000), tensor(11, &[m, h], 200_000));
    let trace = model.trace(&[x_in.clone(), gate_in.clone()]).unwrap();
    let divisor = 1i128 << (3 * SCALE);
    for i in 0..m {
        for c in 0..n {
            let exact: i128 = (0..h)
                .map(|j| {
                    let up: i128 = (0..k)
                        .map(|a| x_in.data()[i * k + a] as i128 * w_up.data()[a * h + j] as i128)
                        .sum();
                    up * gate_in.data()[i * h + j] as i128 * w_down.data()[j * n + c] as i128
                })
                .sum();
            assert_eq!(trace.wide(wide_down).unwrap().data()[i * n + c], exact);
            assert_eq!(
                trace.output(out).data()[i * n + c] as i128,
                exact.div_euclid(divisor)
            );
            assert_eq!(
                trace.rescale_remainder(out).unwrap().data()[i * n + c] as i128,
                exact.rem_euclid(divisor)
            );
        }
    }
}

/// One deferred contraction rescaled at once is the fused contraction.
#[test]
fn a_deferred_contraction_rescaled_is_the_fused_one() {
    let w = tensor(1, &[4, 4], 30_000);
    let build = |deferred: bool| {
        let mut b = GraphBuilder::new();
        let x = b.input(&[2, 4]);
        let w = b.constant(w.clone());
        let mut y = b.push(einsum(deferred), &[x, w]);
        if deferred {
            y = b.push(Op::Rescale(op::Rescale), &[y]);
        }
        b.output(y);
        b.build(CompileOptions::default()).unwrap()
    };
    let x = tensor(9, &[2, 4], 1_000_000);
    assert_eq!(
        build(true).forward(std::slice::from_ref(&x)).unwrap(),
        build(false).forward(&[x]).unwrap()
    );
}

/// A quotient outside `i32` is an execution error, and the largest one inside is not.
#[test]
fn a_rescale_quotient_outside_i32_is_an_error() {
    let build = |weight: i32| {
        let mut b = GraphBuilder::new();
        let x = b.input(&[1, 1]);
        let w = b.constant(Tensor::new(vec![weight], &[1, 1]).unwrap());
        let wide = b.push(einsum(true), &[x, w]);
        let out = b.push(Op::Rescale(op::Rescale), &[wide]);
        b.output(out);
        b.build(CompileOptions::default()).unwrap()
    };
    let x = Tensor::new(vec![i32::MAX], &[1, 1]).unwrap();
    // x · 2^14 / 2^14 = x: the largest quotient.
    let at_edge = build(1 << SCALE).forward(std::slice::from_ref(&x)).unwrap();
    assert_eq!(at_edge[0].data(), &[i32::MAX]);
    let over = build((1 << SCALE) + 1).forward(&[x]);
    assert!(matches!(over, Err(ExecError::Kernel { .. })), "{over:?}");
}

fn compile_error(build: impl FnOnce(&mut GraphBuilder) -> NodeId) -> CompileError {
    let mut b = GraphBuilder::new();
    let out = build(&mut b);
    b.output(out);
    b.build(CompileOptions::default()).unwrap_err()
}

/// A wide value is not a graph output, not an operand of an op without a wide form, and not
/// the second operand of a contraction.
#[test]
fn levels_are_checked_at_compile() {
    let wide = |b: &mut GraphBuilder| {
        let x = b.input(&[2, 4]);
        let w = b.constant(tensor(1, &[4, 4], 100));
        b.push(einsum(true), &[x, w])
    };
    assert!(matches!(compile_error(wide), CompileError::Level { .. }));
    let into_relu = |b: &mut GraphBuilder| {
        let y = wide(b);
        b.push(Op::ReLU(op::ReLU), &[y])
    };
    assert!(matches!(
        compile_error(into_relu),
        CompileError::Level { .. }
    ));
    let as_weights = |b: &mut GraphBuilder| {
        let y = wide(b);
        let x = b.input(&[2, 2]);
        let z = b.push(einsum(true), &[x, y]);
        b.push(Op::Rescale(op::Rescale), &[z])
    };
    assert!(matches!(
        compile_error(as_weights),
        CompileError::Level { .. }
    ));
    let narrow_rescale = |b: &mut GraphBuilder| {
        let x = b.input(&[2, 4]);
        b.push(Op::Rescale(op::Rescale), &[x])
    };
    assert!(matches!(
        compile_error(narrow_rescale),
        CompileError::Level { .. }
    ));
}

/// A deferred contraction of two level-1 values that are not constants is bounded by
/// `terms · 2^62`, past what its `i64` accumulation holds; against a constant it fits.
#[test]
fn bounds_are_checked_at_compile() {
    let unbounded = |b: &mut GraphBuilder| {
        let x = b.input(&[2, 4]);
        let y = b.input(&[4, 4]);
        let z = b.push(einsum(true), &[x, y]);
        b.push(Op::Rescale(op::Rescale), &[z])
    };
    assert!(matches!(
        compile_error(unbounded),
        CompileError::Bound { limit: 62, .. }
    ));
    let mut b = GraphBuilder::new();
    let x = b.input(&[2, 4]);
    let weights = tensor(1, &[4, 4], 1 << 16);
    let largest = weights
        .data()
        .iter()
        .map(|w| w.unsigned_abs())
        .max()
        .unwrap() as u128;
    let w = b.constant(weights);
    let z = b.push(einsum(true), &[x, w]);
    let out = b.push(Op::Rescale(op::Rescale), &[z]);
    b.output(out);
    let model = b.build(CompileOptions::default()).unwrap();
    // 4 terms, each at most 2^31 times the largest weight.
    assert_eq!(model.bound(z), 4 * (1u128 << 31) * largest);
    assert_eq!(model.bound(out), 1u128 << 31);
}

/// The placement pass on `down(relu(gate(x)) · up(x))` followed by a table read: the gate
/// projection stays fused (a ReLU reads it), the up projection, the multiply and the down
/// projection are deferred, and one rescale follows the down projection. The result is the
/// exact chain value rounded once.
#[test]
fn the_placement_pass_rescales_a_chain_once() {
    use atlas_trace::{Scale, defer::defer_rescales};
    let (m, k, h, n) = (2usize, 4, 8, 4);
    let weights = [
        tensor(1, &[k, h], 20_000),
        tensor(2, &[k, h], 20_000),
        tensor(3, &[h, n], 20_000),
    ];
    let fused_mul = Op::Mul(op::Mul {
        rebase: true,
        deferred: false,
    });
    let mut b = GraphBuilder::new();
    let x = b.input(&[m, k]);
    let [w_gate, w_up, w_down] = weights.clone().map(|w| b.constant(w));
    let gate = b.push(einsum(false), &[x, w_gate]);
    let gate = b.push(Op::ReLU(op::ReLU), &[gate]);
    let up = b.push(einsum(false), &[x, w_up]);
    let gated = b.push(fused_mul, &[up, gate]);
    let down = b.push(einsum(false), &[gated, w_down]);
    let out = b.push(Op::ReLU(op::ReLU), &[down]);
    b.output(out);
    let (graph, report) = defer_rescales(b.build_graph().unwrap(), Scale::new(14).unwrap());
    assert_eq!((report.deferred, report.rescales), (3, 1));
    let forms: Vec<String> = graph.nodes.iter().map(|node| node.op.to_string()).collect();
    assert_eq!(
        forms[4..],
        [
            "Einsum(mk,kn->mn)",
            "ReLU",
            "Einsum(mk,kn->mn, deferred)",
            "Mul(deferred)",
            "Einsum(mk,kn->mn, deferred)",
            "Rescale",
            "ReLU",
        ]
    );
    let model = atlas_trace::Model::compile(graph, CompileOptions::default()).unwrap();
    let x_in = tensor(7, &[m, k], 2_000_000);
    let got = model.forward(std::slice::from_ref(&x_in)).unwrap();
    let scale = 1i128 << SCALE;
    for i in 0..m {
        for c in 0..n {
            let exact: i128 = (0..h)
                .map(|j| {
                    let dot = |w: &Tensor<i32>| -> i128 {
                        (0..k)
                            .map(|a| x_in.data()[i * k + a] as i128 * w.data()[a * h + j] as i128)
                            .sum()
                    };
                    let gate = dot(&weights[0]).div_euclid(scale).max(0);
                    dot(&weights[1]) * gate * weights[2].data()[j * n + c] as i128
                })
                .sum();
            let want = exact.div_euclid(scale * scale * scale).max(0);
            assert_eq!(got[0].data()[i * n + c] as i128, want);
        }
    }
}

/// A deferred contraction plus a bias, less a level-1 value, rescaled: the sums lift their
/// level-1 operands by the scale, and the result is the exact value rounded once.
#[test]
fn a_wide_sum_lifts_its_narrow_operand() {
    let (m, k, n) = (2usize, 4, 4);
    let (w, bias, offset) = (
        tensor(1, &[k, n], 30_000),
        tensor(2, &[m, n], 500_000),
        tensor(3, &[m, n], 500_000),
    );
    let mut b = GraphBuilder::new();
    let x = b.input(&[m, k]);
    let [w_id, bias_id, offset_id] =
        [w.clone(), bias.clone(), offset.clone()].map(|t| b.constant(t));
    let wide = b.push(einsum(true), &[x, w_id]);
    let plus = b.push(Op::Add(op::Add), &[wide, bias_id]);
    let less = b.push(Op::Sub(op::Sub), &[offset_id, plus]);
    let out = b.push(Op::Rescale(op::Rescale), &[less]);
    b.output(out);
    let model = b.build(CompileOptions::default()).unwrap();
    assert_eq!(
        [wide, plus, less, out].map(|id| model.level(id)),
        [2, 2, 2, 1]
    );
    let x_in = tensor(7, &[m, k], 3_000_000);
    let got = model.forward(std::slice::from_ref(&x_in)).unwrap();
    let scale = 1i128 << SCALE;
    for i in 0..m {
        for c in 0..n {
            let dot: i128 = (0..k)
                .map(|a| x_in.data()[i * k + a] as i128 * w.data()[a * n + c] as i128)
                .sum();
            let exact = offset.data()[i * n + c] as i128 * scale
                - (dot + bias.data()[i * n + c] as i128 * scale);
            assert_eq!(got[0].data()[i * n + c] as i128, exact.div_euclid(scale));
        }
    }
}
