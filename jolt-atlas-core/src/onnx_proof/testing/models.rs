//! Small multi-op models for the soundness sweeps. Each is a single-output
//! graph a few dozen elements wide, so a prove costs a fraction of a second
//! and a verify a few milliseconds.

use atlas_trace::{Model, Tensor};
use common::consts::MODEL_SCALE;
use rand::{rngs::StdRng, SeedableRng};

use super::ModelBuilder;

const S: u8 = MODEL_SCALE as u8;

/// A named model with its inputs.
pub(crate) struct ProbeModel {
    pub name: &'static str,
    pub model: Model,
    pub inputs: Vec<Tensor<i32>>,
}

/// Every probe model, each on its own seed.
pub(crate) fn probe_models() -> Vec<ProbeModel> {
    vec![
        attention_like(0x5EED_0001),
        reductions(0x5EED_0002),
        tables_and_layout(0x5EED_0003),
        relation_flows(0x5EED_0004),
        scalars(0x5EED_0006),
        smooth_relus(0x5EED_0007),
        deferred_chains(0x5EED_0008),
    ]
}

/// Same models, different inputs: material for commitment / opening splices.
pub(crate) fn probe_models_alt() -> Vec<ProbeModel> {
    vec![
        attention_like(0xA17_0001),
        reductions(0xA17_0002),
        tables_and_layout(0xA17_0003),
        relation_flows(0xA17_0004),
        scalars(0xA17_0006),
        smooth_relus(0xA17_0007),
        deferred_chains(0xA17_0008),
    ]
}

/// Deferred products with one rescale per chain: `down(gate · up(x))` through levels 2, 3
/// and 4, where the gate is a ReLU of a fused projection, and a deferred projection with a
/// bias added and a constant it is subtracted from, rescaled from level 2. The two level-1 results are summed into one output.
fn deferred_chains(seed: u64) -> ProbeModel {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = ModelBuilder::new();
    let x = b.input(vec![4, 8]);
    let w_gate = b.constant(Tensor::random_small(&mut rng, &[8, 16]));
    let w_up = b.constant(Tensor::random_small(&mut rng, &[8, 16]));
    let w_down = b.constant(Tensor::random_small(&mut rng, &[16, 8]));
    let w_side = b.constant(Tensor::random_small(&mut rng, &[8, 8]));
    let gate = b.einsum("mk,kn->mn", vec![x, w_gate], vec![4, 16]);
    let gate = b.relu(gate);
    let up = b.einsum_deferred("mk,kn->mn", vec![x, w_up], vec![4, 16]);
    let gated = b.mul_deferred(up, gate);
    let down = b.einsum_deferred("mk,kn->mn", vec![gated, w_down], vec![4, 8]);
    let chain = b.rescale(down);
    // A projection plus a bias and less a constant, both lifted to the projection's level.
    let side = b.einsum_deferred("mk,kn->mn", vec![x, w_side], vec![4, 8]);
    let bias = b.constant(Tensor::random_small(&mut rng, &[4, 8]));
    let side = b.add(side, bias);
    let offset = b.constant(Tensor::random_small(&mut rng, &[4, 8]));
    let side = b.sub(offset, side);
    let side = b.rescale(side);
    let out = b.add(chain, side);
    b.mark_output(out);
    let inputs = vec![Tensor::random_range(
        &mut rng,
        &[4, 8],
        -(1 << 12)..(1 << 12),
    )];
    ProbeModel {
        name: "deferred_chains",
        model: b.build(),
        inputs,
    }
}

/// einsum → add → relu (consumed twice: fan-out) → softmax → scalar_const_div → reshape →
/// sub, plus an embedding gather and a fused mul, summed into one output.
fn attention_like(seed: u64) -> ProbeModel {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = ModelBuilder::new();
    let x = b.input(vec![4, 8]);
    let w = b.constant(Tensor::random_small(&mut rng, &[8, 8]));
    let h = b.einsum("mk,kn->mn", vec![x, w], vec![4, 8]);
    let bias = b.constant(Tensor::random_small(&mut rng, &[4, 8]));
    let a = b.add(h, bias);
    let r = b.relu(a);
    let s = b.softmax_last_axis(r);
    let d = b.scalar_const_div(s, 1 << 8);
    let q = b.reshape(d, vec![32]);
    let c1 = b.constant(Tensor::random_small(&mut rng, &[32]));
    let e = b.sub(q, c1);
    let ids = b.input(vec![8]);
    let dict = b.constant(Tensor::random_small(&mut rng, &[16, 4]));
    let g = b.gather(dict, ids, 0, vec![8, 4]);
    let g2 = b.reshape(g, vec![32]);
    let rf = b.reshape(r, vec![32]);
    let c2 = b.constant(Tensor::random_small(&mut rng, &[32]));
    let m = b.mul(rf, c2);
    let t = b.add(e, g2);
    let out = b.add(t, m);
    b.mark_output(out);
    let inputs = vec![
        Tensor::random_range(&mut rng, &[4, 8], -(1 << 12)..(1 << 12)),
        Tensor::random_range(&mut rng, &[8], 0..16),
    ];
    ProbeModel {
        name: "attention_like",
        model: b.build(),
        inputs,
    }
}

/// The claim flows a relation-defined op takes part in (`Mul`, see `onnx_proof::relation`):
/// a fused product, whose input claim reads the node's reduced output and remainder advice and
/// whose range claims are proven in the packed buckets after the nodes' steps; a raw product with two
/// consumers, whose input claim is the evaluation reduction's instance for its output; and an
/// operand shared by a product and an add, whose reduction folds the relation's produced
/// opening with the add's.
fn relation_flows(seed: u64) -> ProbeModel {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = ModelBuilder::new();
    let x = b.input(vec![16]);
    let y = b.input(vec![16]);
    let shared = b.add(x, y);
    let fused = b.mul(shared, y);
    let raw = b.mul_raw(shared, x);
    let first = b.add(raw, fused);
    let second = b.sub(first, raw);
    let out = b.add(second, shared);
    b.mark_output(out);
    let inputs = vec![
        Tensor::random_range(&mut rng, &[16], -(1 << 7)..(1 << 7)),
        Tensor::random_range(&mut rng, &[16], -(1 << 7)..(1 << 7)),
    ];
    ProbeModel {
        name: "relation_flows",
        model: b.build(),
        inputs,
    }
}

/// One-element range-checked nodes only: a fused `Mul`, two `Add`s, and the output `Sub`. With
/// no larger member the bucket capacity is one cell, so every hidden output and the `Mul`'s
/// remainder is alone in a one-cell range bucket: each is the first member of its bucket,
/// where the digit tamper sites land. The values keep a non-zero digit above the least
/// significant one in every `out + 2^31` and in the remainder (`scalars_probe_layout` pins
/// the layout and the values).
fn scalars(seed: u64) -> ProbeModel {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = ModelBuilder::new();
    let x = b.input(vec![1]);
    let c = b.constant(Tensor::from_slice(&[SCALARS_PROBE_FACTOR], &[1]).unwrap());
    let m = b.mul(x, c);
    let y = b.input(vec![1]);
    let a = b.add(m, y);
    let top = b.constant(Tensor::from_slice(&[1 << 20], &[1]).unwrap());
    let s = b.add(a, top);
    let z = b.input(vec![1]);
    let out = b.sub(s, z);
    b.mark_output(out);
    // A remainder of `x·factor` below 4 has no digit for the tamper to move.
    let x = loop {
        let x = Tensor::random_range(&mut rng, &[1], 900..1100);
        let product = i64::from(x.data()[0]) * i64::from(SCALARS_PROBE_FACTOR);
        if product.rem_euclid(1 << MODEL_SCALE) >= 4 {
            break x;
        }
    };
    let inputs = vec![
        x,
        Tensor::random_range(&mut rng, &[1], 1..1000),
        Tensor::random_range(&mut rng, &[1], 1..1000),
    ];
    ProbeModel {
        name: "scalars",
        model: b.build(),
        inputs,
    }
}

/// The constant factor of the `scalars` probe's fused `Mul`.
pub(crate) const SCALARS_PROBE_FACTOR: i32 = 3001;

/// rsqrt / cos / sin / iff / square / mean_of_squares / sum / cube /
/// scalar_const_div on a 64-element input.
fn reductions(seed: u64) -> ProbeModel {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = ModelBuilder::new();
    let x = b.input(vec![64]);
    let rs = b.rsqrt(x);
    let c = b.cos(x);
    let s = b.sin(x);
    let mask = b.constant(Tensor::random_boolean(&mut rng, &[64]));
    let sel = b.iff(mask, c, s);
    let sq = b.square(sel);
    let y = b.reshape(sq, vec![8, 8]);
    let mos = b.mean_of_squares(y, vec![1], vec![8, 1]);
    let sm = b.sum(y, vec![1], vec![8]);
    let sm_flat = b.reshape(sm, vec![8]);
    let mos_flat = b.reshape(mos, vec![8]);
    let red = b.add(mos_flat, sm_flat);
    let cu = b.cube(rs, S);
    let sd = b.scalar_const_div(cu, 7);
    let sd8 = b.slice(sd, 0, 0, 8);
    let out = b.sub(red, sd8);
    b.mark_output(out);
    let inputs = vec![Tensor::random_range(&mut rng, &[64], 1..(1 << 12))];
    ProbeModel {
        name: "reductions",
        model: b.build(),
        inputs,
    }
}

/// tanh / erf / sigmoid / neg / concat / slice / moveaxis / broadcast /
/// is_nan / and / identity on a 16-element input.
fn tables_and_layout(seed: u64) -> ProbeModel {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = ModelBuilder::new();
    let x = b.input(vec![16]);
    let t = b.tanh(x);
    let e = b.erf(x);
    let sg = b.sigmoid(x);
    let n = b.neg(t);
    // Raw (un-rescaled) cube must stay inside i32: erf is ±2^14 wide.
    let e_small = b.scalar_const_div(e, 1 << 10);
    let cu = b.cube(e_small, 0);
    // Three operands tiling the axis (16 + 8 + 8), two of them slices, one at a nonzero start.
    let cu_tail = b.slice(cu, 0, 8, 16);
    let sg_head = b.slice(sg, 0, 0, 8);
    let cat = b.concat(&[n, cu_tail, sg_head], 0);
    let rs = b.reshape(cat, vec![4, 8]);
    let mv = b.moveaxis(rs, 0, 1);
    let head = b.slice(x, 0, 0, 4);
    let head_row = b.reshape(head, vec![1, 4]);
    let bc = b.broadcast(head_row, vec![8, 4]);
    let a = b.add(mv, bc);
    let nan = b.is_nan(a);
    let m1 = b.input(vec![8, 4]);
    let m2 = b.constant(Tensor::random_boolean(&mut rng, &[8, 4]));
    let an = b.and(m1, m2);
    let z = b.add(an, nan);
    let sel = b.iff(z, a, bc);
    let out = b.identity(sel);
    b.mark_output(out);
    let inputs = vec![
        Tensor::random_range(&mut rng, &[16], -(1 << 16)..(1 << 16)),
        Tensor::random_boolean(&mut rng, &[8, 4]),
    ];
    ProbeModel {
        name: "tables_and_layout",
        model: b.build(),
        inputs,
    }
}

/// The three smooth ReLUs on one 16-element input reaching past both clamp bounds, their
/// outputs added so each has a consumer.
fn smooth_relus(seed: u64) -> ProbeModel {
    use atlas_trace::op::SmoothReluForm;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut b = ModelBuilder::new();
    let x = b.input(vec![16]);
    let gelu_erf = b.smooth_relu(x, SmoothReluForm::GeluErf);
    let gelu_tanh = b.smooth_relu(x, SmoothReluForm::GeluTanh);
    let silu = b.smooth_relu(x, SmoothReluForm::Silu);
    let gelus = b.add(gelu_erf, gelu_tanh);
    let out = b.add(gelus, silu);
    b.mark_output(out);
    let inputs = vec![Tensor::random_range(&mut rng, &[16], -(1 << 19)..(1 << 19))];
    ProbeModel {
        name: "smooth_relus",
        model: b.build(),
        inputs,
    }
}

/// `sub(input, c)` with `c` fixed; `M'` differs from `M` in one coordinate.
pub(crate) fn sub_const_model(constant: &[i32]) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![constant.len()]);
    let c = b.constant(Tensor::from_slice(constant, &[constant.len()]).unwrap());
    let o = b.sub(i, c);
    b.mark_output(o);
    b.build()
}

/// `square(input)`: `x` and `-x` share an output.
pub(crate) fn square_model(t: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let o = b.square(i);
    b.mark_output(o);
    b.build()
}

/// `add(input, c)` on 1000 elements, built with the production padding pass.
pub(crate) fn padded_add_model(rng: &mut StdRng) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![1000]);
    let c = b.constant(Tensor::random_small(rng, &[1000]));
    let o = b.add(i, c);
    b.mark_output(o);
    b.build_padded()
}

/// `[2,1] → Broadcast [2,3] → Sum(axis 1)`, padded: a broadcast into padded lanes.
pub(crate) fn padded_broadcast_sum_model() -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![2, 1]);
    let bc = b.broadcast(i, vec![2, 3]);
    let o = b.sum(bc, vec![1], vec![2]);
    b.mark_output(o);
    b.build_padded()
}
