//! Small multi-op models for the soundness sweeps. Each is a single-output
//! graph a few dozen elements wide, so a prove costs a fraction of a second
//! and a verify a few milliseconds.

use atlas_trace::{Model, Tensor};
use common::consts::MODEL_SCALE;
use joltworks::lookup_tables::clamp::CLAMP_BOUND;
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
    ]
}

/// Same models, different inputs: material for commitment / opening splices.
pub(crate) fn probe_models_alt() -> Vec<ProbeModel> {
    vec![
        attention_like(0xA17_0001),
        reductions(0xA17_0002),
        tables_and_layout(0xA17_0003),
    ]
}

/// einsum → add → relu (consumed twice: fan-out) → softmax → div → reshape →
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
    let pos = b.constant(Tensor::from_slice(&[1 << 8; 32], &[4, 8]).unwrap());
    let d = b.div(s, pos);
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

/// rsqrt / cos / sin / iff / clamp / square / mean_of_squares / sum / cube /
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
    let cl = b.clamp(sel, CLAMP_BOUND);
    let sq = b.square(cl);
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
    let cat = b.concat(&[n, cu, sg], 0);
    let sl = b.slice(cat, 0, 8, 40);
    let rs = b.reshape(sl, vec![4, 8]);
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

/// `[2,1] → Broadcast [2,3] → Sum(axis 1)`, padded: the known padded-lane gap.
pub(crate) fn padded_broadcast_sum_model() -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![2, 1]);
    let bc = b.broadcast(i, vec![2, 3]);
    let o = b.sum(bc, vec![1], vec![2]);
    b.mark_output(o);
    b.build_padded()
}
