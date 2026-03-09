use ark_bn254::{Bn254, Fr};
use atlas_onnx_tracer::{
    model::{test::ModelBuilder, Model},
    node::ComputationNode,
    ops::Operator,
    tensor::Tensor,
};
use joltworks::{poly::commitment::hyperkzg::HyperKZG, transcripts::Blake2bTranscript};
use rand::{rngs::StdRng, SeedableRng};

pub(super) type TestPCS = HyperKZG<Bn254>;
pub(super) type TestField = Fr;
pub(super) type TestTranscript = Blake2bTranscript;

mod nodeoutput_opening_collapse;
mod tau_rangecheck_bypass;
mod virtual_operand_claim_tamper;

pub(super) fn sub_model(rng: &mut StdRng, t: usize) -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let c = b.constant(Tensor::random_small(rng, &[t]));
    let out = b.sub(i, c);
    b.mark_output(out);
    b.build()
}

pub(super) fn sub_model_const_2() -> Model {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![1]);
    let c = b.constant(Tensor::new(Some(&[2]), &[1]).expect("constant tensor should be valid"));
    let out = b.sub(i, c);
    b.mark_output(out);
    b.build()
}

pub(super) fn find_sub_node(model: &Model) -> ComputationNode {
    model
        .graph
        .nodes
        .values()
        .find(|node| matches!(node.operator, Operator::Sub(_)))
        .cloned()
        .expect("sub node should exist")
}

pub(super) fn fanout_sub_model(rng: &mut StdRng, t: usize) -> (Model, usize, usize, usize) {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let c0 = b.constant(Tensor::random_small(rng, &[t]));
    let c1 = b.constant(Tensor::random_small(rng, &[t]));
    let c2 = b.constant(Tensor::random_small(rng, &[t]));

    let x = b.sub(i, c0);
    let y = b.sub(x, c1);
    let z = b.sub(x, c2);
    let o = b.add(y, z);
    b.mark_output(o);
    (b.build(), x, y, z)
}

pub(super) fn duplicate_operand_sub_model(rng: &mut StdRng, t: usize) -> (Model, usize, usize) {
    let mut b = ModelBuilder::new();
    let i = b.input(vec![t]);
    let c0 = b.constant(Tensor::random_small(rng, &[t]));
    let x = b.sub(i, c0);
    let y = b.sub(x, x);
    b.mark_output(y);
    (b.build(), x, y)
}

pub(super) fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}
