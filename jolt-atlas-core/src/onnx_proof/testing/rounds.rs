//! Native sumcheck members driven round by round, off the transcript, where the split-eq factor
//! vanishes.
//!
//! An eq-point coordinate of 0 (or 1) makes the Gruen linear factor vanish at 1 (or 0) in the
//! round that binds it, so the round polynomial cannot be recovered from the running claim by
//! division and the member evaluates its inner polynomial at that endpoint directly. Random
//! challenges land there with probability about 2⁻¹²⁵ per coordinate, so no end-to-end test
//! reaches it.

use atlas_trace::{Model, Tensor};
use jolt_field::{Fr, Ring};
use jolt_sumcheck::ProveRounds;
use joltworks::{
    field::JoltField,
    poly::{eq_poly::EqPolynomial, opening_proof::OpeningPoint},
};
use num_traits::{One, Zero};
use rand::{rngs::StdRng, SeedableRng};

use super::ModelBuilder;
use crate::{
    node_view::ModelView,
    onnx_proof::ops::{
        cube::{CubeParams, CubeProver},
        mul::{MulParams, MulProver},
        square::{SquareParams, SquareProver},
    },
};

const NUM_VARS: usize = 3;

/// Run `member`'s rounds on `challenges` from `claim`, checking each round polynomial against
/// the running claim; returns the final claim.
fn run_rounds(member: &mut dyn ProveRounds<Fr>, mut claim: Fr, challenges: &[Fr]) -> Fr {
    let mut bind = None;
    for (round, &r) in challenges.iter().enumerate() {
        let s = member
            .prove_round(bind, round, claim)
            .unwrap_or_else(|e| panic!("round {round}: {e:?}"));
        assert_eq!(s.evaluate(Fr::zero()) + s.evaluate(Fr::one()), claim);
        claim = s.evaluate(r);
        bind = Some(r);
    }
    member
        .finish_rounds(*challenges.last().expect("at least one round"))
        .expect("final bind");
    claim
}

/// `Σ_x eq(r, x) · Π_i operands[i](x)` over the hypercube (big-endian `r`).
fn eq_product_sum(r: &[Fr], operands: &[Vec<Fr>]) -> Fr {
    EqPolynomial::<Fr>::evals(r)
        .iter()
        .enumerate()
        .map(|(x, eq)| operands.iter().fold(*eq, |acc, op| acc * op[x]))
        .sum()
}

fn random_challenge(rng: &mut StdRng) -> Fr {
    <Fr as JoltField>::Challenge::random(rng).into()
}

/// A model with one node applied to `NUM_VARS`-variable inputs, its trace's operands as field
/// vectors, an eq point whose last coordinate is 1 and first is 0 (the first and last rounds of a
/// low-to-high member bind them), and challenges.
struct Fixture {
    model: Model,
    trace: atlas_trace::Trace,
    node: usize,
    operands: Vec<Vec<Fr>>,
    r_node: Vec<Fr>,
    challenges: Vec<Fr>,
}

impl Fixture {
    fn new(seed: u64, arity: usize, build: impl Fn(&mut ModelBuilder, &[usize]) -> usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut b = ModelBuilder::new();
        let inputs: Vec<usize> = (0..arity).map(|_| b.input(vec![1 << NUM_VARS])).collect();
        let node = build(&mut b, &inputs);
        b.mark_output(node);
        let model = b.build();
        let values: Vec<Tensor<i32>> = (0..arity)
            .map(|_| Tensor::random_range(&mut rng, &[1 << NUM_VARS], -(1 << 12)..(1 << 12)))
            .collect();
        let trace = model.trace(&values).expect("honest execution");
        let operands = values
            .iter()
            .map(|t| t.data().iter().map(|&v| Fr::from_i64(v as i64)).collect())
            .collect();
        let mut r_node: Vec<Fr> = (0..NUM_VARS).map(|_| random_challenge(&mut rng)).collect();
        r_node[0] = Fr::zero();
        r_node[NUM_VARS - 1] = Fr::one();
        let challenges = (0..NUM_VARS).map(|_| random_challenge(&mut rng)).collect();
        Self {
            model,
            trace,
            node,
            operands,
            r_node,
            challenges,
        }
    }

    /// Run `member` and check its final claim against `eq(r, r′) · Π_i operands[i](r′)`.
    fn check(&self, member: &mut dyn ProveRounds<Fr>, operands: &[&Vec<Fr>]) {
        let operands: Vec<Vec<Fr>> = operands.iter().map(|op| (*op).clone()).collect();
        let claim = eq_product_sum(&self.r_node, &operands);
        let final_claim = run_rounds(member, claim, &self.challenges);
        // Low-to-high: the first challenge binds the last (lowest) big-endian coordinate.
        let r_prime: Vec<Fr> = self.challenges.iter().rev().copied().collect();
        let expected = EqPolynomial::<Fr>::mle(&self.r_node, &r_prime)
            * operands
                .iter()
                .map(|op| eq_product_sum(&r_prime, std::slice::from_ref(op)))
                .product::<Fr>();
        assert_eq!(final_claim, expected);
    }
}

#[test]
fn mul_rounds_where_eq_vanishes() {
    let f = Fixture::new(0xA17_0401, 2, |b, i| b.mul(i[0], i[1]));
    let params = MulParams {
        r_node_output: OpeningPoint::new(f.r_node.clone()),
        computation_node: f.model.view(f.node),
    };
    let mut prover = MulProver::initialize(&f.trace, params);
    f.check(&mut prover, &[&f.operands[0], &f.operands[1]]);
}

#[test]
fn square_rounds_where_eq_vanishes() {
    let f = Fixture::new(0xA17_0402, 1, |b, i| b.square(i[0]));
    let params = SquareParams {
        r_node_output: OpeningPoint::new(f.r_node.clone()),
        computation_node: f.model.view(f.node),
    };
    let mut prover = SquareProver::initialize(&f.trace, params);
    f.check(&mut prover, &[&f.operands[0], &f.operands[0]]);
}

#[test]
fn cube_rounds_where_eq_vanishes() {
    let f = Fixture::new(0xA17_0403, 1, |b, i| b.cube(i[0], 0));
    let params = CubeParams {
        r_node_output: OpeningPoint::new(f.r_node.clone()),
        computation_node: f.model.view(f.node),
    };
    let mut prover = CubeProver::initialize(&f.trace, params);
    f.check(
        &mut prover,
        &[&f.operands[0], &f.operands[0], &f.operands[0]],
    );
}
