use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::OpLayout;
use crate::{
    onnx_proof::{
        plan::{OpPlan, OpRelation, Plan, Step},
        relation::{
            Incoming, OpRel, OpRelKind, RelationChallenge, RelationDerived, RelationOpening,
        },
        ProofType,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_algebra::{
    field::JoltField,
    utils::{errors::ProofVerifyError, math::Math},
};
use atlas_claims::{challenge, constant, derived, opening, Int};
use atlas_trace::{op::Rsqrt, trace::ModelExecutionIO, Model, Op};
use common::{CommittedPoly, VirtualPoly};

/// Dividend `S³` of the fused rsqrt division relation, where `S = 2^scale` is the
/// input's fixed-point scale.
///
/// The fused kernel computes `output = ⌊√⌊S³ / x̂⌋⌋`, so `S³` is the dividend that
/// both the division and square-root relations are built around.
pub fn rsqrt_dividend(node: &ComputationNode) -> i64 {
    assert!(
        matches!(node.operator, Op::Rsqrt(_)),
        "rsqrt_dividend called on a non-Rsqrt node"
    );
    1i64 << (3 * node.scale)
}

impl OpLayout for Rsqrt {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        Ok(vec![CommittedPoly::RsqrtQuotient(node.idx)])
    }
}

/// The plan: the fused relation's sumcheck (its point drawn from the transcript), then the
/// evaluation reduction. The two remainder bounds are proven in the packed remainder buckets
/// after the node loop (`global_clamp::node_remainders`).
impl OpPlan for Rsqrt {
    fn plan<F: atlas_algebra::field::JoltField>(
        &self,
        node: &ComputationNode,
        _model: &Model,
        _io: &ModelExecutionIO,
    ) -> Result<Plan<F>, ProofVerifyError> {
        Ok(Plan(vec![
            Step::Relation {
                proof: ProofType::Execution,
                relation: OpRelation::Op(reciprocal_sqrt(node)),
            },
            Step::Reduce,
        ]))
    }
}

// FUSED RECIPROCAL-SQUARE-ROOT PROOF
//
// For a quantized input x̂ with scale S = 2^scale, the reciprocal square root at the
// same scale is  S / √(x̂ / S) = √(S³ / x̂).  The tracer fuses the division and the
// square root into a single kernel:
//
//     output = ⌊√⌊S³ / x̂⌋⌋
//
// We witness the intermediate `quotient = ⌊S³ / x̂⌋` and two remainders, and prove
// two relations:
//
//   division:  x̂ · quotient + div_remainder = S³      with 0 ≤ div_remainder < x̂
//   sqrt:      output² + sqrt_remainder = quotient     with 0 ≤ sqrt_remainder ≤ 2·output
//
// The two relations are folded together with a challenge `gamma` into one sumcheck:
//
//     0 = (x̂·quotient + div_remainder − S³) + gamma·(output² + sqrt_remainder − quotient)
//
// The bounds `div_remainder < x̂` and `sqrt_remainder ≤ 2·output` are enforced by the
// range checks in `prove_range_and_onehot`.

/// The fused reciprocal square root as a relation (see above):
///
/// ```text
/// input claim   0
/// output claim  eq(r₀, r′) · (x̂·q + r_div − S³ + γ·(out² + r_sqrt − q))
/// ```
///
/// with `r₀` and `γ` drawn from the transcript, in that order, before the sumcheck.
///
/// **S ⊆ R.** `atlas_trace::semantics::fixed::rsqrt` computes `out = ⌊√q⌋` for
/// `q = ⌊S³ / x̂⌋`; the witness takes `r_div = S³ mod x̂` and `r_sqrt = q − out²`, so both
/// identities vanish at every hypercube point and the summand sums to zero. The bounds
/// `r_div < x̂` and `r_sqrt ≤ 2·out` that make the witness unique are the range checks below,
/// outside the relation. Non-positive inputs (where the semantics return 0) are outside the
/// relation's domain: the division range check rejects them.
pub fn reciprocal_sqrt<F: JoltField>(node: &ComputationNode) -> OpRel<F> {
    let ids = OpeningIdBuilder::new(node);
    let produced = [
        ids.nodeio(Target::Input(0)),
        ids.advice(CommittedPoly::RsqrtQuotient),
        ids.nodeio(Target::Current),
        ids.advice(VirtualPoly::DivRemainder),
        ids.advice(VirtualPoly::SqrtRemainder),
    ];
    let [input, quotient, output, div_remainder, sqrt_remainder] =
        produced.map(|id| opening(RelationOpening::Id(id)));
    let s_cubed = constant(Int(i128::from(rsqrt_dividend(node))));
    let division = input * quotient.clone() + div_remainder - s_cubed;
    let sqrt = output.clone() * output + sqrt_remainder - quotient;
    OpRel {
        rounds: node.pow2_padded_num_output_elements().log_2(),
        degree: 3,
        output: derived(RelationDerived::Eq)
            * (division + challenge(RelationChallenge::Gamma) * sqrt),
        produced: produced.to_vec(),
        incoming: Incoming::Fresh,
        draws: vec![RelationChallenge::Gamma],
        ..OpRel::base(OpRelKind::Rsqrt, node)
    }
}
