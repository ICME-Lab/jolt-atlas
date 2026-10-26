use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{OpLayout, ProveOp, VerifyOp};
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    onnx_proof::deferred_lookups::{DeferredBatch, DeferredOneHot},
    onnx_proof::{
        ops::{eval_reduction::NodeEvalReduction, ReductionFlow},
        range_checking::{
            range_check_operands::{
                RangeCheckOperands, RiRangeCheckOperands, RsRangeCheckOperands,
            },
            RangeCheckProvider,
        },
        relation::{
            AtlasRelation, EngineRelation, RelationChallenge, RelationDerived, RelationExpr,
            RelationId, RelationOpening, RelationParams, RelationProver, RelationSetup,
            RelationVerifier,
        },
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::{OpeningIdBuilder, Target},
};
use atlas_trace::{op::Rsqrt, trace::LayerData, Op, Trace};
use common::{consts::XLEN, CommittedPoly, VirtualPoly};
use jolt_claims::{
    challenge, constant, derived, opening, Expr, NoChallenges, NoInputs, NoOutputs,
    SymbolicSumcheck,
};
use jolt_field::Ring;
use joltworks::{
    field::{IntoOpening, JoltField},
    lookup_tables::unsigned_less_than::UnsignedLessThanTable,
    poly::{
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningAccumulator, OpeningId},
    },
    subprotocols::{
        evaluation_reduction::EvalReductionProof,
        shout::RaOneHotEncoding,
        sumcheck::{Sumcheck, SumcheckInstanceProof},
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, math::Math},
};

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
    fn reduction_flow(&self) -> ReductionFlow {
        ReductionFlow::Custom
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let mut polys = vec![CommittedPoly::RsqrtQuotient(node.idx)];

        let rc_operands = RangeCheckOperands::<RiRangeCheckOperands>::new(node);
        let encoding = rc_operands.get_encoding(node);
        let d = encoding.one_hot_params().instruction_d;
        for i in 0..d {
            polys.push(CommittedPoly::SqrtDivRangeCheckRaD(node.idx, i));
            polys.push(CommittedPoly::SqrtRangeCheckRaD(node.idx, i));
        }
        polys
    }
}

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Rsqrt {
    #[tracing::instrument(skip_all, name = "Rsqrt::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        let mut results = Vec::new();
        // Execution proof
        let params = RelationParams::<F, _>::new(
            RsqrtRelation::new(node.clone()),
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut prover_sumcheck = RelationProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut prover_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), proof));

        results
    }

    fn prove_with_reduction(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> (
        EvalReductionProof<F>,
        Vec<(ProofId, SumcheckInstanceProof<F>)>,
    ) {
        let mut proofs = self.prove(node, prover);
        let eval_reduction_proof = NodeEvalReduction::prove(prover, node);
        proofs.extend(prove_range_and_onehot(node, prover));
        (eval_reduction_proof, proofs)
    }
}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for Rsqrt {
    #[tracing::instrument(skip_all, name = "Rsqrt::prove")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Execution proof
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let verifier_sumcheck = RelationVerifier::<F, _>::new(
            RsqrtRelation::new(node.clone()),
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            proof,
            &verifier_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }

    fn verify_with_reduction(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
        eval_reduction_proof: &EvalReductionProof<F>,
    ) -> Result<(), ProofVerifyError> {
        self.verify(node, verifier)?;
        NodeEvalReduction::verify(verifier, node, eval_reduction_proof)?;
        verify_range_and_onehot(node, verifier)
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

crate::bind_relation!(Rsqrt, RsqrtRelation);

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
pub struct RsqrtRelation {
    node: ComputationNode,
}

impl RsqrtRelation {
    fn ids(&self) -> OpeningIdBuilder<'_> {
        OpeningIdBuilder::new(&self.node)
    }
}

impl SymbolicSumcheck for RsqrtRelation {
    type RelationId = RelationId;
    type OpeningId = RelationOpening;
    type DerivedId = RelationDerived;
    type ChallengeId = RelationChallenge;
    type Shape = ComputationNode;
    type Challenges<F> = NoChallenges<F>;
    type Inputs<C> = NoInputs<C>;
    type Outputs<C> = NoOutputs<C>;

    fn new(node: ComputationNode) -> Self {
        Self { node }
    }

    fn id() -> RelationId {
        RelationId::Rsqrt
    }

    fn rounds(&self) -> usize {
        self.node.pow2_padded_num_output_elements().log_2()
    }

    fn degree(&self) -> usize {
        3
    }

    fn input_expression<F: Ring>(&self) -> RelationExpr<F> {
        Expr::zero()
    }

    fn output_expression<F: Ring>(&self) -> RelationExpr<F> {
        let open = |id| opening(RelationOpening::Id(id));
        let input = open(self.ids().nodeio(Target::Input(0)));
        let quotient = open(self.ids().advice(CommittedPoly::RsqrtQuotient));
        let output = open(self.ids().nodeio(Target::Current));
        let div_remainder = open(self.ids().advice(VirtualPoly::DivRemainder));
        let sqrt_remainder = open(self.ids().advice(VirtualPoly::SqrtRemainder));
        let s_cubed = constant(F::from_i64(rsqrt_dividend(&self.node)));
        let division = input * quotient.clone() + div_remainder - s_cubed;
        let sqrt = output.clone() * output + sqrt_remainder - quotient;
        derived(RelationDerived::Eq) * (division + challenge(RelationChallenge::Gamma) * sqrt)
    }
}

impl AtlasRelation for RsqrtRelation {
    type Spec = Rsqrt;

    fn node(&self) -> &ComputationNode {
        &self.node
    }

    fn setup<F: JoltField, T: Transcript>(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> RelationSetup<F> {
        let incoming_point = transcript
            .challenge_vector_optimized::<F>(self.rounds())
            .into_opening();
        let gamma = transcript.challenge_scalar();
        RelationSetup {
            incoming_point,
            challenges: vec![(RelationChallenge::Gamma, gamma)],
        }
    }

    fn produced(&self) -> Vec<OpeningId> {
        vec![
            self.ids().nodeio(Target::Input(0)),
            self.ids().advice(CommittedPoly::RsqrtQuotient),
            self.ids().nodeio(Target::Current),
            self.ids().advice(VirtualPoly::DivRemainder),
            self.ids().advice(VirtualPoly::SqrtRemainder),
        ]
    }
}

impl EngineRelation for RsqrtRelation {
    fn witnesses<F: JoltField>(
        &self,
        trace: &Trace,
        _setup: &RelationSetup<F>,
    ) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, output } = TraceView::layer_data(trace, &self.node);
        let [input] = operands[..] else {
            panic!("Expected one operand for Rsqrt operation")
        };
        let s_cubed = rsqrt_dividend(&self.node);
        // `quotient = ⌊S³ / x̂⌋` and `out²` can exceed i32 at higher scales, so both
        // are computed in i64; `quotient` is committed as u64. `div_remainder < x̂`
        // and `sqrt_remainder < 2·out+1` are small and stay i32.
        let quotient: Vec<u64> = input.iter().map(|&x| (s_cubed / x as i64) as u64).collect();
        let div_remainder: Vec<i32> = input.iter().map(|&x| (s_cubed % x as i64) as i32).collect();
        let sqrt_remainder: Vec<i32> = quotient
            .iter()
            .zip(output.iter())
            .map(|(&quotient, &out)| (quotient as i64 - (out as i64) * (out as i64)) as i32)
            .collect();
        #[cfg(test)]
        {
            use crate::onnx_proof::testing::probe::{probe_assert, probe_assert_eq};
            for (i, (&x, &q)) in input.iter().zip(&quotient).enumerate() {
                let (x, q, out) = (x as i128, q as i128, output.data()[i] as i128);
                let (div_remainder, sqrt_remainder) =
                    (div_remainder[i] as i128, sqrt_remainder[i] as i128);
                // range checks: div_remainder < x, sqrt_remainder <= 2*output
                probe_assert!(div_remainder >= 0 && div_remainder < x);
                probe_assert!(sqrt_remainder >= 0 && sqrt_remainder <= 2 * out);
                probe_assert_eq!(x * q + div_remainder, s_cubed as i128);
                probe_assert_eq!(out * out + sqrt_remainder, q);
            }
        }
        vec![
            MultilinearPolynomial::from(input.clone()),
            MultilinearPolynomial::from(quotient),
            MultilinearPolynomial::from(output.clone()),
            MultilinearPolynomial::from(div_remainder),
            MultilinearPolynomial::from(sqrt_remainder),
        ]
    }
}

fn prove_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    prover: &mut Prover<F, T>,
) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
    // Both range checks are built now (challenges drawn, operand claims
    // appended) and proven after the node loop in the `RangeCheck` batch; their
    // one-hot checks follow that batch.
    let div_rangecheck_provider = RangeCheckProvider::<RiRangeCheckOperands>::new(node);
    let (div_rc_prover, div_lookup_indices) = div_rangecheck_provider
        .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    prover.defer(DeferredBatch::RangeCheck, Box::new(div_rc_prover));

    let sqrt_rangecheck_provider = RangeCheckProvider::<RsRangeCheckOperands>::new(node);
    let (sqrt_rc_prover, sqrt_lookup_indices) = sqrt_rangecheck_provider
        .read_raf_prove::<F, T, UnsignedLessThanTable<XLEN>>(
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
    prover.defer(DeferredBatch::RangeCheck, Box::new(sqrt_rc_prover));

    let div_operands = RangeCheckOperands::<RiRangeCheckOperands>::new(node);
    prover.deferred_onehots.push((
        DeferredOneHot::RangeCheck(div_operands.get_encoding(node)),
        div_lookup_indices,
    ));
    let sqrt_operands = RangeCheckOperands::<RsRangeCheckOperands>::new(node);
    prover.deferred_onehots.push((
        DeferredOneHot::RangeCheck(sqrt_operands.get_encoding(node)),
        sqrt_lookup_indices,
    ));

    Vec::new()
}

fn verify_range_and_onehot<F: JoltField, T: Transcript>(
    node: &ComputationNode,
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    let div_rangecheck_provider = RangeCheckProvider::<RiRangeCheckOperands>::new(node);
    let div_rc_verifier = div_rangecheck_provider
        .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
    verifier.defer(DeferredBatch::RangeCheck, Box::new(div_rc_verifier));

    let sqrt_rangecheck_provider = RangeCheckProvider::<RsRangeCheckOperands>::new(node);
    let sqrt_rc_verifier = sqrt_rangecheck_provider
        .read_raf_verify::<F, T, UnsignedLessThanTable<XLEN>>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
    verifier.defer(DeferredBatch::RangeCheck, Box::new(sqrt_rc_verifier));

    let div_operands = RangeCheckOperands::<RiRangeCheckOperands>::new(node);
    verifier
        .deferred_onehots
        .push(DeferredOneHot::RangeCheck(div_operands.get_encoding(node)));
    let sqrt_operands = RangeCheckOperands::<RsRangeCheckOperands>::new(node);
    verifier
        .deferred_onehots
        .push(DeferredOneHot::RangeCheck(sqrt_operands.get_encoding(node)));

    Ok(())
}
