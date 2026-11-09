use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::eval_reduction::NodeEvalReductionProve;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::range_checking::RangeCheckProve;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    onnx_proof::deferred_lookups::{DeferredBatch, DeferredOneHot},
    onnx_proof::{
        ops::eval_reduction::NodeEvalReduction,
        range_checking::{
            range_check_operands::{
                RangeCheckOperands, RiRangeCheckOperands, RsRangeCheckOperands,
            },
            RangeCheckProvider,
        },
        relation::{RelationParams, RelationProver, RelationSetup, RelationWitnesses},
        ProofId, ProofType, Prover,
    },
};
use atlas_algebra::field::JoltField;
use atlas_algebra::poly::multilinear_polynomial::MultilinearPolynomial;
use atlas_lookup_tables::unsigned_less_than::UnsignedLessThanTable;
use atlas_sumcheck::relation::prover::prove_relation;
use atlas_sumcheck::{evaluation_reduction::EvalReductionProof, sumcheck::SumcheckInstanceProof};
use atlas_trace::{op::Rsqrt, trace::LayerData, Trace};
use atlas_transcript::Transcript;
use common::consts::XLEN;
pub use jolt_atlas_verifier::onnx_proof::ops::rsqrt::*;
use jolt_claims::SymbolicSumcheck;

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
        let (proof, _) = prove_relation(
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
        Option<EvalReductionProof<F>>,
        Vec<(ProofId, SumcheckInstanceProof<F>)>,
    ) {
        let mut proofs = self.prove(node, prover);
        let eval_reduction_proof = NodeEvalReduction::prove(prover, node);
        proofs.extend(prove_range_and_onehot(node, prover));
        (eval_reduction_proof, proofs)
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

impl<F: JoltField> RelationWitnesses<F> for RsqrtRelation {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
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
