use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::ProveOp;
use atlas_trace::op::Sum;
use joltworks::{
    self,
    field::JoltField,
    subprotocols::sumcheck::{Sumcheck, SumcheckInstanceProof},
    transcripts::Transcript,
};

use crate::onnx_proof::{
    clamp_lookups::{is_scalar, prove_append_acc, prove_clamp_lookup},
    ops::sum::axis::SumAxisRelation,
    relation::{RelationParams, RelationProver},
    ProofId, ProofType, Prover,
};

/// Axis-wise sum implementations for sumcheck protocol.
pub mod axis;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Sum {
    #[tracing::instrument(skip_all, name = "Sum::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        // (1+2) Prove `output = SatClamp(acc)`. Non-scalar: clamp lookup + one-hot.
        // Scalar: just append `acc` in the clear (verifier checks the clamp).
        let mut results = if is_scalar(node) {
            prove_append_acc(
                node,
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
                None,
            );
            vec![]
        } else {
            prove_clamp_lookup(node, prover, None)
        };

        // (3) Axis reduction: acc(r) = Σ_axis operand (input claim = ClampAcc).
        let params = RelationParams::<F, _>::new(
            SumAxisRelation::for_node(&prover.preprocessing.model, node.clone()),
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut reduction = RelationProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut reduction,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::SumReduction), proof));

        results
    }
}
