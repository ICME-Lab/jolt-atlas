use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::ProveOp;
use crate::onnx_proof::{ops::Prover, relation::prove_tie, ProofId};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::Constant;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::constant::*;
use jolt_claims::SymbolicSumcheck;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Constant {
    #[tracing::instrument(skip_all, name = "Constant::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        prove_tie(
            &PublicRelation::<Constant>::new(PublicShape::new(node.clone(), self.0.clone())),
            &prover.trace,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        Vec::new()
    }
}
