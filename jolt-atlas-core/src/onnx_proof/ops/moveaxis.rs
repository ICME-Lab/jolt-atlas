use crate::node_view::ComputationNode;
use crate::{
    impl_tie_proof_api,
    onnx_proof::{ops::Prover, ProofId},
};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::MoveAxis;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::moveaxis::*;

impl_tie_proof_api!(MoveAxis, AxisPermutationRelation);

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, TieClaims};
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::Trace;

impl<F: JoltField> TieClaims<F> for AxisPermutationRelation {
    fn claims(
        &self,
        _trace: &Trace,
        _setup: &RelationSetup<F>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        // The operand at the permuted point is the output at `r`.
        vec![accumulator.get_node_output_opening(self.node().idx).1]
    }
}
