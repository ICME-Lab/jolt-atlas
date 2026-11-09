use crate::node_view::ComputationNode;
use crate::{
    impl_tie_proof_api,
    onnx_proof::{ProofId, Prover},
};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::IsNan;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::is_nan::*;

impl_tie_proof_api!(IsNan, ZeroOutputRelation);

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, TieClaims};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear_polynomial::EvaluateMle;
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::Trace;

impl<F: JoltField> TieClaims<F> for ZeroOutputRelation {
    fn claims(
        &self,
        trace: &Trace,
        setup: &RelationSetup<F>,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        let operand = TraceView::operand_tensors(trace, self.node())[0];
        vec![operand.evaluate_mle(&setup.incoming_point)]
    }
}
