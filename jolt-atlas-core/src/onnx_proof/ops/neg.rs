use crate::node_view::ComputationNode;
use crate::{
    impl_tie_proof_api,
    onnx_proof::{ProofId, Prover},
};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::Neg;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::neg::*;

impl_tie_proof_api!(Neg, SignedCopyRelation<Neg>);

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, TieClaims};
use atlas_sumcheck::opening_proof::OpeningAccumulator;
use atlas_trace::Trace;

impl<F: JoltField, S: CopySign> TieClaims<F> for SignedCopyRelation<S> {
    fn claims(
        &self,
        _trace: &Trace,
        _setup: &RelationSetup<F>,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        // operand(r) = ±out(r): the reduced output claim, signed.
        let output = accumulator.get_node_output_opening(self.node().idx).1;
        vec![F::from_i64(S::SIGN) * output]
    }
}
