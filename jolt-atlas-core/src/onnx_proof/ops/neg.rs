use atlas_algebra::field::JoltField;
pub use jolt_atlas_verifier::onnx_proof::ops::neg::*;

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
        let output = accumulator
            .get_node_output_opening(self.node().idx)
            .expect("the prover holds every claim it reads")
            .1;
        vec![F::from_i64(S::SIGN) * output]
    }
}
