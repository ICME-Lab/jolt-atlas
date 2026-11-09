use crate::node_view::ComputationNode;
use crate::{
    impl_standard_sumcheck_proof_api,
    onnx_proof::{ProofId, ProofType, Prover},
};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_trace::op::Iff;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::ops::iff::*;

impl_standard_sumcheck_proof_api!(Iff, relation IffRelation);

use crate::onnx_proof::relation::{AtlasRelation, RelationSetup, RelationWitnesses};
use crate::onnx_proof::trace_view::TraceView;
use atlas_algebra::poly::multilinear_polynomial::MultilinearPolynomial;
use atlas_trace::{trace::LayerData, Trace};

impl<F: JoltField> RelationWitnesses<F> for IffRelation {
    fn witnesses(&self, trace: &Trace, _setup: &RelationSetup<F>) -> Vec<MultilinearPolynomial<F>> {
        let LayerData { operands, .. } = TraceView::layer_data(trace, self.node());
        let [mask, a, b] = operands[..] else {
            panic!("Expected three operands for Iff operation")
        };
        [mask, a, b]
            .map(|operand| MultilinearPolynomial::from(operand.padded_pow2()))
            .into()
    }
}
