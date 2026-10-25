use crate::node_view::ComputationNode;
use crate::onnx_proof::trace_view::TraceView;
use crate::{
    onnx_proof::{ops::OperatorProofTrait, ProofId, Prover, Verifier},
    utils::opening_access::{AccOpeningAccessor, Target},
};
use atlas_trace::{op::Neg, trace::LayerData};
use joltworks::{
    field::JoltField,
    poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    subprotocols::sumcheck::SumcheckInstanceProof,
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// No-sumcheck proof for element-wise negation.
///
/// `Neg` is linear: `output(x) = -operand(x)` on the shared boolean hypercube,
/// so the identity holds at the random output opening point `r`. The prover
/// opens the operand at the same `r`, and the verifier checks
/// `-operand(r) == output(r)` directly — no sumcheck (cf. [`super::identity`]).
impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Neg {
    #[tracing::instrument(skip_all, name = "Neg::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let (opening_point, _claim) =
            AccOpeningAccessor::new(&prover.accumulator, node).get_reduced_opening();

        let LayerData { operands, .. } = TraceView::layer_data(&prover.trace, node);
        let [operand] = operands[..] else {
            panic!("Expected one operand for Neg operation")
        };
        let operand_claim =
            MultilinearPolynomial::from(operand.padded_pow2()).evaluate(&opening_point.r);

        let mut provider = AccOpeningAccessor::new(&mut prover.accumulator, node)
            .into_provider(&mut prover.transcript, opening_point);
        provider.append_nodeio(Target::Input(0), operand_claim);
        vec![]
    }

    #[tracing::instrument(skip_all, name = "Neg::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let (opening_point, claim) =
            AccOpeningAccessor::new(&verifier.accumulator, node).get_reduced_opening();
        let mut provider = AccOpeningAccessor::new(&mut verifier.accumulator, node)
            .into_provider(&mut verifier.transcript, opening_point);
        provider.append_nodeio(Target::Input(0));
        let operand_claim = provider.get_nodeio(Target::Input(0)).1;

        if -operand_claim != claim {
            return Err(ProofVerifyError::InvalidOpeningProof(
                "Neg output claim should equal the negation of the operand claim".to_string(),
            ));
        }
        Ok(())
    }
}
