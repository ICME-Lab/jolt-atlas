use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{OpLayout, VerifyOp};
use crate::onnx_proof::{
    op_lookups::{OpLookupEncoding, OpLookupProvider},
    ProofId, ProofType, Verifier,
};
use atlas_trace::op::ReLU;
use common::CommittedPoly;
use joltworks::{
    self,
    field::JoltField,
    lookup_tables::relu::ReluTable,
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck},
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

use common::consts::XLEN;

impl OpLayout for ReLU {
    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let encoding: OpLookupEncoding = OpLookupEncoding::new(node);
        let d = encoding.one_hot_params().instruction_d;
        (0..d)
            .map(|i| CommittedPoly::NodeOutputRaD(node.idx, i))
            .collect()
    }
}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for ReLU {
    #[tracing::instrument(skip_all, name = "ReLU::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Verify execution proof
        let provider: OpLookupProvider = OpLookupProvider::new(node.clone());
        let verifier_sumcheck = provider.read_raf_verify::<F, T, ReluTable<XLEN>, XLEN>(
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
        let execution_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        Sumcheck::verify(
            execution_proof,
            &verifier_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        // Verify RaOneHotChecks
        let encoding = provider.encoding();
        let [ra_verifier, bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);
        let ra_one_hot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        BatchedSumcheck::verify(
            ra_one_hot_proof,
            vec![&*ra_verifier, &*bool_verifier],
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }
}
