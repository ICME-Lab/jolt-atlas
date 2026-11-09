use super::*;
use atlas_sumcheck::relation::{verify_relation, verify_relation_batch, DynRelationVerifier};
use atlas_sumcheck::{
    config::{OneHotConfig, OneHotParams},
    opening_proof::OpeningId,
    shout::{self, RaOneHotEncoding},
};
use atlas_trace::op::GatherLarge;
use common::CommittedPoly;

impl OpLayout for GatherLarge {
    fn get_committed_polynomials(
        &self,
        node: &ComputationNode,
    ) -> Result<Vec<CommittedPoly>, ProofVerifyError> {
        let encoding = GatherRaEncoding::new(node)?;
        let d = encoding.one_hot_params().instruction_d;
        Ok((0..d)
            .map(|i| CommittedPoly::GatherRaD(node.idx, i))
            .collect())
    }
}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for GatherLarge {
    #[tracing::instrument(skip_all, name = "GatherLarge::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let proof = verifier
            .proofs
            .take(ProofId(node.idx, ProofType::Execution))?;
        let exec_sumcheck = gather_verifier(node, verifier)?;
        verify_relation(
            proof,
            &exec_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        let ra_one_hot_proof = verifier
            .proofs
            .take(ProofId(node.idx, ProofType::RaOneHotChecks))?;
        let encoding = GatherRaEncoding::new(node)?;
        let [ra_verifier, bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript)?;
        let instances: Vec<&dyn DynRelationVerifier<F, T>> = vec![&*ra_verifier, &*bool_verifier];
        verify_relation_batch(
            ra_one_hot_proof,
            instances,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }
}

/// One-hot encoding of a large gather's read addresses.
pub struct GatherRaEncoding {
    node_idx: usize,
    index_input_idx: usize,
    num_words: usize,
}

impl GatherRaEncoding {
    /// The encoding for the node's dictionary. `Err` if the node is not a large gather with
    /// an index operand.
    pub fn new(computation_node: &ComputationNode) -> Result<Self, ProofVerifyError> {
        let Op::GatherLarge(gather_op) = &computation_node.operator else {
            return Err(unsupported(
                computation_node,
                "expected a GatherLarge operator",
            ));
        };
        let index_input_idx = *computation_node
            .inputs
            .get(1)
            .ok_or_else(|| unsupported(computation_node, "no index operand"))?;
        Ok(Self {
            node_idx: computation_node.idx,
            index_input_idx,
            num_words: gather_op.dict_len.next_power_of_two(),
        })
    }
}

impl RaOneHotEncoding for GatherRaEncoding {
    fn committed_poly(&self, d: usize) -> CommittedPoly {
        CommittedPoly::GatherRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::NodeOutput(self.index_input_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> OpeningId {
        OpeningId::new(
            VirtualPoly::NodeOutputRa(self.node_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn log_k(&self) -> usize {
        self.num_words.log_2()
    }

    fn one_hot_params(&self) -> OneHotParams {
        OneHotParams::from_config_and_log_K(&OneHotConfig::default(), self.log_k())
    }
}
