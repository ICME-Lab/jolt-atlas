use super::*;
use atlas_trace::op::GatherLarge;
use common::CommittedPoly;
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    poly::opening_proof::OpeningId,
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck},
    },
};

impl OpLayout for GatherLarge {
    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let encoding = GatherRaEncoding::new(node);
        let d = encoding.one_hot_params().instruction_d;
        (0..d)
            .map(|i| CommittedPoly::GatherRaD(node.idx, i))
            .collect()
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
            .get(&ProofId(node.idx, ProofType::Execution))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let exec_sumcheck = GatherVerifier::new(
            node.clone(),
            &verifier.preprocessing.model,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            proof,
            &exec_sumcheck,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        let ra_one_hot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let encoding = GatherRaEncoding::new(node);
        let [ra_verifier, bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);
        let instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> =
            vec![&*ra_verifier, &*bool_verifier];
        BatchedSumcheck::verify(
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
    /// The encoding for the node's dictionary.
    pub fn new(computation_node: &ComputationNode) -> Self {
        let gather_op = match &computation_node.operator {
            Op::GatherLarge(gather_op) => gather_op,
            _ => panic!("Expected GatherLarge operator"),
        };
        Self {
            node_idx: computation_node.idx,
            index_input_idx: computation_node.inputs[1],
            num_words: gather_op.dict_len.next_power_of_two(),
        }
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
