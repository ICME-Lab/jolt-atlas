use super::*;
use atlas_onnx_tracer::ops::GatherLarge;
use common::CommittedPolynomial;
use joltworks::{
    config::{OneHotConfig, OneHotParams},
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck},
    },
};

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for GatherLarge {
    #[tracing::instrument(skip_all, name = "GatherLarge::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();

        let params = GatherParams::new(
            node.clone(),
            &prover.preprocessing.model.graph,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut exec_sumcheck = GatherProver::initialize(
            &prover.trace,
            params,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        let (proof, _) = Sumcheck::prove(
            &mut exec_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), proof));

        let encoding = GatherRaEncoding::new(node);
        let lookup_indices = gather_lookup_indices(node, &prover.trace);
        let [ra_prover, hw_prover, bool_prover] = shout::ra_onehot_provers(
            &encoding,
            &lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![ra_prover, hw_prover, bool_prover];

        let (stage2_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );

        results.push((ProofId(node.idx, ProofType::RaOneHotChecks), stage2_proof));

        results
    }

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
            &verifier.preprocessing.model.graph,
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
        let [ra_verifier, hw_verifier, bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);
        let instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> =
            vec![&*ra_verifier, &*hw_verifier, &*bool_verifier];
        BatchedSumcheck::verify(
            ra_one_hot_proof,
            instances,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPolynomial> {
        let encoding = GatherRaEncoding::new(node);
        let d = encoding.one_hot_params().instruction_d;
        (0..d)
            .map(|i| CommittedPolynomial::GatherRaD(node.idx, i))
            .collect()
    }
}

pub(crate) struct GatherRaEncoding {
    node_idx: usize,
    index_input_idx: usize,
    num_words: usize,
}

impl GatherRaEncoding {
    pub(crate) fn new(computation_node: &ComputationNode) -> Self {
        let gather_op = match &computation_node.operator {
            Operator::GatherLarge(gather_op) => gather_op,
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
    fn committed_poly(&self, d: usize) -> CommittedPolynomial {
        CommittedPolynomial::GatherRaD(self.node_idx, d)
    }

    fn r_cycle_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::NodeOutput(self.index_input_idx),
            SumcheckId::NodeExecution(self.node_idx),
        )
    }

    fn ra_source(&self) -> (VirtualPolynomial, SumcheckId) {
        (
            VirtualPolynomial::NodeOutputRa(self.node_idx),
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

pub(crate) fn gather_lookup_indices(
    computation_node: &ComputationNode,
    trace: &Trace,
) -> Vec<usize> {
    let LayerData { operands, .. } = Trace::layer_data(trace, computation_node);
    let [_, indexes] = operands[..] else {
        panic!("Expected two operands for Gather operation")
    };
    indexes
        .padded_next_power_of_two()
        .data()
        .par_iter()
        .map(|&x| x as usize)
        .collect()
}
