use crate::onnx_proof::{
    op_lookups::{OpLookupEncoding, OpLookupProvider},
    ops::OperatorProofTrait,
    ProofId, ProofType, Prover, Verifier,
};
use atlas_onnx_tracer::{node::ComputationNode, ops::ReLU};
use common::CommittedPolynomial;
use joltworks::{
    self,
    field::JoltField,
    lookup_tables::relu::ReluTable,
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

use common::consts::XLEN;

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for ReLU {
    #[tracing::instrument(skip_all, name = "ReLU::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let mut results = Vec::new();

        // Execution proof
        let provider = OpLookupProvider::new(node.clone());
        let (mut execution_sumcheck, lookup_indices) = provider
            .read_raf_prove::<F, T, ReluTable<XLEN>>(
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
            );
        let (execution_proof, _) = Sumcheck::prove(
            &mut execution_sumcheck,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::Execution), execution_proof));

        // RaOneHotChecks proof
        let encoding = OpLookupEncoding::new(node);

        let [ra_prover, hw_prover, bool_prover] = shout::ra_onehot_provers(
            &encoding,
            &lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![ra_prover, hw_prover, bool_prover];
        let (ra_one_hot_proof, _) = BatchedSumcheck::prove(
            instances.iter_mut().map(|v| &mut **v as _).collect(),
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((
            ProofId(node.idx, ProofType::RaOneHotChecks),
            ra_one_hot_proof,
        ));

        results
    }

    #[tracing::instrument(skip_all, name = "ReLU::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // Verify execution proof
        let provider = OpLookupProvider::new(node.clone());
        let verifier_sumcheck = provider.read_raf_verify::<F, T, ReluTable<XLEN>>(
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
        let encoding = OpLookupEncoding::new(node);
        let [ra_verifier, hw_verifier, bool_verifier] =
            shout::ra_onehot_verifiers(&encoding, &verifier.accumulator, &mut verifier.transcript);
        let ra_one_hot_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::RaOneHotChecks))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        BatchedSumcheck::verify(
            ra_one_hot_proof,
            vec![&*ra_verifier, &*hw_verifier, &*bool_verifier],
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPolynomial> {
        let encoding = OpLookupEncoding::new(node);
        let d = encoding.one_hot_params().instruction_d;
        (0..d)
            .map(|i| CommittedPolynomial::NodeOutputRaD(node.idx, i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::onnx_proof::ops::test::unit_test_op;
    use atlas_onnx_tracer::{
        model::{test::ModelBuilder, Model},
        tensor::Tensor,
    };
    use rand::{rngs::StdRng, SeedableRng};

    fn relu_model(T: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let res = b.relu(i);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_relu() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random_small(&mut rng, &[T]);
        let model = relu_model(T);
        unit_test_op(model, &[input]);
    }
}
