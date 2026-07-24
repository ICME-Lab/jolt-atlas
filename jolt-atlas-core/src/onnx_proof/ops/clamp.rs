use crate::onnx_proof::{
    op_lookups::{LookupOperandsTrait, OpLookupEncoding, OpLookupProvider},
    ops::OperatorProofTrait,
    ProofId, ProofType, Prover, Verifier,
};
use atlas_onnx_tracer::{
    node::ComputationNode,
    ops::Clamp,
    tensor::{Tensor, TensorError},
};
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    self,
    field::JoltField,
    lookup_tables::clamp::{ClampTable, CLAMP_BOUND},
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

use common::consts::XLEN;

/// Offsets the input by `+2^CLAMP_BOUND` before the lookup, mapping `Clamp`'s symmetric range
/// onto the table's `[0, 2^CLAMP_TABLE_BOUND]` domain, then offsets the resulting claims back
/// by the same constant. Both offsets are applied algebraically to the claims (never to a
/// committed polynomial), via [`LookupOperandsTrait`].
#[derive(Default)]
pub(crate) struct SymmetricClampOperands;

impl SymmetricClampOperands {
    pub(crate) const OFFSET: i32 = 1 << CLAMP_BOUND;
}

impl LookupOperandsTrait for SymmetricClampOperands {
    fn transform_operand_claims<F: JoltField>(&self, claims: Vec<F>) -> (F, F) {
        (claims[0], claims[1] + F::from_u64(Self::OFFSET as u64))
    }

    fn transform_output_claim<F: JoltField>(&self, claim: F) -> F {
        claim + F::from_u64(Self::OFFSET as u64)
    }

    fn build_lookup_operands(&self, operand_tensors: &[Tensor<i32>]) -> Vec<Tensor<i32>> {
        operand_tensors
            .iter()
            .map(|t| {
                t.par_enum_map(|_, v| Ok::<_, TensorError>(v + Self::OFFSET))
                    .unwrap()
            })
            .collect()
    }

    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        VirtualPoly::SymmetricClampRa(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        CommittedPoly::SymmetricClampRaD(node_idx, d)
    }
}

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Clamp {
    #[tracing::instrument(skip_all, name = "Clamp::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        debug_assert_eq!(self.bound_log, CLAMP_BOUND);
        let mut results = Vec::new();

        // Execution proof
        let provider: OpLookupProvider<SymmetricClampOperands> =
            OpLookupProvider::new(node.clone());
        let (mut execution_sumcheck, lookup_indices) = provider
            .read_raf_prove::<F, T, ClampTable<XLEN>>(
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
        let encoding = provider.encoding();

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

    #[tracing::instrument(skip_all, name = "Clamp::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        debug_assert_eq!(self.bound_log, CLAMP_BOUND);

        // Verify execution proof
        let provider: OpLookupProvider<SymmetricClampOperands> =
            OpLookupProvider::new(node.clone());
        let verifier_sumcheck = provider.read_raf_verify::<F, T, ClampTable<XLEN>>(
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

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let encoding = OpLookupEncoding::<SymmetricClampOperands>::new(node);
        let d = encoding.one_hot_params().instruction_d;
        (0..d)
            .map(|i| CommittedPoly::SymmetricClampRaD(node.idx, i))
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
    use joltworks::lookup_tables::clamp::CLAMP_BOUND;
    use rand::{rngs::StdRng, SeedableRng};

    fn clamp_model(T: usize) -> Model {
        let mut b = ModelBuilder::new();
        let i = b.input(vec![T]);
        let res = b.clamp(i, CLAMP_BOUND);
        b.mark_output(res);
        b.build()
    }

    #[test]
    fn test_clamp() {
        let T = 1 << 16;
        let mut rng = StdRng::seed_from_u64(0x888);
        let input = Tensor::<i32>::random(&mut rng, &[T]);
        let model = clamp_model(T);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_clamp_saturating_range() {
        // Deliberately spans both sides of [-2^CLAMP_BOUND, 2^CLAMP_BOUND] to exercise
        // both saturation directions as well as the unsaturated middle.
        let t = 8;
        let bound = 1i32 << CLAMP_BOUND;
        let mut rng = StdRng::seed_from_u64(0x890);
        let input = Tensor::<i32>::random_range(&mut rng, &[t], -2 * bound..2 * bound);
        let model = clamp_model(t);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_clamp_non_power_of_two_input_len() {
        let t = 1000;
        let mut rng = StdRng::seed_from_u64(0x889);
        let input = Tensor::<i32>::random(&mut rng, &[t]);
        let model = clamp_model(t);
        unit_test_op(model, &[input]);
    }
}
