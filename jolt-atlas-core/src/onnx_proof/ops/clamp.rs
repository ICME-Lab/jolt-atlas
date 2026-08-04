use crate::{
    onnx_proof::{
        op_lookups::{
            DefaultLookupOperands, LookupOperandsTrait, OpLookupEncoding, OpLookupProvider,
        },
        ops::OperatorProofTrait,
        ProofId, ProofType, Prover, Verifier,
    },
    utils::compute_lookup_indices_from_operands,
};
use atlas_onnx_tracer::{model::trace::Trace, node::ComputationNode, ops::Clamp, tensor::Tensor};
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    field::JoltField,
    lookup_tables::clamp::{ClampTable, CLAMP_BOUND},
    poly::opening_proof::{OpeningAccumulator, OpeningId, OpeningPoint, BIG_ENDIAN},
    subprotocols::{
        shout::{self, RaOneHotEncoding},
        sumcheck::{BatchedSumcheck, Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, lookup_bits::LookupBits},
};

use common::consts::XLEN;

/// Lookup helper for the ONNX `Clamp` op, backed by `joltworks::lookup_tables::clamp`'s
/// symmetric clamp table (`[-2^CLAMP_BOUND, 2^CLAMP_BOUND - 1]`).
#[derive(Default)]
pub(crate) struct SymmetricClampOperands;

impl LookupOperandsTrait for SymmetricClampOperands {
    const LOG_K: usize = DefaultLookupOperands::LOG_K;

    fn rv_claim<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> F {
        DefaultLookupOperands::rv_claim(node, accumulator)
    }

    fn ra_virtual_poly(node_idx: usize) -> VirtualPoly {
        VirtualPoly::SymmetricClampRa(node_idx)
    }

    fn ra_committed_poly(node_idx: usize, d: usize) -> CommittedPoly {
        CommittedPoly::SymmetricClampRaD(node_idx, d)
    }

    fn witness_opening_id(node: &ComputationNode) -> OpeningId {
        DefaultLookupOperands::witness_opening_id(node)
    }

    fn r_cycle<F: JoltField>(
        node: &ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        DefaultLookupOperands::r_cycle(node, accumulator)
    }

    fn r_cycle_source(node_idx: usize) -> OpeningId {
        DefaultLookupOperands::r_cycle_source(node_idx)
    }

    fn witness(&self, node: &ComputationNode, trace: &Trace) -> Tensor<i64> {
        DefaultLookupOperands.witness(node, trace)
    }

    fn lookup_bits(witness: &Tensor<i64>) -> Vec<LookupBits> {
        let operand = witness.map(|v| v as i32);
        compute_lookup_indices_from_operands(&[&operand], false)
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
            .read_raf_prove::<F, T, ClampTable<XLEN>, XLEN>(
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
        let verifier_sumcheck = provider.read_raf_verify::<F, T, ClampTable<XLEN>, XLEN>(
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
        // Deliberately spans both sides of [-2^CLAMP_BOUND, 2^CLAMP_BOUND - 1] to exercise
        // both saturation directions as well as the unsaturated middle.
        let t = 8;
        let bound = 1i32 << CLAMP_BOUND;
        let mut rng = StdRng::seed_from_u64(0x890);
        let input = Tensor::<i32>::random_range(&mut rng, &[t], -2 * bound..2 * bound);
        let model = clamp_model(t);
        unit_test_op(model, &[input]);
    }

    #[test]
    fn test_clamp_near_i32_max_saturates_correctly() {
        let t = 4;
        let input = Tensor::<i32>::new(
            Some(&[i32::MAX, i32::MAX - 50, i32::MIN, i32::MIN + 50]),
            &[t],
        )
        .unwrap();
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
