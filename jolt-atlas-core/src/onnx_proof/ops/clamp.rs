use crate::node_view::ComputationNode;
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
use atlas_trace::{op::Clamp, Tensor, Trace};
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

    fn lookup_bits(&self, witness: &Tensor<i64>) -> Vec<LookupBits> {
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
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        debug_assert_eq!(self.bound_log as usize, CLAMP_BOUND);
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

        let [ra_prover, bool_prover] = shout::ra_onehot_provers(
            &encoding,
            &lookup_indices,
            &prover.accumulator,
            &mut prover.transcript,
        );

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<_, _>>> =
            vec![ra_prover, bool_prover];
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
        debug_assert_eq!(self.bound_log as usize, CLAMP_BOUND);

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

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        let encoding = OpLookupEncoding::<SymmetricClampOperands>::new(node);
        let d = encoding.one_hot_params().instruction_d;
        (0..d)
            .map(|i| CommittedPoly::SymmetricClampRaD(node.idx, i))
            .collect()
    }
}
