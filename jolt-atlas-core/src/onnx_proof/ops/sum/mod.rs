use crate::node_view::ComputationNode;
use atlas_trace::op::Sum;
use common::{CommittedPoly, VirtualPoly};
use joltworks::{
    self,
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, OpeningId, SumcheckId},
    subprotocols::sumcheck::{Sumcheck, SumcheckInstanceProof},
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

use crate::{
    onnx_proof::{
        clamp_lookups::{
            clamp_committed_polys, is_scalar, prove_append_acc, prove_clamp_lookup,
            verify_append_acc, verify_clamp_lookup, verify_scalar_clamp,
        },
        ops::{sum::axis::SumAxisRelation, OperatorProofTrait},
        relation::{RelationParams, RelationProver, RelationVerifier},
        ProofId, ProofType, Prover, Verifier,
    },
    utils::opening_access::AccOpeningAccessor,
};

crate::bind_relation!(Sum, SumAxisRelation);

/// Axis-wise sum implementations for sumcheck protocol.
pub mod axis;

/// Saturating axis-sum proof: `output = SatClamp(acc)`, `acc[k] = Σ_axis operand[k, ·]`.
///
/// Reuses the [`Add`](super::add)/[`Sub`](super::sub) clamp infrastructure: the
/// per-output-element clamp `output = SatClamp(acc)` is identical (an element-wise
/// `i64 → i32` saturation), only the accumulation differs. So the proof is:
///
/// 1. **Clamp lookup** ([`ClampLookupProvider`], `ProofType::Execution`) +
///    **one-hot checks** (`ProofType::RaOneHotChecks`): prove
///    `output(r) = SatClamp(acc(r))` and tie `acc` ([`VirtualPoly::ClampAcc`]).
/// 2. **Axis reduction** ([`SumAxisRelation`], `ProofType::SumReduction`): prove
///    `acc(r) = Σ_axis operand`, reducing to the operand opening. Its input claim
///    is `acc(r)` (the lookup's `raf`), not the node output.
///
/// Scalar (`1×1`) outputs skip the clamp lookup (its one-hot reduction degenerates
/// at `log_T = 0`): `acc` opens in the clear, so the verifier checks
/// `output == SatClamp(acc)` directly — and the axis reduction still binds `acc`
/// to the operand.
impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Sum {
    #[tracing::instrument(skip_all, name = "Sum::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        // (1+2) Prove `output = SatClamp(acc)`. Non-scalar: clamp lookup + one-hot.
        // Scalar: just append `acc` in the clear (verifier checks the clamp).
        let mut results = if is_scalar(node) {
            prove_append_acc(
                node,
                &prover.trace,
                &mut prover.accumulator,
                &mut prover.transcript,
                None,
            );
            vec![]
        } else {
            prove_clamp_lookup(node, prover, None)
        };

        // (3) Axis reduction: acc(r) = Σ_axis operand (input claim = ClampAcc).
        let params = RelationParams::<F, _>::new(
            SumAxisRelation::for_node(&prover.preprocessing.model, node.clone()),
            &prover.accumulator,
            &mut prover.transcript,
        );
        let mut reduction = RelationProver::initialize(&prover.trace, params);
        let (proof, _) = Sumcheck::prove(
            &mut reduction,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
        results.push((ProofId(node.idx, ProofType::SumReduction), proof));

        results
    }

    #[tracing::instrument(skip_all, name = "Sum::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        // (1+2) Verify `output = SatClamp(acc)`.
        if is_scalar(node) {
            // `acc` opens in the clear: recover it and check the clamp directly.
            verify_append_acc(node, &mut verifier.accumulator, &mut verifier.transcript);
            let acc_id = OpeningId::new(
                VirtualPoly::ClampAcc(node.idx),
                SumcheckId::NodeExecution(node.idx),
            );
            let acc_claim = verifier
                .accumulator
                .get_virtual_polynomial_opening(acc_id)
                .1;
            let output_claim = AccOpeningAccessor::new(&verifier.accumulator, node)
                .get_reduced_opening()
                .1;
            verify_scalar_clamp(acc_claim, output_claim, "Sum")?;
        } else {
            verify_clamp_lookup(node, verifier)?;
        }

        // (3) Axis reduction.
        let proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::SumReduction))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let reduction = RelationVerifier::<F, _>::new(
            SumAxisRelation::for_node(&verifier.preprocessing.model, node.clone()),
            &verifier.accumulator,
            &mut verifier.transcript,
        );
        Sumcheck::verify(
            proof,
            &reduction,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        clamp_committed_polys(node)
    }
}
