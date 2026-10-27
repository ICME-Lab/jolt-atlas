use crate::node_view::ComputationNode;
use crate::onnx_proof::ops::{OpLayout, VerifyOp};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_sumcheck::opening_proof::{OpeningAccumulator, OpeningId, SumcheckId};
use atlas_sumcheck::relation::verify_relation;
use atlas_trace::op::Sum;
use atlas_transcript::Transcript;
use common::{CommittedPoly, VirtualPoly};

use crate::{
    onnx_proof::{
        clamp_lookups::{
            clamp_committed_polys, is_scalar, verify_append_acc, verify_clamp_lookup,
            verify_scalar_clamp,
        },
        ops::sum::axis::SumAxisRelation,
        relation::RelationVerifier,
        ProofId, ProofType, Verifier,
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
impl OpLayout for Sum {
    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        clamp_committed_polys(node)
    }
}

impl<F: JoltField, T: Transcript> VerifyOp<F, T> for Sum {
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
        verify_relation(
            proof,
            &reduction,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;
        Ok(())
    }
}
