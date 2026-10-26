use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    clamp_lookups::clamp_committed_polys,
    ops::{
        add::{prove_accumulation, verify_accumulation, AccumulationRelation},
        OperatorProofTrait,
    },
    ProofId, Prover, Verifier,
};
use atlas_trace::op::Sub;
use common::CommittedPoly;
use joltworks::{
    field::JoltField, subprotocols::sumcheck::SumcheckInstanceProof, transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

crate::bind_relation!(Sub, AccumulationRelation<Sub>);

/// Proof for saturating element-wise subtraction: `output = SatClamp(left - right)`.
///
/// Identical in structure to [`Add`](super::add) but with a subtraction tie:
/// the accumulation `acc = left - right` is widened to `i64`, the clamp lookup
/// proves `output(r) = SatClamp(acc(r))` and ties `acc` ([`VirtualPoly::ClampAcc`])
/// to the one-hot read-address poly ([`VirtualPoly::ClampRa`]), the one-hot
/// checks discharge `ClampRa` against [`CommittedPoly::ClampRaD`], and the
/// operand tie ([`AccumulationRelation`]) checks `left(r) - right(r) == acc(r)`. See
/// [`Add`](super::add) for the full three-part breakdown.
///
/// [`VirtualPoly::ClampAcc`]: common::VirtualPoly::ClampAcc
/// [`VirtualPoly::ClampRa`]: common::VirtualPoly::ClampRa
impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Sub {
    #[tracing::instrument(skip_all, name = "Sub::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        prove_accumulation::<Sub, F, T>(node, prover)
    }

    #[tracing::instrument(skip_all, name = "Sub::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        verify_accumulation::<Sub, F, T>(node, verifier)
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        clamp_committed_polys(node)
    }
}
