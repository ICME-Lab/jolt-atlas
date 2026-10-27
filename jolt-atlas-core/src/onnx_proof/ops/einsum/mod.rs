use crate::onnx_proof::ops::ProveOp;
use atlas_sumcheck::relation::prover::prove_relation;
use atlas_trace::op::Einsum;
pub use jolt_atlas_verifier::onnx_proof::ops::einsum::*;

use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    clamp_lookups::is_scalar,
    fused_rebase,
    ops::{einsum::dot::EinsumDotProver, Prover},
    relation::RelationParams,
    ProofId, ProofType,
};
use atlas_algebra::field::JoltField;
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_transcript::Transcript;

pub mod abmk_abnk_abmn;
/// Shared sumcheck engine for einsum contraction patterns.
pub mod dot;
/// `k,nk->n` layout (vector-matrix multiply).
pub mod k_nk_n;
/// `mk,kn->mn` layout (matrix-matrix multiply).
pub mod mk_kn_mn;

impl<F: JoltField, T: Transcript> ProveOp<F, T> for Einsum {
    #[tracing::instrument(skip_all, name = "Einsum::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F>)> {
        let fused = fused_rebase::fuses_rebase(node);

        let mut proofs = Vec::new();
        let remainder = if fused {
            let (pre_proofs, remainder) = fused_rebase::prove_pre(node, prover);
            proofs.extend(pre_proofs);
            Some(remainder)
        } else {
            None
        };

        let relation = ContractionRelation::for_node(&prover.preprocessing.model, node.clone());
        let params = RelationParams::new(relation, &prover.accumulator, &mut prover.transcript);
        let mut matmul = EinsumDotProver::initialize(&prover.trace, params);
        let (matmul_proof, _) =
            prove_relation(&mut matmul, &mut prover.accumulator, &mut prover.transcript);
        proofs.push((ProofId(node.idx, ProofType::EinsumMatmul), matmul_proof));

        if fused && !is_scalar(node) {
            proofs.extend(fused_rebase::prove_remainder_rc(
                node,
                prover,
                remainder.as_ref().expect("fused"),
            ));
        }

        proofs
    }
}
