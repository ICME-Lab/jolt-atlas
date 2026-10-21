use std::sync::Arc;

use atlas_trace::Model;

use atlas_trace::Op;

use atlas_trace::Trace;

use atlas_trace::op::Einsum;

use crate::node_view::ComputationNode;
use crate::{
    onnx_proof::{
        clamp_lookups::is_scalar,
        fused_rebase,
        ops::{
            einsum::{
                abmk_abnk_abmn::AbmkAbnkAbmnLayout,
                dot::{EinsumDotParams, EinsumDotProver, EinsumDotVerifier, EinsumLayout},
                k_nk_n::KNkNLayout,
                mk_kn_mn::MkKnMnLayout,
            },
            OperatorProofTrait, Prover, Verifier,
        },
        ProofId, ProofType,
    },
    utils::dims::{lookup_einsum_config, EinsumDims},
};
use common::CommittedPoly;
use joltworks::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, VerifierOpeningAccumulator},
    subprotocols::{
        sumcheck::{Sumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};

/// `abmk,abnk->abmn` layout (batched matmul; also covers `amk,ank->amn`).
pub mod abmk_abnk_abmn;
/// Shared sumcheck engine for einsum contraction patterns.
pub mod dot;
/// `k,nk->n` layout (vector-matrix multiply).
pub mod k_nk_n;
/// `mk,kn->mn` layout (matrix-matrix multiply).
pub mod mk_kn_mn;

// atlas-onnx-import lowers every einsum to one of four canonical full-letter
// equations (`CANONICAL_EINSUMS` in its lower.rs); EINSUM_REGISTRY maps each
// onto one of the three layouts above:
// - mk,kn->mn       → mk_kn_mn
// - mk,nk->mn       → k_nk_n   (vector-matrix; m == 1 enforced at dispatch)
// - amk,ank->amn    → abmk_abnk_abmn (batch pack 1×a)
// - abmk,abnk->abmn → abmk_abnk_abmn

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Einsum {
    #[tracing::instrument(skip_all, name = "Einsum::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let fused = fused_rebase::fuses_rebase(node);

        // The fused einsum proves `Σ_k L·R = rescaled·2^S + R` (division) and
        // `output = SatClamp(rescaled)` (clamp), both handled by the shared
        // `fused_rebase` stages; only the contraction (matmul) sumcheck is
        // einsum-specific. See [`fused_rebase`] for the seam.
        let mut proofs = Vec::new();
        let remainder = if fused {
            let (pre_proofs, remainder) = fused_rebase::prove_pre(node, prover);
            proofs.extend(pre_proofs);
            Some(remainder)
        } else {
            None
        };

        // EinsumMatmul: the contraction sumcheck (initial claim `acc(r)` via
        // `fused_rebase::fused_input_claim`).
        let mut matmul = EinsumProver::sumcheck(
            &prover.preprocessing.model,
            &prover.trace,
            node.clone(),
            &prover.accumulator,
        );
        let (matmul_proof, _) = Sumcheck::prove(
            &mut *matmul,
            &mut prover.accumulator,
            &mut prover.transcript,
        );
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

    #[tracing::instrument(skip_all, name = "Einsum::verify")]
    fn verify(
        &self,
        node: &ComputationNode,
        verifier: &mut Verifier<'_, F, T>,
    ) -> Result<(), ProofVerifyError> {
        let fused = fused_rebase::fuses_rebase(node);

        // Mirror the prover: remainder advice + clamp, then the matmul, then the
        // remainder range-check (or the scalar clear checks).
        if fused {
            fused_rebase::verify_pre(node, verifier)?;
        }

        let matmul_proof = verifier
            .proofs
            .get(&ProofId(node.idx, ProofType::EinsumMatmul))
            .ok_or(ProofVerifyError::MissingProof(node.idx))?;
        let matmul = EinsumVerifier::sumcheck(
            &verifier.preprocessing.model,
            node.clone(),
            &verifier.accumulator,
        );
        Sumcheck::verify(
            matmul_proof,
            &*matmul,
            &mut verifier.accumulator,
            &mut verifier.transcript,
        )?;

        if fused {
            fused_rebase::verify_post(node, verifier)?;
        }

        Ok(())
    }

    fn get_committed_polynomials(&self, node: &ComputationNode) -> Vec<CommittedPoly> {
        fused_rebase::committed_polys(node)
    }
}

/// Returns the canonical einsum equation of a node, panicking if the node is
/// not an Einsum operator.
fn einsum_equation(computation_node: &ComputationNode) -> String {
    match &computation_node.operator {
        Op::Einsum(Einsum { equation, .. }) => equation.to_string(),
        _ => panic!("Unexpected operator"),
    }
}

/// Build the [`EinsumLayout`] for a canonical equation; every registered
/// equation is handled by the shared [`dot`] engine.
fn einsum_layout<F: JoltField>(
    equation: &str,
    einsum_dims: &EinsumDims,
) -> Arc<dyn EinsumLayout<F>> {
    match equation {
        "mk,kn->mn" => Arc::new(MkKnMnLayout::new(einsum_dims)),
        "k,nk->n" => Arc::new(KNkNLayout::new(einsum_dims)),
        "abmk,abnk->abmn" => Arc::new(AbmkAbnkAbmnLayout::new(einsum_dims)),
        other => panic!("unexpected canonical einsum equation: {other}"),
    }
}

/// Prover dispatcher for the Einsum contraction (matmul) sumcheck.
///
/// Every contraction pattern routes through the shared [`dot`] engine via its
/// [`EinsumLayout`].
pub struct EinsumProver;

impl EinsumProver {
    /// Create a sumcheck prover for the specified Einsum equation.
    pub fn sumcheck<F: JoltField, T: Transcript>(
        model: &Model,
        trace: &Trace,
        computation_node: ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Box<dyn SumcheckInstanceProver<F, T>> {
        let config = lookup_einsum_config(&einsum_equation(&computation_node));
        let einsum_dims = (config.dims_extractor)(&computation_node, model);
        let layout = einsum_layout::<F>(config.equation, &einsum_dims);
        let params = EinsumDotParams::new(computation_node, layout, accumulator);
        Box::new(EinsumDotProver::initialize(trace, params))
    }
}

/// Verifier dispatcher for the Einsum contraction (matmul) sumcheck.
pub struct EinsumVerifier;

impl EinsumVerifier {
    /// Create a sumcheck verifier for the specified Einsum equation.
    pub fn sumcheck<F: JoltField, T: Transcript>(
        model: &Model,
        computation_node: ComputationNode,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Box<dyn SumcheckInstanceVerifier<F, T>> {
        let config = lookup_einsum_config(&einsum_equation(&computation_node));
        let einsum_dims = (config.dims_extractor)(&computation_node, model);
        let layout = einsum_layout::<F>(config.equation, &einsum_dims);
        Box::new(EinsumDotVerifier::new(
            computation_node,
            layout,
            accumulator,
        ))
    }
}
