use std::sync::Arc;

use crate::{
    onnx_proof::{
        clamp_lookups::is_scalar,
        fused_rebase,
        ops::{
            einsum::{
                bmk_rhs_mbn::{BmkRhs, BmkRhsMbnLayout},
                dot::{EinsumDotParams, EinsumDotProver, EinsumDotVerifier, EinsumLayout},
                k_nk_n::KNkNLayout,
                m_an_a1nm::{MAnA1nmParams, MAnA1nmProver, MAnA1nmVerifier},
                mbk_rhs_bmn::{MbkRhs, MbkRhsBmnLayout},
                mk_kn_mn::MkKnMnLayout,
                rbmk_rbnk_bmn::RbmkRbnkBmnLayout,
            },
            OperatorProofTrait, Prover, Verifier,
        },
        ProofId, ProofType,
    },
    utils::dims::{lookup_einsum_config, EinsumDims},
};
use atlas_onnx_tracer::{
    model::{trace::Trace, Model},
    node::ComputationNode,
    ops::{Einsum, Operator},
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

/// `bmk,?->mbn` layouts (`bmk,bkn->mbn`, `bmk,kbn->mbn`).
pub mod bmk_rhs_mbn;
/// Shared sumcheck engine for einsum contraction patterns.
pub mod dot;
/// `k,nk->n` layout (vector-matrix multiply).
pub mod k_nk_n;
/// Family for outer-product / broadcast pattern: m,an->a1nm
pub mod m_an_a1nm;
/// `mbk,?->bmn` layouts (`mbk,bnk->bmn`, `mbk,nbk->bmn`).
pub mod mbk_rhs_bmn;
/// `mk,kn->mn` layout (matrix-matrix multiply).
pub mod mk_kn_mn;
/// `rbmk,rbnk->bmn`-family layouts (shared retained/reduced batch-pack patterns).
pub mod rbmk_rbnk_bmn;

// Qwen einsum pattern coverage:
//
// Normalized through EINSUM_REGISTRY into existing provers:
// - amk,kn->amn
// - amk,kn->mn
// - mk,kn->amn
// - m,an->abnm   (canonicalized as m,an->a1nm)
//
// Routed through the shared rbmk,rbnk->bmn prover family:
// - abmk,abnk->abmn
// - acbmk,kcn->cbmn
// - cbmk,cbkn->amn

impl<F: JoltField, T: Transcript> OperatorProofTrait<F, T> for Einsum {
    #[tracing::instrument(skip_all, name = "Einsum::prove")]
    fn prove(
        &self,
        node: &ComputationNode,
        prover: &mut Prover<F, T>,
    ) -> Vec<(ProofId, SumcheckInstanceProof<F, T>)> {
        let fused = fused_rebase::fuses_rebase(&node.operator);

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
        let fused = fused_rebase::fuses_rebase(&node.operator);

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

/// Returns the raw ONNX einsum equation for a computation node, panicking if the
/// node is not an Einsum operator.
fn einsum_equation(computation_node: &ComputationNode) -> &str {
    match &computation_node.operator {
        Operator::Einsum(Einsum { equation, .. }) => equation.as_str(),
        _ => panic!("Unexpected operator"),
    }
}

/// Build the [`EinsumLayout`] for a canonical equation handled by the shared
/// [`dot`] engine, or `None` for patterns with a bespoke prover (e.g. `m,an->a1nm`).
fn einsum_layout<F: JoltField>(
    equation: &str,
    computation_node: &ComputationNode,
    einsum_dims: &EinsumDims,
) -> Option<Arc<dyn EinsumLayout<F>>> {
    let layout: Arc<dyn EinsumLayout<F>> = match equation {
        "mk,kn->mn" => Arc::new(MkKnMnLayout::new(einsum_dims)),
        "k,nk->n" => Arc::new(KNkNLayout::new(einsum_dims)),
        "bmk,kbn->mbn" => Arc::new(BmkRhsMbnLayout::new(einsum_dims, BmkRhs::Kbn)),
        "bmk,bkn->mbn" => Arc::new(BmkRhsMbnLayout::new(einsum_dims, BmkRhs::Bkn)),
        "mbk,nbk->bmn" => Arc::new(MbkRhsBmnLayout::new(einsum_dims, MbkRhs::Nbk)),
        "mbk,bnk->bmn" => Arc::new(MbkRhsBmnLayout::new(einsum_dims, MbkRhs::Bnk)),
        "rbmk,rbnk->bmn" => Arc::new(RbmkRbnkBmnLayout::new(computation_node, einsum_dims)),
        _ => return None,
    };
    Some(layout)
}

/// Prover dispatcher for the Einsum contraction (matmul) sumcheck.
///
/// Contraction patterns route through the shared [`dot`] engine via their
/// [`EinsumLayout`]; `m,an->a1nm` has a bespoke (zero-round) prover.
pub struct EinsumProver;

impl EinsumProver {
    /// Create a sumcheck prover for the specified Einsum equation.
    pub fn sumcheck<F: JoltField, T: Transcript>(
        model: &Model,
        trace: &Trace,
        computation_node: ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Box<dyn SumcheckInstanceProver<F, T>> {
        let config = lookup_einsum_config(einsum_equation(&computation_node));
        let einsum_dims = (config.dims_extractor)(&computation_node, model);
        if let Some(layout) = einsum_layout::<F>(config.equation, &computation_node, &einsum_dims) {
            let params = EinsumDotParams::new(computation_node, layout, accumulator);
            return Box::new(EinsumDotProver::initialize(trace, params));
        }
        match config.equation {
            "m,an->a1nm" => {
                let params = MAnA1nmParams::new(computation_node, einsum_dims, accumulator);
                Box::new(MAnA1nmProver::initialize(trace, params))
            }
            other => panic!("unexpected equation: {other}"),
        }
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
        let config = lookup_einsum_config(einsum_equation(&computation_node));
        let einsum_dims = (config.dims_extractor)(&computation_node, model);
        if let Some(layout) = einsum_layout::<F>(config.equation, &computation_node, &einsum_dims) {
            return Box::new(EinsumDotVerifier::new(
                computation_node,
                layout,
                accumulator,
            ));
        }
        match config.equation {
            "m,an->a1nm" => Box::new(MAnA1nmVerifier::new(
                computation_node,
                einsum_dims,
                accumulator,
            )),
            other => panic!("unexpected equation: {other}"),
        }
    }
}
