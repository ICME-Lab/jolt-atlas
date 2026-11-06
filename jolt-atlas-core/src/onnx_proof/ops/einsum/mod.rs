use atlas_trace::{op::Einsum, Model, Op, Trace};

use crate::node_view::{ComputationNode, ModelView};
use crate::{
    onnx_proof::{
        clamp_lookups::is_scalar,
        fused_rebase,
        ops::{
            einsum::contraction::{
                Contraction, ContractionParams, ContractionProver, ContractionVerifier,
            },
            OperatorProofTrait, Prover, Verifier,
        },
        ProofId, ProofType,
    },
    utils::dims::lookup_einsum_config,
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

/// Shared sumcheck engine for einsum contractions.
pub mod contraction;
mod named_point;

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

/// Derive the [`Contraction`] of an einsum node.
///
/// The registry lookup and its dims extractor only gate the accepted equations
/// (and `m == 1` for `mk,nk->mn`); this gate is temporary.
fn einsum_contraction(computation_node: &ComputationNode, model: &Model) -> Contraction {
    let Op::Einsum(Einsum { equation, .. }) = &computation_node.operator else {
        panic!("Unexpected operator")
    };
    let config = lookup_einsum_config(&equation.to_string());
    (config.dims_extractor)(computation_node, model);
    let [left, right] = computation_node.inputs[..] else {
        panic!("Expected two operands for einsum operation")
    };
    Contraction::new(
        equation,
        &model.view(left).physical_outdims(),
        &model.view(right).physical_outdims(),
    )
}

/// Prover dispatcher for the Einsum contraction (matmul) sumcheck.
pub struct EinsumProver;

impl EinsumProver {
    /// Create a sumcheck prover for the specified Einsum equation.
    pub fn sumcheck<F: JoltField, T: Transcript>(
        model: &Model,
        trace: &Trace,
        computation_node: ComputationNode,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Box<dyn SumcheckInstanceProver<F, T>> {
        let contraction = einsum_contraction(&computation_node, model);
        let params = ContractionParams::new(computation_node, contraction, accumulator);
        Box::new(ContractionProver::initialize(trace, params))
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
        let contraction = einsum_contraction(&computation_node, model);
        Box::new(ContractionVerifier::new(
            computation_node,
            contraction,
            accumulator,
        ))
    }
}
