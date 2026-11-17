//! The proof object and its batched opening proof.

use std::collections::BTreeMap;

use atlas_algebra::field::JoltField;
use atlas_pcs::commitment_scheme::CommitmentScheme;
use atlas_sumcheck::{evaluation_reduction::EvalReductionProof, sumcheck::SumcheckInstanceProof};

use crate::ProofId;

/// Proof for an ONNX neural network computation.
///
/// Contains all sumcheck proofs, polynomial commitments, and opening proofs
/// needed to verify the correct execution of a neural network; the opening claims are in the
/// NARG string, each where the verifier binds it. Fields are
/// private: the prover builds a proof with [`ONNXProof::from_parts`] and
/// readers go through the `&` accessors, so a finished proof is immutable.
#[derive(Debug, Clone)]
pub struct ONNXProof<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Map of proof IDs to sumcheck instance proofs.
    proofs: BTreeMap<ProofId, SumcheckInstanceProof<F>>,
    /// Commitment data shared by every witness polynomial (empty for homomorphic schemes).
    batch_commitment: PCS::BatchCommitment,
    /// Polynomial commitments for witness polynomials.
    commitments: Vec<PCS::Commitment>,
    /// Evaluation reduction proofs h polynomials for each opening claim.
    eval_reduction_proofs: BTreeMap<usize, EvalReductionProof<F>>,
    /// Batched opening proof using reduction sum-check protocol to reduce all polynomial openings to the same point.
    reduced_opening_proof: Option<ReducedOpeningProof<F, PCS>>,
    /// The transcript's NARG string: the opening claims and the proof-of-work nonces, in the
    /// order the verifier's transcript reads them back; it rejects a string it does not consume
    /// exactly.
    narg: Vec<u8>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> ONNXProof<F, PCS> {
    /// Assemble a proof from every field.
    pub fn from_parts(
        proofs: BTreeMap<ProofId, SumcheckInstanceProof<F>>,
        batch_commitment: PCS::BatchCommitment,
        commitments: Vec<PCS::Commitment>,
        eval_reduction_proofs: BTreeMap<usize, EvalReductionProof<F>>,
        reduced_opening_proof: Option<ReducedOpeningProof<F, PCS>>,
        narg: Vec<u8>,
    ) -> Self {
        Self {
            proofs,
            batch_commitment,
            commitments,
            eval_reduction_proofs,
            reduced_opening_proof,
            narg,
        }
    }

    /// Sumcheck instance proofs, keyed by proof id.
    pub fn proofs(&self) -> &BTreeMap<ProofId, SumcheckInstanceProof<F>> {
        &self.proofs
    }

    /// Commitment data shared by every witness polynomial.
    pub fn batch_commitment(&self) -> &PCS::BatchCommitment {
        &self.batch_commitment
    }

    /// Witness polynomial commitments, in sorted `CommittedPoly` order.
    pub fn commitments(&self) -> &[PCS::Commitment] {
        &self.commitments
    }

    /// Evaluation-reduction proofs, keyed by node index.
    pub fn eval_reduction_proofs(&self) -> &BTreeMap<usize, EvalReductionProof<F>> {
        &self.eval_reduction_proofs
    }

    /// The batched opening proof, absent when nothing committed was opened.
    pub fn reduced_opening_proof(&self) -> Option<&ReducedOpeningProof<F, PCS>> {
        self.reduced_opening_proof.as_ref()
    }

    /// The transcript's NARG string (opening claims and proof-of-work nonces).
    pub fn narg(&self) -> &[u8] {
        &self.narg
    }
}

/// Mutable field access for the soundness harness (`test-util`).
#[cfg(feature = "test-util")]
impl<F: JoltField, PCS: CommitmentScheme<Field = F>> ONNXProof<F, PCS> {
    /// Mutable sumcheck proofs.
    pub fn proofs_mut(&mut self) -> &mut BTreeMap<ProofId, SumcheckInstanceProof<F>> {
        &mut self.proofs
    }

    /// Mutable batch commitment.
    pub fn batch_commitment_mut(&mut self) -> &mut PCS::BatchCommitment {
        &mut self.batch_commitment
    }

    /// Mutable commitments.
    pub fn commitments_mut(&mut self) -> &mut Vec<PCS::Commitment> {
        &mut self.commitments
    }

    /// Mutable evaluation-reduction proofs.
    pub fn eval_reduction_proofs_mut(&mut self) -> &mut BTreeMap<usize, EvalReductionProof<F>> {
        &mut self.eval_reduction_proofs
    }

    /// Mutable NARG string.
    pub fn narg_mut(&mut self) -> &mut Vec<u8> {
        &mut self.narg
    }

    /// Mutable batched opening proof.
    pub fn reduced_opening_proof_mut(&mut self) -> &mut Option<ReducedOpeningProof<F, PCS>> {
        &mut self.reduced_opening_proof
    }
}

/// Batched polynomial opening proof using sumcheck reduction.
///
/// Reduces multiple polynomial openings to a single joint opening using sumcheck.
#[derive(Debug, Clone)]
pub struct ReducedOpeningProof<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Sumcheck proof for batching multiple openings.
    sumcheck_proof: SumcheckInstanceProof<F>,
    /// Joint opening proof for the batched polynomial.
    joint_opening_proof: PCS::Proof,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> ReducedOpeningProof<F, PCS> {
    /// Assemble a reduced opening proof from every field.
    pub fn from_parts(
        sumcheck_proof: SumcheckInstanceProof<F>,
        joint_opening_proof: PCS::Proof,
    ) -> Self {
        Self {
            sumcheck_proof,
            joint_opening_proof,
        }
    }

    /// The opening-reduction sumcheck proof.
    pub fn sumcheck_proof(&self) -> &SumcheckInstanceProof<F> {
        &self.sumcheck_proof
    }

    /// The PCS opening proof of the batched polynomial.
    pub fn joint_opening_proof(&self) -> &PCS::Proof {
        &self.joint_opening_proof
    }
}

/// Mutable field access for the soundness harness (`test-util`).
#[cfg(feature = "test-util")]
impl<F: JoltField, PCS: CommitmentScheme<Field = F>> ReducedOpeningProof<F, PCS> {
    /// Mutable reduction sumcheck proof.
    pub fn sumcheck_proof_mut(&mut self) -> &mut SumcheckInstanceProof<F> {
        &mut self.sumcheck_proof
    }

    /// Mutable PCS opening proof.
    pub fn joint_opening_proof_mut(&mut self) -> &mut PCS::Proof {
        &mut self.joint_opening_proof
    }
}
