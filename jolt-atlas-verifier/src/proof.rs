//! The proof object and its batched opening proof.

use std::collections::BTreeMap;

use joltworks::{
    field::JoltField,
    poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::{evaluation_reduction::EvalReductionProof, sumcheck::SumcheckInstanceProof},
    transcripts::Transcript,
};

use crate::{Claims, ProofId};

/// Proof for an ONNX neural network computation.
///
/// Contains all sumcheck proofs, polynomial commitments, and opening proofs
/// needed to verify the correct execution of a neural network. Fields are
/// private: the prover builds a proof with [`ONNXProof::from_parts`] and
/// readers go through the `&` accessors, so a finished proof is immutable.
#[derive(Debug, Clone)]
pub struct ONNXProof<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> {
    /// Opening claims for committed polynomials.
    opening_claims: Claims<F>,
    /// Map of proof IDs to sumcheck instance proofs.
    proofs: BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
    /// Polynomial commitments for witness polynomials.
    commitments: Vec<PCS::Commitment>,
    /// Evaluation reduction proofs h polynomials for each opening claim.
    eval_reduction_proofs: BTreeMap<usize, EvalReductionProof<F>>,
    /// Batched opening proof using reduction sum-check protocol to reduce all polynomial openings to the same point.
    reduced_opening_proof: Option<ReducedOpeningProof<F, T, PCS>>,
}

impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ONNXProof<F, T, PCS> {
    /// Assemble a proof from every field.
    pub fn from_parts(
        opening_claims: Claims<F>,
        proofs: BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
        commitments: Vec<PCS::Commitment>,
        eval_reduction_proofs: BTreeMap<usize, EvalReductionProof<F>>,
        reduced_opening_proof: Option<ReducedOpeningProof<F, T, PCS>>,
    ) -> Self {
        Self {
            opening_claims,
            proofs,
            commitments,
            eval_reduction_proofs,
            reduced_opening_proof,
        }
    }

    /// Opening claims for committed and virtual polynomials.
    pub fn opening_claims(&self) -> &Claims<F> {
        &self.opening_claims
    }

    /// Sumcheck instance proofs, keyed by proof id.
    pub fn proofs(&self) -> &BTreeMap<ProofId, SumcheckInstanceProof<F, T>> {
        &self.proofs
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
    pub fn reduced_opening_proof(&self) -> Option<&ReducedOpeningProof<F, T, PCS>> {
        self.reduced_opening_proof.as_ref()
    }
}

/// Mutable field access for the soundness harness (`test-util`).
#[cfg(feature = "test-util")]
impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ONNXProof<F, T, PCS> {
    /// Mutable opening claims.
    pub fn opening_claims_mut(&mut self) -> &mut Claims<F> {
        &mut self.opening_claims
    }

    /// Mutable sumcheck proofs.
    pub fn proofs_mut(&mut self) -> &mut BTreeMap<ProofId, SumcheckInstanceProof<F, T>> {
        &mut self.proofs
    }

    /// Mutable commitments.
    pub fn commitments_mut(&mut self) -> &mut Vec<PCS::Commitment> {
        &mut self.commitments
    }

    /// Mutable evaluation-reduction proofs.
    pub fn eval_reduction_proofs_mut(&mut self) -> &mut BTreeMap<usize, EvalReductionProof<F>> {
        &mut self.eval_reduction_proofs
    }

    /// Mutable batched opening proof.
    pub fn reduced_opening_proof_mut(&mut self) -> &mut Option<ReducedOpeningProof<F, T, PCS>> {
        &mut self.reduced_opening_proof
    }
}

/// Batched polynomial opening proof using sumcheck reduction.
///
/// Reduces multiple polynomial openings to a single joint opening using sumcheck.
#[derive(Debug, Clone)]
pub struct ReducedOpeningProof<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> {
    /// Sumcheck proof for batching multiple openings.
    sumcheck_proof: SumcheckInstanceProof<F, T>,
    /// Evaluation claims at the sumcheck point.
    sumcheck_claims: Vec<F>,
    /// Joint opening proof for the batched polynomial.
    joint_opening_proof: PCS::Proof,
}

impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ReducedOpeningProof<F, T, PCS> {
    /// Assemble a reduced opening proof from every field.
    pub fn from_parts(
        sumcheck_proof: SumcheckInstanceProof<F, T>,
        sumcheck_claims: Vec<F>,
        joint_opening_proof: PCS::Proof,
    ) -> Self {
        Self {
            sumcheck_proof,
            sumcheck_claims,
            joint_opening_proof,
        }
    }

    /// The opening-reduction sumcheck proof.
    pub fn sumcheck_proof(&self) -> &SumcheckInstanceProof<F, T> {
        &self.sumcheck_proof
    }

    /// Evaluation claims at the reduction sumcheck point.
    pub fn sumcheck_claims(&self) -> &[F] {
        &self.sumcheck_claims
    }

    /// The PCS opening proof of the batched polynomial.
    pub fn joint_opening_proof(&self) -> &PCS::Proof {
        &self.joint_opening_proof
    }
}

/// Mutable field access for the soundness harness (`test-util`).
#[cfg(feature = "test-util")]
impl<F: JoltField, T: Transcript, PCS: CommitmentScheme<Field = F>> ReducedOpeningProof<F, T, PCS> {
    /// Mutable reduction sumcheck proof.
    pub fn sumcheck_proof_mut(&mut self) -> &mut SumcheckInstanceProof<F, T> {
        &mut self.sumcheck_proof
    }

    /// Mutable reduction claims.
    pub fn sumcheck_claims_mut(&mut self) -> &mut Vec<F> {
        &mut self.sumcheck_claims
    }

    /// Mutable PCS opening proof.
    pub fn joint_opening_proof_mut(&mut self) -> &mut PCS::Proof {
        &mut self.joint_opening_proof
    }
}
