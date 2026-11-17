//! The proof object and its batched opening proof.

use atlas_algebra::field::JoltField;
use atlas_pcs::commitment_scheme::CommitmentScheme;

/// Proof for an ONNX neural network computation.
///
/// Contains the polynomial commitments, the joint opening proof and the NARG string, which
/// holds every opening claim, sumcheck round message and evaluation-reduction polynomial where
/// the verifier reads it. Fields are
/// private: the prover builds a proof with [`ONNXProof::from_parts`] and
/// readers go through the `&` accessors, so a finished proof is immutable.
#[derive(Debug, Clone)]
pub struct ONNXProof<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Commitment data shared by every witness polynomial (empty for homomorphic schemes).
    batch_commitment: PCS::BatchCommitment,
    /// Polynomial commitments for witness polynomials.
    commitments: Vec<PCS::Commitment>,
    /// The joint opening proof of the committed polynomials at the opening reduction's point.
    reduced_opening_proof: Option<ReducedOpeningProof<F, PCS>>,
    /// The transcript's NARG string: the opening claims, the round messages, the reduction
    /// polynomials and the proof-of-work nonces, in the order the verifier's transcript reads
    /// them back; it rejects a string it does not consume exactly.
    narg: Vec<u8>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> ONNXProof<F, PCS> {
    /// Assemble a proof from every field.
    pub fn from_parts(
        batch_commitment: PCS::BatchCommitment,
        commitments: Vec<PCS::Commitment>,
        reduced_opening_proof: Option<ReducedOpeningProof<F, PCS>>,
        narg: Vec<u8>,
    ) -> Self {
        Self {
            batch_commitment,
            commitments,
            reduced_opening_proof,
            narg,
        }
    }

    /// Commitment data shared by every witness polynomial.
    pub fn batch_commitment(&self) -> &PCS::BatchCommitment {
        &self.batch_commitment
    }

    /// Witness polynomial commitments, in sorted `CommittedPoly` order.
    pub fn commitments(&self) -> &[PCS::Commitment] {
        &self.commitments
    }

    /// The batched opening proof, absent when nothing committed was opened.
    pub fn reduced_opening_proof(&self) -> Option<&ReducedOpeningProof<F, PCS>> {
        self.reduced_opening_proof.as_ref()
    }

    /// The transcript's NARG string.
    pub fn narg(&self) -> &[u8] {
        &self.narg
    }
}

/// Mutable field access for the soundness harness (`test-util`).
#[cfg(feature = "test-util")]
impl<F: JoltField, PCS: CommitmentScheme<Field = F>> ONNXProof<F, PCS> {
    /// Mutable batch commitment.
    pub fn batch_commitment_mut(&mut self) -> &mut PCS::BatchCommitment {
        &mut self.batch_commitment
    }

    /// Mutable commitments.
    pub fn commitments_mut(&mut self) -> &mut Vec<PCS::Commitment> {
        &mut self.commitments
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

/// The joint opening proof of the committed polynomials. The opening reduction's sumcheck and
/// its reduced claims are in the NARG string.
#[derive(Debug, Clone)]
pub struct ReducedOpeningProof<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Joint opening proof for the batched polynomial.
    joint_opening_proof: PCS::Proof,
    _field: std::marker::PhantomData<F>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> ReducedOpeningProof<F, PCS> {
    /// Assemble a reduced opening proof from every field.
    pub fn from_parts(joint_opening_proof: PCS::Proof) -> Self {
        Self {
            joint_opening_proof,
            _field: std::marker::PhantomData,
        }
    }

    /// The PCS opening proof of the batched polynomial.
    pub fn joint_opening_proof(&self) -> &PCS::Proof {
        &self.joint_opening_proof
    }
}

/// Mutable field access for the soundness harness (`test-util`).
#[cfg(feature = "test-util")]
impl<F: JoltField, PCS: CommitmentScheme<Field = F>> ReducedOpeningProof<F, PCS> {
    /// Mutable PCS opening proof.
    pub fn joint_opening_proof_mut(&mut self) -> &mut PCS::Proof {
        &mut self.joint_opening_proof
    }
}
