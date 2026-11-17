use atlas_algebra::serialize::Canonical;
use std::{borrow::Borrow, fmt::Debug};

use atlas_algebra::{
    field::JoltField, poly::multilinear::MultilinearPolynomial, utils::errors::ProofVerifyError,
};
use atlas_transcript::{ProvingTranscript, VerifyingTranscript};

/// The result of committing a whole batch: the scheme-level batch commitment and one
/// `(commitment, hint)` per polynomial, in input order.
pub type BatchCommitOutput<PCS> = (
    <PCS as CommitmentScheme>::BatchCommitment,
    Vec<(
        <PCS as CommitmentScheme>::Commitment,
        <PCS as CommitmentScheme>::OpeningProofHint,
    )>,
);

/// A commitment to every polynomial of a proof at once, opened once: at one point, as a
/// linear combination of the polynomials.
pub trait CommitmentScheme: Clone + Sync + Send + 'static {
    type Field: JoltField + Sized;
    type ProverSetup: Clone + Sync + Send + Debug + Canonical;
    type VerifierSetup: Clone + Sync + Send + Debug + Canonical;
    /// What identifies one polynomial inside the batch commitment.
    type Commitment: Default + Debug + Sync + Send + PartialEq + Canonical + Clone;
    type Proof: Sync + Send + Canonical + Clone + Debug;
    /// What the prover keeps of a commitment to open it.
    type OpeningProofHint: Sync + Send + Clone + Debug + PartialEq;
    /// The commitment to the whole batch.
    type BatchCommitment: Default + Debug + Sync + Send + PartialEq + Canonical + Clone;

    /// Generates the prover setup. `max_num_vars` is the largest number of variables of any
    /// polynomial that will be committed with it.
    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup;

    /// Generates the verifier setup from the prover setup.
    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup;

    /// Commits to every polynomial of a proof: the batch commitment, plus one
    /// `(commitment, hint)` per polynomial in input order.
    fn batch_commit_all<U>(polys: &[U], setup: &Self::ProverSetup) -> BatchCommitOutput<Self>
    where
        U: Borrow<MultilinearPolynomial<Self::Field>> + Sync;

    /// Opens the linear combination `Σ_i γ_i · f_i` at `opening_point`, every `f_i` taken at
    /// the last `n_i` coordinates of the point and zero-padded above its own `2^{n_i}`
    /// values, given the polynomials' commit hints in the order of `polynomials` and `coeffs`.
    fn prove_rlc<T: ProvingTranscript>(
        setup: &Self::ProverSetup,
        polynomials: &std::collections::BTreeMap<
            common::CommittedPoly,
            MultilinearPolynomial<Self::Field>,
        >,
        coeffs: &[Self::Field],
        hints: Vec<Self::OpeningProofHint>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        transcript: &mut T,
    ) -> Self::Proof;

    /// Verifies the joint opening `Σ_i γ_i · f_i(r) = joint_claim` of [`Self::prove_rlc`],
    /// given the opened polynomials' commitments, coefficients and variable counts, all
    /// aligned, and the batch commitment.
    #[allow(clippy::too_many_arguments)]
    fn verify_rlc<T: VerifyingTranscript>(
        proof: &Self::Proof,
        setup: &Self::VerifierSetup,
        transcript: &mut T,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        joint_claim: &Self::Field,
        commitments: &[&Self::Commitment],
        coeffs: &[Self::Field],
        num_vars: &[usize],
        batch_commitment: &Self::BatchCommitment,
    ) -> Result<(), ProofVerifyError>;
}
