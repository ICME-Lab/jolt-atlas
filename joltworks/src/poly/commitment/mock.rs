use std::borrow::Borrow;
use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand_core::{CryptoRng, RngCore};

use crate::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
    transcripts::{AppendToTranscript, Transcript},
    utils::{errors::ProofVerifyError, small_scalar::SmallScalar},
};

use super::commitment_scheme::{CommitmentScheme, HidingCommitmentScheme};

#[derive(Clone)]
pub struct MockCommitScheme<F: JoltField> {
    _marker: PhantomData<F>,
}

#[derive(Default, Debug, PartialEq, Clone, CanonicalDeserialize, CanonicalSerialize)]
pub struct MockCommitment<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> AppendToTranscript for MockCommitment<F> {
    fn append_to_transcript<ProofTranscript: Transcript>(&self, transcript: &mut ProofTranscript) {
        transcript.append_message(b"mocker");
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Clone, Debug)]
pub struct MockProof<F: JoltField> {
    opening_point: Vec<F::Challenge>,
}

impl<F> CommitmentScheme for MockCommitScheme<F>
where
    F: JoltField,
{
    type Field = F;
    type ProverSetup = ();
    type VerifierSetup = ();
    type Commitment = MockCommitment<F>;
    type Proof = MockProof<F>;
    type BatchedProof = MockProof<F>;
    type OpeningProofHint = ();

    fn setup_prover(_num_vars: usize) -> Self::ProverSetup {}

    fn setup_verifier(_setup: &Self::ProverSetup) -> Self::VerifierSetup {}

    fn commit(
        _poly: &MultilinearPolynomial<Self::Field>,
        _setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        (MockCommitment::default(), ())
    }

    fn batch_commit<P>(
        polys: &[P],
        gens: &Self::ProverSetup,
    ) -> Vec<(Self::Commitment, Self::OpeningProofHint)>
    where
        P: Borrow<MultilinearPolynomial<Self::Field>>,
    {
        polys
            .iter()
            .map(|poly| (Self::commit(poly.borrow(), gens).0, ()))
            .collect()
    }

    fn combine_commitments<C: Borrow<Self::Commitment>>(
        _commitments: &[C],
        _coeffs: &[Self::Field],
    ) -> Self::Commitment {
        MockCommitment::default()
    }

    fn combine_hints(
        _hints: Vec<Self::OpeningProofHint>,
        _coeffs: &[Self::Field],
    ) -> Self::OpeningProofHint {
    }

    fn prove<ProofTranscript: Transcript>(
        _setup: &Self::ProverSetup,
        _poly: &MultilinearPolynomial<Self::Field>,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        _hint: Option<Self::OpeningProofHint>,
        _transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        MockProof {
            opening_point: opening_point.to_owned(),
        }
    }

    fn verify<ProofTranscript: Transcript>(
        proof: &Self::Proof,
        _setup: &Self::VerifierSetup,
        _transcript: &mut ProofTranscript,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        _opening: &Self::Field,
        _commitment: &Self::Commitment,
    ) -> Result<(), ProofVerifyError> {
        assert_eq!(proof.opening_point, opening_point);
        Ok(())
    }

    fn protocol_name() -> &'static [u8] {
        b"mock_commit"
    }
}

impl<F> super::commitment_scheme::StreamingCommitmentScheme for MockCommitScheme<F>
where
    F: JoltField,
{
    type ChunkState = ();

    fn process_chunk<T: SmallScalar>(_setup: &Self::ProverSetup, _chunk: &[T]) -> Self::ChunkState {
    }

    fn process_chunk_onehot(
        _setup: &Self::ProverSetup,
        _onehot_k: usize,
        _chunk: &[Option<usize>],
    ) -> Self::ChunkState {
    }

    fn aggregate_chunks(
        _setup: &Self::ProverSetup,
        _onehot_k: Option<usize>,
        _tier1_commitments: &[Self::ChunkState],
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        (MockCommitment::default(), ())
    }
}

/// Mock blinding factor for testing HidingCommitmentScheme
#[derive(Clone, Debug, Default)]
pub struct MockBlindingFactor<F: JoltField> {
    _marker: PhantomData<F>,
}

impl<F> HidingCommitmentScheme for MockCommitScheme<F>
where
    F: JoltField,
{
    type BlindingFactor = MockBlindingFactor<F>;

    fn commit_hiding(
        poly: &MultilinearPolynomial<Self::Field>,
        _blinding: &Self::BlindingFactor,
        setup: &Self::ProverSetup,
    ) -> (Self::Commitment, Self::OpeningProofHint) {
        // Mock: just delegate to regular commit
        Self::commit(poly, setup)
    }

    fn sample_blinding<R: RngCore + CryptoRng>(_rng: &mut R) -> Self::BlindingFactor {
        MockBlindingFactor {
            _marker: PhantomData,
        }
    }

    fn combine_blindings(
        _blindings: &[Self::BlindingFactor],
        _coeffs: &[Self::Field],
    ) -> Self::BlindingFactor {
        MockBlindingFactor {
            _marker: PhantomData,
        }
    }

    fn prove_hiding<ProofTranscript: Transcript>(
        setup: &Self::ProverSetup,
        poly: &MultilinearPolynomial<Self::Field>,
        _blinding: &Self::BlindingFactor,
        opening_point: &[<Self::Field as JoltField>::Challenge],
        hint: Option<Self::OpeningProofHint>,
        transcript: &mut ProofTranscript,
    ) -> Self::Proof {
        // Mock: delegate to regular prove
        Self::prove(setup, poly, opening_point, hint, transcript)
    }
}
