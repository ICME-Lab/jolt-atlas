//! Prover debug state compared against the verifier in tests.

use joltworks::{
    field::JoltField, poly::opening_proof::ProverOpeningAccumulator, transcripts::Transcript,
};

// ---------------------------------------------------------------------------
// ProverDebugInfo
// ---------------------------------------------------------------------------

/// Debug information from the prover for testing and verification.
///
/// Contains the transcript and opening accumulator state for comparing
/// prover and verifier execution in tests.
#[allow(dead_code)]
pub struct ProverDebugInfo<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    pub(crate) transcript: ProofTranscript,
    pub(crate) opening_accumulator: ProverOpeningAccumulator<F>,
}
