//! Prover debug state compared against the verifier in tests.

use atlas_algebra::field::JoltField;
use atlas_sumcheck::opening_proof::prover::ProverOpeningAccumulator;

// ---------------------------------------------------------------------------
// ProverDebugInfo
// ---------------------------------------------------------------------------

/// Debug information from the prover for testing and verification.
///
/// Contains the transcript's state history and the opening accumulator for
/// comparing prover and verifier execution in tests.
#[allow(dead_code)]
pub struct ProverDebugInfo<F: JoltField> {
    pub(crate) transcript_history: Vec<[u8; 32]>,
    pub(crate) opening_accumulator: ProverOpeningAccumulator<F>,
}
