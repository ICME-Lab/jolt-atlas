//! Preprocessing for ONNX proof generation and verification.
//!
//! Preprocessing converts an ONNX [`Model`] into the data structures needed by
//! the prover and verifier: a polynomial commitment scheme setup (SRS / verification key) and
//! the model's verification [`Schedule`].
//!
//! Three structs form a hierarchy:
//!
//! 1. [`AtlasSharedPreprocessing`] — the model alone, shared by both parties and serializable.
//! 2. `AtlasProverPreprocessing` (in `jolt-atlas-core`) — adds the PCS prover setup (SRS) and
//!    the schedule.
//! 3. [`AtlasVerifierPreprocessing`] — adds the PCS verifier setup (VK) and the schedule.
//!
//! The schedule is compiled from the model when the prover's or the verifier's preprocessing
//! is constructed, and is never serialized.

use crate::node_view::ModelView;
use crate::onnx_proof::schedule::Schedule;
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_pcs::commitment_scheme::CommitmentScheme;
use atlas_trace::Model;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

/// Shared preprocessing data for both prover and verifier.
///
/// Contains the ONNX model structure that is used by both parties.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtlasSharedPreprocessing {
    /// The ONNX neural network model.
    pub model: Model,
}

impl AtlasSharedPreprocessing {
    /// Preprocess an ONNX model for proving and verification.
    #[tracing::instrument(skip_all, name = "AtlasSharedPreprocessing::preprocess")]
    pub fn preprocess(model: Model) -> Self {
        Self { model }
    }

    /// Get the model
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Get the model's scale.
    pub fn scale(&self) -> u8 {
        self.model().scale_log2()
    }
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verifier-specific preprocessing with commitment scheme verification key.
///
/// Contains the polynomial commitment scheme setup for the verifier.
#[derive(Debug, Clone)]
pub struct AtlasVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// Polynomial commitment scheme verifier setup (verification key).
    pub generators: PCS::VerifierSetup,
    /// Shared preprocessing data.
    pub shared: AtlasSharedPreprocessing,
    /// The model's verification schedule: what `verify` checks, in order.
    pub schedule: Schedule<F>,
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> AtlasVerifierPreprocessing<F, PCS> {
    /// The verifier preprocessing of `shared`'s model under the verification key `generators`,
    /// with the model's schedule compiled. `Err` if a node does not have the shape its operator
    /// proves.
    pub fn new(
        generators: PCS::VerifierSetup,
        shared: AtlasSharedPreprocessing,
    ) -> Result<Self, ProofVerifyError> {
        let schedule = Schedule::compile(shared.model())?;
        Ok(Self {
            generators,
            shared,
            schedule,
        })
    }

    /// Get a reference to the ONNX model.
    pub fn model(&self) -> &Model {
        &self.shared.model
    }

    /// Get the model's scale.
    pub fn scale(&self) -> u8 {
        self.model().scale_log2()
    }
}
