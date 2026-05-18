//! Preprocessing for ONNX proof generation and verification.
//!
//! Preprocessing converts an ONNX [`Model`] into the data structures needed by
//! the prover and verifier — most importantly, a polynomial commitment scheme
//! setup (SRS / verification key).
//!
//! Three structs form a hierarchy:
//!
//! 1. [`AtlasSharedPreprocessing`] — model-only, shared by both parties.
//! 2. [`AtlasProverPreprocessing`]  — adds the PCS prover setup (SRS).
//! 3. [`AtlasVerifierPreprocessing`] — adds the PCS verifier setup (VK).

use atlas_onnx_tracer::model::Model;
use common::CommittedPoly;
use joltworks::{
    field::JoltField, poly::commitment::commitment_scheme::CommitmentScheme,
    transcripts::Transcript,
};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

/// Shared preprocessing data for both prover and verifier.
///
/// Contains the ONNX model structure that is used by both parties.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

    /// Get all committed polynomials required by the model's operations.
    pub fn get_models_committed_polynomials<F: JoltField, T: Transcript>(
        &self,
    ) -> Vec<CommittedPoly> {
        use crate::onnx_proof::ops::NodeCommittedPolynomials;
        self.model
            .graph
            .nodes
            .values()
            .flat_map(|node| NodeCommittedPolynomials::get_committed_polynomials::<F, T>(node))
            .collect()
    }

    /// Get the model
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Get the model's scale.
    pub fn scale(&self) -> i32 {
        self.model().scale
    }
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Prover-specific preprocessing with commitment scheme generators.
///
/// Contains the polynomial commitment scheme setup for the prover.
#[derive(Clone)]
pub struct AtlasProverPreprocessing<F: JoltField, PCS: CommitmentScheme<Field = F>> {
    /// Polynomial commitment scheme prover setup (SRS generators).
    pub generators: PCS::ProverSetup,
    /// Shared preprocessing data.
    pub shared: AtlasSharedPreprocessing,
}

impl<F, PCS> AtlasProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// Create new prover preprocessing from shared preprocessing.
    ///
    /// Generates the polynomial commitment scheme setup (SRS) based on the
    /// maximum number of variables in the model.
    #[tracing::instrument(skip_all, name = "AtlasProverPreprocessing::gen")]
    pub fn new(shared: AtlasSharedPreprocessing) -> AtlasProverPreprocessing<F, PCS> {
        let model = &shared.model;
        let max_num_vars = model.max_num_vars();
        tracing::info!("Prover preprocessing: max_num_vars = {max_num_vars}");
        let generators = PCS::setup_prover(max_num_vars);
        AtlasProverPreprocessing { generators, shared }
    }

    /// Get a reference to the ONNX model.
    pub fn model(&self) -> &Model {
        &self.shared.model
    }

    /// Get the models scale
    pub fn scale(&self) -> i32 {
        self.model().scale
    }
}

#[cfg(feature = "zk")]
impl
    AtlasProverPreprocessing<
        ark_bn254::Fr,
        joltworks::poly::commitment::hyperkzg::HyperKZG<ark_bn254::Bn254>,
    >
{
    /// Derive Pedersen generators from the HyperKZG SRS for BlindFold ZK proofs.
    pub fn pedersen_generators(
        &self,
        count: usize,
    ) -> joltworks::poly::commitment::pedersen::PedersenGenerators<joltworks::curve::Bn254Curve>
    {
        self.generators.pedersen_generators(count)
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
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> From<&AtlasProverPreprocessing<F, PCS>>
    for AtlasVerifierPreprocessing<F, PCS>
{
    fn from(prover_preprocessing: &AtlasProverPreprocessing<F, PCS>) -> Self {
        let generators = PCS::setup_verifier(&prover_preprocessing.generators);
        Self {
            generators,
            shared: prover_preprocessing.shared.clone(),
        }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> AtlasVerifierPreprocessing<F, PCS> {
    /// Get a reference to the ONNX model.
    pub fn model(&self) -> &Model {
        &self.shared.model
    }

    /// Get the model's scale.
    pub fn scale(&self) -> i32 {
        self.model().scale
    }
}
