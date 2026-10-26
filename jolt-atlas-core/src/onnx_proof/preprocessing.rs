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

use crate::node_view::ModelView;
use atlas_trace::Model;
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtlasSharedPreprocessing {
    /// The ONNX neural network model.
    pub model: Model,
}

impl AtlasSharedPreprocessing {
    /// Preprocess an ONNX model for proving and verification.
    ///
    /// Panics on models the proof layer cannot prove yet, so an unprovable
    /// model fails here — before commitments and witness generation — with
    /// the node named, instead of inside a prover thread mid-IOP.
    #[tracing::instrument(skip_all, name = "AtlasSharedPreprocessing::preprocess")]
    pub fn preprocess(model: Model) -> Self {
        for (id, node, _) in model.nodes() {
            if let atlas_trace::Op::SoftmaxLastAxis(s) = &node.op {
                let last_dim = *node.output_dims.last().expect("softmax rank >= 1");
                assert!(
                    s.active_len.is_none_or(|n| n == last_dim),
                    "node {id:?}: masked softmax (active_len {:?} of {last_dim}) is not yet \
                     provable; use a power-of-two softmax axis",
                    s.active_len
                );
            }
        }
        Self { model }
    }

    /// Get all committed polynomials required by the model's operations.
    pub fn get_models_committed_polynomials<F: JoltField, T: Transcript>(
        &self,
    ) -> Vec<CommittedPoly> {
        use crate::onnx_proof::ops::NodeCommittedPolynomials;
        let mut polys: Vec<CommittedPoly> = self
            .model
            .views()
            .iter()
            .flat_map(|node| NodeCommittedPolynomials::get_committed_polynomials::<F, T>(node))
            .collect();
        polys.extend(crate::onnx_proof::global_clamp::bucket_committed_polys(
            &self.model,
        ));
        polys
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
    /// Create new prover preprocessing from shared preprocessing: the
    /// commitment scheme's setup for the model's maximum number of variables.
    #[tracing::instrument(skip_all, name = "AtlasProverPreprocessing::gen")]
    pub fn new(shared: AtlasSharedPreprocessing) -> AtlasProverPreprocessing<F, PCS> {
        let max_num_vars = shared.model.max_num_vars();
        tracing::info!("Prover preprocessing: max_num_vars = {max_num_vars}");
        let generators = PCS::setup_prover(max_num_vars);
        AtlasProverPreprocessing { generators, shared }
    }

    /// Get a reference to the ONNX model.
    pub fn model(&self) -> &Model {
        &self.shared.model
    }

    /// Get the models scale
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
    pub fn scale(&self) -> u8 {
        self.model().scale_log2()
    }
}
