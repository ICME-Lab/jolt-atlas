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
use atlas_algebra::field::JoltField;
use atlas_pcs::commitment_scheme::CommitmentScheme;
use atlas_trace::Model;
use common::CommittedPoly;
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
    pub fn get_models_committed_polynomials(&self) -> Vec<CommittedPoly> {
        use crate::onnx_proof::ops::NodeCommittedPolynomials;
        let mut polys: Vec<CommittedPoly> = self
            .model
            .views()
            .iter()
            .flat_map(NodeCommittedPolynomials::get_committed_polynomials)
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
