//! Prover preprocessing: the shared model preprocessing plus the commitment scheme's prover
//! setup. The shared and verifier halves live in `jolt-atlas-verifier`.

use atlas_algebra::field::JoltField;
use atlas_pcs::commitment_scheme::CommitmentScheme;
use atlas_trace::Model;
use jolt_atlas_verifier::node_view::ModelView;

pub use jolt_atlas_verifier::onnx_proof::preprocessing::*;

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
