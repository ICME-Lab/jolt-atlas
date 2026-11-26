//! The fold round: send the column folds, then take the challenge.

use crate::BitZProver;
use common::{
    BitTable, BitzClaimField, Fold, FoldError, LinearClaim, SecurityLevel, column_images,
    fold_columns, row_images,
};
use field::F128;
use num_traits::ToBytes;
use transcript::ProverState;

/// A fold the prover cannot produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendError {
    /// The witness is not shaped the way the configuration says it is.
    ShapeMismatch,
    /// The round's parts disagree about the shape.
    Fold(FoldError),
}

impl<F: BitzClaimField> BitZProver<F> {
    /// Runs the fold round.
    ///
    /// Only the folds `eta_j` are sent. Their images `g^{eta_j}` are what the
    /// grand product consumes, but the verifier derives them for itself, so
    /// transmitting both would be sending the same information twice: at
    /// `k_2 = 2^21` that second copy is 32 MiB.
    ///
    /// The folds go out on the narg channel, which writes and absorbs the same
    /// bytes, so the challenge below depends on them. That binds the images just
    /// as tightly, because `u -> g^u` is injective over the range the verifier
    /// admits — which is exactly what the range check and the full order of `g`
    /// establish.
    ///
    /// Each fold is one record. The count is not itself absorbed, which is safe
    /// only because it derives from the configured shape; the wire profile is
    /// where a length-delimited vector record would belong.
    #[tracing::instrument(name = "Fold columns", skip_all)]
    pub fn send_fold(
        &self,
        claim: &LinearClaim<F>,
        table: &BitTable<'_>,
        transcript: &mut ProverState,
        security: SecurityLevel,
    ) -> Result<Fold<F::Integer>, SendError> {
        let (folds, row_images, zeta) = self.send_row_fold(claim, table, transcript, security)?;
        let images = column_images(self.comb(), &folds);
        Fold::new(table.shape(), folds, images, row_images, zeta).map_err(SendError::Fold)
    }

    #[tracing::instrument(name = "Fold columns for GKR", skip_all)]
    pub(crate) fn send_row_fold(
        &self,
        claim: &LinearClaim<F>,
        table: &BitTable<'_>,
        transcript: &mut ProverState,
        security: SecurityLevel,
    ) -> Result<(Vec<F::Integer>, Vec<F128>, Vec<F128>), SendError> {
        // The weights are sized by the configured row count while the bits are
        // read at the table's. Disagreement is a panic, a silently wrong fold, or
        // a desynchronised transcript depending on which way it goes.
        if table.shape() != self.params().shape() {
            return Err(SendError::ShapeMismatch);
        }
        let shape = self.params().shape();

        // Lifted once: the fold reads it per set bit across every column.
        let exponents = claim.row_exponents();
        let folds = fold_columns(table, &exponents);

        for fold in &folds {
            transcript.prover_message(&fold.to_le_bytes().as_ref());
        }

        let row_images = row_images(self.comb(), &exponents);
        transcript.grind(
            b"bitz/columns/v1",
            security.grinding_bits(shape.log_columns()),
        );
        let zeta = (0..shape.log_columns())
            .map(|_| transcript.verifier_message())
            .collect();

        Ok((folds, row_images, zeta))
    }
}
