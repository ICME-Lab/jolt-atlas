use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::transcripts::Transcript;
use jolt_sumcheck::ProveRounds;

use crate::{
    field::{JoltField, MaybeAllocative},
    poly::opening_proof::ProverOpeningAccumulator,
};

/// A sumcheck instance on the prover side: its rounds run as an upstream batch member
/// ([`ProveRounds`], which also gives its number of rounds), and this trait carries the rest of
/// its lifecycle — degree, input claim, and the opening claims it caches once bound.
pub trait SumcheckInstanceProver<F: JoltField, T: Transcript>:
    ProveRounds<F> + Send + Sync + MaybeAllocative
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        unimplemented!(
            "If get_params is unimplemented, degree and input_claim should be \
            implemented directly"
        )
    }

    /// Returns the maximum degree of the sumcheck polynomial.
    fn degree(&self) -> usize {
        self.get_params().degree()
    }

    /// Returns the initial claim of this sumcheck instance.
    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.get_params().input_claim(accumulator)
    }

    /// Caches polynomial opening claims needed after the sumcheck protocol completes.
    /// These openings will later be proven using either an opening proof or another sumcheck.
    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    );

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder);
}
