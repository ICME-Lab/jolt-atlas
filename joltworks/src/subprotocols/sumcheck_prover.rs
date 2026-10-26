use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::transcripts::Transcript;
use jolt_sumcheck::ProveRounds;

use crate::{
    field::{JoltField, MaybeAllocative},
    poly::opening_proof::ProverOpeningAccumulator,
};

pub trait SumcheckInstanceProver<F: JoltField, T: Transcript>:
    Send + Sync + MaybeAllocative
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        unimplemented!(
            "If get_params is unimplemented, degree, num_rounds, and \
            input_claim should be implemented directly"
        )
    }

    /// Returns the maximum degree of the sumcheck polynomial.
    fn degree(&self) -> usize {
        self.get_params().degree()
    }

    /// Returns the number of rounds/variables in this sumcheck instance.
    fn num_rounds(&self) -> usize {
        self.get_params().num_rounds()
    }

    /// Returns the initial claim of this sumcheck instance.
    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        self.get_params().input_claim(accumulator)
    }

    /// Computes the round offset for this instance within a batched sumcheck.
    /// Instances with fewer rounds start later in the batched protocol.
    fn round_offset(&self, max_num_rounds: usize) -> usize {
        max_num_rounds - self.num_rounds()
    }

    /// The instance's rounds as an upstream batch member, for an instance that implements
    /// [`ProveRounds`] itself. Such an instance leaves [`Self::compute_message`],
    /// [`Self::ingest_challenge`] and [`Self::finalize`] unimplemented.
    fn rounds(&mut self) -> Option<&mut dyn ProveRounds<F>> {
        None
    }

    /// Computes the prover's message for a specific round of the sumcheck protocol.
    fn compute_message(&mut self, _round: usize, _previous_claim: F) -> UniPoly<F> {
        unimplemented!("an instance without compute_message runs its rounds through rounds()")
    }

    /// Ingest the verifier's challenge for a sumcheck round.
    fn ingest_challenge(&mut self, _r_j: F::Challenge, _round: usize) {
        unimplemented!("an instance without ingest_challenge runs its rounds through rounds()")
    }

    /// Finalize prover state after the last challenge has been ingested, but before
    /// [`Self::cache_openings`] is called.
    ///
    /// This hook is useful for sumcheck implementations that need to perform
    /// end-of-protocol work (e.g., flushing delayed/streaming bindings or releasing
    /// intermediate state) once all challenges are known.
    ///
    /// Default implementation is a no-op.
    fn finalize(&mut self) {}

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
