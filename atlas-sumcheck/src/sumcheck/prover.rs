//! The prover side of [`super`].

#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use super::*;
use crate::sumcheck_verifier::SumcheckInstanceParams;
use atlas_algebra::field::MaybeAllocative;

use crate::{
    opening_proof::prover::ProverOpeningAccumulator,
    upstream_sumcheck::{
        compressed_from_upstream, expect_engine, FacadeTranscript, InstanceRounds,
    },
};
use atlas_algebra::field::JoltField;
use atlas_transcript::Transcript;
use jolt_sumcheck::{prove_batch, ClearSumcheckRecorder, ProveRounds, SumcheckRecorder};

impl BatchedSumcheck {
    #[tracing::instrument(skip_all, name = "BatchedSumcheck::prove")]
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F>, Vec<F::Challenge>) {
        Self::prove_inner(sumcheck_instances, opening_accumulator, transcript, false)
    }

    /// Like [`Self::prove`], but computes each round's per-instance messages
    /// *and* ingests each challenge in parallel across instances (nested with
    /// each instance's own parallelism).
    ///
    /// Only for batches of mutually independent instances: instances that
    /// share state behind an `RwLock` (e.g. the opening-reduction instances'
    /// shared eq tables) deadlock when a rayon thread holding a write guard
    /// steals a job that takes the same lock. Use [`Self::prove`] for those.
    #[tracing::instrument(skip_all, name = "BatchedSumcheck::prove_parallel_instances")]
    pub fn prove_parallel_instances<F: JoltField, ProofTranscript: Transcript>(
        sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F>, Vec<F::Challenge>) {
        Self::prove_inner(sumcheck_instances, opening_accumulator, transcript, true)
    }

    fn prove_inner<F: JoltField, ProofTranscript: Transcript>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
        parallel_instances: bool,
    ) -> (SumcheckInstanceProof<F>, Vec<F::Challenge>) {
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();
        let max_degree = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.degree())
            .max()
            .unwrap();
        let input_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.input_claim(opening_accumulator))
            .collect();

        let mut recorder = ClearSumcheckRecorder::<F>::new();
        absorb_input_claims::<F, _>(&mut recorder, &input_claims, transcript);
        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());
        let prelude = batch_prelude(
            &sumcheck_instances
                .iter()
                .map(|sumcheck| sumcheck.num_rounds())
                .collect::<Vec<_>>(),
            &input_claims,
            &batching_coeffs,
            max_num_rounds,
            max_degree,
        );

        // Spread per-round message computation across instances rather than
        // inside each one when asked to and worthwhile.
        const MANY_INSTANCES: usize = 8;
        let many_instances = parallel_instances && sumcheck_instances.len() >= MANY_INSTANCES;
        let challenges = {
            let mut member_refs: Vec<&mut dyn ProveRounds<F>> = sumcheck_instances
                .iter_mut()
                .map(|sumcheck| &mut **sumcheck as &mut dyn ProveRounds<F>)
                .collect();
            let mut scheduler = InstanceRounds::new(many_instances);
            let ((proved, recorded), pow) = FacadeTranscript::<ProofTranscript, F>::scope(
                transcript,
                round_pow_bits::<F>(max_degree),
                |facade| {
                    let proved = expect_engine(prove_batch(
                        &prelude,
                        &mut member_refs,
                        &mut scheduler,
                        &mut recorder,
                        facade,
                    ));
                    // Instances cache their own opening claims (below); no output values here.
                    let recorded = expect_engine(recorder.finish(&[], facade));
                    (proved, recorded)
                },
            );
            pow.expect("proof-of-work nonce search");
            let proof = SumcheckInstanceProof::new(
                recorded
                    .proof
                    .round_polynomials
                    .iter()
                    .map(compressed_from_upstream)
                    .collect(),
            );
            (proof, proved.challenges)
        };
        let (proof, challenges) = challenges;
        let r_sumcheck: Vec<F::Challenge> = challenges
            .into_iter()
            .map(F::challenge_from_field)
            .collect();

        for sumcheck in sumcheck_instances.iter() {
            // A member with fewer than `max_num_rounds` rounds is bound in the
            // last `sumcheck.num_rounds()` rounds, so it uses that suffix of
            // `r_sumcheck`.
            let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];

            // Cache polynomial opening claims, to be proven using either an
            // opening proof or sumcheck (in the case of virtual polynomials).
            sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
        }

        (proof, r_sumcheck)
    }
}

impl Sumcheck {
    /// One instance on its own: a batch of one with batching coefficient one.
    #[tracing::instrument(skip_all, name = "Sumcheck::prove")]
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        sumcheck_instance: &mut dyn SumcheckInstanceProver<F, ProofTranscript>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F>, Vec<F::Challenge>) {
        let num_rounds = sumcheck_instance.num_rounds();
        let degree = sumcheck_instance.degree();
        let input_claim = sumcheck_instance.input_claim(opening_accumulator);
        let mut recorder = ClearSumcheckRecorder::<F>::new();
        absorb_input_claims::<F, _>(&mut recorder, &[input_claim], transcript);
        let prelude = batch_prelude(
            &[num_rounds],
            &[input_claim],
            &[F::one()],
            num_rounds,
            degree,
        );

        let (proof, challenges) = {
            let mut member_refs: Vec<&mut dyn ProveRounds<F>> =
                vec![&mut *sumcheck_instance as &mut dyn ProveRounds<F>];
            let ((proved, recorded), pow) = FacadeTranscript::<ProofTranscript, F>::scope(
                transcript,
                round_pow_bits::<F>(degree),
                |facade| {
                    let proved = expect_engine(prove_batch(
                        &prelude,
                        &mut member_refs,
                        &mut InstanceRounds::new(false),
                        &mut recorder,
                        facade,
                    ));
                    let recorded = expect_engine(recorder.finish(&[], facade));
                    (proved, recorded)
                },
            );
            pow.expect("proof-of-work nonce search");
            let proof = SumcheckInstanceProof::new(
                recorded
                    .proof
                    .round_polynomials
                    .iter()
                    .map(compressed_from_upstream)
                    .collect(),
            );
            (proof, proved.challenges)
        };
        let r_sumcheck: Vec<F::Challenge> = challenges
            .into_iter()
            .map(F::challenge_from_field)
            .collect();
        sumcheck_instance.cache_openings(opening_accumulator, transcript, &r_sumcheck);
        (proof, r_sumcheck)
    }
}

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
