//! The prover side of [`super`]: a [`RelationWitness`] computes its own round messages, and the
//! relation supplies everything else the batched sumcheck asks of an instance (degree, input
//! claim, where the produced openings sit).

use super::*;
use crate::{
    opening_proof::prover::ProverOpeningAccumulator,
    sumcheck::{absorb_input_claims, batch_prelude, round_pow_bits, SumcheckInstanceProof},
    upstream_sumcheck::{
        compressed_from_upstream, expect_engine, FacadeTranscript, InstanceRounds,
    },
};
use atlas_algebra::field::MaybeAllocative;
use jolt_sumcheck::ProveRounds;
use jolt_sumcheck::{prove_batch, BatchPrelude, ClearSumcheckRecorder, SumcheckRecorder};

/// Append the relation's produced openings, with their `claims`, at their points.
pub fn append_produced<F: JoltField, T: Transcript, R: Relation<F>>(
    relation: &R,
    setup: &RelationSetup<F>,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    sumcheck_challenges: &[F::Challenge],
    claims: &[F],
) {
    let points = relation.opening_points(setup, &sumcheck_challenges.to_vec().into_opening());
    if let Some(groups) = relation.sparse_groups() {
        let mut claims = claims.iter().copied();
        for (group, point) in group_points(groups, points) {
            let (address, cycle) = point.split_at(group.address_vars);
            let group_claims = claims.by_ref().take(group.polynomials.len()).collect();
            accumulator.append_sparse(
                transcript,
                group.polynomials,
                group.sumcheck,
                address.to_vec(),
                cycle.to_vec(),
                group_claims,
            );
        }
        return;
    }
    for ((&id, point), &claim) in relation.produced().iter().zip(points).zip(claims) {
        match id.polynomial {
            PolynomialId::Virtual(_) => accumulator.append_virtual(
                transcript,
                id,
                OpeningPoint::<BIG_ENDIAN, F>::new(point),
                claim,
            ),
            PolynomialId::Committed(_) => accumulator.append_dense(transcript, id, point, claim),
        }
    }
}

/// A prover's witness for a relation: its own round messages over the relation's hypercube,
/// and the produced openings' claims once every round is bound.
pub trait RelationWitness<F: JoltField>: ProveRounds<F> + Send + Sync + MaybeAllocative {
    /// The relation proven.
    type Relation: Relation<F>;

    /// The relation.
    fn relation(&self) -> &Self::Relation;

    /// The relation's setup.
    fn setup(&self) -> &RelationSetup<F>;

    /// The produced openings' claims at the bound point, aligned with [`Produces::produced`].
    fn produced_claims(&self) -> Vec<F>;

    /// Append the produced openings with their claims, mirroring
    /// [`Relation::append_produced_requests`]. Defaults to [`append_produced`].
    fn append_claims<T: Transcript>(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        append_produced(
            self.relation(),
            self.setup(),
            accumulator,
            transcript,
            sumcheck_challenges,
            &self.produced_claims(),
        );
    }
}

/// A relation witness as a member of a batch, its relation type erased. Every
/// [`RelationWitness`] is one.
pub trait DynRelationWitness<F: JoltField, T: Transcript>:
    ProveRounds<F> + Send + Sync + MaybeAllocative
{
    /// The degree of the relation's round polynomials.
    fn degree(&self) -> usize;

    /// The relation's input claim, from the accumulator's openings and the setup.
    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F;

    /// Append the produced openings with their claims ([`RelationWitness::append_claims`]).
    fn append_produced(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    );
}

impl<F: JoltField, T: Transcript, W: RelationWitness<F>> DynRelationWitness<F, T> for W {
    fn degree(&self) -> usize {
        self.relation().degree()
    }

    fn input_claim(&self, accumulator: &ProverOpeningAccumulator<F>) -> F {
        relation_input_claim(self.relation(), self.setup(), accumulator)
    }

    fn append_produced(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        self.append_claims(accumulator, transcript, sumcheck_challenges);
    }
}

/// Prove a batch of relations as one front-loaded batched sumcheck, the prover side of
/// [`super::verify_relation_batch`]. Returns the proof and the batch's challenges.
#[tracing::instrument(skip_all, name = "prove_relation_batch")]
pub fn prove_relation_batch<F: JoltField, T: Transcript>(
    witnesses: Vec<&mut dyn DynRelationWitness<F, T>>,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> (SumcheckInstanceProof<F>, Vec<F::Challenge>) {
    prove_batch_inner(witnesses, accumulator, transcript, false)
}

/// Like [`prove_relation_batch`], but computes each round's per-witness messages *and* ingests
/// each challenge in parallel across witnesses (nested with each witness's own parallelism).
///
/// Only for batches of mutually independent witnesses: witnesses that share state behind an
/// `RwLock` (the opening-reduction groups' shared eq tables) deadlock when a rayon thread holding
/// a write guard steals a job that takes the same lock. Use [`prove_relation_batch`] for those.
#[tracing::instrument(skip_all, name = "prove_relation_batch_parallel")]
pub fn prove_relation_batch_parallel<F: JoltField, T: Transcript>(
    witnesses: Vec<&mut dyn DynRelationWitness<F, T>>,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> (SumcheckInstanceProof<F>, Vec<F::Challenge>) {
    prove_batch_inner(witnesses, accumulator, transcript, true)
}

fn prove_batch_inner<F: JoltField, T: Transcript>(
    mut witnesses: Vec<&mut dyn DynRelationWitness<F, T>>,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    parallel_witnesses: bool,
) -> (SumcheckInstanceProof<F>, Vec<F::Challenge>) {
    let max_rounds = witnesses.iter().map(|w| w.num_rounds()).max().unwrap();
    let max_degree = witnesses.iter().map(|w| w.degree()).max().unwrap();
    let input_claims: Vec<F> = witnesses
        .iter()
        .map(|w| w.input_claim(accumulator))
        .collect();

    let mut recorder = ClearSumcheckRecorder::<F>::new();
    absorb_input_claims(&mut recorder, &input_claims, transcript);
    let coefficients: Vec<F> = transcript.challenge_vector(witnesses.len());
    let rounds: Vec<usize> = witnesses.iter().map(|w| w.num_rounds()).collect();
    let prelude = batch_prelude(
        &rounds,
        &input_claims,
        &coefficients,
        max_rounds,
        max_degree,
    );

    // Spread per-round message computation across witnesses rather than inside each one when
    // asked to and worthwhile.
    const MANY_WITNESSES: usize = 8;
    let many = parallel_witnesses && witnesses.len() >= MANY_WITNESSES;
    let (proof, r_sumcheck) = {
        let mut members: Vec<&mut dyn ProveRounds<F>> = witnesses
            .iter_mut()
            .map(|w| &mut **w as &mut dyn ProveRounds<F>)
            .collect();
        run_engine(
            &prelude,
            &mut members,
            many,
            max_degree,
            recorder,
            transcript,
        )
    };

    for w in witnesses.iter() {
        // A witness with fewer rounds than the batch is bound in its last rounds, so it uses
        // that suffix of the challenges.
        let r_slice = &r_sumcheck[max_rounds - w.num_rounds()..];
        w.append_produced(accumulator, transcript, r_slice);
    }
    (proof, r_sumcheck)
}

/// Prove one relation as its own sumcheck, the prover side of [`super::verify_relation`]: a
/// batch of one with coefficient one and no coefficient drawn.
#[tracing::instrument(skip_all, name = "prove_relation")]
pub fn prove_relation<F: JoltField, T: Transcript>(
    witness: &mut dyn DynRelationWitness<F, T>,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
) -> (SumcheckInstanceProof<F>, Vec<F::Challenge>) {
    let rounds = witness.num_rounds();
    let degree = witness.degree();
    let input_claim = witness.input_claim(accumulator);
    let mut recorder = ClearSumcheckRecorder::<F>::new();
    absorb_input_claims(&mut recorder, &[input_claim], transcript);
    let prelude = batch_prelude(&[rounds], &[input_claim], &[F::one()], rounds, degree);

    let (proof, r_sumcheck) = {
        let mut members: Vec<&mut dyn ProveRounds<F>> =
            vec![&mut *witness as &mut dyn ProveRounds<F>];
        run_engine(&prelude, &mut members, false, degree, recorder, transcript)
    };
    witness.append_produced(accumulator, transcript, &r_sumcheck);
    (proof, r_sumcheck)
}

/// Run upstream's batch prover over `members` and record the round polynomials.
fn run_engine<F: JoltField, T: Transcript>(
    prelude: &BatchPrelude<F>,
    members: &mut [&mut dyn ProveRounds<F>],
    many_members: bool,
    max_degree: usize,
    mut recorder: ClearSumcheckRecorder<F>,
    transcript: &mut T,
) -> (SumcheckInstanceProof<F>, Vec<F::Challenge>) {
    let mut scheduler = InstanceRounds::new(many_members);
    let ((proved, recorded), pow) =
        FacadeTranscript::<T, F>::scope(transcript, round_pow_bits::<F>(max_degree), |facade| {
            let proved = expect_engine(prove_batch(
                prelude,
                members,
                &mut scheduler,
                &mut recorder,
                facade,
            ));
            // Witnesses append their own openings; no output values here.
            let recorded = expect_engine(recorder.finish(&[], facade));
            (proved, recorded)
        });
    pow.expect("proof-of-work nonce search");
    let proof = SumcheckInstanceProof::new(
        recorded
            .proof
            .round_polynomials
            .iter()
            .map(compressed_from_upstream)
            .collect(),
    );
    let r_sumcheck = proved
        .challenges
        .into_iter()
        .map(F::challenge_from_field)
        .collect();
    (proof, r_sumcheck)
}
