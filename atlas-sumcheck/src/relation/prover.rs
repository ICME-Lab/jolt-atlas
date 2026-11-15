//! The prover side of [`super`]: a [`RelationWitness`] computes its own round messages, and the
//! relation supplies everything else the batched sumcheck asks of an instance (degree, input
//! claim, where the produced openings sit).

use super::*;
use crate::{
    engine::{
        absorb_input_claims,
        prover::{
            prove_batch, ClearSumcheckRecorder, MemberFinish, MemberRound, ProveRounds,
            RoundScheduler,
        },
        BatchMember, BatchPrelude, SumcheckError,
    },
    opening_proof::prover::ProverOpeningAccumulator,
    sumcheck::SumcheckInstanceProof,
};
#[cfg(not(target_arch = "wasm32"))]
use atlas_algebra::utils::profiling::print_current_memory_usage;
use atlas_algebra::{field::MaybeAllocative, poly::univariate::CompressedUniPoly};
use rayon::prelude::*;

/// Append the relation's produced openings, with their `claims`, at their points.
pub fn append_produced<F: JoltField, T: Transcript, R: Relation<F>>(
    relation: &R,
    setup: &RelationSetup<F>,
    accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut T,
    sumcheck_challenges: &[F::Challenge],
    claims: &[F],
) {
    let points = relation
        .opening_points(setup, &sumcheck_challenges.to_vec().into_opening())
        .expect("the prover's relation places its own openings");
    if let Some(groups) = relation.sparse_groups() {
        let mut claims = claims.iter().copied();
        let groups = group_points(groups, points).expect("a point per produced opening");
        for (group, point) in groups {
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
            .expect("the prover holds every claim its relation reads")
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

    absorb_input_claims(&input_claims, transcript);
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
        run_engine(&prelude, &mut members, many, transcript)
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
    absorb_input_claims(&[input_claim], transcript);
    let prelude = batch_prelude(&[rounds], &[input_claim], &[F::one()], rounds, degree);

    let (proof, r_sumcheck) = {
        let mut members: Vec<&mut dyn ProveRounds<F>> =
            vec![&mut *witness as &mut dyn ProveRounds<F>];
        run_engine(&prelude, &mut members, false, transcript)
    };
    witness.append_produced(accumulator, transcript, &r_sumcheck);
    (proof, r_sumcheck)
}

/// Run the engine's batch prover over `members` and record the round polynomials.
fn run_engine<F: JoltField, T: Transcript>(
    prelude: &BatchPrelude<F>,
    members: &mut [&mut dyn ProveRounds<F>],
    many_members: bool,
    transcript: &mut T,
) -> (SumcheckInstanceProof<F>, Vec<F::Challenge>) {
    let mut scheduler = InstanceRounds::new(many_members);
    let mut recorder = ClearSumcheckRecorder::new();
    let proved = expect_engine(prove_batch(
        prelude,
        members,
        &mut scheduler,
        &mut recorder,
        transcript,
    ));
    let proof = SumcheckInstanceProof::new(
        recorder
            .into_proof()
            .round_polynomials
            .iter()
            .map(|poly| CompressedUniPoly {
                coeffs_except_linear_term: poly.coeffs_except_linear_term().to_vec(),
            })
            .collect(),
    );
    let r_sumcheck = proved
        .challenges
        .into_iter()
        .map(F::challenge_from_field)
        .collect();
    (proof, r_sumcheck)
}

/// The engine's batch description of tail-aligned members (a member with fewer rounds is bound
/// in the last `rounds` rounds) with the given input claims and batching coefficients.
fn batch_prelude<F: JoltField>(
    rounds: &[usize],
    input_claims: &[F],
    coefficients: &[F],
    max_num_rounds: usize,
    max_degree: usize,
) -> BatchPrelude<F> {
    let members = rounds
        .iter()
        .zip(input_claims)
        .zip(coefficients)
        .map(|((&rounds, &input_claim), &coefficient)| BatchMember {
            input_claim,
            coefficient,
            rounds,
            offset: max_num_rounds - rounds,
        })
        .collect();
    expect_engine(BatchPrelude::try_new(members, max_num_rounds, max_degree))
}

/// Map an engine error into the prover's panic (the engine's errors are internal
/// inconsistencies of the prover: a round polynomial that fails its own running-claim check).
fn expect_engine<F: JoltField, R>(res: Result<R, SumcheckError<F>>) -> R {
    res.unwrap_or_else(|e| panic!("sumcheck prover: {e:?}"))
}

/// Round scheduling for our instances: in declaration order, or in parallel across members for
/// batches of mutually independent instances (each member's own parallelism stays on; rayon's
/// work stealing absorbs the nesting).
///
/// Private: parallel scheduling relies on every member being a [`DynRelationWitness`] (which is
/// `Send + Sync`) seen as its [`ProveRounds`] supertrait (see [`SendRound`]), which only
/// [`run_engine`]'s callers guarantee.
struct InstanceRounds {
    parallel: bool,
    round: usize,
}

impl InstanceRounds {
    fn new(parallel: bool) -> Self {
        Self { parallel, round: 0 }
    }
}

/// A round handle sent to a rayon worker.
struct SendRound<'s, T>(&'s mut T);

// SAFETY: `InstanceRounds` only schedules members that are `DynRelationWitness: Send + Sync`
// instances upcast to `ProveRounds`; the remaining handle fields are field elements and indices.
// Each handle is visited by exactly one worker.
unsafe impl<T> Send for SendRound<'_, T> {}

impl<F: JoltField> RoundScheduler<F> for InstanceRounds {
    fn batch_prove_round(
        &mut self,
        work: &mut [MemberRound<'_, F>],
    ) -> Result<(), SumcheckError<F>> {
        #[cfg(not(target_arch = "wasm32"))]
        print_current_memory_usage(format!("Sumcheck round {}", self.round).as_str());
        self.round += 1;
        if self.parallel {
            work.iter_mut()
                .map(SendRound)
                .collect::<Vec<_>>()
                .into_par_iter()
                .try_for_each(|SendRound(item)| item.run())
        } else {
            work.iter_mut().try_for_each(|item| item.run())
        }
    }

    fn batch_finish_rounds(
        &mut self,
        finishes: &mut [MemberFinish<'_, F>],
    ) -> Result<(), SumcheckError<F>> {
        if self.parallel {
            finishes
                .iter_mut()
                .map(SendRound)
                .collect::<Vec<_>>()
                .into_par_iter()
                .try_for_each(|SendRound(item)| item.run())
        } else {
            finishes.iter_mut().try_for_each(|item| item.run())
        }
    }
}

impl<F: JoltField, R: SetupRelation<F>> RelationParams<F, R> {
    /// `relation`, set up against the prover's accumulator and transcript. The prover holds
    /// every claim its setup reads: a missing one is a prover bug and panics.
    pub fn new<T: Transcript>(
        relation: R,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let setup = relation
            .setup(accumulator, transcript)
            .expect("the prover holds every claim its setup reads");
        Self::at(relation, setup)
    }
}
