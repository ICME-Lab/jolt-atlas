//! The prover side of [`super`]: a [`RelationWitness`] computes its own round messages, and the
//! relation supplies everything else the batched sumcheck asks of an instance (degree, input
//! claim, where the produced openings sit).

use super::*;
use crate::opening_proof::prover::ProverOpeningAccumulator;
use atlas_algebra::field::MaybeAllocative;
use jolt_sumcheck::ProveRounds;

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

/// [`crate::sumcheck::prover::SumcheckInstanceProver`] for a [`RelationWitness`]: degree and input claim from the
/// relation, the produced openings through [`RelationWitness::append_claims`].
///
/// `relation_witness_instance!([F: JoltField, X: Bound] Witness<F, X>)`
#[macro_export]
macro_rules! relation_witness_instance {
    ([$($generics:tt)*] $witness:ty) => {
        impl<$($generics)*, T: atlas_transcript::Transcript>
            $crate::sumcheck::prover::SumcheckInstanceProver<F, T> for $witness
        {
            fn degree(&self) -> usize {
                jolt_claims::SymbolicSumcheck::degree(
                    $crate::relation::prover::RelationWitness::relation(self),
                )
            }

            fn input_claim(
                &self,
                accumulator: &$crate::opening_proof::prover::ProverOpeningAccumulator<F>,
            ) -> F {
                $crate::relation::relation_input_claim(
                    $crate::relation::prover::RelationWitness::relation(self),
                    $crate::relation::prover::RelationWitness::setup(self),
                    accumulator,
                )
            }

            fn cache_openings(
                &self,
                accumulator: &mut $crate::opening_proof::prover::ProverOpeningAccumulator<F>,
                transcript: &mut T,
                sumcheck_challenges: &[<F as atlas_algebra::field::JoltField>::Challenge],
            ) {
                $crate::relation::prover::RelationWitness::append_claims(
                    self,
                    accumulator,
                    transcript,
                    sumcheck_challenges,
                );
            }

            $crate::relation_witness_flamegraph!();
        }
    };
}

/// The instance's flamegraph hook, present when this crate's `allocative` feature is.
#[cfg(feature = "allocative")]
#[doc(hidden)]
#[macro_export]
macro_rules! relation_witness_flamegraph {
    () => {
        fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
            flamegraph.visit_root(self);
        }
    };
}

/// The instance's flamegraph hook, present when this crate's `allocative` feature is.
#[cfg(not(feature = "allocative"))]
#[doc(hidden)]
#[macro_export]
macro_rules! relation_witness_flamegraph {
    () => {};
}
