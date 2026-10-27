//! The prover side of [`super`].

use super::*;
use crate::relation::prover::prove_relation_batch;
use crate::{
    evaluation_reduction::ReducedInstance,
    opening_reduction::{
        prover::{OpeningProofReductionSumcheckProver, ProverOpening},
        EqAddressState, EqCycleState, SharedDensePolynomial,
    },
    sumcheck::SumcheckInstanceProof,
};
use allocative::Allocative;
pub use atlas_algebra::poly::opening_point::{Endianness, OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN};
use atlas_algebra::{
    field::{IntoOpening, JoltField},
    poly::multilinear_polynomial::MultilinearPolynomial,
};
use atlas_transcript::Transcript;
use common::{CommittedPoly, VirtualPoly};
#[cfg(any(test, feature = "test-feature"))]
use std::cell::RefCell;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::Mul,
    sync::{Arc, RwLock},
};

/// Accumulates openings computed by the prover over the course of Jolt,
/// so that they can all be reduced to a single opening proof using sumcheck.
#[derive(Clone, Allocative)]
pub struct ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    /// One batch-opening-reduction instance per *opening* (a committed
    /// polynomial opened at several points gets several instances; they all
    /// reduce to the same evaluation at the joint point, deduplicated per
    /// polynomial in [`Self::finalize_batch_opening_sumcheck`]).
    pub sumchecks: BTreeMap<OpeningId, OpeningProofReductionSumcheckProver<F>>,
    /// Openings for polynomials claimed during proving.
    /// Identified by the kind of polynomial, the node to which it corresponds (both held in Committed/VirtualPoly variant),
    /// and the sumcheck for which the opening is claimed.
    /// NOTE:
    /// If a node output is fed twice as index in a same node, it would lead to two different openings.
    /// However, in practice we always have that both openings are at the same point,
    /// hence they can be treated as a single opening.  
    pub openings: Openings<F>,
    /// Mapping from reduced source opening keys to evaluation-reduction line restriction `h`.
    pub reduced_evaluations: BTreeMap<usize, ReducedInstance<F>>,
    dense_polynomial_map: HashMap<CommittedPoly, Arc<RwLock<SharedDensePolynomial<F>>>>,
    eq_cycle_map: HashMap<Vec<F>, Arc<RwLock<EqCycleState<F>>>>,
    #[cfg(any(test, feature = "test-feature"))]
    pub appended_virtual_openings: RefCell<Vec<OpeningId>>,
    /// Reduced claims per instance (group), keyed like `sumchecks`.
    pub cached_opening_claims: BTreeMap<OpeningId, F>,
}

impl<F: JoltField> OpeningAccumulator<F> for ProverOpeningAccumulator<F> {
    fn try_get_virtual_polynomial_opening(
        &self,
        opening_id: OpeningId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        self.openings
            .get(&opening_id)
            .map(|(point, claim)| (point.clone(), *claim))
    }

    fn get_virtual_polynomial_opening(
        &self,
        opening_id: OpeningId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        opening_id
            .virtual_poly()
            .expect("expected virtual polynomial");
        match self.openings.get(&opening_id) {
            Some((point, claim)) => {
                #[cfg(any(test, feature = "test-feature"))]
                {
                    let mut virtual_openings = self.appended_virtual_openings.borrow_mut();
                    if let Some(index) = virtual_openings.iter().position(|id| id == &opening_id) {
                        virtual_openings.remove(index);
                    }
                }
                (point.clone(), *claim)
            }
            None => {
                panic!("opening for {opening_id:?} not found")
            }
        }
    }

    fn get_committed_polynomial_opening(
        &self,
        opening_id: OpeningId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        opening_id
            .committed_poly()
            .expect("expected committed polynomial");
        let (point, claim) = self
            .openings
            .get(&opening_id)
            .unwrap_or_else(|| panic!("opening for {opening_id:?} not found"));
        (point.clone(), *claim)
    }

    fn get_node_output_opening(&self, node_idx: usize) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        let reduced_instance = self
            .reduced_evaluations
            .get(&node_idx)
            .unwrap_or_else(|| panic!("reduced evaluation for node {node_idx} not found"))
            .clone();

        (reduced_instance.r.into(), reduced_instance.claim)
    }
}

impl<F> ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    pub fn new() -> Self {
        Self {
            sumchecks: BTreeMap::new(),
            openings: BTreeMap::new(),
            reduced_evaluations: BTreeMap::new(),
            eq_cycle_map: HashMap::new(),
            dense_polynomial_map: HashMap::new(),
            #[cfg(any(test, feature = "test-feature"))]
            appended_virtual_openings: std::cell::RefCell::new(vec![]),
            cached_opening_claims: BTreeMap::new(),
        }
    }

    /// The distinct committed polynomials with a registered opening.
    pub fn opened_polynomials(&self) -> BTreeSet<CommittedPoly> {
        self.sumchecks
            .values()
            .flat_map(|s| s.polynomials.iter().copied())
            .collect()
    }

    /// Caches an instance's reduced claim from the opening reduction sumcheck.
    /// Called from `OpeningProofReductionSumcheckProver::cache_openings`.
    pub fn cache_opening_reduction_claim(&mut self, key: OpeningId, claim: F) {
        let prev = self.cached_opening_claims.insert(key, claim);
        debug_assert!(prev.is_none(), "duplicate reduced claim for {key:?}");
    }

    pub fn evaluation_openings_mut(&mut self) -> &mut Openings<F> {
        &mut self.openings
    }

    /// Get the value of an opening by key
    pub fn get_opening(&self, key: OpeningId) -> F {
        self.openings
            .get(&key)
            .unwrap_or_else(|| panic!("opening should exist for {key:?}"))
            .1
    }

    pub fn get_node_openings(&self, node_idx: usize) -> Vec<&Opening<F>> {
        let lo = OpeningId::new(
            VirtualPoly::NodeOutput(node_idx),
            // Only consider openings for consumer indices greater or equal to node index,
            // since claims are only produced for current or later nodes
            SumcheckId::NodeExecution(node_idx),
        );
        let hi = OpeningId::new(
            VirtualPoly::NodeOutput(node_idx),
            SumcheckId::NodeExecution(usize::MAX),
        );

        self.openings
            .range(lo..=hi)
            .map(|(_, opening)| opening)
            .collect::<Vec<_>>()
    }

    /// Adds an opening of a dense polynomial to the accumulator.
    /// The given `polynomial` is opened at `opening_point`, yielding the claimed
    /// evaluation `claim`.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_dense")]
    pub fn append_dense<T, U>(
        &mut self,
        transcript: &mut T,
        opening_id: OpeningId,
        opening_point: Vec<U>,
        claim: F,
    ) where
        T: Transcript,
        U: Copy + Send + Sync + Into<F>,
        F: Mul<U, Output = F>,
    {
        #[cfg(any(test, feature = "test-feature"))]
        let claim = forgery::apply(opening_id, claim);
        transcript.append_scalar(&claim);

        let shared_eq = self
            .eq_cycle_map
            .entry(opening_point.iter().map(|&u| u.into()).collect())
            .or_insert_with(|| Arc::new(RwLock::new(EqCycleState::new(&opening_point))));

        // Add opening to map
        self.openings.insert(
            opening_id,
            (
                OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
                claim,
            ),
        );

        let polynomial = opening_id
            .committed_poly()
            .expect("expected committed polynomial");

        let sumcheck = OpeningProofReductionSumcheckProver::new_dense(
            polynomial,
            opening_id.sumcheck,
            shared_eq.clone(),
            opening_point,
            claim,
        );
        self.sumchecks.insert(opening_id, sumcheck);
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_sparse")]
    pub fn append_sparse<T, U>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPoly>,
        sumcheck: SumcheckId,
        r_address: Vec<U>,
        r_cycle: Vec<U>,
        claims: Vec<F>,
    ) where
        T: Transcript,
        U: Copy + Send + Sync + Into<F>,
        F: Mul<U, Output = F>,
    {
        #[cfg(any(test, feature = "test-feature"))]
        let claims: Vec<F> = polynomials
            .iter()
            .zip(claims)
            .map(|(&poly, claim)| forgery::apply(OpeningId::new(poly, sumcheck), claim))
            .collect();
        claims.iter().for_each(|claim| {
            transcript.append_scalar(claim);
        });
        let r_concat = [r_address.as_slice(), r_cycle.as_slice()].concat();

        let shared_eq_address = Arc::new(RwLock::new(EqAddressState::new(&r_address)));
        let shared_eq_cycle = self
            .eq_cycle_map
            .entry(r_cycle.clone().into_opening())
            .or_insert(Arc::new(RwLock::new(EqCycleState::new(&r_cycle))));

        // Add openings to map
        for (&label, claim) in polynomials.iter().zip(claims.iter()) {
            let opening_point_struct = OpeningPoint::<BIG_ENDIAN, F>::new(r_concat.clone());
            let key = OpeningId::new(label, sumcheck);
            self.openings
                .insert(key, (opening_point_struct.clone(), *claim));
        }

        let key = OpeningId::new(polynomials[0], sumcheck);
        let sumcheck_instance = OpeningProofReductionSumcheckProver::new_one_hot(
            polynomials.clone(),
            sumcheck,
            shared_eq_address.clone(),
            shared_eq_cycle.clone(),
            r_concat.clone(),
            claims.clone(),
        );
        self.sumchecks.insert(key, sumcheck_instance);
    }

    pub fn append_virtual<T: Transcript>(
        &mut self,
        transcript: &mut T,
        opening_id: OpeningId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        let virtual_poly = opening_id
            .virtual_poly()
            .expect("expected virtual polynomial");
        if let VirtualPoly::NodeOutput(node_idx) = virtual_poly {
            debug_assert!(
                !self.reduced_evaluations.contains_key(&node_idx),
                "cannot append opening for node {node_idx} after evaluation reduction"
            );
        }
        #[cfg(any(test, feature = "test-feature"))]
        let claim = forgery::apply(opening_id, claim);

        transcript.append_scalar(&claim);

        self.openings.insert(opening_id, (opening_point, claim));
        #[cfg(any(test, feature = "test-feature"))]
        self.appended_virtual_openings.borrow_mut().push(opening_id);
    }

    /// Take the openings, removing the points to reduce proof size
    ///
    /// # Returns:
    /// `Openings<F>` - The openings with points removed
    pub fn take(&mut self) -> Openings<F> {
        // to reduce proof size, remove all the opening points from accumulator.openings and just leave the claims
        for (_, opening) in self.openings.iter_mut() {
            opening.0.r.clear();
        }
        std::mem::take(&mut self.openings)
    }

    pub fn assert_virtual_polynomial_opening_exists(
        &self,
        opening_id: OpeningId,
    ) -> Option<&(OpeningPoint<false, F>, F)> {
        opening_id
            .virtual_poly()
            .expect("expected virtual polynomial");
        self.openings.get(&opening_id)
    }
}

impl<F> ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    pub fn num_sumchecks(&self) -> usize {
        self.sumchecks.len()
    }

    // ========== Batch Opening Reduction Sumcheck ==========

    /// Prepares sumcheck instances for the batch opening reduction.
    /// Must be called before `prove_batch_opening_sumcheck`.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::prepare_for_sumcheck")]
    pub fn prepare_for_sumcheck<T: Transcript>(
        &mut self,
        polynomials: &BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
        transcript: &mut T,
    ) {
        // In-group batching coefficient (powers per position in the group).
        let rho: F = transcript.challenge_scalar();
        self.sumchecks
            .values_mut()
            .for_each(|s| s.set_coefficients(rho));

        {
            // A committed polynomial nobody opened is simply left out of the
            // joint opening (it constrains nothing); an opening of an
            // uncommitted polynomial is a bug.
            let extra: Vec<CommittedPoly> = self
                .opened_polynomials()
                .into_iter()
                .filter(|p| !polynomials.contains_key(p))
                .collect();
            assert!(
                extra.is_empty(),
                "batch opening reduction: openings of uncommitted polynomials: {extra:?}"
            );
        }

        tracing::debug!(
            "{} sumcheck instances in batched opening proof reduction",
            self.sumchecks.len()
        );
        let prepare_span = tracing::span!(
            tracing::Level::INFO,
            "prepare_all_sumchecks",
            count = self.sumchecks.len()
        );
        let _enter = prepare_span.enter();

        // Populate dense_polynomial_map
        for (_, sumcheck) in self.sumchecks.iter() {
            if let ProverOpening::Dense(_) = &sumcheck.prover_state {
                let poly_id = sumcheck.polynomials[0];
                self.dense_polynomial_map.entry(poly_id).or_insert_with(|| {
                    let poly = polynomials.get(&poly_id).unwrap().clone();
                    Arc::new(RwLock::new(SharedDensePolynomial::new(poly)))
                });
            }
        }

        self.sumchecks.par_iter_mut().for_each(|(_, sumcheck)| {
            sumcheck.prepare_sumcheck(polynomials, &self.dense_polynomial_map);
        });
    }

    /// Proves the batch opening reduction sumcheck (Stage 7).
    /// Returns the sumcheck proof and challenges.
    #[tracing::instrument(
        skip_all,
        name = "ProverOpeningAccumulator::prove_batch_opening_sumcheck"
    )]
    pub fn prove_batch_opening_sumcheck<T: Transcript>(
        &mut self,
        transcript: &mut T,
    ) -> (SumcheckInstanceProof<F>, Vec<F::Challenge>) {
        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("Opening accumulator", &(*self));
            let mut flamegraph = FlameGraphBuilder::default();
            flamegraph.visit_root(&(*self));
            write_flamegraph_svg(flamegraph, "stage7_start_flamechart.svg");
        }

        // Temporarily take sumchecks so we can pass self to prove_relation_batch
        let mut sumchecks = std::mem::take(&mut self.sumchecks);
        let instances = sumchecks
            .iter_mut()
            .map(|(_, opening)| opening as &mut _)
            .collect();

        let (sumcheck_proof, r_sumcheck) = prove_relation_batch(instances, self, transcript);

        // Restore sumchecks (with cached claims from cache_openings)
        self.sumchecks = sumchecks;

        #[cfg(feature = "allocative")]
        {
            let mut flamegraph = FlameGraphBuilder::default();
            flamegraph.visit_root(&(*self));
            write_flamegraph_svg(flamegraph, "stage7_end_flamechart.svg");
        }

        (sumcheck_proof, r_sumcheck)
    }

    /// Finalizes the batch opening reduction sumcheck.
    /// Uses cached claims from `cache_openings`, appends them to transcript, derives gamma powers,
    /// and cleans up sumcheck instances.
    /// Returns the state needed for Stage 8.
    #[tracing::instrument(
        skip_all,
        name = "ProverOpeningAccumulator::finalize_batch_opening_sumcheck"
    )]
    pub fn finalize_batch_opening_sumcheck<T: Transcript>(
        &mut self,
        r_sumcheck: Vec<F::Challenge>,
        transcript: &mut T,
    ) -> OpeningReductionState<F> {
        // Reduced claims per instance (populated by cache_openings), in key order
        let (keys, sumcheck_claims): (Vec<OpeningId>, Vec<F>) =
            std::mem::take(&mut self.cached_opening_claims)
                .into_iter()
                .unzip();
        debug_assert!(keys.iter().eq(self.sumchecks.keys()));

        transcript.append_scalars(&sumcheck_claims);
        let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(sumcheck_claims.len());

        // Per-polynomial coefficient of the joint RLC: Σ over the instances
        // containing it of `gamma_instance · ρ^position`.
        let (polynomials, poly_coeffs) = joint_coefficients(
            self.sumchecks
                .values()
                .map(|s| (&s.polynomials, &s.coefficients)),
            &gamma_powers,
        );

        // Drop sumchecks in background - they're no longer needed
        {
            let sumchecks = std::mem::take(&mut self.sumchecks);
            atlas_algebra::utils::thread::drop_in_background_thread(sumchecks);
        }

        OpeningReductionState {
            r_sumcheck,
            gamma_powers,
            sumcheck_claims,
            polynomials,
            poly_coeffs,
        }
    }
}

impl<F> Default for ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    /// Compare this accumulator to the corresponding `ProverOpeningAccumulator` and panic
    /// if the openings appended differ from the prover's openings.
    #[cfg(any(test, feature = "test-feature"))]
    pub fn compare_to(&mut self, prover_openings: ProverOpeningAccumulator<F>) {
        self.prover_opening_accumulator = Some(prover_openings);
    }
}
