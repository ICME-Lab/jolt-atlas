//! This is a port of the sumcheck-based batch opening proof protocol implemented
//! in Nova: https://github.com/microsoft/Nova/blob/2772826ba296b66f1cd5deecf7aca3fd1d10e1f4/src/spartan/snark.rs#L410-L424
//! and such code is Copyright (c) Microsoft Corporation.
//! For additively homomorphic commitment schemes (including Zeromorph, HyperKZG) we
//! can use a sumcheck to reduce multiple opening proofs (multiple polynomials, not
//! necessarily of the same size, each opened at a different point) into a single opening.

use crate::{
    field::{IntoOpening, JoltField},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
    },
    subprotocols::{
        evaluation_reduction::ReducedInstance,
        opening_reduction::{
            EqAddressState, EqCycleState, OpeningProofReductionSumcheckProver,
            OpeningProofReductionSumcheckVerifier, ProverOpening, SharedDensePolynomial,
        },
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use allocative::Allocative;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use common::{CommittedPoly, VirtualPoly};
use rayon::prelude::*;

#[cfg(any(test, feature = "test-feature"))]
use std::cell::RefCell;
use std::{
    collections::{BTreeMap, HashMap},
    ops::Mul,
    sync::{Arc, RwLock},
};

pub type Endianness = bool;
pub const BIG_ENDIAN: Endianness = false;
pub const LITTLE_ENDIAN: Endianness = true;

/// (point, claim)
pub type Opening<F> = (OpeningPoint<BIG_ENDIAN, F>, F);
pub type Openings<F> = BTreeMap<OpeningId, Opening<F>>;

/// Accumulates openings computed by the prover over the course of Jolt,
/// so that they can all be reduced to a single opening proof using sumcheck.
#[derive(Clone, Allocative)]
pub struct ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    pub sumchecks: BTreeMap<CommittedPoly, OpeningProofReductionSumcheckProver<F>>,
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
    pub cached_opening_claims: BTreeMap<CommittedPoly, F>,
    #[cfg(feature = "zk")]
    pending_claims: Vec<F>,
    #[cfg(feature = "zk")]
    pending_claim_ids: Vec<OpeningId>,
    /// In ZK mode the prover suppresses cleartext transcript appends in
    /// `append_virtual` / `append_dense`; transcript synchronisation comes
    /// from the Pedersen commitments emitted by `BatchedSumcheck::prove_zk`
    /// (or via auxiliary_claims for public scalars that are intentionally
    /// re-appended with `zk_mode` toggled off on both sides — see the
    /// Softmax aux-scalar path). `pending_claims` is still collected so the
    /// downstream commit_chunked step can hide the claim values.
    #[cfg(feature = "zk")]
    pub zk_mode: bool,
}

/// Accumulates openings encountered by the verifier over the course of Jolt,
/// so that they can all be reduced to a single opening proof verification using sumcheck.
pub struct VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    sumchecks: BTreeMap<CommittedPoly, OpeningProofReductionSumcheckVerifier<F>>,
    pub openings: Openings<F>,
    /// Mapping for nodes reduced opening points
    pub reduced_evaluations: BTreeMap<usize, ReducedInstance<F>>,
    /// In testing, the Jolt verifier may be provided the prover's openings so that we
    /// can detect any places where the openings don't match up.
    #[cfg(any(test, feature = "test-feature"))]
    prover_opening_accumulator: Option<ProverOpeningAccumulator<F>>,
    #[cfg(feature = "zk")]
    pending_claims: Vec<F>,
    /// When true, `append_virtual` inserts new keys with placeholder claims
    /// instead of panicking. Claims are verified by BlindFold, not individually.
    pub zk_mode: bool,
}

pub trait OpeningAccumulator<F: JoltField> {
    fn try_get_virtual_polynomial_opening(
        &self,
        opening_id: OpeningId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)>;

    fn get_virtual_polynomial_opening(
        &self,
        opening_id: OpeningId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F);

    fn get_committed_polynomial_opening(
        &self,
        opening_id: OpeningId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F);

    /// Gets the reduced opening claim for the given node index.
    /// The reduced opening claim is either the unique existing opening claim,
    /// Or obtained from the evaluation reduction protocol if multiple openings need to be reduced.
    fn get_node_output_opening(&self, node_idx: usize) -> (OpeningPoint<BIG_ENDIAN, F>, F);
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
            #[cfg(feature = "zk")]
            pending_claims: Vec::new(),
            #[cfg(feature = "zk")]
            pending_claim_ids: Vec::new(),
            #[cfg(feature = "zk")]
            zk_mode: false,
        }
    }

    /// Takes and returns the pending ZK claims, leaving the internal vector empty.
    #[cfg(feature = "zk")]
    pub fn take_pending_claims(&mut self) -> Vec<F> {
        std::mem::take(&mut self.pending_claims)
    }

    /// Takes and returns the pending ZK claim IDs, leaving the internal vector empty.
    #[cfg(feature = "zk")]
    pub fn take_pending_claim_ids(&mut self) -> Vec<OpeningId> {
        std::mem::take(&mut self.pending_claim_ids)
    }

    /// Caches an opening claim from the opening reduction sumcheck.
    /// Called from `OpeningProofReductionSumcheckProver::cache_openings`.
    pub fn cache_opening_reduction_claim(&mut self, polynomial: CommittedPoly, claim: F) {
        self.cached_opening_claims.insert(polynomial, claim);
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
        #[cfg(feature = "zk")]
        let suppress = self.zk_mode;
        #[cfg(not(feature = "zk"))]
        let suppress = false;
        if !suppress {
            transcript.append_scalar(&claim);
        }

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
        self.sumchecks.insert(polynomial, sumcheck);

        #[cfg(feature = "zk")]
        {
            self.pending_claims.push(claim);
            self.pending_claim_ids.push(opening_id);
        }
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
        #[cfg(feature = "zk")]
        let suppress = self.zk_mode;
        #[cfg(not(feature = "zk"))]
        let suppress = false;
        if !suppress {
            claims.iter().for_each(|claim| {
                transcript.append_scalar(claim);
            });
        }
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

        for (label, claim) in polynomials.iter().copied().zip(claims.into_iter()) {
            let sumcheck_instance = OpeningProofReductionSumcheckProver::new_one_hot(
                label,
                sumcheck,
                shared_eq_address.clone(),
                shared_eq_cycle.clone(),
                r_concat.clone(),
                claim,
            );
            self.sumchecks.insert(label, sumcheck_instance);
            #[cfg(feature = "zk")]
            {
                let key = OpeningId::new(label, sumcheck);
                self.pending_claims.push(claim);
                self.pending_claim_ids.push(key);
            }
        }
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

        #[cfg(feature = "zk")]
        let suppress = self.zk_mode;
        #[cfg(not(feature = "zk"))]
        let suppress = false;
        if !suppress {
            transcript.append_scalar(&claim);
        }

        self.openings.insert(opening_id, (opening_point, claim));
        #[cfg(any(test, feature = "test-feature"))]
        self.appended_virtual_openings.borrow_mut().push(opening_id);
        #[cfg(feature = "zk")]
        {
            self.pending_claims.push(claim);
            self.pending_claim_ids.push(opening_id);
        }
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
    pub fn prepare_for_sumcheck(
        &mut self,
        polynomials: &BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
    ) {
        if self.sumchecks.len() != polynomials.len() {
            let missing_sumchecks: Vec<CommittedPoly> = polynomials
                .keys()
                .filter(|poly| !self.sumchecks.contains_key(poly))
                .cloned()
                .collect();
            println!("Missing sumcheck instances for polynomials: {missing_sumchecks:?}");
            panic!(
                "Expected {} sumcheck instances, but found {}",
                polynomials.len(),
                self.sumchecks.len()
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
                self.dense_polynomial_map
                    .entry(sumcheck.polynomial)
                    .or_insert_with(|| {
                        let poly = polynomials.get(&sumcheck.polynomial).unwrap().clone();
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
    ) -> (SumcheckInstanceProof<F, T>, Vec<F::Challenge>) {
        #[cfg(feature = "allocative")]
        {
            print_data_structure_heap_usage("Opening accumulator", &(*self));
            let mut flamegraph = FlameGraphBuilder::default();
            flamegraph.visit_root(&(*self));
            write_flamegraph_svg(flamegraph, "stage7_start_flamechart.svg");
        }

        // Temporarily take sumchecks so we can pass self to BatchedSumcheck::prove
        let mut sumchecks = std::mem::take(&mut self.sumchecks);
        let instances = sumchecks
            .iter_mut()
            .map(|(_, opening)| opening as &mut _)
            .collect();

        let (sumcheck_proof, r_sumcheck) = BatchedSumcheck::prove(instances, self, transcript);

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

    /// ZK variant of `prove_batch_opening_sumcheck`. Runs `BatchedSumcheck::prove_zk`
    /// so that round polynomials are Pedersen-committed rather than sent in the clear,
    /// and the resulting stage data is pushed into `blindfold_accumulator` for later
    /// inclusion in the BlindFold R1CS witness. The reduced opening claim is no longer
    /// leaked through the transcript; binding to `y_com` happens via BlindFold.
    #[cfg(feature = "zk")]
    #[tracing::instrument(
        skip_all,
        name = "ProverOpeningAccumulator::prove_batch_opening_sumcheck_zk"
    )]
    pub fn prove_batch_opening_sumcheck_zk<T, C, R>(
        &mut self,
        blindfold_accumulator: &mut crate::subprotocols::blindfold::BlindFoldAccumulator<F, C>,
        stage_configs: &mut Vec<crate::subprotocols::blindfold::StageConfig>,
        pedersen_gens: &crate::poly::commitment::pedersen::PedersenGenerators<C>,
        rng: &mut R,
        transcript: &mut T,
    ) -> (
        crate::subprotocols::sumcheck::ZkSumcheckProof<F, C, T>,
        Vec<F::Challenge>,
    )
    where
        T: Transcript,
        C: crate::curve::JoltCurve<F = F>,
        R: rand_core::CryptoRngCore,
    {
        use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;

        let mut sumchecks = std::mem::take(&mut self.sumchecks);
        // Capture stage shape before borrowing the instances mutably.
        let max_rounds = sumchecks
            .values()
            .map(|opening| <Opening<F> as SumcheckInstanceParams<F>>::num_rounds(&opening.opening))
            .max()
            .unwrap_or(1);
        let max_degree = sumchecks
            .values()
            .map(|opening| <Opening<F> as SumcheckInstanceParams<F>>::degree(&opening.opening))
            .max()
            .unwrap_or(1);

        let instances = sumchecks
            .iter_mut()
            .map(|(_, opening)| opening as &mut _)
            .collect();

        let (zk_proof, r_sumcheck, _final_claim) = BatchedSumcheck::prove_zk::<F, C, T, _>(
            instances,
            self,
            blindfold_accumulator,
            transcript,
            pedersen_gens,
            rng,
        );

        self.sumchecks = sumchecks;
        stage_configs.push(crate::subprotocols::blindfold::StageConfig::new_chain(
            max_rounds, max_degree,
        ));
        (zk_proof, r_sumcheck)
    }

    /// Finalizes the batch opening reduction sumcheck.
    /// Uses cached claims from `cache_openings`, appends them to transcript, derives gamma powers,
    /// and cleans up sumcheck instances.
    /// Returns the state needed for Stage 8.
    ///
    /// In ZK mode the cleartext claim append is suppressed when the prover
    /// accumulator's `zk_mode` flag is on; the gamma_powers are still derived
    /// from the transcript (which by then carries the batched-opening stage's
    /// round commitments via `BatchedSumcheck::prove_zk`). The verifier
    /// reproduces the same gamma derivation, since both sides skip the
    /// cleartext append symmetrically.
    #[tracing::instrument(
        skip_all,
        name = "ProverOpeningAccumulator::finalize_batch_opening_sumcheck"
    )]
    pub fn finalize_batch_opening_sumcheck<T: Transcript>(
        &mut self,
        r_sumcheck: Vec<F::Challenge>,
        transcript: &mut T,
    ) -> OpeningReductionState<F> {
        // Extract claims and polynomials from cached opening claims (populated by cache_openings)
        let (polynomials, sumcheck_claims): (Vec<CommittedPoly>, Vec<F>) =
            std::mem::take(&mut self.cached_opening_claims)
                .into_iter()
                .unzip();

        #[cfg(feature = "zk")]
        let suppress = self.zk_mode;
        #[cfg(not(feature = "zk"))]
        let suppress = false;
        if !suppress {
            transcript.append_scalars(&sumcheck_claims);
        }
        let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(sumcheck_claims.len());

        // Drop sumchecks in background - they're no longer needed
        {
            let sumchecks = std::mem::take(&mut self.sumchecks);
            crate::utils::thread::drop_in_background_thread(sumchecks);
        }

        OpeningReductionState {
            r_sumcheck,
            gamma_powers,
            sumcheck_claims,
            polynomials,
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

/// Intermediate state between Stage 7 (batch opening reduction sumcheck) and Stage 8 (Dory opening).
/// Stored in prover/verifier state to bridge the two stages.
#[derive(Clone, Allocative)]
pub struct OpeningReductionState<F: JoltField> {
    pub r_sumcheck: Vec<F::Challenge>,
    pub gamma_powers: Vec<F>,
    pub sumcheck_claims: Vec<F>,
    pub polynomials: Vec<CommittedPoly>,
}

impl<F: JoltField> OpeningAccumulator<F> for VerifierOpeningAccumulator<F> {
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
        let key = opening_id;
        match self.openings.get(&key) {
            Some((point, claim)) => (point.clone(), *claim),
            None if self.zk_mode => (OpeningPoint::default(), F::zero()),
            None => {
                panic!("No opening found for {opening_id:?}")
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
            .unwrap_or_else(|| panic!("No opening found for {opening_id:?}"));
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

impl<F> VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    pub fn new() -> Self {
        Self {
            sumchecks: BTreeMap::new(),
            openings: BTreeMap::new(),
            reduced_evaluations: BTreeMap::new(),
            #[cfg(any(test, feature = "test-feature"))]
            prover_opening_accumulator: None,
            #[cfg(feature = "zk")]
            pending_claims: Vec::new(),
            zk_mode: false,
        }
    }

    /// Create a verifier accumulator in ZK mode. In ZK mode, `append_virtual`
    /// inserts new keys with placeholder claims instead of panicking.
    pub fn new_zk() -> Self {
        Self {
            zk_mode: true,
            ..Self::new()
        }
    }

    /// Takes and returns the pending ZK claims, leaving the internal vector empty.
    #[cfg(feature = "zk")]
    pub fn take_pending_claims(&mut self) -> Vec<F> {
        std::mem::take(&mut self.pending_claims)
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
            .collect()
    }

    /// Compare this accumulator to the corresponding `ProverOpeningAccumulator` and panic
    /// if the openings appended differ from the prover's openings.
    #[cfg(any(test, feature = "test-feature"))]
    pub fn compare_to(&mut self, prover_openings: ProverOpeningAccumulator<F>) {
        self.prover_opening_accumulator = Some(prover_openings);
    }

    /// Adds an opening of a dense polynomial the accumulator.
    /// The given `polynomial` is opened at `opening_point`.
    pub fn append_dense<T, U>(
        &mut self,
        transcript: &mut T,
        opening_id: OpeningId,
        opening_point: Vec<U>,
    ) where
        T: Transcript,
        U: Copy + Send + Sync + Into<F>,
    {
        let claim = if self.zk_mode {
            // ZK mode: stay transcript-quiet to mirror the prover, even if the
            // key already exists from a prior append (see the matching comment
            // in `append_virtual` above).
            self.openings.insert(
                opening_id,
                (
                    OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
                    F::zero(),
                ),
            );
            F::zero()
        } else if let Some((_, claim)) = self.openings.get(&opening_id) {
            // Standard mode: claim was pre-loaded.
            let claim = *claim;
            transcript.append_scalar(&claim);
            self.openings.insert(
                opening_id,
                (
                    OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
                    claim,
                ),
            );
            claim
        } else {
            panic!("Tried to populate dense opening for non-existent key: {opening_id:?}");
        };

        let polynomial = opening_id
            .committed_poly()
            .expect("expected committed polynomial");
        self.sumchecks.insert(
            polynomial,
            OpeningProofReductionSumcheckVerifier::new(polynomial, opening_point, claim),
        );
    }

    /// Adds openings to the accumulator. The polynomials underlying the given
    /// `commitments` are opened at `opening_point`, yielding the claimed evaluations
    /// `claims`.
    /// Multiple sparse polynomials opened at a single point are NOT batched into
    /// a single polynomial opened at the same point.
    pub fn append_sparse<T, U>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPoly>,
        sumcheck: SumcheckId,
        opening_point: Vec<U>,
    ) where
        T: Transcript,
        U: Copy + Send + Sync + Into<F>,
    {
        for label in polynomials.into_iter() {
            let key = OpeningId::new(label, sumcheck);
            let claim = if self.zk_mode {
                // ZK mode: stay transcript-quiet to mirror the prover, even
                // if the key already exists from a prior append (see the
                // matching comment in `append_virtual` above).
                F::zero()
            } else if let Some((_, c)) = self.openings.get(&key) {
                let c = *c;
                transcript.append_scalar(&c);
                c
            } else {
                panic!("Tried to populate sparse opening for non-existent key: {key:?}");
            };

            self.openings.insert(
                key,
                (
                    OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
                    claim,
                ),
            );

            self.sumchecks.insert(
                label,
                OpeningProofReductionSumcheckVerifier::new(label, opening_point.clone(), claim),
            );
        }
    }

    /// Populates the opening point for an existing claim in the evaluation_openings map.
    pub fn append_virtual<T: Transcript>(
        &mut self,
        transcript: &mut T,
        opening_id: OpeningId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
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

        if self.zk_mode {
            // ZK mode: stay transcript-quiet to mirror the prover (whose
            // zk_mode also suppresses cleartext appends). We always (re)write
            // a placeholder claim, even when the key already exists from a
            // prior `append_virtual` call. This matters when the same
            // `opening_id` is registered twice in a sumcheck batch (e.g.
            // `SoftmaxExpQ` from both RecipMult and ExpSum cache_openings):
            // without this, the second call would fall into the standard
            // branch and emit a stray `transcript.append_scalar(F::zero())`
            // that the prover does not produce. The actual claim is verified
            // by BlindFold, not individually.
            //
            // Callers that publish a real cleartext claim (the Softmax
            // auxiliary-scalar path) pre-populate `openings` AND toggle
            // `zk_mode = false`, so they fall through to the standard branch
            // below.
            self.openings
                .insert(opening_id, (opening_point.clone(), F::zero()));
        } else if let Some((_, claim)) = self.openings.get(&opening_id) {
            // Standard mode: claim was pre-loaded. Append to transcript.
            transcript.append_scalar(claim);
            let claim = *claim;
            self.openings
                .insert(opening_id, (opening_point.clone(), claim));
        } else {
            panic!("Tried to populate opening point for non-existent key: {opening_id:?}");
        }
    }
}

impl<F> VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    pub fn num_sumchecks(&self) -> usize {
        self.sumchecks.len()
    }

    /// Iterates the `CommittedPoly` keys for which a sumcheck instance has been
    /// registered (one per committed polynomial that some operator opened).
    /// `BTreeMap` iteration is sorted, so the order matches the prover's
    /// `state.polynomials` produced by `finalize_batch_opening_sumcheck`.
    pub fn sumchecks_keys(&self) -> impl Iterator<Item = CommittedPoly> + '_ {
        self.sumchecks.keys().copied()
    }

    // ========== Batch Opening Reduction Sumcheck ==========

    #[tracing::instrument(skip_all, name = "VerifierOpeningAccumulator::prepare_for_sumcheck")]
    /// Prepares the verifier for the batch opening reduction sumcheck.
    /// Populates sumcheck claims from the proof.
    pub fn prepare_for_sumcheck(&mut self, sumcheck_claims: &[F]) {
        // #[cfg(any(test, feature = "test-feature"))]
        // if let Some(prover_openings) = &self.prover_opening_accumulator {
        //     assert_eq!(prover_openings.num_sumchecks(), self.num_sumchecks());
        // }

        self.sumchecks
            .values_mut()
            .zip(sumcheck_claims.iter())
            .for_each(|(opening, claim)| opening.sumcheck_claim = Some(*claim));
    }

    /// Verifies the batch opening reduction sumcheck (Stage 7).
    #[tracing::instrument(
        skip_all,
        name = "VerifierOpeningAccumulator::verify_batch_opening_sumcheck"
    )]
    pub fn verify_batch_opening_sumcheck<T: Transcript>(
        &self,
        sumcheck_proof: &SumcheckInstanceProof<F, T>,
        transcript: &mut T,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> = self
            .sumchecks
            .values()
            .map(|opening| {
                let instance: &dyn SumcheckInstanceVerifier<F, T> = opening;
                instance
            })
            .collect();
        BatchedSumcheck::verify(
            sumcheck_proof,
            instances,
            &mut VerifierOpeningAccumulator::new(),
            transcript,
        )
    }

    #[tracing::instrument(
        skip_all,
        name = "VerifierOpeningAccumulator::finalize_batch_opening_sumcheck"
    )]
    /// Finalizes the batch opening reduction sumcheck verification.
    /// Returns the state needed for verify joint opening.
    pub fn finalize_batch_opening_sumcheck<T: Transcript>(
        &self,
        r_sumcheck: Vec<F::Challenge>,
        sumcheck_claims: &[F],
        transcript: &mut T,
    ) -> OpeningReductionState<F> {
        // Extract polynomial labels
        let polynomials: Vec<CommittedPoly> =
            self.sumchecks.values().map(|s| s.polynomial).collect();

        // Append claims and derive gamma powers
        transcript.append_scalars(sumcheck_claims);
        let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(self.sumchecks.len());

        OpeningReductionState {
            r_sumcheck,
            gamma_powers,
            sumcheck_claims: sumcheck_claims.to_vec(),
            polynomials,
        }
    }

    // ========== Batch Opening Verification ==========

    /// Computes the joint commitment by homomorphically combining individual commitments.
    pub fn compute_joint_commitment<PCS: CommitmentScheme<Field = F>>(
        commitment_map: &mut BTreeMap<CommittedPoly, PCS::Commitment>,
        state: &OpeningReductionState<F>,
    ) -> PCS::Commitment {
        let commitments: Vec<PCS::Commitment> = state
            .polynomials
            .iter()
            .map(|poly| commitment_map.remove(poly).unwrap())
            .collect();
        PCS::combine_commitments(&commitments, &state.gamma_powers)
    }

    /// Computes the joint claim for the batch opening verification.
    pub fn compute_joint_claim<T: Transcript>(&self, state: &OpeningReductionState<F>) -> F {
        let num_sumcheck_rounds = self
            .sumchecks
            .values()
            .map(|opening| SumcheckInstanceVerifier::<F, T>::num_rounds(opening))
            .max()
            .unwrap();

        state
            .gamma_powers
            .iter()
            .zip(state.sumcheck_claims.iter())
            .zip(self.sumchecks.values())
            .map(|((coeff, claim), opening)| {
                let r_slice = &state.r_sumcheck
                    [..num_sumcheck_rounds - SumcheckInstanceVerifier::<F, T>::num_rounds(opening)];
                let lagrange_eval: F = r_slice.iter().map(|r| F::one() - r).product();
                *coeff * claim * lagrange_eval
            })
            .sum()
    }

    #[tracing::instrument(skip_all, name = "VerifierOpeningAccumulator::verify_joint_opening")]
    /// Verifies the joint opening proof (Stage 8).
    pub fn verify_joint_opening<T: Transcript, PCS: CommitmentScheme<Field = F>>(
        &self,
        pcs_setup: &PCS::VerifierSetup,
        joint_opening_proof: &PCS::Proof,
        joint_commitment: &PCS::Commitment,
        state: &OpeningReductionState<F>,
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        let joint_claim = self.compute_joint_claim::<T>(state);

        PCS::verify(
            joint_opening_proof,
            pcs_setup,
            transcript,
            &state.r_sumcheck,
            &joint_claim,
            joint_commitment,
        )
    }
}

impl<F> Default for VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    fn default() -> Self {
        Self::new()
    }
}

// TODO(AntoineF4C5): Using Vec<F> rather than Vec<F::Challenge> denies us from the performance gains of F::Challenge.
// The goal would be to use an enum that can be either F or F::Challenge.
// Might also be interesting to make a difference between an opening point, which allows to evaluate a poly at a point,
// and an array of sumcheck challenges, which might need reordering to correspond to a polynomial evaluation point.
#[derive(Clone, Debug, PartialEq, Default, Allocative)]
pub struct OpeningPoint<const E: Endianness, F: JoltField> {
    pub r: Vec<F>,
}

impl<const E: Endianness, F: JoltField> std::ops::Index<usize> for OpeningPoint<E, F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.r[index]
    }
}

impl<const E: Endianness, F: JoltField> std::ops::Index<std::ops::RangeFull>
    for OpeningPoint<E, F>
{
    type Output = [F];

    fn index(&self, _index: std::ops::RangeFull) -> &Self::Output {
        &self.r[..]
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn len(&self) -> usize {
        self.r.len()
    }

    pub fn split_at_r(&self, mid: usize) -> (&[F], &[F]) {
        self.r.split_at(mid)
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        let (left, right) = self.r.split_at(mid);
        (Self::new(left.to_vec()), Self::new(right.to_vec()))
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn new<C: Into<F>>(r: Vec<C>) -> Self {
        Self {
            r: r.into_iter().map(|x| x.into()).collect(),
        }
    }

    pub fn endianness(&self) -> &'static str {
        if E == BIG_ENDIAN {
            "big"
        } else {
            "little"
        }
    }

    pub fn match_endianness<const SWAPPED_E: Endianness>(&self) -> OpeningPoint<SWAPPED_E, F>
    where
        F: Clone,
    {
        let mut reversed = self.r.clone();
        if E != SWAPPED_E {
            reversed.reverse();
        }
        OpeningPoint::<SWAPPED_E, F>::new(reversed)
    }
}

impl<F: JoltField, U: Into<F>> From<Vec<U>> for OpeningPoint<LITTLE_ENDIAN, F> {
    fn from(r: Vec<U>) -> Self {
        Self::new(r.into_iter().map(|x| x.into()).collect())
    }
}

impl<F: JoltField, U: Into<F>> From<Vec<U>> for OpeningPoint<BIG_ENDIAN, F> {
    fn from(r: Vec<U>) -> Self {
        Self::new(r.into_iter().map(|x| x.into()).collect())
    }
}

impl<const E: Endianness, F: JoltField> Into<Vec<F>> for OpeningPoint<E, F> {
    fn into(self) -> Vec<F> {
        self.r
    }
}

impl<const E: Endianness, F: JoltField> Into<Vec<F>> for &OpeningPoint<E, F>
where
    F: Clone,
{
    fn into(self) -> Vec<F> {
        self.r.clone()
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum SumcheckId {
    /// A node-execution sumcheck for a specific consumer node.
    /// The payload is the consumer's node index.
    NodeExecution(usize),
    Raf,
    RaVirtualization,
    RamHammingBooleanity,
    RamHammingWeight,
    Booleanity,
    HammingWeight,
    /// RLC sumcheck for the execution of a specific node.
    RLC(usize),
    /// Batch opening reduction (used by BlindFold y_com constraint).
    BlindFoldBatchOpening,
    /// Evaluation reduction sum-check used in neural teleport ops
    NTReduction,
}

impl CanonicalSerialize for SumcheckId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::NodeExecution(idx) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                idx.serialize_with_mode(&mut writer, compress)?;
            }
            Self::Raf => 1u8.serialize_with_mode(&mut writer, compress)?,
            Self::RaVirtualization => 2u8.serialize_with_mode(&mut writer, compress)?,
            Self::RamHammingBooleanity => 3u8.serialize_with_mode(&mut writer, compress)?,
            Self::RamHammingWeight => 4u8.serialize_with_mode(&mut writer, compress)?,
            Self::Booleanity => 5u8.serialize_with_mode(&mut writer, compress)?,
            Self::HammingWeight => 6u8.serialize_with_mode(&mut writer, compress)?,
            Self::RLC(idx) => {
                7u8.serialize_with_mode(&mut writer, compress)?;
                idx.serialize_with_mode(&mut writer, compress)?;
            }
            Self::BlindFoldBatchOpening => 8u8.serialize_with_mode(&mut writer, compress)?,
            Self::NTReduction => 9u8.serialize_with_mode(&mut writer, compress)?,
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Self::NodeExecution(idx) | Self::RLC(idx) => {
                1u8.serialized_size(compress) + idx.serialized_size(compress)
            }
            _ => 1u8.serialized_size(compress),
        }
    }
}

impl Valid for SumcheckId {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for SumcheckId {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match tag {
            0 => {
                let idx = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(Self::NodeExecution(idx))
            }
            1 => Ok(Self::Raf),
            2 => Ok(Self::RaVirtualization),
            3 => Ok(Self::RamHammingBooleanity),
            4 => Ok(Self::RamHammingWeight),
            5 => Ok(Self::Booleanity),
            6 => Ok(Self::HammingWeight),
            7 => {
                let idx = usize::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(Self::RLC(idx))
            }
            8 => Ok(Self::BlindFoldBatchOpening),
            9 => Ok(Self::NTReduction),
            _ => Err(SerializationError::InvalidData),
        }
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum PolynomialId {
    Virtual(VirtualPoly),
    Committed(CommittedPoly),
}

impl From<VirtualPoly> for PolynomialId {
    fn from(poly: VirtualPoly) -> Self {
        Self::Virtual(poly)
    }
}

impl From<CommittedPoly> for PolynomialId {
    fn from(poly: CommittedPoly) -> Self {
        Self::Committed(poly)
    }
}

impl PolynomialId {
    fn into_virtual(self) -> Option<VirtualPoly> {
        match self {
            Self::Virtual(poly) => Some(poly),
            _ => None,
        }
    }

    fn into_committed(self) -> Option<CommittedPoly> {
        match self {
            Self::Committed(poly) => Some(poly),
            _ => None,
        }
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub struct OpeningId {
    pub polynomial: PolynomialId,
    pub sumcheck: SumcheckId,
}

impl OpeningId {
    pub fn new(polynomial: impl Into<PolynomialId>, sumcheck: SumcheckId) -> Self {
        Self {
            polynomial: polynomial.into(),
            sumcheck,
        }
    }

    pub fn virtual_poly(&self) -> Option<VirtualPoly> {
        self.polynomial.into_virtual()
    }

    pub fn committed_poly(&self) -> Option<CommittedPoly> {
        self.polynomial.into_committed()
    }
}

impl CanonicalSerialize for OpeningId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self.polynomial {
            PolynomialId::Committed(polynomial) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                polynomial.serialize_with_mode(&mut writer, compress)?;
                self.sumcheck.serialize_with_mode(&mut writer, compress)?;
            }
            PolynomialId::Virtual(polynomial) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                polynomial.serialize_with_mode(&mut writer, compress)?;
                self.sumcheck.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + self.sumcheck.serialized_size(compress)
            + match self.polynomial {
                PolynomialId::Virtual(polynomial) => polynomial.serialized_size(compress),
                PolynomialId::Committed(polynomial) => polynomial.serialized_size(compress),
            }
    }
}

impl Valid for OpeningId {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for OpeningId {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match tag {
            0 => {
                let poly = CommittedPoly::deserialize_with_mode(&mut reader, compress, validate)?;
                let sc = SumcheckId::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(Self::new(poly, sc))
            }
            1 => {
                let poly = VirtualPoly::deserialize_with_mode(&mut reader, compress, validate)?;
                let sc = SumcheckId::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(Self::new(poly, sc))
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}

#[cfg(test)]
mod guardrail_tests {
    use super::{
        OpeningAccumulator, OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    };
    use crate::{
        subprotocols::evaluation_reduction::ReducedInstance,
        transcripts::{Blake2bTranscript, Transcript},
    };
    use ark_bn254::Fr;
    use common::VirtualPoly;

    #[test]
    #[should_panic(expected = "reduced evaluation for node 7 not found")]
    fn get_node_output_opening_panics_before_reduction() {
        let acc = ProverOpeningAccumulator::<Fr>::new();
        let _ = acc.get_node_output_opening(7);
    }

    #[test]
    #[should_panic(expected = "cannot append opening for node 3 after evaluation reduction")]
    fn append_virtual_panics_after_reduction() {
        let mut acc = ProverOpeningAccumulator::<Fr>::new();
        acc.reduced_evaluations.insert(
            3,
            ReducedInstance {
                r: vec![Fr::from(1u64)],
                claim: Fr::from(2u64),
            },
        );

        let mut transcript = Blake2bTranscript::new(b"guardrail_test");
        let id = OpeningId::new(VirtualPoly::NodeOutput(3), SumcheckId::NodeExecution(9));
        acc.append_virtual(
            &mut transcript,
            id,
            OpeningPoint::new(vec![Fr::from(1u64)]),
            Fr::from(2u64),
        );
    }

    #[should_panic(expected = "cannot append opening for node 3 after evaluation reduction")]
    #[test]
    fn append_virtual_node_output_raf_rejected_after_reduction() {
        let mut acc = ProverOpeningAccumulator::<Fr>::new();
        acc.reduced_evaluations.insert(
            3,
            ReducedInstance {
                r: vec![Fr::from(1u64)],
                claim: Fr::from(2u64),
            },
        );

        let mut transcript = Blake2bTranscript::new(b"guardrail_test");
        let id = OpeningId::new(VirtualPoly::NodeOutput(3), SumcheckId::Raf);
        acc.append_virtual(
            &mut transcript,
            id,
            OpeningPoint::new(vec![Fr::from(1u64)]),
            Fr::from(5u64),
        );

        let key = OpeningId::new(VirtualPoly::NodeOutput(3), SumcheckId::Raf);
        assert!(acc.openings.contains_key(&key));
    }
}
