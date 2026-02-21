//! This is a port of the sumcheck-based batch opening proof protocol implemented
//! in Nova: https://github.com/microsoft/Nova/blob/2772826ba296b66f1cd5deecf7aca3fd1d10e1f4/src/spartan/snark.rs#L410-L424
//! and such code is Copyright (c) Microsoft Corporation.
//! For additively homomorphic commitment schemes (including Zeromorph, HyperKZG) we
//! can use a sumcheck to reduce multiple opening proofs (multiple polynomials, not
//! necessarily of the same size, each opened at a different point) into a single opening.

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
    },
    subprotocols::{
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
use atlas_onnx_tracer::node::ComputationNode;
use common::{CommittedPolynomial, VirtualPolynomial};
use num_derive::FromPrimitive;
use num_traits::FromPrimitive as _;
use rayon::prelude::*;

#[cfg(any(test, feature = "test-feature"))]
use std::cell::RefCell;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, RwLock},
};

pub type Endianness = bool;
pub const BIG_ENDIAN: Endianness = false;
pub const LITTLE_ENDIAN: Endianness = true;

/// (point, claim)
pub type Opening<F> = (OpeningPoint<BIG_ENDIAN, F>, F);
pub type Openings<F> = BTreeMap<OpeningId, Opening<F>>;
pub type VirtualOperandClaims<F> = BTreeMap<usize, Vec<F>>;

/// Accumulates openings computed by the prover over the course of Jolt,
/// so that they can all be reduced to a single opening proof using sumcheck.
#[derive(Clone, Allocative)]
pub struct ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    pub sumchecks: BTreeMap<CommittedPolynomial, OpeningProofReductionSumcheckProver<F>>,
    pub openings: Openings<F>,
    dense_polynomial_map: HashMap<CommittedPolynomial, Arc<RwLock<SharedDensePolynomial<F>>>>,
    eq_cycle_map: HashMap<Vec<F::Challenge>, Arc<RwLock<EqCycleState<F>>>>,
    #[cfg(any(test, feature = "test-feature"))]
    pub appended_virtual_openings: RefCell<Vec<OpeningId>>,
    pub cached_opening_claims: BTreeMap<CommittedPolynomial, F>,
    pub virtual_operand_claims: VirtualOperandClaims<F>,
}

/// Accumulates openings encountered by the verifier over the course of Jolt,
/// so that they can all be reduced to a single opening proof verification using sumcheck.
pub struct VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    sumchecks: BTreeMap<CommittedPolynomial, OpeningProofReductionSumcheckVerifier<F>>,
    pub openings: Openings<F>,
    pub virtual_operand_claims: VirtualOperandClaims<F>,
    /// In testing, the Jolt verifier may be provided the prover's openings so that we
    /// can detect any places where the openings don't match up.
    #[cfg(any(test, feature = "test-feature"))]
    prover_opening_accumulator: Option<ProverOpeningAccumulator<F>>,
}

pub trait OpeningAccumulator<F: JoltField> {
    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F);

    fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F);
}

impl<F: JoltField> OpeningAccumulator<F> for ProverOpeningAccumulator<F> {
    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        let (point, claim) = self
            .openings
            .get(&OpeningId::Virtual(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("opening for {sumcheck:?} {polynomial:?} not found"));
        #[cfg(any(test, feature = "test-feature"))]
        {
            let mut virtual_openings = self.appended_virtual_openings.borrow_mut();
            if let Some(index) = virtual_openings
                .iter()
                .position(|id| id == &OpeningId::Virtual(polynomial, sumcheck))
            {
                virtual_openings.remove(index);
            }
        }
        (point.clone(), *claim)
    }

    fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        let (point, claim) = self
            .openings
            .get(&OpeningId::Committed(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("opening for {sumcheck:?} {polynomial:?} not found"));
        (point.clone(), *claim)
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
            eq_cycle_map: HashMap::new(),
            dense_polynomial_map: HashMap::new(),
            #[cfg(any(test, feature = "test-feature"))]
            appended_virtual_openings: std::cell::RefCell::new(vec![]),
            cached_opening_claims: BTreeMap::new(),
            virtual_operand_claims: BTreeMap::new(),
        }
    }

    /// Caches an opening claim from the opening reduction sumcheck.
    /// Called from `OpeningProofReductionSumcheckProver::cache_openings`.
    pub fn cache_opening_reduction_claim(&mut self, polynomial: CommittedPolynomial, claim: F) {
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

    /// Adds an opening of a dense polynomial to the accumulator.
    /// The given `polynomial` is opened at `opening_point`, yielding the claimed
    /// evaluation `claim`.
    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_dense")]
    pub fn append_dense<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
        claim: F,
    ) {
        transcript.append_scalar(&claim);

        let shared_eq = self
            .eq_cycle_map
            .entry(opening_point.clone())
            .or_insert_with(|| Arc::new(RwLock::new(EqCycleState::new(&opening_point))));

        // Add opening to map
        let key = OpeningId::Committed(polynomial, sumcheck);
        self.openings.insert(
            key,
            (
                OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
                claim,
            ),
        );

        let sumcheck = OpeningProofReductionSumcheckProver::new_dense(
            polynomial,
            sumcheck,
            shared_eq.clone(),
            opening_point,
            claim,
        );
        self.sumchecks.insert(polynomial, sumcheck);
    }

    #[tracing::instrument(skip_all, name = "ProverOpeningAccumulator::append_sparse")]
    pub fn append_sparse<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        r_address: Vec<F::Challenge>,
        r_cycle: Vec<F::Challenge>,
        claims: Vec<F>,
    ) {
        claims.iter().for_each(|claim| {
            transcript.append_scalar(claim);
        });
        let r_concat = [r_address.as_slice(), r_cycle.as_slice()].concat();

        let shared_eq_address = Arc::new(RwLock::new(EqAddressState::new(&r_address)));
        let shared_eq_cycle = self
            .eq_cycle_map
            .entry(r_cycle.clone())
            .or_insert(Arc::new(RwLock::new(EqCycleState::new(&r_cycle))));

        // Add openings to map
        for (label, claim) in polynomials.iter().zip(claims.iter()) {
            let opening_point_struct = OpeningPoint::<BIG_ENDIAN, F>::new(r_concat.clone());
            let key = OpeningId::Committed(*label, sumcheck);
            self.openings
                .insert(key, (opening_point_struct.clone(), *claim));
        }

        for (label, claim) in polynomials.into_iter().zip(claims.into_iter()) {
            let sumcheck = OpeningProofReductionSumcheckProver::new_one_hot(
                label,
                sumcheck,
                shared_eq_address.clone(),
                shared_eq_cycle.clone(),
                r_concat.clone(),
                claim,
            );
            self.sumchecks.insert(label, sumcheck);
        }
    }

    pub fn append_virtual<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        // Don't add to transcript if node_output_poly as we will append in cache operand claims
        if !matches!(polynomial, VirtualPolynomial::NodeOutput(_)) {
            transcript.append_scalar(&claim);
        }

        // assert!(
        //     self.openings
        //         .insert(
        //             OpeningId::Virtual(polynomial, sumcheck),
        //             (opening_point, claim),
        //         )
        //         .is_none(),
        //     "Key ({polynomial:?}, {sumcheck:?}) is already in opening map"
        // );

        // TODO: Allow a node to have multiple openings that need to be reduced
        //       See #138 for details
        self.openings.insert(
            OpeningId::Virtual(polynomial, sumcheck),
            (opening_point, claim),
        );
        #[cfg(test)]
        self.appended_virtual_openings
            .borrow_mut()
            .push(OpeningId::Virtual(polynomial, sumcheck));
    }

    pub fn cache_virtual_operand_claims<T: Transcript>(
        &mut self,
        transcript: &mut T,
        computation_node: &ComputationNode,
    ) {
        let mut operand_claims = vec![];
        for operand_node_index in computation_node.inputs.iter() {
            let claim = self.get_opening(OpeningId::Virtual(
                VirtualPolynomial::NodeOutput(*operand_node_index),
                SumcheckId::Execution,
            ));
            transcript.append_scalar(&claim);
            operand_claims.push(claim);
        }
        self.virtual_operand_claims
            .insert(computation_node.idx, operand_claims);
    }

    /// Take the openings, removing the points to reduce proof size
    ///
    /// # Returns:
    /// `Openings<F>` - The openings with points removed
    /// `VirtualOperandClaims<F>` - The virtual operand claims that were cached separately
    pub fn take(&mut self) -> (Openings<F>, VirtualOperandClaims<F>) {
        // to reduce proof size, remove all the opening points from accumulator.openings and just leave the claims
        for (_, opening) in self.openings.iter_mut() {
            opening.0.r.clear();
        }
        (
            std::mem::take(&mut self.openings),
            std::mem::take(&mut self.virtual_operand_claims),
        )
    }

    pub fn assert_virtual_polynomial_opening_exists(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> Option<&(OpeningPoint<false, F>, F)> {
        self.openings.get(&OpeningId::Virtual(polynomial, sumcheck))
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
        polynomials: &BTreeMap<CommittedPolynomial, MultilinearPolynomial<F>>,
    ) {
        if self.sumchecks.len() != polynomials.len() {
            let missing_sumchecks: Vec<CommittedPolynomial> = polynomials
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
        // Extract claims and polynomials from cached opening claims (populated by cache_openings)
        let (polynomials, sumcheck_claims): (Vec<CommittedPolynomial>, Vec<F>) =
            std::mem::take(&mut self.cached_opening_claims)
                .into_iter()
                .unzip();

        // Append claims and derive gamma powers
        transcript.append_scalars(&sumcheck_claims);
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
    pub polynomials: Vec<CommittedPolynomial>,
}

impl<F: JoltField> OpeningAccumulator<F> for VerifierOpeningAccumulator<F> {
    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        let (point, claim) = self
            .openings
            .get(&OpeningId::Virtual(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("No opening found for {sumcheck:?} {polynomial:?}"));
        (point.clone(), *claim)
    }

    fn get_committed_polynomial_opening(
        &self,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F) {
        let (point, claim) = self
            .openings
            .get(&OpeningId::Committed(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("No opening found for {sumcheck:?} {polynomial:?}"));
        (point.clone(), *claim)
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
            virtual_operand_claims: BTreeMap::new(),
            #[cfg(any(test, feature = "test-feature"))]
            prover_opening_accumulator: None,
        }
    }

    pub fn append_operand_claims<T: Transcript>(&self, transcript: &mut T, node_index: usize) {
        let claims = self.operand_claims(node_index);
        claims.iter().for_each(|claim| {
            transcript.append_scalar(claim);
        });
    }

    pub fn get_operand_claims<const NUM_OPERANDS: usize>(
        &self,
        node_index: usize,
    ) -> [F; NUM_OPERANDS] {
        self.operand_claims(node_index)
            .try_into()
            .unwrap_or_else(|claims: Vec<F>| {
                panic!("Expected {NUM_OPERANDS} operand claims for node index {node_index}, but found {}", claims.len())
            })
    }

    pub fn operand_claims(&self, node_index: usize) -> Vec<F> {
        self.virtual_operand_claims
            .get(&node_index)
            .unwrap_or_else(|| {
                panic!("No virtual operand claims found for node index {node_index}")
            })
            .clone()
    }

    /// Compare this accumulator to the corresponding `ProverOpeningAccumulator` and panic
    /// if the openings appended differ from the prover's openings.
    #[cfg(any(test, feature = "test-feature"))]
    pub fn compare_to(&mut self, prover_openings: ProverOpeningAccumulator<F>) {
        self.prover_opening_accumulator = Some(prover_openings);
    }

    /// Adds an opening of a dense polynomial the accumulator.
    /// The given `polynomial` is opened at `opening_point`.
    pub fn append_dense<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: CommittedPolynomial,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
    ) {
        let key = OpeningId::Committed(polynomial, sumcheck);
        let claim = self.openings.get(&key).unwrap().1;
        transcript.append_scalar(&claim);

        // Update the opening point in self.openings (it was initialized with default empty point)
        self.openings.insert(
            key,
            (
                OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
                claim,
            ),
        );

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
    pub fn append_sparse<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPolynomial>,
        sumcheck: SumcheckId,
        opening_point: Vec<F::Challenge>,
    ) {
        for label in polynomials.into_iter() {
            let key = OpeningId::Committed(label, sumcheck);
            let claim = self.openings.get(&key).unwrap().1;
            transcript.append_scalar(&claim);

            // Update the opening point in self.openings (it was initialized with default empty point)
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
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let key = OpeningId::Virtual(polynomial, sumcheck);
        if let Some((_, claim)) = self.openings.get(&key) {
            // Don't add to transcript if node_output_poly as we will append in append operand claims
            if !matches!(polynomial, VirtualPolynomial::NodeOutput(_)) {
                transcript.append_scalar(claim);
            }
            let claim = *claim; // Copy the claim value
            self.openings.insert(key, (opening_point.clone(), claim));
        } else {
            panic!("Tried to populate opening point for non-existent key: {key:?}");
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
        let polynomials: Vec<CommittedPolynomial> =
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
        commitment_map: &mut BTreeMap<CommittedPolynomial, PCS::Commitment>,
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

#[derive(Clone, Debug, PartialEq, Default, Allocative)]
pub struct OpeningPoint<const E: Endianness, F: JoltField> {
    pub r: Vec<F::Challenge>,
}

impl<const E: Endianness, F: JoltField> std::ops::Index<usize> for OpeningPoint<E, F> {
    type Output = F::Challenge;

    fn index(&self, index: usize) -> &Self::Output {
        &self.r[index]
    }
}

impl<const E: Endianness, F: JoltField> std::ops::Index<std::ops::RangeFull>
    for OpeningPoint<E, F>
{
    type Output = [F::Challenge];

    fn index(&self, _index: std::ops::RangeFull) -> &Self::Output {
        &self.r[..]
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn len(&self) -> usize {
        self.r.len()
    }

    pub fn split_at_r(&self, mid: usize) -> (&[F::Challenge], &[F::Challenge]) {
        self.r.split_at(mid)
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        let (left, right) = self.r.split_at(mid);
        (Self::new(left.to_vec()), Self::new(right.to_vec()))
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn new(r: Vec<F::Challenge>) -> Self {
        Self { r }
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

impl<F: JoltField> From<Vec<F::Challenge>> for OpeningPoint<LITTLE_ENDIAN, F> {
    fn from(r: Vec<F::Challenge>) -> Self {
        Self::new(r)
    }
}

impl<F: JoltField> From<Vec<F::Challenge>> for OpeningPoint<BIG_ENDIAN, F> {
    fn from(r: Vec<F::Challenge>) -> Self {
        Self::new(r)
    }
}

impl<const E: Endianness, F: JoltField> Into<Vec<F::Challenge>> for OpeningPoint<E, F> {
    fn into(self) -> Vec<F::Challenge> {
        self.r
    }
}

impl<const E: Endianness, F: JoltField> Into<Vec<F::Challenge>> for &OpeningPoint<E, F>
where
    F: Clone,
{
    fn into(self) -> Vec<F::Challenge> {
        self.r.clone()
    }
}

#[derive(
    Hash,
    PartialEq,
    Eq,
    Copy,
    Clone,
    Debug,
    PartialOrd,
    Ord,
    FromPrimitive,
    Allocative,
    strum_macros::EnumCount,
)]
#[repr(u8)]
pub enum SumcheckId {
    Execution,
    Raf,
    RaVirtualization,
    RamHammingBooleanity,
    RamHammingWeight,
    Booleanity,
    HammingWeight,
}

impl CanonicalSerialize for SumcheckId {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (*self as u8).serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        (*self as u8).serialized_size(compress)
    }
}

impl Valid for SumcheckId {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for SumcheckId {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let v = u8::deserialize_with_mode(reader, compress, validate)?;
        Self::from_u8(v).ok_or(SerializationError::InvalidData)
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum OpeningId {
    Committed(CommittedPolynomial, SumcheckId),
    Virtual(VirtualPolynomial, SumcheckId),
}

impl CanonicalSerialize for OpeningId {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::Committed(poly, sc) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                poly.serialize_with_mode(&mut writer, compress)?;
                sc.serialize_with_mode(&mut writer, compress)?;
            }
            Self::Virtual(poly, sc) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                poly.serialize_with_mode(&mut writer, compress)?;
                sc.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            Self::Committed(poly, sc) => {
                poly.serialized_size(compress) + sc.serialized_size(compress)
            }
            Self::Virtual(poly, sc) => {
                poly.serialized_size(compress) + sc.serialized_size(compress)
            }
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
                let poly =
                    CommittedPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                let sc = SumcheckId::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(Self::Committed(poly, sc))
            }
            1 => {
                let poly =
                    VirtualPolynomial::deserialize_with_mode(&mut reader, compress, validate)?;
                let sc = SumcheckId::deserialize_with_mode(&mut reader, compress, validate)?;
                Ok(Self::Virtual(poly, sc))
            }
            _ => Err(SerializationError::InvalidData),
        }
    }
}
