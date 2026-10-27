//! This is a port of the sumcheck-based batch opening proof protocol implemented
//! in Nova: https://github.com/microsoft/Nova/blob/2772826ba296b66f1cd5deecf7aca3fd1d10e1f4/src/spartan/snark.rs#L410-L424
//! and such code is Copyright (c) Microsoft Corporation.
//! A sumcheck reduces multiple opening proofs (multiple polynomials, not
//! necessarily of the same size, each opened at a different point) into a single opening.

use crate::{
    evaluation_reduction::ReducedInstance,
    opening_reduction::OpeningReductionGroup,
    sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
    sumcheck_verifier::SumcheckInstanceVerifier,
};
use allocative::Allocative;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_transcript::Transcript;
use common::{CommittedPoly, VirtualPoly};
use rayon::prelude::*;

use std::collections::{BTreeMap, BTreeSet};

pub use atlas_algebra::poly::opening_point::{Endianness, OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN};

/// (point, claim)
pub type Opening<F> = (OpeningPoint<BIG_ENDIAN, F>, F);
pub type Openings<F> = BTreeMap<OpeningId, Opening<F>>;

/// Test-only opening forgery hook (feature `test-feature`).
///
/// Soundness sweeps set a target [`OpeningId`]; every claim the prover
/// appends under that id (dense, sparse or virtual) is bumped by one before
/// it reaches the transcript, so the forged value flows into the prover's
/// own Fiat-Shamir state and downstream claims exactly as a malicious prover
/// would arrange. The verifier must still reject. This is the single choke
/// point every opening passes through, including those appended from inside
/// the subprotocols.
#[cfg(any(test, feature = "test-feature"))]
pub mod forgery {
    use super::OpeningId;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        RwLock,
    };

    static TARGET: RwLock<Option<OpeningId>> = RwLock::new(None);
    static HITS: AtomicUsize = AtomicUsize::new(0);

    /// Target an opening id (or clear with `None`); resets the hit counter.
    pub fn set(target: Option<OpeningId>) {
        *TARGET.write().unwrap() = target;
        HITS.store(0, Ordering::SeqCst);
    }

    /// How many appended claims were forged since the last [`set`].
    pub fn hits() -> usize {
        HITS.load(Ordering::SeqCst)
    }

    /// The claim to actually append for `id`.
    pub fn apply<F: atlas_algebra::field::JoltField>(id: OpeningId, claim: F) -> F {
        if *TARGET.read().unwrap() == Some(id) {
            HITS.fetch_add(1, Ordering::SeqCst);
            claim + F::one()
        } else {
            claim
        }
    }
}

/// Accumulates openings encountered by the verifier over the course of Jolt,
/// so that they can all be reduced to a single opening proof verification using sumcheck.
pub struct VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    /// See `ProverOpeningAccumulator::sumchecks`.
    sumchecks: BTreeMap<OpeningId, OpeningReductionGroup<F>>,
    pub openings: Openings<F>,
    /// Ids bound by an `append_*` call, to reject one id bound at two
    /// different points (see [`Self::mark_appended`]).
    appended: BTreeSet<OpeningId>,
    /// Mapping for nodes reduced opening points
    pub reduced_evaluations: BTreeMap<usize, ReducedInstance<F>>,
    /// In testing, the Jolt verifier may be provided the prover's openings so that we
    /// can detect any places where the openings don't match up.
    #[cfg(any(test, feature = "test-feature"))]
    prover_opening_accumulator: Option<prover::ProverOpeningAccumulator<F>>,
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

/// Intermediate state between the batch opening reduction sumcheck and the joint PCS opening.
/// Stored in prover/verifier state to bridge the two stages.
#[derive(Clone, Allocative)]
pub struct OpeningReductionState<F: JoltField> {
    pub r_sumcheck: Vec<F::Challenge>,
    /// Per-instance (group) batching coefficients.
    pub gamma_powers: Vec<F>,
    /// Per-instance reduced claims.
    pub sumcheck_claims: Vec<F>,
    /// The distinct opened polynomials, sorted …
    pub polynomials: Vec<CommittedPoly>,
    /// … and their coefficients in the joint RLC (aligned with `polynomials`).
    pub poly_coeffs: Vec<F>,
}

/// Per-polynomial coefficients of the joint RLC from the instances'
/// `(polynomials, in-group coefficients)` and their batching coefficients.
pub fn joint_coefficients<'a, F: JoltField>(
    instances: impl Iterator<Item = (&'a Vec<CommittedPoly>, &'a Vec<F>)>,
    gamma_powers: &[F],
) -> (Vec<CommittedPoly>, Vec<F>) {
    let mut map: BTreeMap<CommittedPoly, F> = BTreeMap::new();
    for ((polys, coeffs), gamma) in instances.zip(gamma_powers) {
        for (p, c) in polys.iter().zip(coeffs) {
            *map.entry(*p).or_insert(F::zero()) += *gamma * *c;
        }
    }
    map.into_iter().unzip()
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
            appended: BTreeSet::new(),
            reduced_evaluations: BTreeMap::new(),
            #[cfg(any(test, feature = "test-feature"))]
            prover_opening_accumulator: None,
        }
    }

    /// Record that `opening_id` was bound at `new_point`. A re-append is only
    /// legal when it re-binds the same point (the same opening registered by
    /// two instances of one sumcheck batch, e.g. `SoftmaxExpQ` from both
    /// RecipMult and ExpSum): binding one id at two different points would
    /// silently discard the first binding, leaving a claim some check believes
    /// it constrained but the accumulator no longer holds. Appends are driven
    /// by verifier code (the model), never by proof data, so a violation is a
    /// code bug and panics.
    fn mark_appended(&mut self, opening_id: OpeningId, new_point: &[F]) {
        if !self.appended.insert(opening_id) {
            if let Some((existing, _)) = self.openings.get(&opening_id) {
                assert_eq!(
                    existing.r, new_point,
                    "opening {opening_id:?} appended twice at different points"
                );
            }
        }
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
        let point = OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone());
        self.mark_appended(opening_id, &point.r);
        let Some(&(_, claim)) = self.openings.get(&opening_id) else {
            panic!("Tried to populate dense opening for non-existent key: {opening_id:?}");
        };
        // The claim was pre-loaded from the proof.
        transcript.append_scalar(&claim);
        self.openings.insert(opening_id, (point, claim));

        let polynomial = opening_id
            .committed_poly()
            .expect("expected committed polynomial");
        self.sumchecks.insert(
            opening_id,
            OpeningReductionGroup::new(vec![polynomial], opening_id.sumcheck, opening_point),
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
        let point_f: Vec<F> = opening_point.iter().map(|u| (*u).into()).collect();
        for label in polynomials.iter().copied() {
            let key = OpeningId::new(label, sumcheck);
            self.mark_appended(key, &point_f);
            let claim = if let Some((_, c)) = self.openings.get(&key) {
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
        }
        let key = OpeningId::new(polynomials[0], sumcheck);
        self.sumchecks.insert(
            key,
            OpeningReductionGroup::new(polynomials, sumcheck, opening_point),
        );
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

        if self.openings.contains_key(&opening_id) {
            self.mark_appended(opening_id, &opening_point.r);
            // The claim was pre-loaded from the proof. Append to transcript.
            let (_, claim) = &self.openings[&opening_id];
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

    /// The distinct committed polynomials with a registered opening, sorted —
    /// the order of the proof's reduced claims and of the prover's
    /// `state.polynomials` from `finalize_batch_opening_sumcheck`.
    /// Committed polynomials without an opening are not part of the joint
    /// opening.
    pub fn sumchecks_keys(&self) -> impl Iterator<Item = CommittedPoly> + '_ {
        let polys: BTreeSet<CommittedPoly> = self
            .sumchecks
            .values()
            .flat_map(|s| s.polynomials.iter().copied())
            .collect();
        polys.into_iter()
    }

    // ========== Batch Opening Reduction Sumcheck ==========

    #[tracing::instrument(skip_all, name = "VerifierOpeningAccumulator::prepare_for_sumcheck")]
    /// Prepares the verifier for the batch opening reduction sumcheck: draws
    /// the in-group coefficient and populates the per-instance claims from the proof.
    pub fn prepare_for_sumcheck<T: Transcript>(
        &mut self,
        sumcheck_claims: &[F],
        transcript: &mut T,
    ) -> Result<(), ProofVerifyError> {
        let rho: F = transcript.challenge_scalar();
        if sumcheck_claims.len() != self.sumchecks.len() {
            return Err(ProofVerifyError::InvalidOpeningProof(format!(
                "expected {} reduced claims, found {}",
                self.sumchecks.len(),
                sumcheck_claims.len()
            )));
        }
        self.sumchecks
            .values_mut()
            .zip(sumcheck_claims)
            .for_each(|(opening, claim)| {
                opening.set_coefficients(rho);
                opening.sumcheck_claim = Some(*claim);
            });
        Ok(())
    }

    /// Verifies the batch opening reduction sumcheck (Stage 7).
    #[tracing::instrument(
        skip_all,
        name = "VerifierOpeningAccumulator::verify_batch_opening_sumcheck"
    )]
    pub fn verify_batch_opening_sumcheck<T: Transcript>(
        &mut self,
        sumcheck_proof: &SumcheckInstanceProof<F>,
        transcript: &mut T,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let verifiers: Vec<_> = self
            .sumchecks
            .values()
            .map(|group| group.verifier())
            .collect();
        let instances: Vec<&dyn SumcheckInstanceVerifier<F, T>> = verifiers
            .iter()
            .map(|verifier| verifier as &dyn SumcheckInstanceVerifier<F, T>)
            .collect();
        BatchedSumcheck::verify(sumcheck_proof, instances, self, transcript)
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
        // Append claims and derive per-instance gamma powers
        transcript.append_scalars(sumcheck_claims);
        let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(self.sumchecks.len());
        let (polynomials, poly_coeffs) = joint_coefficients(
            self.sumchecks
                .values()
                .map(|s| (&s.polynomials, &s.coefficients)),
            &gamma_powers,
        );

        OpeningReductionState {
            r_sumcheck,
            gamma_powers,
            sumcheck_claims: sumcheck_claims.to_vec(),
            polynomials,
            poly_coeffs,
        }
    }

    // ========== Batch Opening Verification ==========

    /// Computes the joint claim for the batch opening verification.
    pub fn compute_joint_claim(&self, state: &OpeningReductionState<F>) -> F {
        let num_sumcheck_rounds = self
            .sumchecks
            .values()
            .map(OpeningReductionGroup::num_vars)
            .max()
            .unwrap();

        state
            .gamma_powers
            .iter()
            .zip(state.sumcheck_claims.iter())
            .zip(self.sumchecks.values())
            .map(|((coeff, claim), opening)| {
                let r_slice = &state.r_sumcheck[..num_sumcheck_rounds - opening.num_vars()];
                let lagrange_eval: F = r_slice.iter().map(|r| F::one() - r).product();
                *coeff * claim * lagrange_eval
            })
            .sum()
    }

    /// The variable count of each opened polynomial (aligned with `state.polynomials`): the round
    /// count of the reduction instances that open it.
    pub fn opened_num_vars(&self, state: &OpeningReductionState<F>) -> Vec<usize> {
        let mut num_vars = BTreeMap::new();
        for opening in self.sumchecks.values() {
            let rounds = opening.num_vars();
            for poly in &opening.polynomials {
                num_vars.insert(*poly, rounds);
            }
        }
        state.polynomials.iter().map(|p| num_vars[p]).collect()
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
    /// Eval-shift sumcheck used in neural teleport ops
    NTEvalShift,
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
            Self::NTEvalShift => 9u8.serialize_with_mode(&mut writer, compress)?,
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
            9 => Ok(Self::NTEvalShift),
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
        prover::ProverOpeningAccumulator, OpeningAccumulator, OpeningId, OpeningPoint, SumcheckId,
    };
    use crate::evaluation_reduction::ReducedInstance;
    use atlas_algebra::field::fp128::Fp128 as Fr;
    use atlas_transcript::{Blake2bProver, ProvingTranscript};
    use common::VirtualPoly;
    use jolt_field::Ring as _;

    #[test]
    #[should_panic(expected = "reduced evaluation for node 7 not found")]
    fn get_node_output_opening_panics_before_reduction() {
        let acc = ProverOpeningAccumulator::<Fr>::new();
        let _ = acc.get_node_output_opening(7);
    }

    /// One opening id bound at two different points would silently discard
    /// the first binding, leaving a check constraining a claim the
    /// accumulator no longer holds.
    #[test]
    #[should_panic(expected = "appended twice at different points")]
    fn verifier_append_rejects_rebinding_at_a_different_point() {
        let mut acc = super::VerifierOpeningAccumulator::<Fr>::new();
        let id = OpeningId::new(
            VirtualPoly::HammingWeight(0),
            SumcheckId::RamHammingBooleanity,
        );
        acc.openings
            .insert(id, (OpeningPoint::default(), Fr::from_u64(1u64)));

        let mut transcript = Blake2bProver::new(b"guardrail_test");
        acc.append_virtual(
            &mut transcript,
            id,
            OpeningPoint::new(vec![Fr::from_u64(1u64), Fr::from_u64(2u64)]),
        );
        acc.append_virtual(
            &mut transcript,
            id,
            OpeningPoint::new(vec![Fr::from_u64(3u64), Fr::from_u64(4u64)]),
        );
    }

    /// Re-binding the same id at the same point is legal (two instances of
    /// one batch registering a shared opening).
    #[test]
    fn verifier_append_allows_rebinding_at_the_same_point() {
        let mut acc = super::VerifierOpeningAccumulator::<Fr>::new();
        let id = OpeningId::new(
            VirtualPoly::HammingWeight(0),
            SumcheckId::RamHammingBooleanity,
        );
        acc.openings
            .insert(id, (OpeningPoint::default(), Fr::from_u64(1u64)));

        let mut transcript = Blake2bProver::new(b"guardrail_test");
        let point = OpeningPoint::new(vec![Fr::from_u64(1u64), Fr::from_u64(2u64)]);
        acc.append_virtual(&mut transcript, id, point.clone());
        acc.append_virtual(&mut transcript, id, point);
    }

    #[test]
    #[should_panic(expected = "cannot append opening for node 3 after evaluation reduction")]
    fn append_virtual_panics_after_reduction() {
        let mut acc = ProverOpeningAccumulator::<Fr>::new();
        acc.reduced_evaluations.insert(
            3,
            ReducedInstance {
                r: vec![Fr::from_u64(1u64)],
                claim: Fr::from_u64(2u64),
            },
        );

        let mut transcript = Blake2bProver::new(b"guardrail_test");
        let id = OpeningId::new(VirtualPoly::NodeOutput(3), SumcheckId::NodeExecution(9));
        acc.append_virtual(
            &mut transcript,
            id,
            OpeningPoint::new(vec![Fr::from_u64(1u64)]),
            Fr::from_u64(2u64),
        );
    }

    #[should_panic(expected = "cannot append opening for node 3 after evaluation reduction")]
    #[test]
    fn append_virtual_node_output_raf_rejected_after_reduction() {
        let mut acc = ProverOpeningAccumulator::<Fr>::new();
        acc.reduced_evaluations.insert(
            3,
            ReducedInstance {
                r: vec![Fr::from_u64(1u64)],
                claim: Fr::from_u64(2u64),
            },
        );

        let mut transcript = Blake2bProver::new(b"guardrail_test");
        let id = OpeningId::new(VirtualPoly::NodeOutput(3), SumcheckId::Raf);
        acc.append_virtual(
            &mut transcript,
            id,
            OpeningPoint::new(vec![Fr::from_u64(1u64)]),
            Fr::from_u64(5u64),
        );

        let key = OpeningId::new(VirtualPoly::NodeOutput(3), SumcheckId::Raf);
        assert!(acc.openings.contains_key(&key));
    }
}

pub mod prover;
