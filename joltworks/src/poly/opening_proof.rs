//! This is a port of the sumcheck-based batch opening proof protocol implemented
//! in Nova: https://github.com/microsoft/Nova/blob/2772826ba296b66f1cd5deecf7aca3fd1d10e1f4/src/spartan/snark.rs#L410-L424
//! and such code is Copyright (c) Microsoft Corporation.
//! For additively homomorphic commitment schemes (including Zeromorph, HyperKZG) we
//! can use a sumcheck to reduce multiple opening proofs (multiple polynomials, not
//! necessarily of the same size, each opened at a different point) into a single opening.

use allocative::Allocative;
use common::{CommittedPolynomial, VirtualPolynomial};
use num_derive::FromPrimitive;
#[cfg(test)]
use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::{field::JoltField, transcripts::Transcript};

pub type Endianness = bool;
pub const BIG_ENDIAN: Endianness = false;
pub const LITTLE_ENDIAN: Endianness = true;

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
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum OpeningId {
    Committed(CommittedPolynomial, SumcheckId),
    Virtual(VirtualPolynomial, SumcheckId),
    /// Untrusted advice opened at r_address derived from the given sumcheck.
    /// - `RamReadWriteChecking`: opened at r_address from RamVal (used by ValEvaluation)
    /// - `RamOutputCheck`: opened at r_address from RamValFinal (used by ValFinal)
    UntrustedAdvice(SumcheckId),
    /// Trusted advice opened at r_address derived from the given sumcheck.
    /// - `RamReadWriteChecking`: opened at r_address from RamVal (used by ValEvaluation)
    /// - `RamOutputCheck`: opened at r_address from RamValFinal (used by ValFinal)
    TrustedAdvice(SumcheckId),
}

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
    pub openings: Openings<F>,
    #[cfg(test)]
    pub appended_virtual_openings: RefCell<Vec<OpeningId>>,
    pub log_T: usize,
}

/// Accumulates openings encountered by the verifier over the course of Jolt,
/// so that they can all be reduced to a single opening proof verification using sumcheck.
pub struct VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    pub openings: Openings<F>,
    /// In testing, the Jolt verifier may be provided the prover's openings so that we
    /// can detect any places where the openings don't match up.
    #[cfg(test)]
    prover_opening_accumulator: Option<ProverOpeningAccumulator<F>>,
    pub log_T: usize,
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

/// State for Dory batch opening (Stage 8).
/// This is a generic interface for batch opening proofs.
#[derive(Clone, Allocative)]
pub struct DoryOpeningState<F: JoltField> {
    /// Unified opening point for all polynomials (length = log_k_chunk + log_T)
    pub opening_point: Vec<F::Challenge>,
    /// γ^i coefficients for the RLC polynomial
    pub gamma_powers: Vec<F>,
    /// (polynomial, claim) pairs at the opening point
    /// (with Lagrange factors already applied for shorter polys)
    pub polynomial_claims: Vec<(CommittedPolynomial, F)>,
}

impl<F: JoltField> DoryOpeningState<F> {}

impl<F> Default for ProverOpeningAccumulator<F>
where
    F: JoltField,
{
    fn default() -> Self {
        Self::new(0)
    }
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
        #[cfg(test)]
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
    pub fn new(log_T: usize) -> Self {
        Self {
            openings: BTreeMap::new(),
            #[cfg(test)]
            appended_virtual_openings: std::cell::RefCell::new(vec![]),
            log_T,
        }
    }

    pub fn evaluation_openings_mut(&mut self) -> &mut Openings<F> {
        &mut self.openings
    }

    /// Get the value of an opening by key
    pub fn get_opening(&self, key: OpeningId) -> F {
        self.openings.get(&key).unwrap().1
    }

    pub fn get_untrusted_advice_opening(
        &self,
        sumcheck_id: SumcheckId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self
            .openings
            .get(&OpeningId::UntrustedAdvice(sumcheck_id))?;
        Some((point.clone(), *claim))
    }

    pub fn get_trusted_advice_opening(
        &self,
        sumcheck_id: SumcheckId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self.openings.get(&OpeningId::TrustedAdvice(sumcheck_id))?;
        Some((point.clone(), *claim))
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

        // Add opening to map
        let key = OpeningId::Committed(polynomial, sumcheck);
        self.openings.insert(
            key,
            (
                OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
                claim,
            ),
        );
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

        // Add openings to map
        for (label, claim) in polynomials.iter().zip(claims.iter()) {
            let opening_point_struct = OpeningPoint::<BIG_ENDIAN, F>::new(r_concat.clone());
            let key = OpeningId::Committed(*label, sumcheck);
            self.openings
                .insert(key, (opening_point_struct.clone(), *claim));
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
        transcript.append_scalar(&claim);
        assert!(
            self.openings
                .insert(
                    OpeningId::Virtual(polynomial, sumcheck),
                    (opening_point, claim),
                )
                .is_none(),
            "Key ({polynomial:?}, {sumcheck:?}) is already in opening map"
        );
        #[cfg(test)]
        self.appended_virtual_openings
            .borrow_mut()
            .push(OpeningId::Virtual(polynomial, sumcheck));
    }

    pub fn append_untrusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        transcript.append_scalar(&claim);
        self.openings.insert(
            OpeningId::UntrustedAdvice(sumcheck_id),
            (opening_point, claim),
        );
    }

    pub fn append_trusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
        claim: F,
    ) {
        transcript.append_scalar(&claim);
        self.openings.insert(
            OpeningId::TrustedAdvice(sumcheck_id),
            (opening_point, claim),
        );
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
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) {
        let _ = self
            .openings
            .get(&OpeningId::Virtual(polynomial, sumcheck))
            .unwrap_or_else(|| panic!("opening for {sumcheck:?} {polynomial:?} not found"));
    }
}

impl<F> Default for VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    fn default() -> Self {
        Self::new(0)
    }
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
    pub fn new(log_T: usize) -> Self {
        Self {
            openings: BTreeMap::new(),
            #[cfg(test)]
            prover_opening_accumulator: None,
            log_T,
        }
    }

    /// Compare this accumulator to the corresponding `ProverOpeningAccumulator` and panic
    /// if the openings appended differ from the prover's openings.
    #[cfg(test)]
    pub fn compare_to(&mut self, prover_openings: ProverOpeningAccumulator<F>) {
        self.prover_opening_accumulator = Some(prover_openings);
    }

    pub fn get_untrusted_advice_opening(
        &self,
        sumcheck_id: SumcheckId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self
            .openings
            .get(&OpeningId::UntrustedAdvice(sumcheck_id))?;
        Some((point.clone(), *claim))
    }

    pub fn get_trusted_advice_opening(
        &self,
        sumcheck_id: SumcheckId,
    ) -> Option<(OpeningPoint<BIG_ENDIAN, F>, F)> {
        let (point, claim) = self.openings.get(&OpeningId::TrustedAdvice(sumcheck_id))?;
        Some((point.clone(), *claim))
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
            transcript.append_scalar(claim);
            let claim = *claim; // Copy the claim value
            self.openings.insert(key, (opening_point.clone(), claim));
        } else {
            panic!("Tried to populate opening point for non-existent key: {key:?}");
        }
    }

    pub fn append_untrusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let key = OpeningId::UntrustedAdvice(sumcheck_id);
        if let Some((_, claim)) = self.openings.get(&key) {
            transcript.append_scalar(claim);
            let claim = *claim;
            self.openings.insert(key, (opening_point.clone(), claim));
        } else {
            panic!("Tried to populate opening point for non-existent key: {key:?}");
        }
    }

    pub fn append_trusted_advice<T: Transcript>(
        &mut self,
        transcript: &mut T,
        sumcheck_id: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let key = OpeningId::TrustedAdvice(sumcheck_id);
        if let Some((_, claim)) = self.openings.get(&key) {
            transcript.append_scalar(claim);
            let claim = *claim;
            self.openings.insert(key, (opening_point.clone(), claim));
        } else {
            panic!("Tried to populate opening point for non-existent key: {key:?}");
        }
    }
}
