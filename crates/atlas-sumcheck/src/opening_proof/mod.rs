//! This is a port of the sumcheck-based batch opening proof protocol implemented
//! in Nova: https://github.com/microsoft/Nova/blob/2772826ba296b66f1cd5deecf7aca3fd1d10e1f4/src/spartan/snark.rs#L410-L424
//! and such code is Copyright (c) Microsoft Corporation.
//! A sumcheck reduces multiple opening proofs (multiple polynomials, not
//! necessarily of the same size, each opened at a different point) into a single opening.

use crate::relation::{verify_relation_batch, DynRelationVerifier};
use crate::{evaluation_reduction::ReducedInstance, opening_reduction::OpeningReductionGroup};
use allocative::Allocative;
use atlas_algebra::serialize::{Canonical, Error};
use atlas_algebra::{field::JoltField, utils::errors::ProofVerifyError};
use atlas_transcript::{Transcript, VerifyingTranscript};
use common::{CommittedPoly, VirtualPoly};
use rayon::prelude::*;
use std::io::{Read, Write};

use std::collections::{BTreeMap, BTreeSet};

pub use atlas_algebra::poly::point::{Endianness, OpeningPoint, BIG_ENDIAN, LITTLE_ENDIAN};

/// (point, claim)
pub type Opening<F> = (OpeningPoint<BIG_ENDIAN, F>, F);
pub type Openings<F> = BTreeMap<OpeningId, Opening<F>>;

/// Test-only opening forgery hook (feature `test-feature`).
///
/// Soundness sweeps set a target [`OpeningId`]; every claim the prover
/// appends under that id (dense, sparse or virtual) is bumped by one before
/// it is sent, so the forged value flows into the proof, the prover's
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
///
/// A claim enters the accumulator when an `append_*` call binds its id to a point: the call
/// reads the claim from the transcript, which absorbs it. The accumulator holds no claim that
/// was not bound, and the stream carries each claim once.
pub struct VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    /// See `ProverOpeningAccumulator::sumchecks`.
    sumchecks: BTreeMap<OpeningId, OpeningReductionGroup<F>>,
    /// The claims bound so far, each at its point.
    openings: Openings<F>,
    /// Mapping for nodes reduced opening points
    pub reduced_evaluations: BTreeMap<usize, ReducedInstance<F>>,
    /// The nodes whose entry in `reduced_evaluations` is a claim at a point of the node's
    /// own, not a reduction of its consumers' claims: a consumer may append its claim later.
    pub own_points: BTreeSet<usize>,
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

    /// The opening of a virtual polynomial; `Err` if the accumulator holds no claim for it.
    fn get_virtual_polynomial_opening(
        &self,
        opening_id: OpeningId,
    ) -> Result<(OpeningPoint<BIG_ENDIAN, F>, F), ProofVerifyError>;

    /// The opening of a committed polynomial; `Err` if the accumulator holds no claim for it.
    fn get_committed_polynomial_opening(
        &self,
        opening_id: OpeningId,
    ) -> Result<(OpeningPoint<BIG_ENDIAN, F>, F), ProofVerifyError>;

    /// Gets the reduced opening claim for the given node index.
    /// The reduced opening claim is either the unique existing opening claim,
    /// Or obtained from the evaluation reduction protocol if multiple openings need to be reduced.
    /// `Err` if the node's output has not been reduced.
    fn get_node_output_opening(
        &self,
        node_idx: usize,
    ) -> Result<(OpeningPoint<BIG_ENDIAN, F>, F), ProofVerifyError>;

    /// Every claim appended on node `node_idx`'s output, each with its point, in opening-id
    /// order: its consumers' claims.
    fn node_output_claims(&self, node_idx: usize) -> Vec<(Vec<F>, F)>;
}

/// The error of a claim the accumulator does not hold.
pub fn missing_claim(opening_id: OpeningId) -> ProofVerifyError {
    ProofVerifyError::MissingClaim(format!("{opening_id:?}"))
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
    ) -> Result<(OpeningPoint<BIG_ENDIAN, F>, F), ProofVerifyError> {
        if opening_id.virtual_poly().is_none() {
            return Err(missing_claim(opening_id));
        }
        self.openings
            .get(&opening_id)
            .map(|(point, claim)| (point.clone(), *claim))
            .ok_or_else(|| missing_claim(opening_id))
    }

    fn get_committed_polynomial_opening(
        &self,
        opening_id: OpeningId,
    ) -> Result<(OpeningPoint<BIG_ENDIAN, F>, F), ProofVerifyError> {
        if opening_id.committed_poly().is_none() {
            return Err(missing_claim(opening_id));
        }
        self.openings
            .get(&opening_id)
            .map(|(point, claim)| (point.clone(), *claim))
            .ok_or_else(|| missing_claim(opening_id))
    }

    fn get_node_output_opening(
        &self,
        node_idx: usize,
    ) -> Result<(OpeningPoint<BIG_ENDIAN, F>, F), ProofVerifyError> {
        let reduced_instance = self.reduced_evaluations.get(&node_idx).ok_or_else(|| {
            ProofVerifyError::MissingClaim(format!("reduced evaluation of node {node_idx}"))
        })?;
        Ok((reduced_instance.r.clone().into(), reduced_instance.claim))
    }

    fn node_output_claims(&self, node_idx: usize) -> Vec<(Vec<F>, F)> {
        self.get_node_openings(node_idx)
            .into_iter()
            .map(|(point, claim)| (point.r.clone(), *claim))
            .collect()
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
            own_points: BTreeSet::new(),
            #[cfg(any(test, feature = "test-feature"))]
            prover_opening_accumulator: None,
        }
    }

    /// The claims bound so far, each at its point.
    pub fn openings(&self) -> &Openings<F> {
        &self.openings
    }

    /// Bind `opening_id` at `point`: read its claim from the transcript and hold it there.
    ///
    /// An id is bound once. A second bind at the same point (one opening named twice by a
    /// relation whose operands are one node, or by two relations of one step) reads nothing:
    /// the stream carries the claim once and every reader sees that claim. A second bind at
    /// another point is an `Err`: it would replace a claim some check has already read.
    fn bind<T: VerifyingTranscript>(
        &mut self,
        transcript: &mut T,
        opening_id: OpeningId,
        point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Result<(), ProofVerifyError> {
        if let Some((bound, _)) = self.openings.get(&opening_id) {
            return if bound.r == point.r {
                Ok(())
            } else {
                Err(ProofVerifyError::Malformed(format!(
                    "opening {opening_id:?} is bound at two points"
                )))
            };
        }
        let claim = transcript.read_scalar()?;
        self.openings.insert(opening_id, (point, claim));
        Ok(())
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

    /// Adds an opening of a dense polynomial to the accumulator: `polynomial` is opened at
    /// `opening_point`, and its claim is read from the transcript. `Err` on a failed read.
    pub fn append_dense<T, U>(
        &mut self,
        transcript: &mut T,
        opening_id: OpeningId,
        opening_point: Vec<U>,
    ) -> Result<(), ProofVerifyError>
    where
        T: VerifyingTranscript,
        U: Copy + Send + Sync + Into<F>,
    {
        let polynomial = opening_id
            .committed_poly()
            .ok_or_else(|| missing_claim(opening_id))?;
        let point = OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone());
        self.bind(transcript, opening_id, point)?;

        self.sumchecks.insert(
            opening_id,
            OpeningReductionGroup::new(vec![polynomial], opening_id.sumcheck, opening_point),
        );
        Ok(())
    }

    /// Adds a group of openings to the accumulator: the `polynomials`, one-hot or dense, are
    /// opened at `opening_point`, and their claims are read from the transcript, in order. The
    /// opening reduction reduces the group as one member. `Err` on a failed read.
    pub fn append_group<T, U>(
        &mut self,
        transcript: &mut T,
        polynomials: Vec<CommittedPoly>,
        sumcheck: SumcheckId,
        opening_point: Vec<U>,
    ) -> Result<(), ProofVerifyError>
    where
        T: VerifyingTranscript,
        U: Copy + Send + Sync + Into<F>,
    {
        let Some(&first) = polynomials.first() else {
            return Ok(());
        };
        for label in polynomials.iter().copied() {
            self.bind(
                transcript,
                OpeningId::new(label, sumcheck),
                OpeningPoint::<BIG_ENDIAN, F>::new(opening_point.clone()),
            )?;
        }
        let key = OpeningId::new(first, sumcheck);
        self.sumchecks.insert(
            key,
            OpeningReductionGroup::new(polynomials, sumcheck, opening_point),
        );
        Ok(())
    }

    /// Adds an opening of a virtual polynomial at `opening_point`; its claim is read from the
    /// transcript. `Err` on a failed read.
    pub fn append_virtual<T: VerifyingTranscript>(
        &mut self,
        transcript: &mut T,
        opening_id: OpeningId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) -> Result<(), ProofVerifyError> {
        let virtual_poly = opening_id
            .virtual_poly()
            .ok_or_else(|| missing_claim(opening_id))?;
        if let VirtualPoly::NodeOutput(node_idx) = virtual_poly {
            debug_assert!(
                !self.reduced_evaluations.contains_key(&node_idx)
                    || self.own_points.contains(&node_idx),
                "cannot append opening for node {node_idx} after evaluation reduction"
            );
        }
        self.bind(transcript, opening_id, opening_point)
    }
}

impl<F> VerifierOpeningAccumulator<F>
where
    F: JoltField,
{
    // ========== Batch Opening Reduction Sumcheck ==========

    /// Whether any committed polynomial was opened: whether the proof carries an opening
    /// reduction and a joint opening proof.
    pub fn has_committed_openings(&self) -> bool {
        !self.sumchecks.is_empty()
    }

    #[tracing::instrument(skip_all, name = "VerifierOpeningAccumulator::prepare_for_sumcheck")]
    /// Prepares the verifier for the batch opening reduction sumcheck: draws the in-group
    /// coefficient.
    pub fn prepare_for_sumcheck<T: Transcript>(&mut self, transcript: &mut T) {
        let rho: F = transcript.challenge_scalar();
        self.sumchecks
            .values_mut()
            .for_each(|opening| opening.set_coefficients(rho));
    }

    /// Verifies the batch opening reduction sumcheck (Stage 7): each group's reduced claim is
    /// read from the transcript once the rounds are bound, and kept with its group.
    #[tracing::instrument(
        skip_all,
        name = "VerifierOpeningAccumulator::verify_batch_opening_sumcheck"
    )]
    pub fn verify_batch_opening_sumcheck<T: VerifyingTranscript>(
        &mut self,
        transcript: &mut T,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let verifiers: Vec<_> = self
            .sumchecks
            .values()
            .map(|group| group.verifier())
            .collect();
        let instances: Vec<&dyn DynRelationVerifier<F, T>> = verifiers
            .iter()
            .map(|verifier| verifier as &dyn DynRelationVerifier<F, T>)
            .collect();
        let r_sumcheck = verify_relation_batch(instances, self, transcript)?;
        for (group, verifier) in self.sumchecks.values_mut().zip(&verifiers) {
            group.sumcheck_claim = verifier.reduced_claim();
        }
        Ok(r_sumcheck)
    }

    #[tracing::instrument(
        skip_all,
        name = "VerifierOpeningAccumulator::finalize_batch_opening_sumcheck"
    )]
    /// Finalizes the batch opening reduction sumcheck verification: draws the per-group
    /// batching coefficients. Returns the state needed to verify the joint opening. `Err` if a
    /// group's reduced claim was not read.
    pub fn finalize_batch_opening_sumcheck<T: Transcript>(
        &self,
        r_sumcheck: Vec<F::Challenge>,
        transcript: &mut T,
    ) -> Result<OpeningReductionState<F>, ProofVerifyError> {
        let sumcheck_claims: Vec<F> = self
            .sumchecks
            .values()
            .map(|group| {
                group.sumcheck_claim.ok_or_else(|| {
                    ProofVerifyError::Malformed(
                        "an opening group has no reduced claim before the joint opening".into(),
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(self.sumchecks.len());
        let (polynomials, poly_coeffs) = joint_coefficients(
            self.sumchecks
                .values()
                .map(|s| (&s.polynomials, &s.coefficients)),
            &gamma_powers,
        );

        Ok(OpeningReductionState {
            r_sumcheck,
            gamma_powers,
            sumcheck_claims,
            polynomials,
            poly_coeffs,
        })
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
    HammingWeight,
    /// RLC sumcheck for the execution of a specific node.
    RLC(usize),
    /// Eval-shift sumcheck used in the trig ops
    NTEvalShift,
    #[cfg(feature = "akita")]
    AkitaBooleanity,
    #[cfg(feature = "akita")]
    AkitaDigitRange,
}

impl Canonical for SumcheckId {
    fn encode<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        match self {
            Self::NodeExecution(idx) => {
                0u8.encode(&mut writer)?;
                idx.encode(&mut writer)?;
            }
            Self::Raf => 1u8.encode(&mut writer)?,
            Self::RaVirtualization => 2u8.encode(&mut writer)?,
            Self::RamHammingBooleanity => 3u8.encode(&mut writer)?,
            Self::RamHammingWeight => 4u8.encode(&mut writer)?,
            Self::HammingWeight => 6u8.encode(&mut writer)?,
            Self::RLC(idx) => {
                7u8.encode(&mut writer)?;
                idx.encode(&mut writer)?;
            }
            Self::NTEvalShift => 9u8.encode(&mut writer)?,
            #[cfg(feature = "akita")]
            Self::AkitaBooleanity => 10u8.encode(&mut writer)?,
            #[cfg(feature = "akita")]
            Self::AkitaDigitRange => 11u8.encode(&mut writer)?,
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        match self {
            Self::NodeExecution(idx) | Self::RLC(idx) => 1u8.encoded_len() + idx.encoded_len(),
            _ => 1u8.encoded_len(),
        }
    }

    fn decode<R: Read>(mut reader: R) -> Result<Self, Error> {
        let tag = u8::decode(&mut reader)?;
        match tag {
            0 => {
                let idx = usize::decode(&mut reader)?;
                Ok(Self::NodeExecution(idx))
            }
            1 => Ok(Self::Raf),
            2 => Ok(Self::RaVirtualization),
            3 => Ok(Self::RamHammingBooleanity),
            4 => Ok(Self::RamHammingWeight),
            6 => Ok(Self::HammingWeight),
            7 => {
                let idx = usize::decode(&mut reader)?;
                Ok(Self::RLC(idx))
            }
            9 => Ok(Self::NTEvalShift),
            #[cfg(feature = "akita")]
            10 => Ok(Self::AkitaBooleanity),
            #[cfg(feature = "akita")]
            11 => Ok(Self::AkitaDigitRange),
            _ => Err(Error::InvalidData),
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

impl Canonical for OpeningId {
    fn encode<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        match self.polynomial {
            PolynomialId::Committed(polynomial) => {
                0u8.encode(&mut writer)?;
                polynomial.encode(&mut writer)?;
                self.sumcheck.encode(&mut writer)?;
            }
            PolynomialId::Virtual(polynomial) => {
                1u8.encode(&mut writer)?;
                polynomial.encode(&mut writer)?;
                self.sumcheck.encode(&mut writer)?;
            }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        1 + self.sumcheck.encoded_len()
            + match self.polynomial {
                PolynomialId::Virtual(polynomial) => polynomial.encoded_len(),
                PolynomialId::Committed(polynomial) => polynomial.encoded_len(),
            }
    }

    fn decode<R: Read>(mut reader: R) -> Result<Self, Error> {
        let tag = u8::decode(&mut reader)?;
        match tag {
            0 => {
                let poly = CommittedPoly::decode(&mut reader)?;
                let sc = SumcheckId::decode(&mut reader)?;
                Ok(Self::new(poly, sc))
            }
            1 => {
                let poly = VirtualPoly::decode(&mut reader)?;
                let sc = SumcheckId::decode(&mut reader)?;
                Ok(Self::new(poly, sc))
            }
            _ => Err(Error::InvalidData),
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
    use atlas_algebra::field::Ring as _;
    use atlas_transcript::{
        Blake2bProver, Blake2bVerifier, ProvingTranscript, VerifyingTranscript,
    };
    use common::VirtualPoly;

    #[test]
    #[should_panic(expected = "reduced evaluation for node 7 not found")]
    fn get_node_output_opening_panics_before_reduction() {
        let acc = ProverOpeningAccumulator::<Fr>::new();
        let _ = acc.get_node_output_opening(7);
    }

    fn hamming_weight_id() -> OpeningId {
        OpeningId::new(
            VirtualPoly::HammingWeight(0),
            SumcheckId::RamHammingBooleanity,
        )
    }

    fn point(seed: u64) -> OpeningPoint<{ super::BIG_ENDIAN }, Fr> {
        OpeningPoint::new(vec![Fr::from_u64(seed), Fr::from_u64(seed + 1)])
    }

    /// A stream of two claims, and the verifier's transcript over it.
    fn two_claims() -> Blake2bVerifier {
        let mut prover = Blake2bProver::new(b"guardrail_test");
        prover.send_scalar(&Fr::from_u64(5));
        prover.send_scalar(&Fr::from_u64(6));
        Blake2bVerifier::new(b"guardrail_test", prover.into_narg())
    }

    /// A claim is held only from its bind: no getter returns one before.
    #[test]
    fn verifier_holds_no_claim_before_its_bind() {
        let mut acc = super::VerifierOpeningAccumulator::<Fr>::new();
        let id = hamming_weight_id();
        assert!(acc.openings().is_empty());
        assert!(acc.get_virtual_polynomial_opening(id).is_err());
        assert!(acc.try_get_virtual_polynomial_opening(id).is_none());

        let mut transcript = two_claims();
        acc.append_virtual(&mut transcript, id, point(1)).unwrap();
        assert_eq!(
            acc.get_virtual_polynomial_opening(id).unwrap(),
            (point(1), Fr::from_u64(5))
        );
    }

    /// One opening id bound at two different points would discard the first binding, leaving
    /// a check constraining a claim the accumulator does not hold.
    #[test]
    fn verifier_append_rejects_rebinding_at_a_different_point() {
        let mut acc = super::VerifierOpeningAccumulator::<Fr>::new();
        let id = hamming_weight_id();
        let mut transcript = two_claims();
        acc.append_virtual(&mut transcript, id, point(1)).unwrap();
        assert!(acc.append_virtual(&mut transcript, id, point(3)).is_err());
        // The first binding stands.
        assert_eq!(
            acc.get_virtual_polynomial_opening(id).unwrap(),
            (point(1), Fr::from_u64(5))
        );
    }

    /// Binding an id again at its point reads nothing: the stream carries the claim once, so
    /// the next read is the next claim.
    #[test]
    fn verifier_append_at_the_same_point_reads_the_claim_once() {
        let mut acc = super::VerifierOpeningAccumulator::<Fr>::new();
        let id = hamming_weight_id();
        let mut transcript = two_claims();
        acc.append_virtual(&mut transcript, id, point(1)).unwrap();
        acc.append_virtual(&mut transcript, id, point(1)).unwrap();
        assert_eq!(
            acc.get_virtual_polynomial_opening(id).unwrap().1,
            Fr::from_u64(5)
        );
        assert_eq!(transcript.read_scalar::<Fr>().unwrap(), Fr::from_u64(6));
        transcript.finish().unwrap();
    }

    /// A bind with nothing left to read is an error, not a default claim.
    #[test]
    fn verifier_append_rejects_a_short_stream() {
        let mut acc = super::VerifierOpeningAccumulator::<Fr>::new();
        let mut transcript = Blake2bVerifier::new(b"guardrail_test", Vec::new());
        assert!(acc
            .append_virtual(&mut transcript, hamming_weight_id(), point(1))
            .is_err());
        assert!(acc.openings().is_empty());
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

#[cfg(feature = "akita")]
mod akita;
#[cfg(feature = "akita")]
pub use akita::AkitaOpeningPadding;

pub mod prover;
