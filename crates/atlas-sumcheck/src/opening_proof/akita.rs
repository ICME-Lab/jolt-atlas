//! Lift opening requests into the padded Akita commitment domains.

use super::{prover::ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::opening_reduction::{
    prover::{OpeningProofReductionSumcheckProver, ProverOpening},
    EqAddressState, EqCycleState,
};
use atlas_algebra::{
    field::JoltField, poly::multilinear::MultilinearPolynomial, utils::errors::ProofVerifyError,
};
use atlas_transcript::Transcript;
use common::CommittedPoly;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, RwLock},
};

/// The original and committed domains of a padded witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AkitaOpeningPadding {
    address_vars: Option<usize>,
    original_vars: usize,
    committed_vars: usize,
}

impl AkitaOpeningPadding {
    /// Keep address variables first for native one-hot witnesses. Dense witnesses
    /// use a zero prefix, including one-hot witnesses committed as dense.
    pub fn new(
        address_vars: Option<usize>,
        original_vars: usize,
        committed_vars: usize,
    ) -> Option<Self> {
        (address_vars.is_none_or(|vars| vars <= original_vars)
            && original_vars < committed_vars
            && committed_vars < usize::BITS as usize)
            .then_some(Self {
                address_vars,
                original_vars,
                committed_vars,
            })
    }

    fn lift<F: JoltField>(
        polynomials: &[CommittedPoly],
        point: &[F],
        padding: &BTreeMap<CommittedPoly, Self>,
    ) -> Result<Option<(Self, Vec<F>)>, ProofVerifyError> {
        let expected = polynomials.first().and_then(|p| padding.get(p));
        if polynomials.iter().any(|poly| padding.get(poly) != expected) {
            return Err(ProofVerifyError::Malformed(
                "opening group mixes Akita padding layouts".into(),
            ));
        }
        let Some(&layout) = expected else {
            return Ok(None);
        };
        if point.len() != layout.original_vars {
            return Err(ProofVerifyError::Malformed(
                "opening does not match its original Akita domain".into(),
            ));
        }

        let (prefix, suffix) = point.split_at(layout.address_vars.unwrap_or(0));
        let mut lifted = Vec::with_capacity(layout.committed_vars);
        lifted.extend_from_slice(prefix);
        lifted.resize(
            prefix.len() + layout.committed_vars - layout.original_vars,
            F::zero(),
        );
        lifted.extend_from_slice(suffix);
        Ok(Some((layout, lifted)))
    }
}

impl<F: JoltField> ProverOpeningAccumulator<F> {
    /// Prepare the Akita batch reduction using commitment domains from the group plan.
    ///
    /// Call once after registering the original openings. The witness map contains the
    /// polynomials used for commitment, including any empty rows.
    pub fn prepare_for_akita_sumcheck(
        &mut self,
        polynomials: &BTreeMap<CommittedPoly, MultilinearPolynomial<F>>,
        padding: &BTreeMap<CommittedPoly, AkitaOpeningPadding>,
        transcript: &mut impl Transcript,
    ) -> Result<(), ProofVerifyError> {
        let mut lifted = Vec::new();
        for (&key, member) in &self.sumchecks {
            if let Some((layout, point)) =
                AkitaOpeningPadding::lift(&member.polynomials, &member.point, padding)?
            {
                if layout.address_vars.is_some()
                    && !matches!(&member.prover_state, ProverOpening::OneHot(_))
                {
                    return Err(ProofVerifyError::Malformed(
                        "one-hot padding was assigned to a dense opening".into(),
                    ));
                }
                lifted.push((key, layout, point));
            }
        }

        let mut row_states = HashMap::new();
        for (key, layout, point) in lifted {
            let member = self.sumchecks.get_mut(&key).unwrap();
            let polynomials = std::mem::take(&mut member.polynomials);
            let sumcheck_id = member.sumcheck_id;
            let claims = std::mem::take(&mut member.claims);
            *member = if let Some(address_vars) = layout.address_vars {
                let (address, row) = point.split_at(address_vars);
                let eq_address = Arc::new(RwLock::new(EqAddressState::new(address)));
                let eq_cycle = row_states
                    .entry(row.to_vec())
                    .or_insert_with(|| Arc::new(RwLock::new(EqCycleState::new(row))))
                    .clone();
                OpeningProofReductionSumcheckProver::new_one_hot(
                    polynomials,
                    sumcheck_id,
                    eq_address,
                    eq_cycle,
                    point,
                    claims,
                )
            } else {
                let eq_cycle = row_states
                    .entry(point.clone())
                    .or_insert_with(|| Arc::new(RwLock::new(EqCycleState::new(&point))))
                    .clone();
                OpeningProofReductionSumcheckProver::new_dense_group(
                    polynomials,
                    sumcheck_id,
                    eq_cycle,
                    point,
                    claims,
                )
            };
        }
        self.prepare_for_sumcheck(polynomials, transcript);
        Ok(())
    }
}

impl<F: JoltField> VerifierOpeningAccumulator<F> {
    /// Prepare the Akita batch reduction using commitment domains from the trusted group plan.
    ///
    /// Call once after registering the original openings. Only the reduction points are lifted.
    pub fn prepare_for_akita_sumcheck(
        &mut self,
        padding: &BTreeMap<CommittedPoly, AkitaOpeningPadding>,
        transcript: &mut impl Transcript,
    ) -> Result<(), ProofVerifyError> {
        let mut lifted = Vec::new();
        for (&key, member) in &self.sumchecks {
            if let Some((_, point)) =
                AkitaOpeningPadding::lift(&member.polynomials, &member.point, padding)?
            {
                lifted.push((key, point));
            }
        }
        for (key, point) in lifted {
            self.sumchecks.get_mut(&key).unwrap().point = point;
        }
        self.prepare_for_sumcheck(transcript);
        Ok(())
    }
}
