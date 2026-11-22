//! Convert Atlas witness polynomials to Jolt's Akita commitment and opening API.

use atlas_algebra::{
    field::akita::AkitaField,
    poly::multilinear::{MultilinearPolynomial, PolynomialBinding},
};
use jolt_akita::{
    AkitaBackendFlavor, AkitaField as Native, AkitaNativeBatching, AKITA_ONE_HOT_K16,
    AKITA_ONE_HOT_K256,
};
pub use jolt_akita::{
    AkitaBatchProof, AkitaCommitment, AkitaProverHint, AkitaProverSetup, AkitaScheduleArtifacts,
    AkitaScheme, AkitaSetupParams, AkitaVerifierSetup,
};
pub use jolt_openings::CommitmentScheme;
use jolt_openings::{BatchOpeningScheme, EvaluationClaim, OpeningsError, VerifierOpeningClaim};
use jolt_poly::{MultilinearPoly, OneHotPolynomial, Polynomial};
use jolt_transcript::FieldTranscript;
pub use jolt_transcript::{protocol_tag, prover_transcript, verifier_transcript, Keccak};
use std::borrow::Cow;

mod grouping;
mod schedules;
pub use grouping::{plan_groups, plan_groups_with_padding, PlannedGroup};
pub use schedules::schedule_artifacts;

/// Arity and coordinate order of an Atlas polynomial group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GroupLayout {
    one_hot_width: Option<usize>,
    num_vars: usize,
}

impl GroupLayout {
    /// Determine the layout the Akita adapter would use for an Atlas witness.
    pub fn for_polynomial(
        polynomial: &MultilinearPolynomial<AkitaField>,
    ) -> Result<Self, OpeningsError> {
        witness_layout(polynomial)
    }

    /// The number of variables in the committed polynomial.
    pub fn num_vars(self) -> usize {
        self.num_vars
    }

    /// The column count of a native one-hot polynomial.
    pub fn one_hot_width(self) -> Option<usize> {
        self.one_hot_width
    }

    /// Extend a witness to its commitment domain, using empty rows or zero coefficients.
    pub fn pad_polynomial(
        self,
        polynomial: &MultilinearPolynomial<AkitaField>,
    ) -> Result<Cow<'_, MultilinearPolynomial<AkitaField>>, OpeningsError> {
        let original = Self::for_polynomial(polynomial)?;
        if original == self {
            return Ok(Cow::Borrowed(polynomial));
        }
        match (self.one_hot_width, polynomial) {
            (None, _) if original.one_hot_width.is_none() && self.num_vars > original.num_vars => {
                let mut coefficients = polynomial.coeffs();
                coefficients.resize(1 << self.num_vars, AkitaField::default());
                Ok(Cow::Owned(MultilinearPolynomial::from(coefficients)))
            }
            (Some(k), MultilinearPolynomial::OneHot(poly))
                if original.one_hot_width == Some(k) && self.num_vars > original.num_vars =>
            {
                let mut indices = poly.nonzero_indices.to_vec();
                indices.resize(1 << (self.num_vars - k.ilog2() as usize), None);
                Ok(Cow::Owned(MultilinearPolynomial::OneHot(
                    atlas_algebra::poly::one_hot::OneHotPolynomial::from_indices(indices, k),
                )))
            }
            _ => Err(OpeningsError::InvalidBatch(
                "polynomial cannot be padded to the Akita group layout".into(),
            )),
        }
    }

    /// Construct the layout used by the prover and verifier for one group.
    pub fn new(num_vars: usize, one_hot_width: Option<usize>) -> Result<Self, OpeningsError> {
        if num_vars >= usize::BITS as usize
            || one_hot_width.is_some_and(|width| {
                !matches!(width, AKITA_ONE_HOT_K16 | AKITA_ONE_HOT_K256)
                    || width.trailing_zeros() as usize > num_vars
            })
        {
            return Err(OpeningsError::InvalidBatch(
                "invalid Akita polynomial group layout".into(),
            ));
        }
        Ok(Self {
            num_vars,
            one_hot_width,
        })
    }

    fn point(self, point: &[AkitaField]) -> Result<Vec<Native>, OpeningsError> {
        if point.len() != self.num_vars {
            return Err(OpeningsError::InvalidBatch(
                "opening point does not match the Akita group arity".into(),
            ));
        }
        // Atlas uses address-major indexing. Jolt's sparse commitment takes row-major indices.
        // A dense conversion keeps the original coefficient order, even for a gather.
        let address_vars = self
            .one_hot_width
            .map_or(0, |width| width.trailing_zeros() as usize);
        let (address, row) = point.split_at(address_vars);
        Ok(row
            .iter()
            .chain(address)
            .copied()
            .map(AkitaField::into_native)
            .collect())
    }

    fn statement(
        self,
        commitment: &AkitaCommitment,
        point: &[AkitaField],
        evaluations: &[AkitaField],
    ) -> Result<Vec<VerifierOpeningClaim<Native, AkitaCommitment>>, OpeningsError> {
        let flavor = if self.one_hot_width.is_some() {
            AkitaBackendFlavor::OneHot
        } else {
            AkitaBackendFlavor::Dense
        };
        if commitment.backend_flavor() != flavor
            || commitment.one_hot_k() != self.one_hot_width.unwrap_or(0)
            || commitment.num_vars() != self.num_vars
            || commitment.poly_count() != evaluations.len()
        {
            return Err(OpeningsError::InvalidBatch(
                "commitment does not match the Akita opening layout".into(),
            ));
        }
        let point = self.point(point)?;
        Ok(evaluations
            .iter()
            .map(|value| VerifierOpeningClaim {
                commitment: commitment.clone(),
                evaluation: EvaluationClaim::new(point.clone(), value.into_native()),
            })
            .collect())
    }
}

enum Polynomials {
    Dense(Vec<Polynomial<Native>>),
    OneHot(Vec<OneHotPolynomial>),
}

/// A group of witness polynomials in Jolt's commitment representation, in Atlas column order.
pub struct Group {
    layout: GroupLayout,
    polynomials: Polynomials,
}

impl Group {
    /// Convert a group of unbound, equally sized Atlas witness polynomials.
    ///
    /// Supported one-hot widths use Jolt's sparse format. Other widths use the same coefficient
    /// table via the bounded dense path. A group must have a single representation.
    pub fn new(polynomials: &[&MultilinearPolynomial<AkitaField>]) -> Result<Self, OpeningsError> {
        let layouts = polynomials
            .iter()
            .map(|poly| witness_layout(poly))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(&layout) = layouts.first() else {
            return Err(OpeningsError::InvalidBatch("empty Akita group".into()));
        };
        if layouts.iter().any(|&other| other != layout) {
            return Err(OpeningsError::InvalidBatch(
                "mixed Akita group layouts".into(),
            ));
        }
        let polynomials = match layout.one_hot_width {
            Some(width) => Polynomials::OneHot(
                polynomials
                    .iter()
                    .map(|poly| {
                        let MultilinearPolynomial::OneHot(poly) = poly else {
                            unreachable!("the validated group consists of one-hot polynomials")
                        };
                        Ok(OneHotPolynomial::new(
                            width,
                            poly.nonzero_indices
                                .iter()
                                .map(|&index| index.map(u8::try_from).transpose())
                                .collect::<Result<_, _>>()
                                .map_err(|error| OpeningsError::InvalidBatch(error.to_string()))?,
                        ))
                    })
                    .collect::<Result<_, OpeningsError>>()?,
            ),
            None => Polynomials::Dense(
                polynomials
                    .iter()
                    .map(|poly| {
                        Polynomial::new(
                            poly.coeffs()
                                .into_iter()
                                .map(AkitaField::into_native)
                                .collect(),
                        )
                    })
                    .collect(),
            ),
        };
        Ok(Self {
            layout,
            polynomials,
        })
    }

    /// The layout used to construct and verify this group's opening statement.
    pub fn layout(&self) -> GroupLayout {
        self.layout
    }

    /// Commit the group's polynomials with a shape-matched Jolt setup.
    pub fn commit(
        &self,
        setup: &AkitaProverSetup,
        layout_digest: [u8; 32],
    ) -> Result<(AkitaCommitment, AkitaProverHint), OpeningsError> {
        match &self.polynomials {
            Polynomials::Dense(polys) => AkitaScheme::commit_group(setup, layout_digest, polys),
            Polynomials::OneHot(polys) => {
                AkitaScheme::commit_one_hot_group(setup, layout_digest, polys)
            }
        }
    }

    /// Prove an opening of every column at the supplied Atlas evaluation point.
    pub fn prove_opening(
        &self,
        setup: &AkitaProverSetup,
        commitment: &AkitaCommitment,
        hint: AkitaProverHint,
        point: &[AkitaField],
        evaluations: &[AkitaField],
        transcript: &mut impl FieldTranscript<Native>,
    ) -> Result<AkitaBatchProof, OpeningsError> {
        let statement = self.layout.statement(commitment, point, evaluations)?;
        AkitaNativeBatching::prove_batch(setup, statement, self.polynomials(), hint, transcript)
    }

    fn polynomials(&self) -> Vec<&dyn MultilinearPoly<Native>> {
        match &self.polynomials {
            Polynomials::Dense(polys) => polys
                .iter()
                .map(|poly| poly as &dyn MultilinearPoly<Native>)
                .collect(),
            Polynomials::OneHot(polys) => polys
                .iter()
                .map(|poly| poly as &dyn MultilinearPoly<Native>)
                .collect(),
        }
    }
}

/// Verify the given evaluations against a group commitment using Jolt's standalone Akita proof.
pub fn verify_opening(
    setup: &AkitaVerifierSetup,
    layout: GroupLayout,
    commitment: &AkitaCommitment,
    point: &[AkitaField],
    evaluations: &[AkitaField],
    proof: &AkitaBatchProof,
    transcript: &mut impl FieldTranscript<Native>,
) -> Result<(), OpeningsError> {
    let statement = layout.statement(commitment, point, evaluations)?;
    AkitaNativeBatching::verify_batch(setup, &statement, proof, transcript)
}

fn witness_layout(poly: &MultilinearPolynomial<AkitaField>) -> Result<GroupLayout, OpeningsError> {
    let is_bound = match poly {
        MultilinearPolynomial::OneHot(hot) => hot.num_variables_bound != 0,
        MultilinearPolynomial::S128Scalars(dense) => dense.is_bound(),
        _ => poly.is_bound(),
    };
    if is_bound {
        return Err(OpeningsError::InvalidBatch(
            "expected an unbound Atlas polynomial".into(),
        ));
    }
    let MultilinearPolynomial::OneHot(hot) = poly else {
        return GroupLayout::new(poly.get_num_vars(), None);
    };
    if !hot.K.is_power_of_two()
        || !hot.nonzero_indices.len().is_power_of_two()
        || hot
            .nonzero_indices
            .iter()
            .flatten()
            .any(|&index| usize::from(index) >= hot.K)
    {
        return Err(OpeningsError::InvalidBatch(
            "invalid Atlas one-hot polynomial".into(),
        ));
    }
    let native_width = matches!(hot.K, AKITA_ONE_HOT_K16 | AKITA_ONE_HOT_K256).then_some(hot.K);
    GroupLayout::new(poly.get_num_vars(), native_width)
}

#[cfg(test)]
mod tests;
