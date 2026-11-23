use super::opening_error;
use crate::node_view::ModelView;
use crate::onnx_proof::coefficient_checks::CoefficientCheck;
use atlas_algebra::{
    field::{akita::AkitaField, One},
    utils::errors::ProofVerifyError,
};
use atlas_pcs::akita::{
    plan_groups_with_padding, AkitaCommitment, AkitaScheduleArtifacts, AkitaScheme,
    AkitaSetupParams, AkitaVerifierSetup, GroupLayout,
};
use atlas_sumcheck::opening_proof::AkitaOpeningPadding;
use atlas_trace::Model;
use common::{consts::LOG_K_CHUNK, CommittedPoly};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

/// The original dimensions of a polynomial, before any commitment padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct WitnessShape {
    num_vars: usize,
    k: Option<usize>,
}

impl WitnessShape {
    pub(super) fn for_model(
        model: &Model,
        poly: CommittedPoly,
        num_vars: usize,
    ) -> Result<Self, ProofVerifyError> {
        let k = match poly {
            CommittedPoly::GatherRa(node) => {
                let shape = super::super::ops::gather::small::GatherSmallShape::for_node(
                    model,
                    &model.view(node),
                )?;
                Some(shape.num_words)
            }
            CommittedPoly::GlobalRemainderDigit(..)
            | CommittedPoly::RsqrtQuotient(..)
            | CommittedPoly::TrigPeriodQuotient(..)
            | CommittedPoly::SoftmaxShift(..) => None,
            CommittedPoly::NodeOutputRaD(..)
            | CommittedPoly::CosRaD(..)
            | CommittedPoly::SinRaD(..)
            | CommittedPoly::TrigDownscaleRaD(..)
            | CommittedPoly::GatherRaD(..)
            | CommittedPoly::SoftmaxAddrRaD(..)
            | CommittedPoly::ActivationClampRaD(..)
            | CommittedPoly::ActivationSmallRaD(..)
            | CommittedPoly::SoftmaxClampRaD(..) => Some(1 << LOG_K_CHUNK),
        };
        Self::new(num_vars, k)
    }

    /// Create the shape of an original dense or sparse witness.
    pub fn new(num_vars: usize, k: Option<usize>) -> Result<Self, ProofVerifyError> {
        if num_vars >= usize::BITS as usize
            || k.is_some_and(|k| !k.is_power_of_two() || k.ilog2() as usize > num_vars)
        {
            return Err(opening_error("invalid original witness shape"));
        }
        Ok(Self { num_vars, k })
    }
}

fn polynomial_layout(shape: WitnessShape) -> Result<GroupLayout, ProofVerifyError> {
    GroupLayout::new(shape.num_vars, shape.k.filter(|k| matches!(k, 16 | 256)))
        .map_err(opening_error)
}

/// A commitment group selected by the shared Atlas Akita planner.
pub struct CommitmentGroup {
    /// The shape actually committed after padding.
    pub layout: GroupLayout,
    /// The group's columns in commitment order.
    pub polynomials: Vec<CommittedPoly>,
    /// Shape-specific setup, reconstructed independently by each side.
    pub setup: AkitaSetupParams,
    /// Domain separator for this group's commitment and claims.
    pub digest: [u8; 32],
    verifier_setup: Mutex<Option<Arc<AkitaVerifierSetup>>>,
}

impl CommitmentGroup {
    pub(super) fn verifier_setup(&self) -> Result<Arc<AkitaVerifierSetup>, ProofVerifyError> {
        let mut cache = self.verifier_setup.lock().map_err(opening_error)?;
        if let Some(setup) = cache.as_ref() {
            return Ok(setup.clone());
        }
        let setup =
            Arc::new(AkitaScheme::verifier_setup_from_params(&self.setup).map_err(opening_error)?);
        *cache = Some(setup.clone());
        Ok(setup)
    }

    /// Get this group's suffix of the batched Atlas opening point.
    pub fn point<'a>(&self, point: &'a [AkitaField]) -> Result<&'a [AkitaField], ProofVerifyError> {
        let prefix = point
            .len()
            .checked_sub(self.layout.num_vars())
            .ok_or_else(|| opening_error("Akita group exceeds the batch opening arity"))?;
        point
            .get(prefix..)
            .ok_or_else(|| opening_error("invalid Akita opening point"))
    }

    pub(super) fn zero_extension_scale(
        &self,
        point: &[AkitaField],
    ) -> Result<AkitaField, ProofVerifyError> {
        let len = self.point(point)?.len();
        let prefix = point
            .get(..point.len() - len)
            .ok_or_else(|| opening_error("invalid Akita opening point"))?;
        Ok(prefix.iter().map(|&r| AkitaField::one() - r).product())
    }

    pub(super) fn validate_commitment(
        &self,
        commitment: &AkitaCommitment,
    ) -> Result<(), ProofVerifyError> {
        if commitment.layout_digest() != self.digest
            || commitment.num_vars() != self.layout.num_vars()
            || commitment.poly_count() != self.polynomials.len()
            || commitment.one_hot_k() != self.layout.one_hot_width().unwrap_or(0)
        {
            return Err(opening_error(
                "Akita commitment does not match the model group",
            ));
        }
        Ok(())
    }
}

/// The commitment layouts and original coefficient domains for an Atlas proof.
pub struct CommitmentPlan {
    /// Groups in transcript order.
    pub groups: Vec<CommitmentGroup>,
    /// Padded domains for the Akita opening reduction.
    pub padding: BTreeMap<CommittedPoly, AkitaOpeningPadding>,
    shapes: BTreeMap<CommittedPoly, WitnessShape>,
}

impl CommitmentPlan {
    /// Choose supported commitment groups from ordered original witness shapes.
    pub fn new(
        shapes: &BTreeMap<CommittedPoly, WitnessShape>,
        artifacts: Arc<AkitaScheduleArtifacts>,
    ) -> Result<Self, ProofVerifyError> {
        let layouts = shapes
            .values()
            .copied()
            .map(polynomial_layout)
            .collect::<Result<Vec<_>, _>>()?;
        let ids: Vec<_> = shapes.keys().copied().collect();
        let mut groups = Vec::new();
        let mut padding = BTreeMap::new();
        for (index, group) in plan_groups_with_padding(&layouts, &artifacts)
            .map_err(opening_error)?
            .into_iter()
            .enumerate()
        {
            let mut digest = [0; 32];
            digest[..8].copy_from_slice(&(index as u64).to_le_bytes());
            let polynomials = group
                .polynomial_indices
                .iter()
                .map(|&i| {
                    ids.get(i)
                        .copied()
                        .ok_or_else(|| opening_error("invalid Akita witness index"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let count = polynomials.len();
            let vars = group.layout.num_vars();
            let width = group.layout.one_hot_width();
            let address_vars = width.map(|k| k.ilog2() as usize);
            for p in &polynomials {
                let shape = shapes
                    .get(p)
                    .ok_or_else(|| opening_error("missing Akita witness shape"))?;
                if let Some(pad) = AkitaOpeningPadding::new(address_vars, shape.num_vars, vars) {
                    padding.insert(*p, pad);
                }
            }
            let setup = match width {
                Some(k) => {
                    AkitaSetupParams::one_hot_only(vars, count, digest, k, artifacts.clone())
                }
                None => AkitaSetupParams::dense_only(vars, count, digest, artifacts.clone()),
            };
            groups.push(CommitmentGroup {
                layout: group.layout,
                polynomials,
                setup,
                digest,
                verifier_setup: Mutex::default(),
            });
        }
        Ok(Self {
            groups,
            padding,
            shapes: shapes.clone(),
        })
    }

    /// Construct the Booleanity and base-four checks in shared transcript order.
    pub fn coefficient_checks(&self) -> Result<Vec<CoefficientCheck>, ProofVerifyError> {
        let mut by_shape = BTreeMap::<WitnessShape, Vec<_>>::new();
        for (&poly, &shape) in &self.shapes {
            if shape.k.is_some() || matches!(poly, CommittedPoly::GlobalRemainderDigit(..)) {
                by_shape.entry(shape).or_default().push(poly);
            }
        }
        by_shape
            .into_iter()
            .map(|(shape, polynomials)| {
                if let Some(k) = shape.k {
                    let address = k.ilog2() as usize;
                    CoefficientCheck::one_hot(polynomials, address, shape.num_vars - address)
                } else {
                    CoefficientCheck::base_four(polynomials, shape.num_vars)
                }
            })
            .collect()
    }
}
