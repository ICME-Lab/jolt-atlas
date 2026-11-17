//! Temporary: the post-loop batches as `jolt-atlas-core`'s per-node walk proves them.
//!
//! The verifier runs the model's schedule ([`schedule`](super::schedule)), whose last steps are
//! the packed buckets' two sumchecks, the lookups sumcheck and the one-hot sumcheck. Until the
//! prover walks that schedule too, it queues a node's table reads for [`DeferredBatch::Lookups`]
//! and its one-hot checks for [`DeferredBatch::OneHot`], and builds the buckets' steps with
//! [`plan`]. This module goes away with [`plan`](super::plan).
use crate::onnx_proof::{
    plan::{Plan, Step},
    range_bucket::range_buckets,
    schedule::OpRelation,
    ProofType,
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::Model;

/// When a deferred relation is set up (its claims read, its challenges drawn).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeferredSetup {
    /// In the node's turn, when the interpreter reaches the `Defer` step.
    InNodeLoop,
    /// When the batch runs, after the batch that produces the claims it reads.
    AtBatch,
}

/// A batch of sumcheck instances proven after the node loop as one batched sumcheck. Batches
/// run in this enum's order, after the bucket plan: a one-hot check reads the read-address
/// claim a member of [`Self::Lookups`] or a node's own read produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeferredBatch {
    /// Every table read whose sumcheck produces only its own read-address claim.
    Lookups,
    /// Every one-hot check of a read address.
    OneHot,
}

impl DeferredBatch {
    /// Every batch, in run order.
    pub const ALL: [Self; 2] = [Self::Lookups, Self::OneHot];

    /// The proof slot this batch's sumcheck is stored under.
    pub fn proof_type(self) -> ProofType {
        match self {
            Self::Lookups => ProofType::DeferredLookups,
            Self::OneHot => ProofType::DeferredOneHot,
        }
    }

    /// When this batch's relations are set up: a read draws its challenges and appends its
    /// operand claim in the node's turn; a one-hot check waits for the claims it reads.
    pub fn setup(self) -> DeferredSetup {
        match self {
            Self::Lookups => DeferredSetup::InNodeLoop,
            Self::OneHot => DeferredSetup::AtBatch,
        }
    }
}

/// The post-loop plan over the packed range buckets, from the model: their value sumchecks as
/// one batch, then their digit range checks as a second. The deferred batches follow, proven
/// from the prover's queue.
pub fn plan<F: atlas_algebra::field::JoltField>(
    model: &Model,
) -> Result<Plan<F>, ProofVerifyError> {
    let mut steps = Vec::new();
    let buckets = range_buckets(model)?;
    if !buckets.is_empty() {
        steps.push(Step::Batch {
            proof: ProofType::DeferredRemainderRC,
            relations: buckets
                .iter()
                .map(|bucket| OpRelation::RemainderBucketValue(bucket.clone()))
                .collect(),
        });
        steps.push(Step::Batch {
            proof: ProofType::DeferredRemainderDigitRange,
            relations: buckets
                .into_iter()
                .map(OpRelation::RemainderBucketDigits)
                .collect(),
        });
    }
    Ok(Plan(steps))
}
