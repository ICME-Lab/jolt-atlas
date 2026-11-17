//! The post-loop plan: the packed buckets, then every lookup in two batches.
//!
//! A table read consumes claims appended during its node's turn and produces a claim only on
//! its own read address, and a one-hot check consumes only that read-address claim, so neither
//! needs to run in the node loop: a plan step [`Step::Defer`](super::plan::Step::Defer) queues
//! the read for [`DeferredBatch::Lookups`] and its one-hot check for
//! [`DeferredBatch::OneHot`], and each batch is proven as one batched sumcheck after the loop.
//! The range claims (remainders, their slacks, the outputs of the arithmetic nodes) are packed
//! into model-level buckets ([`range_bucket`](super::range_bucket)); [`plan`] builds their steps
//! from the model, with nothing to register during the loop, and they run before the two
//! batches.
//!
//! The prover walks the same plan (`jolt_atlas_core::onnx_proof::deferred_lookups`), proving
//! every batch with `prove_relation_batch_parallel`: instance messages are computed in parallel
//! across instances, which turns hundreds of tiny sequential sumchecks into a few large
//! parallel ones.
use crate::onnx_proof::{
    plan::{Plan, Step},
    range_bucket::range_buckets,
    schedule::OpRelation,
    ProofType,
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::Model;

/// Node index under which the deferred (model-wide) lookup proofs are stored.
pub const DEFERRED_PROOF_IDX: usize = usize::MAX;

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
/// one batch, then their digit range checks as a second. The deferred batches follow, run by
/// the interpreter from its queue.
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
