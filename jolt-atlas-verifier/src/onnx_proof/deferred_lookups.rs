//! The post-loop plan: the packed buckets, then every lookup in two batches.
//!
//! A table read consumes claims appended during its node's turn and produces a claim only on
//! its own read address, and a one-hot check consumes only that read-address claim, so neither
//! needs to run in the node loop: a plan step [`Step::Defer`](super::plan::Step::Defer) queues
//! the read for [`DeferredBatch::Lookups`] and its one-hot check for
//! [`DeferredBatch::OneHot`], and each batch is proven as one batched sumcheck after the loop.
//! The saturating clamps and the rescale remainders are packed into model-level buckets
//! ([`global_clamp`](super::global_clamp)); [`plan`] builds their steps from the model and the
//! public IO, with nothing to register during the loop, and they run before the two batches.
//!
//! The prover walks the same plan (`jolt_atlas_core::onnx_proof::deferred_lookups`), proving
//! every batch with `prove_relation_batch_parallel`: instance messages are computed in parallel
//! across instances, which turns hundreds of tiny sequential sumchecks into a few large
//! parallel ones.
use crate::onnx_proof::{
    clamp_split,
    global_clamp::{clamp_buckets, remainder_buckets, ClampBucket},
    plan::{OpOpen, OpRelation, OpTie, Plan, Step},
    ProofType,
};
use atlas_algebra::utils::errors::ProofVerifyError;
use atlas_trace::{trace::ModelExecutionIO, Model};

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
/// claim a member of [`Self::Lookups`], a node's own read or the clamp split batch produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeferredBatch {
    /// Every table read whose sumcheck produces only its own read-address claim.
    Lookups,
    /// Every one-hot check of a read address, and the clamp buckets' slack chunk checks.
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

/// The post-loop plan over the packed clamp and remainder buckets, from the model and the
/// public IO: per live clamp bucket its exact flag and the exact check, then the bucket value
/// sumchecks as one batch and the `OUT` digit range checks as a second, with each bucket's
/// slack chunk check queued for [`DeferredBatch::OneHot`] (it yields no relation when the
/// bucket is exact); then the remainder buckets' value sumchecks and their digit range checks.
/// A clamp bucket is live when one of its nodes is not an unsaturated model output (those tie
/// `acc = out` in their own turn). The deferred batches follow, run by the interpreter from
/// its queue.
pub fn plan(model: &Model, io: &ModelExecutionIO) -> Result<Plan, ProofVerifyError> {
    let mut steps = Vec::new();
    let buckets: Vec<ClampBucket> = clamp_buckets(model)?
        .iter()
        .filter_map(|bucket| {
            clamp_split::live_bucket(bucket, |idx| {
                !clamp_split::exact_output_verifier(model, io, idx)
            })
        })
        .collect();
    if !buckets.is_empty() {
        for bucket in &buckets {
            steps.push(Step::Open(OpOpen::ClampExactFlag(bucket.idx)));
            steps.push(Step::Tie {
                relation: OpTie::ClampBucketExact(bucket.clone()),
            });
        }
        steps.push(Step::Batch {
            proof: ProofType::DeferredClampSplit,
            relations: buckets
                .iter()
                .map(|bucket| OpRelation::ClampBucketValue(bucket.clone()))
                .collect(),
        });
        steps.push(Step::Batch {
            proof: ProofType::DeferredClampDigitRange,
            relations: buckets
                .iter()
                .map(|bucket| OpRelation::ClampBucketDigits(bucket.clone()))
                .collect(),
        });
        steps.extend(buckets.into_iter().map(|bucket| Step::Defer {
            batch: DeferredBatch::OneHot,
            relation: OpRelation::ClampBucketSlack(bucket),
        }));
    }
    let rbuckets = remainder_buckets(model)?;
    if !rbuckets.is_empty() {
        steps.push(Step::Batch {
            proof: ProofType::DeferredRemainderRC,
            relations: rbuckets
                .iter()
                .map(|bucket| OpRelation::RemainderBucketValue(bucket.clone()))
                .collect(),
        });
        steps.push(Step::Batch {
            proof: ProofType::DeferredRemainderDigitRange,
            relations: rbuckets
                .into_iter()
                .map(OpRelation::RemainderBucketDigits)
                .collect(),
        });
    }
    Ok(Plan(steps))
}
