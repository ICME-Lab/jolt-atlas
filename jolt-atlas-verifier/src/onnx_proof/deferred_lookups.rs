//! The post-loop plan: batched lookup sub-protocols.
//!
//! A clamped or fused-rescale node's lookup sumchecks (a clamp read-raf, a remainder range
//! check, their one-hot checks) consume claims appended during the node's own turn in the IOP
//! and produce claims only on their own committed one-hots, so they need not run in the node
//! loop: a plan step [`Step::Defer`](super::plan::Step::Defer) queues its relation for one of
//! the [`DeferredBatch`]es, and every batch is proven as one batched sumcheck after the loop,
//! in this module's order. The saturating clamps and the rescale remainders are packed further,
//! into model-level buckets ([`global_clamp`](super::global_clamp)); [`plan`] builds their steps
//! from the model and the public IO, with nothing to register during the loop.
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
/// run in this enum's order; a one-hot batch directly follows the batch producing the `ra`
/// claims it checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeferredBatch {
    /// Clamped-activation small-table execution sumchecks.
    ActivationExec,
    /// The one-hot checks of [`Self::ActivationExec`]'s read addresses.
    ActivationExecRaChecks,
    /// Clamped-activation clamp read-rafs.
    ActivationClamp,
    /// The one-hot checks of [`Self::ActivationClamp`]'s read addresses.
    ActivationClampRaChecks,
    /// Operand range checks (MeanOfSquares remainder, Rsqrt div / sqrt).
    RangeCheck,
    /// The one-hot checks of [`Self::RangeCheck`]'s read addresses.
    RangeCheckRaChecks,
    /// Softmax stage 3: exp-digit lookups + significance-clamp read-raf.
    SoftmaxStage3,
    /// Softmax stage 4: the one-hot checks of stage 3's read addresses.
    SoftmaxRaChecks,
}

impl DeferredBatch {
    /// Every batch, in run order.
    pub const ALL: [Self; 8] = [
        Self::ActivationExec,
        Self::ActivationExecRaChecks,
        Self::ActivationClamp,
        Self::ActivationClampRaChecks,
        Self::RangeCheck,
        Self::RangeCheckRaChecks,
        Self::SoftmaxStage3,
        Self::SoftmaxRaChecks,
    ];

    /// The proof slot this batch's sumcheck is stored under.
    pub fn proof_type(self) -> ProofType {
        match self {
            Self::ActivationExec => ProofType::DeferredActivationExec,
            Self::ActivationExecRaChecks => ProofType::DeferredActivationSmallRaChecks,
            Self::ActivationClamp => ProofType::DeferredActivationClamp,
            Self::ActivationClampRaChecks => ProofType::DeferredActivationClampRaChecks,
            Self::RangeCheck => ProofType::DeferredRangeCheck,
            Self::RangeCheckRaChecks => ProofType::DeferredRangeCheckRaChecks,
            Self::SoftmaxStage3 => ProofType::DeferredSoftmaxStage3,
            Self::SoftmaxRaChecks => ProofType::DeferredSoftmaxRaChecks,
        }
    }

    /// When this batch's relations are set up: the one-hot batches read the `ra` claim their
    /// parent batch produces, so they wait for it; the others draw their challenges in the
    /// node's turn.
    pub fn setup(self) -> DeferredSetup {
        match self {
            Self::ActivationExec
            | Self::ActivationClamp
            | Self::RangeCheck
            | Self::SoftmaxStage3 => DeferredSetup::InNodeLoop,
            Self::ActivationExecRaChecks
            | Self::ActivationClampRaChecks
            | Self::RangeCheckRaChecks
            | Self::SoftmaxRaChecks => DeferredSetup::AtBatch,
        }
    }
}

/// The post-loop plan over the packed clamp and remainder buckets, from the model and the
/// public IO: per live clamp bucket its exact flag and the exact check, then the bucket
/// value sumchecks as one batch, the `OUT` digit range checks as a second and the slack chunk
/// checks of the buckets not declared exact as a third (absent when every bucket is exact);
/// then the value and chunk-check pair for the remainder buckets. A clamp bucket is live when one of its nodes is not an unsaturated
/// model output (those tie `acc = out` in their own turn). The deferred batches follow,
/// run by the interpreter from its queue.
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
        steps.push(Step::Batch {
            proof: ProofType::DeferredClampChunkChecks,
            relations: buckets
                .into_iter()
                .map(OpRelation::ClampBucketSlack)
                .collect(),
        });
    }
    let rbuckets = remainder_buckets(model);
    if !rbuckets.is_empty() {
        steps.push(Step::Batch {
            proof: ProofType::DeferredRemainderRC,
            relations: rbuckets
                .iter()
                .map(|bucket| OpRelation::RemainderBucketValue(bucket.clone()))
                .collect(),
        });
        steps.push(Step::Batch {
            proof: ProofType::DeferredRemainderRaChecks,
            relations: rbuckets
                .into_iter()
                .map(OpRelation::RemainderBucketChunks)
                .collect(),
        });
    }
    Ok(Plan(steps))
}
