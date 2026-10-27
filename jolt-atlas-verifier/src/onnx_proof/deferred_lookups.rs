//! Deferred, batched lookup sub-protocols.
//!
//! Every clamped / fused-rescale node used to run its own lookup sumchecks
//! inline (a clamp read-raf, its three one-hot checks, a remainder range check
//! and its three one-hot checks): ~5 sumchecks × ~750 nodes, each with 64+
//! rounds whose per-round work is far too small to amortize the per-round
//! overhead (transcript, allocation, rayon fan-out). The claims those lookups
//! consume (the node's output opening, `ClampAcc`, `RescaleRemainder`) are all
//! appended during the node's own turn in the IOP and never change afterwards,
//! and the lookups only *produce* claims on their own committed one-hots — so
//! they can be taken out of the node loop entirely.
//!
//! Nodes now append their claims as before and register a [`ProverLookupJob`];
//! after the IOP, [`prove_all`] runs every clamp read-raf as one
//! [`BatchedSumcheck`], then every clamp one-hot check as one more, and the same
//! pair for the rescale remainders. The verifier mirrors it in [`verify_all`]
//! after its node loop. The batched sumcheck computes instance messages in
//! parallel across instances, which is what turns the ~750 tiny sequential
//! sumchecks into a few large parallel ones.
use crate::node_view::ComputationNode;
use crate::onnx_proof::{
    clamp_split::{self, SplitParams, ValueParams},
    global_clamp::{clamp_buckets, remainder_buckets, ClampBucket},
    ProofId, ProofType, Verifier,
};
use atlas_algebra::{
    field::JoltField, poly::opening_point::OpeningPoint, utils::errors::ProofVerifyError,
};
use atlas_sumcheck::{
    opening_proof::OpeningAccumulator,
    shout::{self, RaOneHotEncoding, RaOneHotParams},
    sumcheck::BatchedSumcheck,
    sumcheck_verifier::SumcheckInstanceVerifier,
};
use atlas_transcript::Transcript;

/// Node index under which the deferred (model-wide) lookup proofs are stored.
pub const DEFERRED_PROOF_IDX: usize = usize::MAX;

/// A batch of sumcheck instances built during the node loop (their transcript
/// challenges drawn there, on both sides) and proven after it as one batched
/// sumcheck. Batches run in this enum's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeferredBatch {
    /// Clamped-activation small-table execution sumchecks.
    ActivationExec,
    /// Clamped-activation clamp read-rafs.
    ActivationClamp,
    /// Operand range checks (MeanOfSquares remainder, Rsqrt div / sqrt).
    RangeCheck,
    /// Softmax stage 3: exp-digit lookups + significance-clamp read-raf.
    SoftmaxStage3,
}

impl DeferredBatch {
    /// The proof slot this batch's sumcheck is stored under.
    pub fn proof_type(self) -> ProofType {
        match self {
            Self::ActivationExec => ProofType::DeferredActivationExec,
            Self::ActivationClamp => ProofType::DeferredActivationClamp,
            Self::RangeCheck => ProofType::DeferredRangeCheck,
            Self::SoftmaxStage3 => ProofType::DeferredSoftmaxStage3,
        }
    }
}

/// A one-hot check whose `ra` claim is produced by a deferred batch; built and
/// proven right after that batch.
#[derive(Clone)]
pub enum DeferredOneHot {
    /// Activation small-table stage of node `idx`.
    ActivationSmall(usize),
    /// Activation clamp stage of `node`.
    ActivationClamp(ComputationNode),
    /// A range check's one-hot encoding.
    RangeCheck(crate::onnx_proof::range_checking::RangeCheckEncoding),
    /// Softmax exp high-digit table of node `idx` (`log2` table size).
    SoftmaxExpHi(usize, usize),
    /// Softmax exp low-digit table of node `idx` (`log2` table size).
    SoftmaxExpLo(usize, usize),
    /// Softmax significance clamp of `node`.
    SoftmaxClamp(ComputationNode),
}

impl DeferredOneHot {
    /// The batch whose sumcheck produces this check's `ra` claim.
    pub fn parent(&self) -> DeferredBatch {
        match self {
            Self::ActivationSmall(_) => DeferredBatch::ActivationExec,
            Self::ActivationClamp(_) => DeferredBatch::ActivationClamp,
            Self::RangeCheck(_) => DeferredBatch::RangeCheck,
            Self::SoftmaxExpHi(..) | Self::SoftmaxExpLo(..) | Self::SoftmaxClamp(_) => {
                DeferredBatch::SoftmaxStage3
            }
        }
    }

    /// The proof slot this check's one-hot sumcheck is stored under.
    pub fn proof_type(&self) -> ProofType {
        match self {
            Self::ActivationSmall(_) => ProofType::DeferredActivationSmallRaChecks,
            Self::ActivationClamp(_) => ProofType::DeferredActivationClampRaChecks,
            Self::RangeCheck(_) => ProofType::DeferredRangeCheckRaChecks,
            Self::SoftmaxExpHi(..) | Self::SoftmaxExpLo(..) | Self::SoftmaxClamp(_) => {
                ProofType::DeferredSoftmaxRaChecks
            }
        }
    }

    /// Draw this check's challenges and resolve its params (sequential part).
    pub fn params<F: JoltField, T: Transcript>(
        &self,
        accumulator: &dyn OpeningAccumulator<F>,
        transcript: &mut T,
    ) -> RaOneHotParams<F> {
        use crate::onnx_proof::{
            op_lookups::OpLookupEncoding,
            ops::{
                activation_clamped::{ActivationClampOperands, SmallTableRaEncoding},
                softmax_last_axis::{
                    rc::SoftmaxRaEncoding, significance_clamp::SoftmaxSignificanceClampOperands,
                },
            },
        };
        fn go<F: JoltField, T: Transcript>(
            enc: &impl RaOneHotEncoding,
            accumulator: &dyn OpeningAccumulator<F>,
            transcript: &mut T,
        ) -> RaOneHotParams<F> {
            let ch = shout::ra_onehot_challenges::<F, T>(enc, transcript);
            shout::ra_onehot_params(enc, accumulator, ch)
        }
        match self {
            Self::ActivationSmall(idx) => go(
                &SmallTableRaEncoding { node_idx: *idx },
                accumulator,
                transcript,
            ),
            Self::ActivationClamp(node) => go(
                &OpLookupEncoding::<ActivationClampOperands>::new(node),
                accumulator,
                transcript,
            ),
            Self::RangeCheck(enc) => go(enc, accumulator, transcript),
            Self::SoftmaxExpHi(idx, log_k) => go(
                &SoftmaxRaEncoding::exp_hi(*idx, *log_k),
                accumulator,
                transcript,
            ),
            Self::SoftmaxExpLo(idx, log_k) => go(
                &SoftmaxRaEncoding::exp_lo(*idx, *log_k),
                accumulator,
                transcript,
            ),
            Self::SoftmaxClamp(node) => go(
                &OpLookupEncoding::<SoftmaxSignificanceClampOperands>::new(node),
                accumulator,
                transcript,
            ),
        }
    }
}

/// Verifier-side counterpart of [`ProverLookupJob`].
pub enum VerifierLookupJob {
    /// See [`ProverLookupJob::Clamp`].
    Clamp {
        /// The clamped node.
        node: ComputationNode,
    },
    /// See [`ProverLookupJob::RescaleRemainder`].
    RescaleRemainder {
        /// The fused node.
        node: ComputationNode,
        /// Width of the rebase.
        bits: i32,
    },
}

fn batched_verify<F: JoltField, T: Transcript>(
    instances: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>>,
    verifier: &mut Verifier<'_, F, T>,
    proof_type: ProofType,
) -> Result<(), ProofVerifyError> {
    let proof = verifier
        .proofs
        .get(&ProofId(DEFERRED_PROOF_IDX, proof_type))
        .ok_or(ProofVerifyError::MissingProof(DEFERRED_PROOF_IDX))?;
    BatchedSumcheck::verify(
        proof,
        instances.iter().map(|v| &**v as _).collect(),
        &mut verifier.accumulator,
        &mut verifier.transcript,
    )?;
    Ok(())
}

/// Verifier counterpart of [`prove_all`].
#[tracing::instrument(skip_all, name = "deferred_lookups::verify_all")]
pub fn verify_all<F: JoltField, T: Transcript>(
    verifier: &mut Verifier<'_, F, T>,
) -> Result<(), ProofVerifyError> {
    let mut clamp: Vec<ComputationNode> = Vec::new();
    let mut remainder: Vec<(ComputationNode, i32)> = Vec::new();
    for job in std::mem::take(&mut verifier.deferred) {
        match job {
            VerifierLookupJob::Clamp { node } => clamp.push(node),
            VerifierLookupJob::RescaleRemainder { node, bits } => remainder.push((node, bits)),
        }
    }

    let model = verifier.preprocessing.model();
    let registered: std::collections::BTreeSet<usize> = clamp.iter().map(|n| n.idx).collect();
    let full_buckets = clamp_buckets(model);
    let bucketed: std::collections::BTreeSet<usize> = full_buckets
        .iter()
        .flat_map(|b| b.nodes.iter().map(|n| n.idx))
        .filter(|&idx| !clamp_split::exact_output_verifier(model, verifier.io, idx))
        .collect();
    if registered != bucketed {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "clamp lookup registrations do not match the model's clamp buckets".to_string(),
        ));
    }
    let buckets: Vec<ClampBucket> = full_buckets
        .iter()
        .filter_map(|b| clamp_split::live_bucket(b, |idx| registered.contains(&idx)))
        .collect();
    if !buckets.is_empty() {
        let mut exact: Vec<bool> = Vec::with_capacity(buckets.len());
        for b in &buckets {
            let id = clamp_split::exact_flag_id(b.idx);
            verifier.accumulator.append_virtual(
                &mut verifier.transcript,
                id,
                OpeningPoint::new(Vec::<F>::new()),
            );
            let flag = verifier.accumulator.get_virtual_polynomial_opening(id).1;
            let e = if flag == F::one() {
                true
            } else if flag == F::zero() {
                false
            } else {
                return Err(ProofVerifyError::InvalidOpeningProof(format!(
                    "clamp bucket {}: exact flag must be 0 or 1",
                    b.idx
                )));
            };
            if e {
                clamp_split::verify_exact_nodes(b, &verifier.accumulator)?;
            }
            exact.push(e);
        }
        let instances: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> = buckets
            .iter()
            .zip(&exact)
            .map(|(b, &e)| {
                if e {
                    Box::new(ValueParams::into_verifier(clamp_split::exact_value_params(
                        b,
                        &verifier.accumulator,
                        &mut verifier.transcript,
                    ))) as Box<dyn SumcheckInstanceVerifier<F, T>>
                } else {
                    Box::new(SplitParams::into_verifier(SplitParams::draw(
                        b,
                        &verifier.accumulator,
                        &mut verifier.transcript,
                    ))) as Box<dyn SumcheckInstanceVerifier<F, T>>
                }
            })
            .collect();
        batched_verify(instances, verifier, ProofType::DeferredClampSplit)?;

        let mut instances: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> =
            Vec::with_capacity(buckets.len());
        for (b, &e) in buckets.iter().zip(&exact) {
            let p = if e {
                clamp_split::exact_chunk_check_params::<F, T>(
                    b,
                    &verifier.accumulator,
                    &mut verifier.transcript,
                )
            } else {
                clamp_split::chunk_check_params::<F, T>(
                    b,
                    &verifier.accumulator,
                    &mut verifier.transcript,
                )
            };
            instances.extend(clamp_split::chunk_check_verifiers::<F, T>(p));
        }
        batched_verify(instances, verifier, ProofType::DeferredClampChunkChecks)?;
    }

    let rbuckets = remainder_buckets(verifier.preprocessing.model());
    let registered: std::collections::BTreeSet<usize> =
        remainder.iter().map(|(n, _)| n.idx).collect();
    let bucketed: std::collections::BTreeSet<usize> = rbuckets
        .iter()
        .flat_map(|b| b.nodes.iter().map(|n| n.idx))
        .collect();
    if registered != bucketed {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "remainder lookup registrations do not match the model's remainder buckets".to_string(),
        ));
    }
    if !rbuckets.is_empty() {
        let model = verifier.preprocessing.model();
        let instances: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> = rbuckets
            .iter()
            .map(|b| {
                let gammas: Vec<F> = verifier.transcript.challenge_scalar_powers(b.nodes.len());
                let (input_claim, cycle) =
                    b.remainder_inputs(&gammas, &verifier.accumulator, model);
                Box::new(ValueParams::into_verifier(ValueParams {
                    bucket_idx: b.idx,
                    value_id: clamp_split::remainder_value_id(b.idx),
                    cycle,
                    input_claim,
                })) as Box<dyn SumcheckInstanceVerifier<F, T>>
            })
            .collect();
        batched_verify(instances, verifier, ProofType::DeferredRemainderRC)?;

        let mut instances: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> =
            Vec::with_capacity(rbuckets.len());
        for b in &rbuckets {
            let p = clamp_split::remainder_chunk_check_params::<F, T>(
                b,
                &verifier.accumulator,
                &mut verifier.transcript,
            );
            instances.extend(clamp_split::chunk_check_verifiers::<F, T>(p));
        }
        batched_verify(instances, verifier, ProofType::DeferredRemainderRaChecks)?;
    }

    let mut batches = std::mem::take(&mut verifier.deferred_batches);
    let onehots = std::mem::take(&mut verifier.deferred_onehots);
    for kind in [
        DeferredBatch::ActivationExec,
        DeferredBatch::ActivationClamp,
        DeferredBatch::RangeCheck,
        DeferredBatch::SoftmaxStage3,
    ] {
        if let Some(instances) = batches.remove(&kind) {
            if !instances.is_empty() {
                batched_verify(instances, verifier, kind.proof_type())?;
            }
        }
        let mine: Vec<&DeferredOneHot> = onehots.iter().filter(|j| j.parent() == kind).collect();
        if mine.is_empty() {
            continue;
        }
        let mut instances: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> =
            Vec::with_capacity(3 * mine.len());
        for j in &mine {
            let p = j.params::<F, T>(&verifier.accumulator, &mut verifier.transcript);
            instances.extend(shout::ra_onehot_verifiers_with_params::<F, T>(p));
        }
        batched_verify(instances, verifier, mine[0].proof_type())?;
    }
    Ok(())
}
