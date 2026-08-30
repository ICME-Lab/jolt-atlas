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
use crate::onnx_proof::{
    clamp_split::{
        self, BucketSplit, NodeSplit, SplitParams, SplitProver, SplitVerifier, ValueParams,
        ValueProver, ValueVerifier,
    },
    global_clamp::{clamp_buckets, remainder_buckets, ClampBucket},
    ProofId, ProofType, Prover, Verifier,
};
use atlas_onnx_tracer::node::ComputationNode;
use joltworks::{
    field::JoltField,
    poly::opening_proof::{OpeningAccumulator, OpeningPoint},
    subprotocols::{
        shout::{self, RaOneHotEncoding, RaOneHotParams},
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, lookup_bits::LookupBits, thread::drop_in_background_thread},
};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::collections::HashMap;

/// Node index under which the deferred (model-wide) lookup proofs are stored.
pub const DEFERRED_PROOF_IDX: usize = usize::MAX;

/// A lookup the prover has registered during a node's turn and will prove in
/// [`prove_all`].
pub enum ProverLookupJob {
    /// `output = SatClamp(acc)` for a non-scalar node.
    Clamp {
        /// The clamped node.
        node: ComputationNode,
        /// The node's interior / saturated split witness.
        split: NodeSplit,
    },
    /// The fused rescale remainder range check `R ∈ [0, 2^bits)`.
    RescaleRemainder {
        /// The fused node.
        node: ComputationNode,
        /// Width of the rebase (`R < 2^bits`).
        bits: i32,
        /// `bits`-wide lookup addresses (the remainder per element).
        lookup_bits: Vec<LookupBits>,
    },
}

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
    fn proof_type(self) -> ProofType {
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
    fn parent(&self) -> DeferredBatch {
        match self {
            Self::ActivationSmall(_) => DeferredBatch::ActivationExec,
            Self::ActivationClamp(_) => DeferredBatch::ActivationClamp,
            Self::RangeCheck(_) => DeferredBatch::RangeCheck,
            Self::SoftmaxExpHi(..) | Self::SoftmaxExpLo(..) | Self::SoftmaxClamp(_) => {
                DeferredBatch::SoftmaxStage3
            }
        }
    }

    fn proof_type(&self) -> ProofType {
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
    fn params<F: JoltField, T: Transcript>(
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

fn batched_prove<F: JoltField, T: Transcript>(
    mut instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>>,
    prover: &mut Prover<F, T>,
    proofs: &mut BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
    proof_type: ProofType,
) {
    let (proof, _) = BatchedSumcheck::prove_parallel_instances(
        instances.iter_mut().map(|v| &mut **v as _).collect(),
        &mut prover.accumulator,
        &mut prover.transcript,
    );
    drop_in_background_thread(instances);
    proofs.insert(ProofId(DEFERRED_PROOF_IDX, proof_type), proof);
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

/// Prove every registered lookup job: clamp read-rafs, clamp one-hot checks,
/// remainder range checks, remainder one-hot checks — one batched sumcheck each.
#[tracing::instrument(skip_all, name = "deferred_lookups::prove_all")]
pub fn prove_all<F: JoltField, T: Transcript>(
    prover: &mut Prover<F, T>,
    proofs: &mut BTreeMap<ProofId, SumcheckInstanceProof<F, T>>,
) {
    let mut clamp: Vec<(ComputationNode, NodeSplit)> = Vec::new();
    let mut remainder: Vec<(ComputationNode, i32, Vec<LookupBits>)> = Vec::new();
    for job in std::mem::take(&mut prover.deferred) {
        match job {
            ProverLookupJob::Clamp { node, split } => clamp.push((node, split)),
            ProverLookupJob::RescaleRemainder {
                node,
                bits,
                lookup_bits,
            } => remainder.push((node, bits, lookup_bits)),
        }
    }

    // Clamp lookups: packed into buckets (see `global_clamp`), proven by the
    // interior / saturated split (see `clamp_split`).
    let model = prover.preprocessing.model();
    let mut split_by_node: HashMap<usize, NodeSplit> =
        clamp.into_iter().map(|(n, s)| (n.idx, s)).collect();
    // Exact outputs (see `clamp_split::exact_output_prover`) register no
    // split and are gaps; a bucket with no live node is skipped entirely.
    let full_buckets = clamp_buckets(model);
    for b in &full_buckets {
        for n in &b.nodes {
            assert!(
                split_by_node.contains_key(&n.idx)
                    || clamp_split::exact_output_prover(model, &prover.trace, n.idx),
                "node {} is in a clamp bucket but registered no clamp lookup",
                n.idx
            );
        }
    }
    let buckets: Vec<ClampBucket> = full_buckets
        .iter()
        .filter_map(|b| clamp_split::live_bucket(b, |idx| split_by_node.contains_key(&idx)))
        .collect();
    if !buckets.is_empty() {
        let splits: Vec<BucketSplit> = {
            let _span =
                tracing::span!(tracing::Level::INFO, "deferred::assemble_buckets").entered();
            buckets
                .iter()
                .map(|b| BucketSplit::assemble(b, |idx| split_by_node.remove(&idx)))
                .collect()
        };
        assert!(
            split_by_node.is_empty(),
            "clamp lookups registered for nodes outside every bucket: {:?}",
            split_by_node.keys().collect::<Vec<_>>()
        );

        // Each bucket first declares (as a public claim) whether it has any
        // saturated element; unsaturated buckets are proven exactly (value
        // sumcheck on OUT + per-node `acc = out` on the verifier), the others
        // by the split sumcheck. Challenges are drawn sequentially in bucket
        // order; the (heavy) instance construction then runs in parallel.
        let exact: Vec<bool> = splits.iter().map(|s| !s.saturated()).collect();
        for (b, &e) in buckets.iter().zip(&exact) {
            prover.accumulator.append_virtual(
                &mut prover.transcript,
                clamp_split::exact_flag_id(b.idx),
                OpeningPoint::new(Vec::<F>::new()),
                if e { F::one() } else { F::zero() },
            );
        }
        enum Params<F: JoltField> {
            Split(SplitParams<F>),
            Exact(ValueParams<F>),
        }
        let params: Vec<Params<F>> = buckets
            .iter()
            .zip(&exact)
            .map(|(b, &e)| {
                if e {
                    Params::Exact(clamp_split::exact_value_params(
                        b,
                        &prover.accumulator,
                        &mut prover.transcript,
                    ))
                } else {
                    Params::Split(SplitParams::draw(
                        b,
                        &prover.accumulator,
                        &mut prover.transcript,
                    ))
                }
            })
            .collect();
        let instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = {
            let _span =
                tracing::span!(tracing::Level::INFO, "deferred::build_clamp_split").entered();
            params
                .into_par_iter()
                .zip(splits.par_iter())
                .map(|(p, s)| match p {
                    Params::Split(p) => {
                        Box::new(SplitProver::new(p, s)) as Box<dyn SumcheckInstanceProver<F, T>>
                    }
                    Params::Exact(p) => Box::new(ValueProver::new(
                        p,
                        s.out.iter().map(|&o| o as u64).collect(),
                    ))
                        as Box<dyn SumcheckInstanceProver<F, T>>,
                })
                .collect()
        };
        batched_prove(instances, prover, proofs, ProofType::DeferredClampSplit);

        let params: Vec<_> = buckets
            .iter()
            .zip(&exact)
            .map(|(b, &e)| {
                if e {
                    clamp_split::exact_chunk_check_params::<F, T>(
                        b,
                        &prover.accumulator,
                        &mut prover.transcript,
                    )
                } else {
                    clamp_split::chunk_check_params::<F, T>(
                        b,
                        &prover.accumulator,
                        &mut prover.transcript,
                    )
                }
            })
            .collect();
        let instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = {
            let _span =
                tracing::span!(tracing::Level::INFO, "deferred::build_clamp_chunks").entered();
            params
                .into_par_iter()
                .zip(splits.par_iter())
                .zip(exact.par_iter())
                .flat_map_iter(|((p, s), &e)| {
                    if e {
                        clamp_split::exact_chunk_check_provers::<F, T>(p, s)
                    } else {
                        clamp_split::chunk_check_provers::<F, T>(p, s)
                    }
                })
                .collect()
        };
        drop_in_background_thread(splits);
        batched_prove(
            instances,
            prover,
            proofs,
            ProofType::DeferredClampChunkChecks,
        );
    }

    // Rescale-remainder range checks: same packing, proven by a value
    // sumcheck + Booleanity with the chunk-value linear term (see `clamp_split`).
    let rbuckets = remainder_buckets(prover.preprocessing.model());
    let mut rbits_by_node: HashMap<usize, Vec<LookupBits>> = remainder
        .into_iter()
        .map(|(n, _bits, lookup_bits)| (n.idx, lookup_bits))
        .collect();
    if !rbuckets.is_empty() {
        let (_assembled, indices) = assemble_buckets(&rbuckets, &mut rbits_by_node);
        assert!(
            rbits_by_node.is_empty(),
            "remainder lookups registered for nodes outside every bucket: {:?}",
            rbits_by_node.keys().collect::<Vec<_>>()
        );
        let values: Vec<Vec<u64>> = indices
            .into_par_iter()
            .map(|idx| idx.into_iter().map(|x| x as u64).collect())
            .collect();
        let model = prover.preprocessing.model();
        let params: Vec<ValueParams<F>> = rbuckets
            .iter()
            .map(|b| {
                let gammas: Vec<F> = prover.transcript.challenge_scalar_powers(b.nodes.len());
                let (input_claim, cycle) = b.remainder_inputs(&gammas, &prover.accumulator, model);
                ValueParams {
                    bucket_idx: b.idx,
                    value_id: clamp_split::remainder_value_id(b.idx),
                    cycle,
                    input_claim,
                }
            })
            .collect();
        let instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = {
            let _span =
                tracing::span!(tracing::Level::INFO, "deferred::build_remainder_rc").entered();
            params
                .into_par_iter()
                .zip(values.par_iter())
                .map(|(p, v)| {
                    Box::new(ValueProver::new(p, v.clone()))
                        as Box<dyn SumcheckInstanceProver<F, T>>
                })
                .collect()
        };
        batched_prove(instances, prover, proofs, ProofType::DeferredRemainderRC);

        let params: Vec<_> = rbuckets
            .iter()
            .map(|b| {
                clamp_split::remainder_chunk_check_params::<F, T>(
                    b,
                    &prover.accumulator,
                    &mut prover.transcript,
                )
            })
            .collect();
        let instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = {
            let _span =
                tracing::span!(tracing::Level::INFO, "deferred::build_remainder_onehot").entered();
            params
                .into_par_iter()
                .zip(values.par_iter())
                .zip(rbuckets.par_iter())
                .flat_map_iter(|((p, v), b)| {
                    clamp_split::remainder_chunk_check_provers::<F, T>(p, v, b.width)
                })
                .collect()
        };
        drop_in_background_thread(values);
        batched_prove(
            instances,
            prover,
            proofs,
            ProofType::DeferredRemainderRaChecks,
        );
    }

    // Instances built during the node loop, batched here; each batch is
    // followed by the one-hot checks that consume its `ra` claims.
    let mut batches = std::mem::take(&mut prover.deferred_batches);
    let onehots = std::mem::take(&mut prover.deferred_onehots);
    for kind in [
        DeferredBatch::ActivationExec,
        DeferredBatch::ActivationClamp,
        DeferredBatch::RangeCheck,
        DeferredBatch::SoftmaxStage3,
    ] {
        if let Some(instances) = batches.remove(&kind) {
            if !instances.is_empty() {
                batched_prove(instances, prover, proofs, kind.proof_type());
            }
        }
        let mine: Vec<&(DeferredOneHot, Vec<usize>)> =
            onehots.iter().filter(|(j, _)| j.parent() == kind).collect();
        if mine.is_empty() {
            continue;
        }
        let params: Vec<RaOneHotParams<F>> = mine
            .iter()
            .map(|(j, _)| j.params::<F, T>(&prover.accumulator, &mut prover.transcript))
            .collect();
        let instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = params
            .into_par_iter()
            .zip(mine.par_iter())
            .flat_map_iter(|(p, (_, idx))| shout::ra_onehot_provers_from_params::<F, T>(p, idx))
            .collect();
        let proof_type = mine[0].0.proof_type();
        batched_prove(instances, prover, proofs, proof_type);
    }
    drop_in_background_thread(onehots);
}

/// Lay every bucket's per-node lookup bits into its cycle space (in parallel
/// across buckets), returning the assembled bits and their `usize` indices.
#[tracing::instrument(skip_all, name = "deferred::assemble_buckets")]
fn assemble_buckets(
    buckets: &[ClampBucket],
    bits_by_node: &mut HashMap<usize, Vec<LookupBits>>,
) -> (Vec<Vec<LookupBits>>, Vec<Vec<usize>>) {
    // Hand each bucket its nodes' bits (moved, no copies) …
    let per_bucket: Vec<Vec<(usize, Vec<LookupBits>)>> = buckets
        .iter()
        .map(|b| {
            b.nodes
                .iter()
                .map(|n| {
                    (
                        n.idx,
                        bits_by_node
                            .remove(&n.idx)
                            .expect("registered clamp lookup"),
                    )
                })
                .collect()
        })
        .collect();
    // … then assemble and index in parallel.
    buckets
        .par_iter()
        .zip(per_bucket.into_par_iter())
        .map(|(b, mut owned)| {
            let mut by_idx: HashMap<usize, Vec<LookupBits>> = owned.drain(..).collect();
            let bits = b.assemble_bits(|idx| by_idx.remove(&idx).unwrap());
            let indices: Vec<usize> = bits.iter().map(|&x| x.into()).collect();
            (bits, indices)
        })
        .unzip()
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
                    Box::new(ValueVerifier::new(clamp_split::exact_value_params(
                        b,
                        &verifier.accumulator,
                        &mut verifier.transcript,
                    ))) as Box<dyn SumcheckInstanceVerifier<F, T>>
                } else {
                    Box::new(SplitVerifier::new(SplitParams::draw(
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
                Box::new(ValueVerifier::new(ValueParams {
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
