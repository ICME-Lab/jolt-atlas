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
    global_clamp::{clamp_buckets, remainder_buckets, ClampBucket, GlobalClampEncoding},
    ProofId, ProofType, Prover, Verifier,
};
use atlas_onnx_tracer::node::ComputationNode;
use joltworks::{
    field::JoltField,
    lookup_tables::clamp::SaturationTable,
    poly::opening_proof::OpeningAccumulator,
    subprotocols::{
        identity_range_check::{
            identity_rangecheck_prover_with_cycle, identity_rangecheck_verifier_with_cycle,
        },
        ps_shout::unary::{ps_read_raf_prover_with_gamma, ps_read_raf_verifier_with_gamma},
        ps_shout::{unary::ReadRafClaims, CycleWeight},
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

/// Monomorphize `$body` over the node's statically derived clamp width `$w`
/// (see `atlas_onnx_tracer::model::clamp_width::CLAMP_WIDTHS`).
macro_rules! with_clamp_width {
    ($bits:expr, $w:ident => $body:expr) => {
        match $bits {
            40 => {
                const $w: usize = 40;
                $body
            }
            48 => {
                const $w: usize = 48;
                $body
            }
            56 => {
                const $w: usize = 56;
                $body
            }
            64 => {
                const $w: usize = 64;
                $body
            }
            other => panic!("unsupported saturating-clamp width {other}"),
        }
    };
}

/// A lookup the prover has registered during a node's turn and will prove in
/// [`prove_all`].
pub enum ProverLookupJob {
    /// `output = SatClamp(acc)` for a non-scalar node.
    Clamp {
        /// The clamped node.
        node: ComputationNode,
        /// 64-bit lookup addresses (the pre-clamp accumulation per element).
        lookup_bits: Vec<LookupBits>,
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
    let mut clamp: Vec<(ComputationNode, Vec<LookupBits>)> = Vec::new();
    let mut remainder: Vec<(ComputationNode, i32, Vec<LookupBits>)> = Vec::new();
    for job in std::mem::take(&mut prover.deferred) {
        match job {
            ProverLookupJob::Clamp { node, lookup_bits } => clamp.push((node, lookup_bits)),
            ProverLookupJob::RescaleRemainder {
                node,
                bits,
                lookup_bits,
            } => remainder.push((node, bits, lookup_bits)),
        }
    }

    // Clamp lookups: packed into buckets (see `global_clamp`).
    let buckets = clamp_buckets(prover.preprocessing.model());
    let mut bits_by_node: HashMap<usize, Vec<LookupBits>> =
        clamp.into_iter().map(|(n, bits)| (n.idx, bits)).collect();
    for b in &buckets {
        for n in &b.nodes {
            assert!(
                bits_by_node.contains_key(&n.idx),
                "node {} is in a clamp bucket but registered no clamp lookup",
                n.idx
            );
        }
    }
    if !buckets.is_empty() {
        let (assembled, indices) = assemble_buckets(&buckets, &mut bits_by_node);
        assert!(
            bits_by_node.is_empty(),
            "clamp lookups registered for nodes outside every bucket: {:?}",
            bits_by_node.keys().collect::<Vec<_>>()
        );

        // Challenges are drawn sequentially in bucket order; the (heavy)
        // instance construction then runs in parallel across buckets.
        let _prep =
            tracing::span!(tracing::Level::INFO, "deferred::prepare_clamp_read_raf").entered();
        let prepared: Vec<(F, ReadRafClaims<F>, CycleWeight<F>)> = buckets
            .iter()
            .map(|b| {
                let gammas: Vec<F> = prover.transcript.challenge_scalar_powers(b.nodes.len());
                let gamma: F = prover.transcript.challenge_scalar();
                let (claims, cycle) = b.read_raf_inputs(&gammas, &prover.accumulator);
                (gamma, claims, cycle)
            })
            .collect();
        drop(_prep);
        let instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = {
            let _span =
                tracing::span!(tracing::Level::INFO, "deferred::build_clamp_read_raf").entered();
            buckets
                .par_iter()
                .zip(assembled.into_par_iter())
                .zip(prepared.into_par_iter())
                .map(|((b, bits), (gamma, claims, cycle))| {
                    let ra_poly = b.ra_poly();
                    with_clamp_width!(b.width, W => Box::new(
                        ps_read_raf_prover_with_gamma::<F, SaturationTable<W>, W>(
                            gamma, claims, cycle, ra_poly, bits,
                        ),
                    )
                        as Box<dyn SumcheckInstanceProver<F, T>>)
                })
                .collect()
        };
        batched_prove(instances, prover, proofs, ProofType::DeferredClampReadRaf);

        let params: Vec<RaOneHotParams<F>> = buckets
            .iter()
            .map(|b| {
                let enc = GlobalClampEncoding(b);
                let ch = shout::ra_onehot_challenges::<F, T>(&enc, &mut prover.transcript);
                shout::ra_onehot_params(&enc, &prover.accumulator, ch)
            })
            .collect();
        let instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = {
            let _span =
                tracing::span!(tracing::Level::INFO, "deferred::build_clamp_onehot").entered();
            params
                .into_par_iter()
                .zip(indices.par_iter())
                .flat_map_iter(|(p, idx)| shout::ra_onehot_provers_from_params::<F, T>(p, idx))
                .collect()
        };
        drop_in_background_thread(indices);
        batched_prove(instances, prover, proofs, ProofType::DeferredClampRaChecks);
    }

    // Rescale-remainder range checks: same packing.
    let rbuckets = remainder_buckets(prover.preprocessing.model());
    let mut rbits_by_node: HashMap<usize, Vec<LookupBits>> = remainder
        .into_iter()
        .map(|(n, _bits, lookup_bits)| (n.idx, lookup_bits))
        .collect();
    if !rbuckets.is_empty() {
        let (assembled, indices) = assemble_buckets(&rbuckets, &mut rbits_by_node);
        assert!(
            rbits_by_node.is_empty(),
            "remainder lookups registered for nodes outside every bucket: {:?}",
            rbits_by_node.keys().collect::<Vec<_>>()
        );
        let prepared: Vec<(F, CycleWeight<F>)> = rbuckets
            .iter()
            .map(|b| {
                let gammas: Vec<F> = prover.transcript.challenge_scalar_powers(b.nodes.len());
                b.remainder_inputs(&gammas, &prover.accumulator, prover.preprocessing.model())
            })
            .collect();
        let instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = {
            let _span =
                tracing::span!(tracing::Level::INFO, "deferred::build_remainder_rc").entered();
            rbuckets
                .par_iter()
                .zip(assembled.into_par_iter())
                .zip(prepared.into_par_iter())
                .map(|((b, bits), (input_claim, cycle))| {
                    Box::new(identity_rangecheck_prover_with_cycle(
                        cycle,
                        b.width,
                        b.ra_poly(),
                        input_claim,
                        bits,
                    )) as Box<dyn SumcheckInstanceProver<F, T>>
                })
                .collect()
        };
        batched_prove(instances, prover, proofs, ProofType::DeferredRemainderRC);

        let params: Vec<RaOneHotParams<F>> = rbuckets
            .iter()
            .map(|b| {
                let enc = GlobalClampEncoding(b);
                let ch = shout::ra_onehot_challenges::<F, T>(&enc, &mut prover.transcript);
                shout::ra_onehot_params(&enc, &prover.accumulator, ch)
            })
            .collect();
        let instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = {
            let _span =
                tracing::span!(tracing::Level::INFO, "deferred::build_remainder_onehot").entered();
            params
                .into_par_iter()
                .zip(indices.par_iter())
                .flat_map_iter(|(p, idx)| shout::ra_onehot_provers_from_params::<F, T>(p, idx))
                .collect()
        };
        drop_in_background_thread(indices);
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

    let buckets = clamp_buckets(verifier.preprocessing.model());
    let registered: std::collections::BTreeSet<usize> = clamp.iter().map(|n| n.idx).collect();
    let bucketed: std::collections::BTreeSet<usize> = buckets
        .iter()
        .flat_map(|b| b.nodes.iter().map(|n| n.idx))
        .collect();
    if registered != bucketed {
        return Err(ProofVerifyError::InvalidOpeningProof(
            "clamp lookup registrations do not match the model's clamp buckets".to_string(),
        ));
    }
    if !buckets.is_empty() {
        let instances: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> = buckets
            .iter()
            .map(|b| {
                let gammas: Vec<F> = verifier.transcript.challenge_scalar_powers(b.nodes.len());
                let gamma: F = verifier.transcript.challenge_scalar();
                let (claims, cycle) = b.read_raf_inputs(&gammas, &verifier.accumulator);
                let ra_poly = b.ra_poly();
                with_clamp_width!(b.width, W => Box::new(
                    ps_read_raf_verifier_with_gamma::<F, SaturationTable<W>, W>(
                        gamma, claims, cycle, ra_poly,
                    ),
                )
                    as Box<dyn SumcheckInstanceVerifier<F, T>>)
            })
            .collect();
        batched_verify(instances, verifier, ProofType::DeferredClampReadRaf)?;

        let mut instances: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> =
            Vec::with_capacity(3 * buckets.len());
        for b in &buckets {
            let encoding = GlobalClampEncoding(b);
            instances.extend(shout::ra_onehot_verifiers(
                &encoding,
                &verifier.accumulator,
                &mut verifier.transcript,
            ));
        }
        batched_verify(instances, verifier, ProofType::DeferredClampRaChecks)?;
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
                Box::new(identity_rangecheck_verifier_with_cycle(
                    cycle,
                    b.width,
                    b.ra_poly(),
                    input_claim,
                )) as Box<dyn SumcheckInstanceVerifier<F, T>>
            })
            .collect();
        batched_verify(instances, verifier, ProofType::DeferredRemainderRC)?;

        let mut instances: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> =
            Vec::with_capacity(3 * rbuckets.len());
        for b in &rbuckets {
            let encoding = GlobalClampEncoding(b);
            instances.extend(shout::ra_onehot_verifiers(
                &encoding,
                &verifier.accumulator,
                &mut verifier.transcript,
            ));
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
