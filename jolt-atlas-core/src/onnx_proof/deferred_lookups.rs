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
    fused_rebase::{RescaleRemainderRCProvider, RescaleRemainderRaEncoding},
    global_clamp::{clamp_buckets, GlobalClampEncoding},
    ProofId, ProofType, Prover, Verifier,
};
use atlas_onnx_tracer::node::ComputationNode;
use common::VirtualPoly;
use joltworks::{
    field::JoltField,
    lookup_tables::clamp::SaturationTable,
    poly::opening_proof::SumcheckId,
    subprotocols::{
        identity_range_check::{identity_rangecheck_prover, identity_rangecheck_verifier},
        ps_shout::unary::{ps_read_raf_prover_with_cycle, ps_read_raf_verifier_with_cycle},
        shout,
        sumcheck::{BatchedSumcheck, SumcheckInstanceProof},
        sumcheck_prover::SumcheckInstanceProver,
        sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    utils::{errors::ProofVerifyError, lookup_bits::LookupBits},
};
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
        let assembled: Vec<Vec<LookupBits>> = buckets
            .iter()
            .map(|b| b.assemble_bits(|idx| bits_by_node.remove(&idx).unwrap()))
            .collect();
        assert!(
            bits_by_node.is_empty(),
            "clamp lookups registered for nodes outside every bucket: {:?}",
            bits_by_node.keys().collect::<Vec<_>>()
        );

        let instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = buckets
            .iter()
            .zip(&assembled)
            .map(|(b, bits)| {
                let gammas: Vec<F> = prover.transcript.challenge_scalar_powers(b.nodes.len());
                let (claims, cycle) = b.read_raf_inputs(&gammas, &prover.accumulator);
                let ra_poly = (
                    VirtualPoly::GlobalClampRa(b.idx),
                    SumcheckId::NodeExecution(DEFERRED_PROOF_IDX),
                );
                with_clamp_width!(b.width, W => Box::new(
                    ps_read_raf_prover_with_cycle::<F, T, SaturationTable<W>, W>(
                        claims,
                        cycle,
                        ra_poly,
                        bits.clone(),
                        &mut prover.transcript,
                    ),
                )
                    as Box<dyn SumcheckInstanceProver<F, T>>)
            })
            .collect();
        batched_prove(instances, prover, proofs, ProofType::DeferredClampReadRaf);

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> =
            Vec::with_capacity(3 * buckets.len());
        for (b, bits) in buckets.iter().zip(&assembled) {
            let indices: Vec<usize> = bits.iter().map(|&x| x.into()).collect();
            let encoding = GlobalClampEncoding(b);
            instances.extend(shout::ra_onehot_provers(
                &encoding,
                &indices,
                &prover.accumulator,
                &mut prover.transcript,
            ));
        }
        drop(assembled);
        batched_prove(instances, prover, proofs, ProofType::DeferredClampRaChecks);
    }

    if !remainder.is_empty() {
        let instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> = remainder
            .iter()
            .map(|(node, bits, lookup_bits)| {
                let rc_provider = RescaleRemainderRCProvider::new(node.clone(), *bits);
                Box::new(identity_rangecheck_prover(
                    &rc_provider,
                    lookup_bits.clone(),
                    &mut prover.accumulator,
                )) as Box<dyn SumcheckInstanceProver<F, T>>
            })
            .collect();
        batched_prove(instances, prover, proofs, ProofType::DeferredRemainderRC);

        let mut instances: Vec<Box<dyn SumcheckInstanceProver<F, T>>> =
            Vec::with_capacity(3 * remainder.len());
        for (node, bits, lookup_bits) in &remainder {
            let indices: Vec<usize> = lookup_bits.iter().map(|&b| b.into()).collect();
            let encoding = RescaleRemainderRaEncoding::new(node.idx, *bits);
            instances.extend(shout::ra_onehot_provers(
                &encoding,
                &indices,
                &prover.accumulator,
                &mut prover.transcript,
            ));
        }
        drop(remainder);
        batched_prove(
            instances,
            prover,
            proofs,
            ProofType::DeferredRemainderRaChecks,
        );
    }
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
                let (claims, cycle) = b.read_raf_inputs(&gammas, &verifier.accumulator);
                let ra_poly = (
                    VirtualPoly::GlobalClampRa(b.idx),
                    SumcheckId::NodeExecution(DEFERRED_PROOF_IDX),
                );
                with_clamp_width!(b.width, W => Box::new(
                    ps_read_raf_verifier_with_cycle::<F, T, SaturationTable<W>, W>(
                        claims,
                        cycle,
                        ra_poly,
                        &mut verifier.transcript,
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

    if !remainder.is_empty() {
        let instances: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> = remainder
            .iter()
            .map(|(node, bits)| {
                let rc_provider = RescaleRemainderRCProvider::new(node.clone(), *bits);
                Box::new(identity_rangecheck_verifier(
                    &rc_provider,
                    &mut verifier.accumulator,
                )) as Box<dyn SumcheckInstanceVerifier<F, T>>
            })
            .collect();
        batched_verify(instances, verifier, ProofType::DeferredRemainderRC)?;

        let mut instances: Vec<Box<dyn SumcheckInstanceVerifier<F, T>>> =
            Vec::with_capacity(3 * remainder.len());
        for (node, bits) in &remainder {
            let encoding = RescaleRemainderRaEncoding::new(node.idx, *bits);
            instances.extend(shout::ra_onehot_verifiers(
                &encoding,
                &verifier.accumulator,
                &mut verifier.transcript,
            ));
        }
        batched_verify(instances, verifier, ProofType::DeferredRemainderRaChecks)?;
    }
    Ok(())
}
