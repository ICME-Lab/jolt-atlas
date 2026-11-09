//! The post-loop plan on the prover's side: the packed clamp and remainder buckets, then the
//! deferral queue, each batch proven as one batched sumcheck with
//! `prove_relation_batch_parallel`. The plan itself is the verifier's
//! ([`plan`](jolt_atlas_verifier::onnx_proof::deferred_lookups::plan)); the bucket witnesses
//! come from the trace.
use crate::node_view::ModelView;
use crate::onnx_proof::{
    clamp_split::{self, BucketSplit},
    fused_rebase::{rebase_remainder, remainder_lookup_bits},
    global_clamp::{assemble_bits, clamp_buckets, remainder_buckets, ClampBucket},
    ops::{prove_deferred_batches, prove_steps},
    ProofId, Prover,
};
use atlas_algebra::{field::JoltField, utils::thread::drop_in_background_thread};
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::deferred_lookups::*;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Prove everything after the node loop: the bucket plan, then the deferral queue.
#[tracing::instrument(skip_all, name = "deferred_lookups::prove_all")]
pub fn prove_all<F: JoltField, T: Transcript>(
    prover: &mut Prover<F, T>,
    proofs: &mut BTreeMap<ProofId, SumcheckInstanceProof<F>>,
) {
    let model = prover.preprocessing.model();
    let trace = &prover.trace;

    // The bucket witnesses, in parallel across buckets. A clamp bucket is live when one of its
    // nodes is not an unsaturated model output (`clamp_split::exact_output_prover`).
    let live: Vec<ClampBucket> = clamp_buckets(model)
        .expect("the model's nodes have the shapes their ops prove")
        .iter()
        .filter_map(|bucket| {
            clamp_split::live_bucket(bucket, |idx| {
                !clamp_split::exact_output_prover(model, trace, idx)
            })
        })
        .collect();
    prover.clamp_splits = {
        let _span = tracing::span!(tracing::Level::INFO, "deferred::assemble_buckets").entered();
        live.par_iter()
            .map(|bucket| (bucket.idx, BucketSplit::from_trace(bucket, trace, model)))
            .collect()
    };
    prover.remainder_values = remainder_buckets(model)
        .par_iter()
        .map(|bucket| {
            let bits = assemble_bits(bucket, |idx| {
                remainder_lookup_bits(
                    &rebase_remainder(&model.view(idx), trace),
                    bucket.width as i32,
                )
            });
            let values: Vec<u64> = bits.into_iter().map(|x| usize::from(x) as u64).collect();
            (bucket.idx, (values, bucket.width))
        })
        .collect();

    let plan = plan(model, &prover.io).expect("the model's nodes have the shapes their ops prove");
    let (_, bucket_proofs) = prove_steps(prover, DEFERRED_PROOF_IDX, None, plan.0);
    proofs.extend(bucket_proofs);
    drop_in_background_thread(std::mem::take(&mut prover.clamp_splits));
    drop_in_background_thread(std::mem::take(&mut prover.remainder_values));

    proofs.extend(prove_deferred_batches(prover));
}
