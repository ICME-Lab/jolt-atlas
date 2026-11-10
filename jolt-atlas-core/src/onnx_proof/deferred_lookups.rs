//! The post-loop plan on the prover's side: the packed clamp and remainder buckets, then the
//! deferral queue, each batch proven as one batched sumcheck with
//! `prove_relation_batch_parallel`. The plan itself is the verifier's
//! ([`plan`](jolt_atlas_verifier::onnx_proof::deferred_lookups::plan)); the bucket witnesses
//! are the ones built for the commitment
//! ([`BucketWitnesses`](crate::onnx_proof::global_clamp::BucketWitnesses)).
use crate::onnx_proof::{
    ops::{prove_deferred_batches, prove_steps},
    ProofId, Prover,
};
use atlas_algebra::{field::JoltField, utils::thread::drop_in_background_thread};
use atlas_sumcheck::sumcheck::SumcheckInstanceProof;
use atlas_transcript::Transcript;
pub use jolt_atlas_verifier::onnx_proof::deferred_lookups::*;
use std::collections::BTreeMap;

/// Prove everything after the node loop: the bucket plan, then the deferral queue.
#[tracing::instrument(skip_all, name = "deferred_lookups::prove_all")]
pub fn prove_all<F: JoltField, T: Transcript>(
    prover: &mut Prover<F, T>,
    proofs: &mut BTreeMap<ProofId, SumcheckInstanceProof<F>>,
) {
    let model = prover.preprocessing.model();
    let plan = plan(model, &prover.io).expect("the model's nodes have the shapes their ops prove");
    let (_, bucket_proofs) = prove_steps(prover, DEFERRED_PROOF_IDX, None, plan.0);
    proofs.extend(bucket_proofs);
    drop_in_background_thread(std::mem::take(&mut prover.buckets));

    proofs.extend(prove_deferred_batches(prover));
}
