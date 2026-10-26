//! Packed ("global") saturating-clamp and rescale-remainder buckets.
//!
//! Every clamped node used to commit its own one-hot chunks and run its own
//! lookup sumchecks. Here nodes of the same width are packed into *buckets*:
//! one shared cycle space per bucket, each node occupying a
//! power-of-two-aligned segment, so a bucket has one set of committed chunks
//! and one set of checks whatever the node count.
//!
//! A bucket sumcheck over the packed cycle space uses the cycle weight
//! `Σ_n γ_n·eq(r_n, t − off_n)` ([`CycleWeight::Packed`]) so its input claim
//! is a `γ`-combination of the member nodes' existing openings. The clamp
//! buckets are proven by the interior / saturated split and the remainder
//! buckets by a value sumcheck, both in [`super::clamp_split`]. Gaps at the
//! end of a bucket hold zero-weight dummy cycles.
//!
//! Bucket capacity is the largest clamped node's (padded) size, so no bucket
//! polynomial is larger than the largest per-node one was and the SRS size is
//! unchanged. The layout is a pure function of the model, so prover and
//! verifier derive it independently.
use super::{
    clamp_split,
    fused_rebase::{rebase_remainder, remainder_lookup_bits},
    witness::build_one_hot_rad_witness,
};
use crate::node_view::ModelView;
use atlas_trace::{Model, Trace};
use common::CommittedPoly;
pub use jolt_atlas_verifier::onnx_proof::global_clamp::*;
use joltworks::{field::JoltField, poly::multilinear_polynomial::MultilinearPolynomial};

/// Witnesses for every bucket's chunk polynomials.
#[tracing::instrument(skip_all, name = "global_clamp::bucket_witnesses")]
pub fn bucket_witnesses<F: JoltField>(
    model: &Model,
    trace: &Trace,
) -> Vec<(CommittedPoly, MultilinearPolynomial<F>)> {
    clamp_buckets(model)
        .iter()
        .chain(remainder_buckets(model).iter())
        .flat_map(|bucket| match bucket.kind {
            BucketKind::Clamp => {
                match clamp_split::live_bucket(bucket, |idx| {
                    !clamp_split::exact_output_prover(model, trace, idx)
                }) {
                    Some(_) => clamp_split::BucketSplit::from_trace(bucket, trace, model)
                        .witness_polys(bucket),
                    None => clamp_split::BucketSplit::empty_witness_polys(bucket),
                }
            }
            BucketKind::Remainder => {
                let bits = bucket.assemble_bits(|idx| {
                    remainder_lookup_bits(
                        &rebase_remainder(&model.view(idx), trace),
                        bucket.width as i32,
                    )
                });
                (0..bucket.num_chunks())
                    .map(|d| {
                        (
                            bucket.committed_poly(d),
                            build_one_hot_rad_witness(&bits, d, bucket.width),
                        )
                    })
                    .collect::<Vec<_>>()
            }
        })
        .collect()
}
