//! Packed ("global") saturating-clamp and rescale-remainder buckets.
//!
//! Every clamped node used to commit its own one-hot chunks and run its own
//! lookup sumchecks. Here nodes of the same width are packed into *buckets*:
//! one shared cycle space per bucket, each node occupying a
//! power-of-two-aligned segment, so a bucket has one set of committed columns
//! (a clamp bucket's dense output digits and one-hot slack chunks, a remainder
//! bucket's one-hot chunks) and one set of checks whatever the node count.
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
    clamp_split::{self, BucketSplit},
    fused_rebase::{rebase_remainder, remainder_lookup_bits},
    witness::build_one_hot_rad_witness,
};
use crate::node_view::ModelView;
use atlas_algebra::{
    field::JoltField, poly::multilinear_polynomial::MultilinearPolynomial,
    utils::lookup_bits::LookupBits,
};
use atlas_trace::{Model, Trace};
use common::CommittedPoly;
pub use jolt_atlas_verifier::onnx_proof::global_clamp::*;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// The packed buckets' witnesses, built once from the trace before the commitment and read
/// again by the post-loop plan's provers: every live clamp bucket's split and every remainder
/// bucket's packed values with its width.
#[derive(Default)]
pub struct BucketWitnesses {
    /// The live clamp buckets' split witnesses, by bucket.
    pub clamp_splits: BTreeMap<usize, BucketSplit>,
    /// The remainder buckets' packed values and widths, by bucket.
    pub remainder_values: BTreeMap<usize, (Vec<u64>, usize)>,
}

impl BucketWitnesses {
    /// Build every bucket's witness, in parallel across buckets. A clamp bucket is live when one
    /// of its nodes is not an unsaturated model output
    /// ([`clamp_split::exact_output_prover`]).
    #[tracing::instrument(skip_all, name = "global_clamp::BucketWitnesses::build")]
    pub fn build(model: &Model, trace: &Trace) -> Self {
        let live: Vec<ClampBucket> = clamp_buckets(model)
            .expect("the model's nodes have the shapes their ops prove")
            .iter()
            .filter_map(|bucket| {
                clamp_split::live_bucket(bucket, |idx| {
                    !clamp_split::exact_output_prover(model, trace, idx)
                })
            })
            .collect();
        let clamp_splits = live
            .par_iter()
            .map(|bucket| (bucket.idx, BucketSplit::from_trace(bucket, trace, model)))
            .collect();
        let remainder_values = remainder_buckets(model)
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
        Self {
            clamp_splits,
            remainder_values,
        }
    }

    /// A live clamp bucket's split witness.
    pub fn clamp_split(&self, bucket_idx: usize) -> &BucketSplit {
        self.clamp_splits
            .get(&bucket_idx)
            .unwrap_or_else(|| panic!("clamp bucket {bucket_idx} has no split witness"))
    }

    /// A remainder bucket's packed values and width.
    pub fn remainder_values(&self, bucket_idx: usize) -> (&[u64], usize) {
        let (values, width) = self
            .remainder_values
            .get(&bucket_idx)
            .unwrap_or_else(|| panic!("remainder bucket {bucket_idx} has no values"));
        (values, *width)
    }

    /// The buckets' committed polynomials with their witnesses: every live clamp bucket's and
    /// every remainder bucket's. A clamp bucket with no live node commits nothing (the
    /// verifier's [`bucket_committed_polys`] leaves it out the same way).
    #[tracing::instrument(skip_all, name = "global_clamp::BucketWitnesses::committed_polys")]
    pub fn committed_polys<F: JoltField>(
        &self,
        model: &Model,
    ) -> Vec<(CommittedPoly, MultilinearPolynomial<F>)> {
        let clamp = clamp_buckets(model)
            .expect("the model's nodes have the shapes their ops prove")
            .into_iter()
            .filter_map(|bucket| {
                self.clamp_splits
                    .get(&bucket.idx)
                    .map(|split| split.witness_polys(&bucket))
            })
            .flatten()
            .collect::<Vec<_>>();
        let remainder = remainder_buckets(model).into_iter().flat_map(|bucket| {
            let (values, width) = self.remainder_values(bucket.idx);
            let bits: Vec<LookupBits> = values.iter().map(|&v| LookupBits::new(v, width)).collect();
            (0..bucket.num_chunks())
                .map(|d| {
                    (
                        bucket.remainder_chunk_poly(d),
                        build_one_hot_rad_witness(&bits, d, width),
                    )
                })
                .collect::<Vec<_>>()
        });
        clamp.into_iter().chain(remainder).collect()
    }
}

/// Lay each node's lookup addresses into the bucket's cycle space; the
/// unused tail holds index-0 lookups.
pub fn assemble_bits(
    bucket: &ClampBucket,
    mut per_node: impl FnMut(usize) -> Vec<LookupBits>,
) -> Vec<LookupBits> {
    let mut bits = vec![LookupBits::new(0, bucket.width); bucket.cycle_len()];
    for n in &bucket.nodes {
        let b = per_node(n.idx);
        assert_eq!(b.len(), 1 << n.log_t, "node {} lookup count", n.idx);
        bits[n.offset..n.offset + b.len()].copy_from_slice(&b);
    }
    bits
}
