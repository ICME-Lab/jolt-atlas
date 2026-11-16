//! Packed ("global") saturating-clamp and rescale-remainder buckets.
//!
//! Every clamped node used to commit its own one-hot chunks and run its own
//! lookup sumchecks. Here nodes of the same width are packed into *buckets*:
//! one shared cycle space per bucket, each node occupying a
//! power-of-two-aligned segment, so a bucket has one set of committed columns
//! (a clamp bucket's dense output digits and one-hot slack chunks, a remainder
//! bucket's dense digits) and one set of checks whatever the node count.
//!
//! A bucket sumcheck over the packed cycle space uses the cycle weight
//! `Σ_n γ_n·eq(r_n, t − off_n)` ([`CycleWeight::Packed`]) so its input claim
//! is a `γ`-combination of the member nodes' existing openings. The clamp
//! buckets are proven by the interior / saturated split and the remainder
//! buckets by a value sumcheck and a digit range check, both in [`super::clamp_split`]. Gaps at the
//! end of a bucket hold zero-weight dummy cycles.
//!
//! Bucket capacity is the largest clamped node's (padded) size, so no bucket
//! polynomial is larger than the largest per-node one was and the SRS size is
//! unchanged. The layout is a pure function of the model, so prover and
//! verifier derive it independently.
#[cfg(test)]
use super::clamp_split::digit_tamper;
use super::{
    clamp_split::{self, BucketSplit},
    fused_rebase::rebase_remainder,
    trace_view::TraceView,
};
use crate::node_view::{ComputationNode, ModelView};
use atlas_algebra::{
    field::JoltField, poly::multilinear::MultilinearPolynomial, utils::lookup_bits::LookupBits,
};
use atlas_trace::{Model, Tensor, Trace};
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
            .expect("the model's remainders have bounds the buckets can check")
            .par_iter()
            .map(|bucket| {
                let bits = assemble_bits(bucket, |member| {
                    let claim = member
                        .claim
                        .expect("a remainder bucket's members carry range claims");
                    range_claim_bits(&model.view(member.idx), &claim, bucket.width, trace)
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

    /// A remainder bucket's packed values as its value sumcheck's witness.
    pub fn remainder_value_column(&self, bucket_idx: usize) -> Vec<u64> {
        #[cfg(test)]
        if let Some((cycle, digit_tamper::DigitTamperForm::RaiseTop)) =
            digit_tamper::for_bucket(BucketKind::Remainder, bucket_idx)
        {
            let (values, width) = self.remainder_values(bucket_idx);
            let mut raised = values.to_vec();
            raised[cycle] += clamp_split::DIGIT_BASE.pow(clamp_split::num_digits(width) as u32);
            return raised;
        }
        self.remainder_values(bucket_idx).0.to_vec()
    }

    /// A remainder bucket's digit columns, most significant first.
    pub fn remainder_digits(&self, bucket_idx: usize) -> Vec<Vec<u8>> {
        let (values, width) = self.remainder_values(bucket_idx);
        let digits = clamp_split::digit_columns(values, clamp_split::num_digits(width));
        #[cfg(test)]
        let digits = {
            let mut digits = digits;
            if let Some((cycle, form)) = digit_tamper::for_bucket(BucketKind::Remainder, bucket_idx)
            {
                digit_tamper::apply(&mut digits, cycle, form);
            }
            digits
        };
        digits
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
        let remainder = remainder_buckets(model)
            .expect("the model's remainders have bounds the buckets can check")
            .into_iter()
            .flat_map(|bucket| {
                clamp_split::remainder_digit_polys(&bucket)
                    .into_iter()
                    .zip(self.remainder_digits(bucket.idx))
                    .map(|(poly, digits)| (poly, MultilinearPolynomial::from(digits)))
                    .collect::<Vec<_>>()
            });
        clamp.into_iter().chain(remainder).collect()
    }
}

/// A range claim's values over its node's padded cycle domain, from the trace: the remainder
/// itself, or its slack below the bound. A value outside `[0, 2^width)` (a remainder that does
/// not satisfy its bound) has no digit decomposition; its low `width` bits are committed, and
/// the proof does not verify.
fn range_claim_bits(
    node: &ComputationNode,
    claim: &RangeClaim,
    width: usize,
    trace: &Trace,
) -> Vec<LookupBits> {
    let len = node.pow2_padded_num_output_elements();
    let padded = |tensor: &Tensor<i32>| -> Vec<i64> {
        tensor.padded_pow2().iter().map(|&v| i64::from(v)).collect()
    };
    let flat = |values: &[i32]| -> Vec<i64> {
        let mut values: Vec<i64> = values.iter().map(|&v| i64::from(v)).collect();
        values.resize(len, 0);
        values
    };
    let softmax = || {
        trace
            .softmax_witness(node.idx)
            .unwrap_or_else(|| panic!("softmax node {} has no trace witness", node.idx))
    };
    let remainder: Vec<i64> = match claim.remainder {
        Remainder::Rescale => padded(&rebase_remainder(node, trace)),
        Remainder::Div | Remainder::ScalarConstDiv => {
            padded(TraceView::division_remainder(trace, node))
        }
        Remainder::RsqrtDiv => padded(&TraceView::rsqrt_advice(trace, node).div_remainder),
        Remainder::RsqrtSqrt => padded(&TraceView::rsqrt_advice(trace, node).sqrt_remainder),
        Remainder::Teleport => padded(&TraceView::teleport_advice(trace, node).remainder),
        Remainder::SoftmaxRecipMult => flat(&softmax().r),
        Remainder::SoftmaxExp => flat(&softmax().r_exp),
    };
    assert_eq!(remainder.len(), len, "node {} remainder length", node.idx);
    let values: Vec<i64> = match claim.slack {
        None => remainder,
        Some(Bound::Const(bound)) => remainder.iter().map(|r| bound - 1 - r).collect(),
        Some(Bound::Node(bound)) => padded(&trace.node_outputs[bound])
            .iter()
            .zip(&remainder)
            .map(|(b, r)| b - 1 - r)
            .collect(),
        Some(Bound::TwiceNodePlusOne(bound)) => padded(&trace.node_outputs[bound])
            .iter()
            .zip(&remainder)
            .map(|(v, r)| 2 * v - r)
            .collect(),
    };
    values
        .into_iter()
        .map(|v| LookupBits::new(v as u64, width))
        .collect()
}

/// Lay each member's lookup addresses into the bucket's cycle space; the
/// unused tail holds index-0 lookups.
pub fn assemble_bits(
    bucket: &ClampBucket,
    mut per_node: impl FnMut(&BucketNode) -> Vec<LookupBits>,
) -> Vec<LookupBits> {
    let mut bits = vec![LookupBits::new(0, bucket.width); bucket.cycle_len()];
    for n in &bucket.nodes {
        let b = per_node(n);
        assert_eq!(b.len(), 1 << n.log_t, "node {} lookup count", n.idx);
        bits[n.offset..n.offset + b.len()].copy_from_slice(&b);
    }
    bits
}
