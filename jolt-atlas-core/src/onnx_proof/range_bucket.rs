//! The prover side of the packed range buckets
//! ([`jolt_atlas_verifier::onnx_proof::range_bucket`] states the layout): each bucket's packed
//! values, from the trace, and their dense digit columns.
#[cfg(test)]
use super::range_check::digit_tamper;
use super::{fused_rebase::rebase_remainder, range_check, trace_view::TraceView};
use crate::node_view::{ComputationNode, ModelView};
use atlas_algebra::{
    field::JoltField, poly::multilinear::MultilinearPolynomial, utils::lookup_bits::LookupBits,
};
use atlas_trace::{Model, Tensor, Trace};
use common::CommittedPoly;
pub use jolt_atlas_verifier::onnx_proof::range_bucket::*;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// The packed buckets' witnesses, built once from the trace before the commitment and read
/// again by the post-loop plan's provers: every bucket's packed values with its width.
#[derive(Default)]
pub struct BucketWitnesses {
    /// The buckets' packed values and widths, by bucket.
    pub remainder_values: BTreeMap<usize, (Vec<u64>, usize)>,
}

impl BucketWitnesses {
    /// Build every bucket's witness, in parallel across buckets.
    #[tracing::instrument(skip_all, name = "range_bucket::BucketWitnesses::build")]
    pub fn build(model: &Model, trace: &Trace) -> Self {
        let remainder_values = remainder_buckets(model)
            .expect("the model's range claims have bounds the buckets can check")
            .par_iter()
            .map(|bucket| {
                let bits = assemble_bits(bucket, |member| {
                    let claim = member.claim.expect("a bucket's members carry range claims");
                    range_claim_bits(&model.view(member.idx), &claim, bucket.width, trace)
                });
                let values: Vec<u64> = bits.into_iter().map(|x| usize::from(x) as u64).collect();
                (bucket.idx, (values, bucket.width))
            })
            .collect();
        Self { remainder_values }
    }

    /// A bucket's packed values and width.
    pub fn remainder_values(&self, bucket_idx: usize) -> (&[u64], usize) {
        let (values, width) = self
            .remainder_values
            .get(&bucket_idx)
            .unwrap_or_else(|| panic!("range bucket {bucket_idx} has no values"));
        (values, *width)
    }

    /// A bucket's packed values as its value sumcheck's witness.
    pub fn remainder_value_column(&self, bucket_idx: usize) -> Vec<u64> {
        #[cfg(test)]
        if let Some((cycle, digit_tamper::DigitTamperForm::RaiseTop)) =
            digit_tamper::for_bucket(bucket_idx)
        {
            let (values, width) = self.remainder_values(bucket_idx);
            let mut raised = values.to_vec();
            raised[cycle] += range_check::DIGIT_BASE.pow(range_check::num_digits(width) as u32);
            return raised;
        }
        self.remainder_values(bucket_idx).0.to_vec()
    }

    /// A bucket's digit columns, most significant first.
    pub fn remainder_digits(&self, bucket_idx: usize) -> Vec<Vec<u8>> {
        let (values, width) = self.remainder_values(bucket_idx);
        let digits = range_check::digit_columns(values, range_check::num_digits(width));
        #[cfg(test)]
        let digits = {
            let mut digits = digits;
            if let Some((cycle, form)) = digit_tamper::for_bucket(bucket_idx) {
                digit_tamper::apply(&mut digits, cycle, form);
            }
            digits
        };
        digits
    }

    /// The buckets' committed polynomials with their witnesses: every bucket's digit columns.
    #[tracing::instrument(skip_all, name = "range_bucket::BucketWitnesses::committed_polys")]
    pub fn committed_polys<F: JoltField>(
        &self,
        model: &Model,
    ) -> Vec<(CommittedPoly, MultilinearPolynomial<F>)> {
        remainder_buckets(model)
            .expect("the model's range claims have bounds the buckets can check")
            .into_iter()
            .flat_map(|bucket| {
                range_check::remainder_digit_polys(&bucket)
                    .into_iter()
                    .zip(self.remainder_digits(bucket.idx))
                    .map(|(poly, digits)| (poly, MultilinearPolynomial::from(digits)))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

/// A range claim's values over its domain ([`Remainder::domain_len`]), from the trace: the remainder
/// itself, or its slack below the bound. A value outside `[0, 2^width)` (a remainder that does
/// not satisfy its bound) has no digit decomposition; its low `width` bits are committed, and
/// the proof does not verify.
fn range_claim_bits(
    node: &ComputationNode,
    claim: &RangeClaim,
    width: usize,
    trace: &Trace,
) -> Vec<LookupBits> {
    let len = claim.remainder.domain_len(node);
    let padded = |tensor: &Tensor<i32>| -> Vec<i64> {
        tensor.padded_pow2().iter().map(|&v| i64::from(v)).collect()
    };
    let per_row = |values: &[i64]| -> Vec<i64> {
        let mut values = values.to_vec();
        values.resize(len, 0);
        values
    };
    let softmax = || {
        trace
            .softmax_witness(node.idx)
            .unwrap_or_else(|| panic!("softmax node {} has no trace witness", node.idx))
    };
    let remainder: Vec<i64> = match claim.remainder {
        // The output offset by `2^31`: an `i32` output lands in `[0, 2^32)`.
        Remainder::Output => padded(&trace.node_outputs[node.idx])
            .into_iter()
            .map(|out| out + OUTPUT_OFFSET as i64)
            .collect(),
        Remainder::Rescale => padded(&rebase_remainder(node, trace)),
        Remainder::Div | Remainder::ScalarConstDiv => {
            padded(TraceView::division_remainder(trace, node))
        }
        Remainder::RsqrtDiv => padded(&TraceView::rsqrt_advice(trace, node).div_remainder),
        Remainder::RsqrtSqrt => padded(&TraceView::rsqrt_advice(trace, node).sqrt_remainder),
        Remainder::Teleport => padded(&TraceView::teleport_advice(trace, node).remainder),
        Remainder::SoftmaxSumGap => per_row(&softmax().sum_gap),
        Remainder::SoftmaxPrevSumGap => per_row(&softmax().prev_sum_gap),
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
    bucket: &RangeBucket,
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
