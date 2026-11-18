//! Integer softmax over the last axis in log-partition form, with witness capture and padding
//! masking.
//!
//! Per row ([`crate::semantics::softmax`]): the shift `c` is the smallest integer at or above
//! the live lanes' maximum whose sum-table row sum does not exceed `S'`; a live lane's output is
//! the output table read at `min(c − x, top)`.
//!
//! Padding lanes (`j ≥ active_len`, present only after power-of-two padding) output 0 and
//! contribute nothing to the row sums. Their address `c − x` is still captured: the prover
//! commits an address on every lane and gates the reads with the public selector
//! `[j < active_len]`.

use std::fmt;

use crate::{
    error::{KernelError, ShapeError},
    id::SemVersion,
    tensor::Tensor,
    trace::{NodeAux, SoftmaxWitness},
};

use super::{Eval, EvalCtx, OpSpec, PadBehavior, PadCtx, SoftmaxLastAxis, same_dims};

impl fmt::Display for SoftmaxLastAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.active_len {
            Some(n) => write!(f, "SoftmaxLastAxis(active {n})"),
            None => f.write_str("SoftmaxLastAxis"),
        }
    }
}

impl OpSpec for SoftmaxLastAxis {
    fn eval(&self, inputs: &[&Tensor<i32>], ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        let input = inputs[0];
        let dims = input.dims();
        let last_dim = *dims.last().ok_or_else(|| {
            KernelError::Shape(ShapeError::Invalid("softmax input is rank 0".into()))
        })?;
        let active_len = self.active_len.unwrap_or(last_dim);
        if active_len == 0 || active_len > last_dim {
            return Err(KernelError::Shape(ShapeError::Invalid(format!(
                "softmax active_len {active_len} invalid for last dim {last_dim}"
            ))));
        }

        if ctx.version == SemVersion::V1 {
            return Err(KernelError::Shape(ShapeError::Invalid(
                "softmax is defined from SemVersion::V2".into(),
            )));
        }
        let tables = ctx.tables.softmax_tables(ctx.scale, ctx.version);
        let sum_prev = tables.sum_prev();
        let top = tables.top();
        let rows = input.len() / last_dim;
        let data = input.data();
        let total = rows * last_dim;
        let overflow = |what: &'static str, value: i64| KernelError::WitnessOverflow {
            op: "softmax",
            what,
            value,
        };

        let mut address = vec![0i32; total];
        let mut output = vec![0i32; total];
        // One row: its shift and its two gaps, with its addresses and outputs written in place.
        let row = |((live_row, address), output): ((&[i32], &mut [i32]), &mut [i32])| {
            let c = tables
                .shift(&live_row[..active_len])
                .expect("active_len >= 1");
            let shift = i32::try_from(c).map_err(|_| overflow("shift c", c))?;
            // The proof commits `c − x` as a signed 32-bit lookup address on every lane.
            for (address, &x) in address.iter_mut().zip(live_row) {
                let u = c - x as i64;
                *address = i32::try_from(u).map_err(|_| overflow("address c - x", u))?;
            }
            let (mut sum, mut prev_sum) = (0i64, 0i64);
            for (output, &address) in output.iter_mut().zip(&*address).take(active_len) {
                // A live lane has `c ≥ x`, so its address is non-negative.
                let address = (address as i64).min(top) as usize;
                *output = tables.out[address];
                sum += tables.sum[address] as i64;
                prev_sum += sum_prev[address] as i64;
            }
            Ok((
                shift,
                tables.sum_scale - sum,
                prev_sum - tables.sum_scale - 1,
            ))
        };
        #[cfg(feature = "rayon")]
        let per_row: Vec<Result<(i32, i64, i64), KernelError>> = {
            use rayon::prelude::*;
            data.par_chunks(last_dim)
                .zip(address.par_chunks_mut(last_dim))
                .zip(output.par_chunks_mut(last_dim))
                .map(row)
                .collect()
        };
        #[cfg(not(feature = "rayon"))]
        let per_row: Vec<Result<(i32, i64, i64), KernelError>> = data
            .chunks(last_dim)
            .zip(address.chunks_mut(last_dim))
            .zip(output.chunks_mut(last_dim))
            .map(row)
            .collect();
        // The first row in order that fails is the error.
        let mut w = SoftmaxWitness {
            last_dim,
            active_len,
            shift: Vec::with_capacity(rows),
            address: Vec::new(),
            sum_gap: Vec::with_capacity(rows),
            prev_sum_gap: Vec::with_capacity(rows),
        };
        for result in per_row {
            let (shift, sum_gap, prev_sum_gap) = result?;
            w.shift.push(shift);
            w.sum_gap.push(sum_gap);
            w.prev_sum_gap.push(prev_sum_gap);
        }
        w.address = address;

        Ok(Eval {
            output: Tensor::new(output, dims)?,
            aux: Some(NodeAux::Softmax(Box::new(w))),
        })
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        let dims = same_dims(inputs, Some(1))?;
        if dims.is_empty() {
            return Err(ShapeError::Invalid("softmax input is rank 0".into()));
        }
        if let Some(n) = self.active_len
            && (n == 0 || n > dims[dims.len() - 1])
        {
            return Err(ShapeError::Invalid(format!(
                "active_len {n} invalid for last dim {}",
                dims[dims.len() - 1]
            )));
        }
        Ok(dims)
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::NeedsMask
    }

    fn apply_padding(&mut self, ctx: &PadCtx<'_>) -> Result<(), String> {
        // Remember the live width when the last axis widens; keep an
        // existing (narrower) mask if one was already set.
        let last = *ctx.output_dims.last().ok_or("softmax output is rank 0")?;
        if self.active_len.is_none() && !last.is_power_of_two() {
            self.active_len = Some(last);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::Scale;
    use crate::op::TableCache;

    fn try_eval(
        input: Tensor<i32>,
        active_len: Option<usize>,
        scale: u8,
        version: SemVersion,
    ) -> Result<(Tensor<i32>, SoftmaxWitness), KernelError> {
        let cache = TableCache::default();
        let ctx = EvalCtx {
            scale: Scale::new(scale).unwrap(),
            version,
            tables: &cache,
        };
        let out = SoftmaxLastAxis { active_len }.eval(&[&input], &ctx)?;
        let NodeAux::Softmax(w) = out.aux.unwrap() else {
            panic!()
        };
        Ok((out.output, *w))
    }

    fn eval(
        input: Tensor<i32>,
        active_len: Option<usize>,
        scale: u8,
    ) -> (Tensor<i32>, SoftmaxWitness) {
        try_eval(input, active_len, scale, SemVersion::V2).unwrap()
    }

    #[test]
    fn equal_logits_give_equal_probabilities() {
        let input = Tensor::new(vec![0i32, 0], &[1, 2]).unwrap();
        let (out, _) = eval(input, None, 15);
        assert_eq!(out.data(), &[1 << 14, 1 << 14]);
    }

    #[test]
    fn a_dominant_lane_takes_the_whole_scale() {
        let s = 1i32 << 14;
        let input = Tensor::new(vec![0, 400_000, -5, 7], &[1, 4]).unwrap();
        let (out, w) = eval(input, None, 14);
        assert_eq!(out.data(), &[0, s, 0, 0]);
        // The shift is the maximum itself, and the minimality gap comes from the zero address.
        assert_eq!(w.shift, vec![400_000]);
        assert_eq!(w.sum_gap, vec![0]);
        assert_eq!(w.prev_sum_gap, vec![0]);
    }

    #[test]
    fn witness_is_the_definition() {
        let s_sum = 1i64 << 22;
        let cache = TableCache::default();
        let tables = cache.softmax_tables(Scale::new(14).unwrap(), SemVersion::V2);
        let sum_prev = tables.sum_prev();
        let data = vec![100, -2000, 5000, 0, 16384, -16384, 7, 7];
        let (out, w) = eval(Tensor::new(data.clone(), &[2, 4]).unwrap(), None, 14);
        for k in 0..2 {
            let row = &data[k * 4..(k + 1) * 4];
            assert_eq!(w.shift[k] as i64, tables.shift(row).unwrap());
            let (mut sum, mut prev) = (0i64, 0i64);
            for (j, &x) in row.iter().enumerate() {
                let u = w.address[k * 4 + j];
                assert_eq!(u, w.shift[k] - x);
                let a = (u as i64).min(tables.top()) as usize;
                assert_eq!(out.data()[k * 4 + j], tables.out[a]);
                sum += tables.sum[a] as i64;
                prev += sum_prev[a] as i64;
            }
            assert_eq!(w.sum_gap[k], s_sum - sum);
            assert_eq!(w.prev_sum_gap[k], prev - s_sum - 1);
            assert!(w.sum_gap[k] >= 0 && w.prev_sum_gap[k] >= 0);
        }
    }

    #[test]
    fn active_len_masks_padding() {
        // A 3-wide row padded to 4: the padded lane must not disturb the live probabilities,
        // whatever value it holds.
        let live = Tensor::new(vec![100, 300, -50], &[1, 3]).unwrap();
        let (expected, _) = eval(live, None, 14);
        for pad in [0, 1 << 20, -(1 << 20)] {
            let padded = Tensor::new(vec![100, 300, -50, pad], &[1, 4]).unwrap();
            let (out, w) = eval(padded, Some(3), 14);
            assert_eq!(&out.data()[..3], expected.data());
            assert_eq!(out.data()[3], 0);
            assert_eq!(w.address[3], w.shift[0] - pad);
        }
    }

    #[test]
    fn out_of_range_addresses_are_typed_errors() {
        // Two logits a full i32 range apart: the low lane's address does not fit 32 signed bits.
        let input = Tensor::new(vec![i32::MAX, i32::MIN], &[1, 2]).unwrap();
        let err = try_eval(input, None, 14, SemVersion::V2).unwrap_err();
        assert!(matches!(err, KernelError::WitnessOverflow { .. }), "{err}");
        // A shift above i32::MAX: many lanes at the top of the range.
        let input = Tensor::new(vec![i32::MAX; 4], &[1, 4]).unwrap();
        let err = try_eval(input, None, 14, SemVersion::V2).unwrap_err();
        assert!(matches!(err, KernelError::WitnessOverflow { .. }), "{err}");
    }

    #[test]
    fn the_first_version_has_no_softmax() {
        let input = Tensor::new(vec![0i32, 0], &[1, 2]).unwrap();
        assert!(try_eval(input, None, 14, SemVersion::V1).is_err());
    }
}
