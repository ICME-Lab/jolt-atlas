//! Integer softmax over the last axis, with full witness capture and
//! padding masking.
//!
//! Pipeline per row (ported bit-for-bit from the old crate on live lanes):
//! 1. `max_k` and `argmax_k` over the **live** lanes `0..active_len`;
//! 2. per live lane: `z = max_k - x`, saturated at the exp-table bound,
//!    split into digits `z = z_hi·B + z_lo`, two sub-table lookups, and
//!    `exp_q = ⌊exp_hi·exp_lo/S⌋` with remainder `r_exp ∈ [0, S)`;
//! 3. `exp_sum_q = Σ exp_q` over live lanes, `inv_sum = ⌊S²/exp_sum_q⌋`;
//! 4. per live lane: `softmax_q = ⌊exp_q·inv_sum/S⌋`, remainder `R ∈ [0, S)`.
//!
//! Masked lanes (`j ≥ active_len`, present only after power-of-two padding)
//! produce 0 for the output and every per-lane witness value, and contribute
//! nothing to `exp_sum_q` — this fixes the old crate's padding corruption,
//! where padded zero lanes entered the row sum as `exp(max)` terms. The
//! prover gates those lanes with the public selector `[j < active_len]`.

use std::fmt;

use crate::{
    error::{KernelError, ShapeError},
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

        let s = ctx.scale.denom_i32();
        // All row-level arithmetic runs in i64: a row sum of exp_q terms (each
        // up to S) overflows i32 once active_len · S exceeds 2^31 (e.g. a
        // vocab-size softmax axis), and s² itself overflows i32 at scale ≥ 16.
        let s_i64 = s as i64;
        let s_sq = s_i64 * s_i64;
        let rows = input.len() / last_dim;
        let data = input.data();

        // Sub-tables; lut_hi zero-padded to a power of two so a saturated
        // z_hi can never index out of range (mirrors the proof side).
        let lut = ctx.tables.exp_lut(ctx.scale, ctx.version);
        let mut lut_hi = lut.lut_hi.clone();
        lut_hi.resize(lut_hi.len().next_power_of_two(), 0);
        let z_bound = (lut_hi.len() * lut.base) as i32;

        let total = rows * last_dim;
        let mut w = SoftmaxWitness {
            scale: s as i64,
            last_dim,
            active_len,
            x: data.to_vec(),
            max_k: Vec::with_capacity(rows),
            argmax_k: Vec::with_capacity(rows),
            exp_q: vec![0; total],
            exp_sum_q: Vec::with_capacity(rows),
            inv_sum: Vec::with_capacity(rows),
            r: vec![0; total],
            z_hi: vec![0; total],
            z_lo: vec![0; total],
            exp_hi: vec![0; total],
            exp_lo: vec![0; total],
            r_exp: vec![0; total],
        };
        let mut softmax_q = vec![0i32; total];

        for k in 0..rows {
            let offset = k * last_dim;
            let live = &data[offset..offset + active_len];

            let mv = *live.iter().max().expect("active_len >= 1");
            w.max_k.push(mv);
            w.argmax_k
                .push(live.iter().position(|&x| x == mv).expect("max exists"));

            let mut sum_exp: i64 = 0;
            for j in 0..active_len {
                let idx = offset + j;
                // z ≥ 0; saturate to the sub-table range (exp decays to 0
                // beyond it anyway).
                let z_c = (mv - data[idx]).min(z_bound - 1);
                let zu = z_c as usize;
                let z_hi = (zu >> lut.log2_base) as i32;
                let z_lo = (zu & (lut.base - 1)) as i32;
                let hi_val = lut_hi[z_hi as usize];
                let lo_val = lut.lut_lo[z_lo as usize];
                let product = hi_val as i64 * lo_val as i64;
                let exp_q = (product / s as i64) as i32;

                w.z_hi[idx] = z_hi;
                w.z_lo[idx] = z_lo;
                w.exp_hi[idx] = hi_val;
                w.exp_lo[idx] = lo_val;
                w.exp_q[idx] = exp_q;
                w.r_exp[idx] = (product - exp_q as i64 * s as i64) as i32;
                sum_exp += exp_q as i64;
            }
            // The witness (and the proof protocol behind it) carries the row
            // sum as an i32; a row whose live exponentials exceed that is a
            // typed error, never a wrapped sum.
            let sum_exp_i32 = i32::try_from(sum_exp).map_err(|_| KernelError::WitnessOverflow {
                op: "softmax",
                what: "exp_sum_q",
                value: sum_exp,
            })?;
            w.exp_sum_q.push(sum_exp_i32);

            // inv ≤ s²/s = s, so it always fits back into i32.
            let inv = (s_sq / sum_exp) as i32;
            w.inv_sum.push(inv);

            for j in 0..active_len {
                let idx = offset + j;
                // exp_q, inv ≤ S: the product needs i64 headroom at scale ≥ 16.
                let product = w.exp_q[idx] as i64 * inv as i64;
                let sq = product / s_i64;
                softmax_q[idx] = sq as i32;
                w.r[idx] = (product - sq * s_i64) as i32;
            }
            // Lanes active_len..last_dim stay 0 everywhere.
        }

        Ok(Eval {
            output: Tensor::new(softmax_q, dims)?,
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
    use crate::id::{Scale, SemVersion};
    use crate::op::TableCache;

    fn eval(
        input: Tensor<i32>,
        active_len: Option<usize>,
        scale: u8,
    ) -> (Tensor<i32>, SoftmaxWitness) {
        let cache = TableCache::default();
        let ctx = EvalCtx {
            scale: Scale::new(scale).unwrap(),
            version: SemVersion::V1,
            tables: &cache,
        };
        let out = SoftmaxLastAxis { active_len }
            .eval(&[&input], &ctx)
            .unwrap();
        let NodeAux::Softmax(w) = out.aux.unwrap() else {
            panic!()
        };
        (out.output, *w)
    }

    #[test]
    fn equal_logits_give_equal_probabilities() {
        // Old crate's pinned test: scale 2^15, two equal logits → S/2 each.
        let input = Tensor::new(vec![0i32, 0], &[1, 2]).unwrap();
        let (out, _) = eval(input, None, 15);
        assert_eq!(out.data(), &[1 << 14, 1 << 14]);
    }

    #[test]
    fn witness_invariants_hold() {
        let s = 1i32 << 14;
        let input = Tensor::new(vec![100, -2000, 5000, 0, 16384, -16384, 7, 7], &[2, 4]).unwrap();
        let (out, w) = eval(input, None, 14);
        for k in 0..2 {
            // inv_sum = ⌊S²/exp_sum⌋.
            assert_eq!(w.inv_sum[k], (s * s) / w.exp_sum_q[k]);
            let row_sum: i32 = w.exp_q[k * 4..(k + 1) * 4].iter().sum();
            assert_eq!(row_sum, w.exp_sum_q[k]);
        }
        for idx in 0..8 {
            assert!((0..s).contains(&w.r[idx]));
            assert!((0..s).contains(&w.r_exp[idx]));
            // exp_hi·exp_lo = exp_q·S + r_exp.
            assert_eq!(
                w.exp_hi[idx] as i64 * w.exp_lo[idx] as i64,
                w.exp_q[idx] as i64 * s as i64 + w.r_exp[idx] as i64
            );
            // softmax_q·S + R = exp_q·inv_sum.
            let k = idx / 4;
            assert_eq!(out.data()[idx] * s + w.r[idx], w.exp_q[idx] * w.inv_sum[k]);
        }
        // Probabilities approximately sum to S per row.
        for k in 0..2 {
            let p: i32 = out.data()[k * 4..(k + 1) * 4].iter().sum();
            assert!((p - s).abs() < 8, "row {k} sums to {p}");
        }
    }

    #[test]
    fn active_len_masks_padding() {
        // A 3-wide row padded to 4: the padded lane must not disturb the
        // live probabilities.
        let live = Tensor::new(vec![100, 300, -50], &[1, 3]).unwrap();
        let (expected, _) = eval(live, None, 14);

        let padded = Tensor::new(vec![100, 300, -50, 0], &[1, 4]).unwrap();
        let (out, w) = eval(padded, Some(3), 14);
        assert_eq!(&out.data()[..3], expected.data());
        assert_eq!(out.data()[3], 0);
        assert_eq!(w.exp_q[3], 0);
        assert_eq!(w.exp_hi[3], 0);
        assert_eq!(w.r_exp[3], 0);
    }

    #[test]
    fn wide_rows_sum_in_i64() {
        // A vocab-width row: 2^17 lanes whose exponentials sum past i32::MAX
        // would overflow an i32 accumulator (an abort under the workspace's
        // release overflow-checks). The sum runs in i64; a row whose total
        // exceeds the i32 witness range is a typed error, never a panic.
        let n = 1 << 18;
        let input = Tensor::new(vec![0i32; n], &[1, n]).unwrap();
        let cache = TableCache::default();
        let ctx = EvalCtx {
            scale: Scale::new(14).unwrap(),
            version: SemVersion::V1,
            tables: &cache,
        };
        let op = SoftmaxLastAxis { active_len: None };
        let err = match op.eval(&[&input], &ctx) {
            Err(e) => e,
            Ok(_) => panic!("overflowing row sum must not evaluate"),
        };
        assert!(matches!(err, KernelError::WitnessOverflow { .. }), "{err}");

        // A wide row whose sum still fits the witness evaluates fine.
        let mut data = vec![-100_000i32; n];
        data[0] = 0;
        let input = Tensor::new(data, &[1, n]).unwrap();
        let (out, _) = eval(input, None, 14);
        assert!(out.data()[0] > 0);
    }

    #[test]
    fn unmasked_padding_reproduces_old_corruption() {
        // Documenting the bug the mask fixes: without active_len, the padded
        // zero lane joins the row and changes the live probabilities.
        let live = Tensor::new(vec![100, 300, -50], &[1, 3]).unwrap();
        let (expected, _) = eval(live, None, 14);
        let padded = Tensor::new(vec![100, 300, -50, 0], &[1, 4]).unwrap();
        let (out, _) = eval(padded, None, 14);
        assert_ne!(&out.data()[..3], expected.data());
    }
}
