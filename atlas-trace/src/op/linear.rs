//! Elementwise and reduction operators without a fused rescale.

use std::fmt;

use crate::{
    error::{KernelError, ShapeError},
    semantics::fixed::{floor_div, rem_floor, rsqrt, sat_clamp_i32},
    tensor::Tensor,
    trace::{NodeAux, RsqrtAdvice},
};

use super::{
    Add, And, BoundCtx, Clamp, Div, Eval, EvalCtx, Iff, IsNan, Neg, OpSpec, PadBehavior, ReLU,
    Rsqrt, ScalarConstDiv, Sub, Sum, same_dims,
};

/// Magnitude bound of an i32 (`2^31`), the induction hypothesis of the
/// clamp-width analysis.
pub(super) const I32_MAG: u128 = 1 << 31;

/// Saturating elementwise chain for Add/Sub: accumulate each consecutive
/// pair in `i64` and clamp back to `i32` per step (per-step clamping is the
/// old crate's variadic semantics).
fn sat_chain(
    inputs: &[&Tensor<i32>],
    combine: impl Fn(i64, i64) -> i64,
) -> Result<Tensor<i32>, KernelError> {
    let mut acc = inputs[0].clone();
    for rhs in &inputs[1..] {
        acc = acc.zip_map(rhs, |a, b| sat_clamp_i32(combine(a as i64, b as i64)))?;
    }
    Ok(acc)
}

impl fmt::Display for Add {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Add")
    }
}

impl OpSpec for Add {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        Ok(Eval::plain(sat_chain(inputs, |a, b| a + b)?))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        if inputs.len() < 2 {
            return Err(ShapeError::WrongArity {
                expected: 2,
                got: inputs.len(),
            });
        }
        same_dims(inputs, None)
    }

    fn clamp_bound(&self, _ctx: &BoundCtx<'_>) -> Option<u128> {
        Some(2 * I32_MAG)
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }
}

impl fmt::Display for Sub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Sub")
    }
}

impl OpSpec for Sub {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        Ok(Eval::plain(sat_chain(inputs, |a, b| a - b)?))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        if inputs.len() < 2 {
            return Err(ShapeError::WrongArity {
                expected: 2,
                got: inputs.len(),
            });
        }
        same_dims(inputs, None)
    }

    fn clamp_bound(&self, _ctx: &BoundCtx<'_>) -> Option<u128> {
        Some(2 * I32_MAG)
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }
}

impl fmt::Display for Neg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Neg")
    }
}

impl OpSpec for Neg {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        // Saturating: -i32::MIN stays representable.
        Ok(Eval::plain(inputs[0].map(|x| x.saturating_neg())))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(1))
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }
}

impl fmt::Display for Sum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sum(axes {:?})", self.axes)
    }
}

/// Reduce `input` over `axes` in `i64`, collapsing each reduced axis to 1.
///
/// Public because the pre-clamp Sum accumulation is part of the proof spec:
/// the prover recovers the saturating-clamp lookup index from it.
pub fn sum_axes_i64(input: &Tensor<i32>, axes: &[usize]) -> Tensor<i64> {
    sum_map_i64(input, axes, |v| v as i64)
}

/// Shared reduction skeleton: fold `f(x)` over the reduced axes.
pub(super) fn sum_map_i64(
    input: &Tensor<i32>,
    axes: &[usize],
    f: impl Fn(i32) -> i64,
) -> Tensor<i64> {
    let dims = input.dims();
    let out_dims: Vec<usize> = dims
        .iter()
        .enumerate()
        .map(|(i, &d)| if axes.contains(&i) { 1 } else { d })
        .collect();
    let mut out = vec![0i64; out_dims.iter().product()];

    // Row-major walk of the input, mapping each coordinate to its reduced
    // output slot.
    let mut coords = vec![0usize; dims.len()];
    for &v in input.data() {
        let mut slot = 0;
        for (axis, (&c, &od)) in coords.iter().zip(&out_dims).enumerate() {
            slot = slot * od + if axes.contains(&axis) { 0 } else { c };
        }
        out[slot] += f(v);
        for axis in (0..dims.len()).rev() {
            coords[axis] += 1;
            if coords[axis] < dims[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
    Tensor::new(out, &out_dims).expect("reduction dims consistent by construction")
}

impl OpSpec for Sum {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        let acc = sum_axes_i64(inputs[0], &self.axes);
        Ok(Eval::plain(acc.map(sat_clamp_i32)))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        let dims = same_dims(inputs, Some(1))?;
        for &a in &self.axes {
            if a >= dims.len() {
                return Err(ShapeError::AxisOutOfRange {
                    axis: a,
                    rank: dims.len(),
                });
            }
        }
        Ok(dims
            .iter()
            .enumerate()
            .map(|(i, &d)| if self.axes.contains(&i) { 1 } else { d })
            .collect())
    }

    fn clamp_bound(&self, ctx: &BoundCtx<'_>) -> Option<u128> {
        let input = ctx.input_dims.first()?;
        let n: u128 = self
            .axes
            .iter()
            .map(|&a| *input.get(a).unwrap_or(&1) as u128)
            .product::<u128>()
            .max(1);
        Some(n * I32_MAG)
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }
}

impl fmt::Display for Div {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Div")
    }
}

impl OpSpec for Div {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        let (a, b) = (inputs[0], inputs[1]);
        if a.dims() != b.dims() {
            return Err(ShapeError::BroadcastIncompatible {
                lhs: a.dims().to_vec(),
                rhs: b.dims().to_vec(),
            }
            .into());
        }
        if let Some(element) = b.data().iter().position(|&d| d == 0) {
            return Err(KernelError::DivByZero { element });
        }
        Ok(Eval {
            output: a.zip_map(b, floor_div)?,
            aux: Some(NodeAux::Remainder(a.zip_map(b, rem_floor)?)),
        })
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(2))
    }

    fn padding(&self) -> PadBehavior {
        // Padded divisor lanes are zero; `eval` rejects them, so padded
        // graphs must keep Div inputs unpadded on live lanes. The importer
        // never emits Div (it rewrites to Mul/Rsqrt); flag it masked so a
        // hand-built padded graph fails loudly rather than silently.
        PadBehavior::NeedsMask
    }
}

impl fmt::Display for ScalarConstDiv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScalarConstDiv(÷{})", self.divisor)
    }
}

impl OpSpec for ScalarConstDiv {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        Ok(Eval {
            output: inputs[0].map(|x| floor_div(x, self.divisor)),
            aux: Some(NodeAux::Remainder(
                inputs[0].map(|x| rem_floor(x, self.divisor)),
            )),
        })
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(1))
    }

    fn validate(&self) -> Result<(), String> {
        if self.divisor == 0 {
            return Err("divisor must be non-zero".into());
        }
        Ok(())
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }
}

impl fmt::Display for Rsqrt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Rsqrt")
    }
}

impl OpSpec for Rsqrt {
    fn eval(&self, inputs: &[&Tensor<i32>], ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        let scale = ctx.scale;
        let input = inputs[0];
        let output = input.map(|x| rsqrt(x, scale));
        let s_cubed = 1i64 << (3 * scale.log2());
        let quotient: Vec<i64> = input
            .data()
            .iter()
            .map(|&x| if x > 0 { s_cubed / x as i64 } else { 0 })
            .collect();
        let div_remainder: Vec<i32> = input
            .data()
            .iter()
            .map(|&x| {
                if x > 0 {
                    (s_cubed % x as i64) as i32
                } else {
                    0
                }
            })
            .collect();
        let sqrt_remainder: Vec<i32> = quotient
            .iter()
            .zip(output.data())
            .map(|(&q, &out)| (q - out as i64 * out as i64) as i32)
            .collect();
        let dims = input.dims();
        Ok(Eval {
            aux: Some(NodeAux::Rsqrt(RsqrtAdvice {
                quotient: Tensor::new(quotient, dims)?,
                div_remainder: Tensor::new(div_remainder, dims)?,
                sqrt_remainder: Tensor::new(sqrt_remainder, dims)?,
            })),
            output,
        })
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(1))
    }

    fn padding(&self) -> PadBehavior {
        // rsqrt(0) = 0, so zero padding is preserved.
        PadBehavior::ZeroPreserving
    }
}

impl fmt::Display for ReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ReLU")
    }
}

impl OpSpec for ReLU {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        Ok(Eval::plain(inputs[0].map(|x| x.max(0))))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(1))
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }
}

impl fmt::Display for Clamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Clamp(±2^{})", self.bound_log)
    }
}

impl OpSpec for Clamp {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        let bound = 1i32 << self.bound_log;
        Ok(Eval::plain(inputs[0].map(|x| x.clamp(-bound, bound - 1))))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(1))
    }

    fn validate(&self) -> Result<(), String> {
        if self.bound_log >= 31 {
            return Err(format!("bound_log {} overflows i32", self.bound_log));
        }
        Ok(())
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }
}

/// Check every element of `t` is 0/1, else report the first offender.
fn check_boolean(t: &Tensor<i32>, op: &'static str) -> Result<(), KernelError> {
    if let Some(element) = t.data().iter().position(|&v| v != 0 && v != 1) {
        return Err(KernelError::NotBoolean {
            op,
            value: t.data()[element],
            element,
        });
    }
    Ok(())
}

impl fmt::Display for Iff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Iff")
    }
}

impl OpSpec for Iff {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        let (mask, a, b) = (inputs[0], inputs[1], inputs[2]);
        check_boolean(mask, "Iff")?;
        let masked_a = mask.zip_map(a, |m, x| m * x)?;
        let masked_b = mask.zip_map(b, |m, x| (1 - m) * x)?;
        Ok(Eval::plain(masked_a.zip_map(&masked_b, |x, y| x + y)?))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(3))
    }

    fn padding(&self) -> PadBehavior {
        // Padded mask lanes are 0 ⇒ output = (1-0)·b = b's padded zeros.
        PadBehavior::ZeroPreserving
    }
}

impl fmt::Display for And {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("And")
    }
}

impl OpSpec for And {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        let (a, b) = (inputs[0], inputs[1]);
        check_boolean(a, "And")?;
        check_boolean(b, "And")?;
        // The prover proves And with the Mul sumcheck: output = a·b.
        Ok(Eval::plain(a.zip_map(b, |x, y| x * y)?))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(2))
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }
}

impl fmt::Display for IsNan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("IsNan")
    }
}

impl OpSpec for IsNan {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        // The quantized domain has no NaN; the output is identically zero
        // (and the prover only checks a zero claim).
        Ok(Eval::plain(Tensor::zeros(inputs[0].dims())))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(1))
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{Scale, SemVersion};
    use crate::op::TableCache;

    /// A spread of i32 values: small, large, both signs, the extremes that do not overflow.
    fn values() -> Vec<i32> {
        let mut v = vec![0, 1, -1, 2, -2, 7, -7, 1000, -1000, i32::MAX, i32::MIN + 1];
        v.extend((0..64).map(|i| (i * 0x0101_3F1Fi64 % 2_000_003 - 1_000_001) as i32));
        v
    }

    fn ctx(cache: &TableCache) -> EvalCtx<'_> {
        EvalCtx {
            scale: Scale::new(14).unwrap(),
            version: SemVersion::V1,
            tables: cache,
        }
    }

    /// `Div` captures `a = out·b + r` with `r` of the divisor's sign and `|r| < |b|`.
    #[test]
    fn div_captures_its_floor_remainder() {
        let cache = TableCache::default();
        let a = values();
        for &b in &[1, 3, 7, 256, -3, i32::MAX] {
            let lhs = Tensor::new(a.clone(), &[a.len()]).unwrap();
            let rhs = Tensor::new(vec![b; a.len()], &[a.len()]).unwrap();
            let eval = Div.eval(&[&lhs, &rhs], &ctx(&cache)).unwrap();
            let Some(NodeAux::Remainder(r)) = eval.aux else {
                panic!("Div captures a remainder");
            };
            for ((&a, &q), &r) in a.iter().zip(eval.output.data()).zip(r.data()) {
                assert_eq!(a as i64, q as i64 * b as i64 + r as i64);
                assert!(if b > 0 {
                    0 <= r && r < b
                } else {
                    b < r && r <= 0
                });
            }
        }
    }

    /// `ScalarConstDiv` captures the same remainder for its constant divisor.
    #[test]
    fn scalar_const_div_captures_its_floor_remainder() {
        let cache = TableCache::default();
        let a = values();
        for divisor in [1, 7, 128, 1 << 14, -3] {
            let input = Tensor::new(a.clone(), &[a.len()]).unwrap();
            let eval = ScalarConstDiv { divisor }
                .eval(&[&input], &ctx(&cache))
                .unwrap();
            let Some(NodeAux::Remainder(r)) = eval.aux else {
                panic!("ScalarConstDiv captures a remainder");
            };
            for ((&a, &q), &r) in a.iter().zip(eval.output.data()).zip(r.data()) {
                assert_eq!(a as i64, q as i64 * divisor as i64 + r as i64);
                assert!(if divisor > 0 {
                    0 <= r && r < divisor
                } else {
                    divisor < r && r <= 0
                });
            }
        }
    }

    /// `Rsqrt` captures `S³ = q·x + r_div`, `q = out² + r_sqrt` with both remainders in range,
    /// and zeros where the input is not positive.
    #[test]
    fn rsqrt_captures_its_division_and_square_root() {
        let cache = TableCache::default();
        let x = values();
        let input = Tensor::new(x.clone(), &[x.len()]).unwrap();
        let eval = Rsqrt.eval(&[&input], &ctx(&cache)).unwrap();
        let Some(NodeAux::Rsqrt(advice)) = eval.aux else {
            panic!("Rsqrt captures its advice");
        };
        let s_cubed = 1i64 << 42;
        for (i, &x) in x.iter().enumerate() {
            let out = eval.output.data()[i] as i64;
            let (q, r_div, r_sqrt) = (
                advice.quotient.data()[i],
                advice.div_remainder.data()[i] as i64,
                advice.sqrt_remainder.data()[i] as i64,
            );
            if x <= 0 {
                assert_eq!((out, q, r_div, r_sqrt), (0, 0, 0, 0));
                continue;
            }
            assert_eq!(s_cubed, q * x as i64 + r_div);
            assert!(0 <= r_div && r_div < x as i64);
            assert_eq!(q, out * out + r_sqrt);
            assert!(0 <= r_sqrt && r_sqrt <= 2 * out);
        }
    }
}
