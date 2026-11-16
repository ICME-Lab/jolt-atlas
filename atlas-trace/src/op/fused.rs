//! Fused-rescale elementwise operators: mul, square, cube, mean-of-squares.
//!
//! Each accumulates in `i64`, splits `acc = quotient·D + remainder`
//! (Euclidean, via [`crate::semantics::fixed::rebase`]), emits
//! the quotient, which must fit `i32`, and captures the remainder in the trace.

use std::fmt;

use crate::{
    error::{KernelError, ShapeError},
    id::Scale,
    semantics::fixed::rebase,
    tensor::Tensor,
    trace::{FusedIntermediates, NodeAux},
};

use super::{Cube, Eval, EvalCtx, MeanOfSquares, Mul, OpSpec, PadBehavior, Square, same_dims};

/// Split an i64 accumulation tensor by `divisor`, producing the output (the quotient, which
/// must fit `i32`) and the captured intermediates. The single implementation of the fused
/// contract for every rescaling op.
pub(super) fn rebase_eval(
    acc: Tensor<i64>,
    divisor: i64,
    op: &'static str,
) -> Result<Eval, KernelError> {
    let mut quotient = Vec::with_capacity(acc.len());
    let mut remainder = Vec::with_capacity(acc.len());
    for &v in acc.data() {
        let (q, r) = rebase(v, divisor);
        quotient.push(q);
        remainder.push(r);
    }
    let dims = acc.dims();
    let quotient = Tensor::new(quotient, dims).expect("same length");
    Ok(Eval {
        output: super::linear::narrow(&quotient, op, "quotient")?,
        aux: Some(NodeAux::Fused(FusedIntermediates {
            remainder: Tensor::new(remainder, dims).expect("same length"),
        })),
    })
}

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.rebase { "Mul" } else { "Mul(raw)" })
    }
}

impl OpSpec for Mul {
    fn eval(&self, inputs: &[&Tensor<i32>], ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        let (a, b) = (inputs[0], inputs[1]);
        if !self.rebase {
            // Raw building-block product: already at target scale, no rescale.
            return Ok(Eval::plain(a.zip_map(b, |x, y| x.wrapping_mul(y))?));
        }
        let acc = a.zip_map(b, |x, y| x as i64 * y as i64)?;
        rebase_eval(
            acc,
            self.rebase_divisor(ctx.scale).expect("rebase on"),
            "mul",
        )
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(2))
    }

    fn checks_output(&self) -> bool {
        self.rebase
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }

    fn rebase_divisor(&self, scale: Scale) -> Option<i64> {
        self.rebase.then(|| scale.denom_i64())
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.rebase { "Square" } else { "Square(raw)" })
    }
}

impl OpSpec for Square {
    fn eval(&self, inputs: &[&Tensor<i32>], ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        let x = inputs[0];
        if !self.rebase {
            return Ok(Eval::plain(x.map(|v| v.wrapping_mul(v))));
        }
        let acc = x.map(|v| v as i64 * v as i64);
        rebase_eval(
            acc,
            self.rebase_divisor(ctx.scale).expect("rebase on"),
            "square",
        )
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(1))
    }

    fn checks_output(&self) -> bool {
        self.rebase
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }

    fn rebase_divisor(&self, scale: Scale) -> Option<i64> {
        self.rebase.then(|| scale.denom_i64())
    }
}

impl fmt::Display for Cube {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.rebase { "Cube" } else { "Cube(raw)" })
    }
}

impl OpSpec for Cube {
    fn eval(&self, inputs: &[&Tensor<i32>], ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        let x = inputs[0];
        if !self.rebase {
            return Ok(Eval::plain(x.map(|v| v.wrapping_mul(v).wrapping_mul(v))));
        }
        // x³ overflows i64 for |x| > 2^21: the operand is assumed to stay below it.
        let acc = x.map(|v| {
            let v = v as i64;
            v * v * v
        });
        rebase_eval(
            acc,
            self.rebase_divisor(ctx.scale).expect("rebase on"),
            "cube",
        )
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(1))
    }

    fn checks_output(&self) -> bool {
        self.rebase
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }

    fn rebase_divisor(&self, scale: Scale) -> Option<i64> {
        // x³ is at scale 3s and rebases to s: divisor 2^(2s).
        self.rebase.then(|| 1i64 << (2 * scale.log2()))
    }
}

impl fmt::Display for MeanOfSquares {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeanOfSquares(axes {:?}, n={})", self.axes, self.count)
    }
}

impl OpSpec for MeanOfSquares {
    fn eval(&self, inputs: &[&Tensor<i32>], ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        // Σx² over the reduced axes; zero-padded lanes contribute nothing,
        // and the divisor uses the semantic `count`, so the mean is exact
        // under padding.
        let acc = super::linear::sum_map_i64(inputs[0], &self.axes, |v| {
            let v = v as i64;
            v * v
        });
        rebase_eval(
            acc,
            self.rebase_divisor(ctx.scale).expect("always fused"),
            "mean_of_squares",
        )
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

    fn validate(&self) -> Result<(), String> {
        if self.count == 0 {
            return Err("count must be positive".into());
        }
        if self.padded_count < self.count {
            return Err(format!(
                "padded_count {} below count {}",
                self.padded_count, self.count
            ));
        }
        Ok(())
    }

    fn checks_output(&self) -> bool {
        true
    }

    fn padding(&self) -> PadBehavior {
        // Masks internally: `count` is the semantic denominator while the
        // reduction runs over the padded cube (`padded_count`).
        PadBehavior::NeedsMask
    }

    fn apply_padding(&mut self, ctx: &super::PadCtx<'_>) -> Result<(), String> {
        // Keep `count` (semantic); recompute `padded_count` from the padded
        // input dims so hand-built graphs stay consistent.
        let input = ctx.input_dims[0];
        self.padded_count = self
            .axes
            .iter()
            .map(|&a| input.get(a).copied().unwrap_or(1).next_power_of_two())
            .product();
        Ok(())
    }

    fn rebase_divisor(&self, scale: Scale) -> Option<i64> {
        Some(scale.denom_i64() * self.count as i64)
    }
}
