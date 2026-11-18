//! Fused-rescale elementwise operators: mul, square, cube, mean-of-squares.
//!
//! Each accumulates in `i64`, splits `acc = quotient·D + remainder`
//! (Euclidean, via [`crate::semantics::fixed::rebase`]), emits
//! the quotient, which must fit `i32`, and captures the remainder in the trace.

use std::fmt;

use crate::{
    error::{KernelError, ShapeError},
    id::Scale,
    semantics::fixed::{narrow_i32, rebase},
    tensor::Tensor,
    trace::{FusedIntermediates, NodeAux},
};

use super::{
    Cube, Eval, EvalCtx, MAX_LEVEL, MeanOfSquares, Mul, OpSpec, Operand, PadBehavior, Rescale,
    Square, same_dims,
};

/// Split an i64 accumulation tensor by `divisor`, producing the output (the quotient, which
/// must fit `i32`) and the captured intermediates. The single implementation of the fused
/// contract for every rescaling op.
pub(super) fn rebase_eval(
    acc: Tensor<i64>,
    divisor: i64,
    op: &'static str,
) -> Result<Eval, KernelError> {
    // One pass: the quotient narrowed and the remainder, with a flag if a quotient does not
    // fit, in which case the first one in order is the error.
    let overflowed = std::sync::atomic::AtomicBool::new(false);
    let split = |&v: &i64| {
        let (quotient, remainder) = rebase(v, divisor);
        if narrow_i32(quotient).is_none() {
            overflowed.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        (quotient as i32, remainder)
    };
    #[cfg(feature = "rayon")]
    let (quotient, remainder): (Vec<i32>, Vec<i32>) = {
        use rayon::prelude::*;
        acc.data()
            .par_iter()
            .with_min_len(crate::tensor::PAR_MIN_LEN)
            .map(split)
            .unzip()
    };
    #[cfg(not(feature = "rayon"))]
    let (quotient, remainder): (Vec<i32>, Vec<i32>) = acc.data().iter().map(split).unzip();
    if overflowed.into_inner() {
        let value = acc
            .data()
            .iter()
            .map(|&v| rebase(v, divisor).0)
            .find(|&quotient| narrow_i32(quotient).is_none())
            .expect("a quotient was flagged");
        return Err(KernelError::WitnessOverflow {
            op,
            what: "quotient",
            value,
        });
    }
    let dims = acc.dims();
    Ok(Eval {
        wide: None,
        output: Tensor::new(quotient, dims).expect("same length"),
        aux: Some(NodeAux::Fused(FusedIntermediates {
            remainder: Tensor::new(remainder, dims).expect("same length"),
        })),
    })
}

impl fmt::Display for Mul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match (self.rebase, self.deferred) {
            (true, _) => "Mul",
            (false, true) => "Mul(deferred)",
            (false, false) => "Mul(raw)",
        })
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

    fn validate(&self) -> Result<(), String> {
        if self.rebase && self.deferred {
            return Err("a product is rescaled or deferred, not both".to_string());
        }
        Ok(())
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

    /// A deferred product is at the sum of its operands' levels.
    fn level(&self, operands: &[u8]) -> Result<u8, String> {
        if self.deferred {
            return Ok(operands.iter().sum());
        }
        if operands.iter().all(|&level| level == 1) {
            Ok(1)
        } else {
            Err(format!(
                "only a deferred product takes operands at levels {operands:?}"
            ))
        }
    }

    fn eval_wide(
        &self,
        inputs: &[Operand<'_>],
        _levels: &[u8],
        _ctx: &EvalCtx<'_>,
    ) -> Result<Eval, KernelError> {
        let [a, b] = inputs else {
            return Err(ShapeError::WrongArity {
                expected: 2,
                got: inputs.len(),
            }
            .into());
        };
        if a.dims() != b.dims() {
            return Err(ShapeError::BroadcastIncompatible {
                lhs: a.dims().to_vec(),
                rhs: b.dims().to_vec(),
            }
            .into());
        }
        let len: usize = a.dims().iter().product();
        let product = |i: usize| a.at(i) * b.at(i);
        #[cfg(feature = "rayon")]
        let data: Vec<i128> = {
            use rayon::prelude::*;
            (0..len)
                .into_par_iter()
                .with_min_len(crate::tensor::PAR_MIN_LEN)
                .map(product)
                .collect()
        };
        #[cfg(not(feature = "rayon"))]
        let data: Vec<i128> = (0..len).map(product).collect();
        Ok(Eval::wide(Tensor::new(data, a.dims())?))
    }

    fn wide_bound(&self, operands: &[u128], _dims: &[&[usize]]) -> u128 {
        operands.iter().fold(1u128, |acc, &b| acc.saturating_mul(b))
    }
}

impl fmt::Display for Rescale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Rescale")
    }
}

impl OpSpec for Rescale {
    fn eval(&self, _inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        Err(KernelError::NoWideForm {
            op: "rescale",
            levels: vec![1],
        })
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(1))
    }

    /// The quotient is an `i32` the proof must show in range.
    fn checks_output(&self) -> bool {
        true
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }

    fn level(&self, operands: &[u8]) -> Result<u8, String> {
        match operands {
            [level] if (2..=MAX_LEVEL).contains(level) => Ok(1),
            _ => Err(format!(
                "a rescale takes one wide operand of level at most {MAX_LEVEL}, not {operands:?}"
            )),
        }
    }

    fn eval_wide(
        &self,
        inputs: &[Operand<'_>],
        levels: &[u8],
        ctx: &EvalCtx<'_>,
    ) -> Result<Eval, KernelError> {
        let ([Operand::Wide(wide)], [level]) = (inputs, levels) else {
            return Err(KernelError::NoWideForm {
                op: "rescale",
                levels: levels.to_vec(),
            });
        };
        let divisor = 1i128 << ((*level as u32 - 1) * ctx.scale.log2());
        // One pass: the quotient narrowed and the remainder, with a flag if a quotient does
        // not fit, in which case the first one in order is the error.
        let overflowed = std::sync::atomic::AtomicBool::new(false);
        let split = |&v: &i128| {
            let quotient = v.div_euclid(divisor);
            if i32::try_from(quotient).is_err() {
                overflowed.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            (quotient as i32, v.rem_euclid(divisor) as i64)
        };
        #[cfg(feature = "rayon")]
        let (quotient, remainder): (Vec<i32>, Vec<i64>) = {
            use rayon::prelude::*;
            wide.data()
                .par_iter()
                .with_min_len(crate::tensor::PAR_MIN_LEN)
                .map(split)
                .unzip()
        };
        #[cfg(not(feature = "rayon"))]
        let (quotient, remainder): (Vec<i32>, Vec<i64>) = wide.data().iter().map(split).unzip();
        if overflowed.into_inner() {
            let value = wide
                .data()
                .iter()
                .map(|&v| v.div_euclid(divisor))
                .find(|&quotient| i32::try_from(quotient).is_err())
                .expect("a quotient was flagged");
            return Err(KernelError::WitnessOverflow {
                op: "rescale",
                what: "quotient",
                value: value.clamp(i64::MIN as i128, i64::MAX as i128) as i64,
            });
        }
        let dims = wide.dims();
        Ok(Eval {
            output: Tensor::new(quotient, dims)?,
            wide: None,
            aux: Some(NodeAux::RescaleRemainder(Tensor::new(remainder, dims)?)),
        })
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
