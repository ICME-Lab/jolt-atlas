//! Structural operators: index plumbing with no arithmetic.

use std::fmt;

use crate::{
    error::{KernelError, ShapeError},
    tensor::{self, Tensor},
};

use super::{
    Broadcast, Concat, Constant, Eval, EvalCtx, GatherLarge, GatherSmall, Identity, Input,
    MoveAxis, OpSpec, PadBehavior, PadCtx, Reshape, Slice, same_dims,
};

/// Pad each dim to its next power of two.
fn pow2_dims(dims: &[usize]) -> Vec<usize> {
    dims.iter().map(|d| d.next_power_of_two()).collect()
}

/// Factor a reshape into balanced groups (standard reshape grouping: the
/// smallest consecutive axis runs with equal products) and verify padding
/// coherence: in every group, each non-leading axis must already be a power
/// of two on both sides.
fn check_pad_coherent_reshape(input: &[usize], target: &[usize]) -> Result<(), String> {
    let (mut i, mut j) = (0usize, 0usize);
    while i < input.len() || j < target.len() {
        // Grow a balanced group: take one axis per side, then extend the
        // smaller side until the products match.
        let (gi, gj) = (i, j);
        let mut pi = 1usize;
        let mut pj = 1usize;
        if i < input.len() {
            pi *= input[i];
            i += 1;
        }
        if j < target.len() {
            pj *= target[j];
            j += 1;
        }
        while pi != pj {
            if pi < pj {
                if i >= input.len() {
                    return Err(format!(
                        "reshape {input:?} -> {target:?} has unbalanced factor groups"
                    ));
                }
                pi *= input[i];
                i += 1;
            } else {
                if j >= target.len() {
                    return Err(format!(
                        "reshape {input:?} -> {target:?} has unbalanced factor groups"
                    ));
                }
                pj *= target[j];
                j += 1;
            }
        }
        // Non-leading axes of a multi-axis group must be unpadded (pow2).
        for &d in &input[(gi + 1).min(i)..i] {
            if !d.is_power_of_two() {
                return Err(format!(
                    "reshape {input:?} -> {target:?}: inner merged axis {d} is not a power of two; \
                     padding would scramble the flattened layout"
                ));
            }
        }
        for &d in &target[(gj + 1).min(j)..j] {
            if !d.is_power_of_two() {
                return Err(format!(
                    "reshape {input:?} -> {target:?}: inner split axis {d} is not a power of two; \
                     padding would scramble the flattened layout"
                ));
            }
        }
    }
    Ok(())
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Input")
    }
}

impl OpSpec for Input {
    fn eval(&self, _inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        // The executor substitutes the supplied tensor; this is unreachable.
        unreachable!("Input nodes are materialized by the executor")
    }

    fn infer_shape(&self, _inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        // No inputs to infer from; the compile pass keeps the declared dims.
        Err(ShapeError::Invalid(
            "Input has no operands; shape is declared".into(),
        ))
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::Structural
    }
}

impl fmt::Display for Constant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Constant{:?}", self.0.dims())
    }
}

impl OpSpec for Constant {
    fn eval(&self, _inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        Ok(Eval::plain(self.0.clone()))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        if !inputs.is_empty() {
            return Err(ShapeError::WrongArity {
                expected: 0,
                got: inputs.len(),
            });
        }
        Ok(self.0.dims().to_vec())
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::Structural
    }

    fn apply_padding(&mut self, _ctx: &PadCtx<'_>) -> Result<(), String> {
        self.0 = self.0.padded_pow2();
        Ok(())
    }
}

impl fmt::Display for Identity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Identity")
    }
}

impl OpSpec for Identity {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        Ok(Eval::plain(inputs[0].clone()))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(1))
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::ZeroPreserving
    }
}

impl fmt::Display for Reshape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reshape{:?}", self.shape)
    }
}

impl OpSpec for Reshape {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        Ok(Eval::plain(inputs[0].clone().reshape(&self.shape)?))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        let input = same_dims(inputs, Some(1))?;
        let in_len: usize = input.iter().product();
        let out_len: usize = self.shape.iter().product();
        if in_len != out_len {
            return Err(ShapeError::LengthMismatch {
                len: in_len,
                dims: self.shape.clone(),
                expected: out_len,
            });
        }
        Ok(self.shape.clone())
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::Structural
    }

    fn apply_padding(&mut self, ctx: &PadCtx<'_>) -> Result<(), String> {
        // A reshape stays coherent under per-dim padding only if, within
        // each balanced factor group of the reshape, every non-leading axis
        // is already a power of two on BOTH sides: then the original values
        // occupy the same flat positions in the padded corner as in the
        // padded target, with padding confined to the tail. Otherwise
        // padding lands *inside* the merged dimension and the layout
        // scrambles (e.g. merging a grouped-heads axis `[2, 7] -> [14]`).
        check_pad_coherent_reshape(ctx.input_dims[0], &self.shape)?;
        let padded_shape = pow2_dims(&self.shape);
        let padded_in: usize = pow2_dims(ctx.input_dims[0]).iter().product();
        let padded_out: usize = padded_shape.iter().product();
        if padded_in != padded_out {
            return Err(format!(
                "padded reshape element count mismatch: input {:?} ({padded_in} elements) vs target {:?} ({padded_out} elements)",
                pow2_dims(ctx.input_dims[0]),
                padded_shape,
            ));
        }
        self.shape = padded_shape;
        Ok(())
    }
}

impl fmt::Display for Broadcast {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Broadcast{:?}", self.shape)
    }
}

impl OpSpec for Broadcast {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        Ok(Eval::plain(inputs[0].expand(&self.shape)?))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        let input = same_dims(inputs, Some(1))?;
        // Must be a valid expansion.
        if input.len() > self.shape.len()
            || input
                .iter()
                .rev()
                .zip(self.shape.iter().rev())
                .any(|(&src, &dst)| src != dst && src != 1)
        {
            return Err(ShapeError::ExpandIncompatible {
                from: input,
                to: self.shape.clone(),
            });
        }
        Ok(self.shape.clone())
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::Structural
    }

    fn apply_padding(&mut self, _ctx: &PadCtx<'_>) -> Result<(), String> {
        self.shape = pow2_dims(&self.shape);
        Ok(())
    }
}

impl fmt::Display for MoveAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MoveAxis({}→{})", self.source, self.destination)
    }
}

impl OpSpec for MoveAxis {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        Ok(Eval::plain(
            inputs[0].move_axis(self.source, self.destination)?,
        ))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        let mut dims = same_dims(inputs, Some(1))?;
        let rank = dims.len();
        if self.source >= rank || self.destination >= rank {
            return Err(ShapeError::AxisOutOfRange {
                axis: self.source.max(self.destination),
                rank,
            });
        }
        let d = dims.remove(self.source);
        dims.insert(self.destination, d);
        Ok(dims)
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::Structural
    }
}

impl fmt::Display for Slice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Slice(axis {}, {}..{})", self.axis, self.start, self.end)
    }
}

impl OpSpec for Slice {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        Ok(Eval::plain(inputs[0].slice(self.axis, self.start, self.end)?))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        let mut dims = same_dims(inputs, Some(1))?;
        if self.axis >= dims.len() {
            return Err(ShapeError::AxisOutOfRange {
                axis: self.axis,
                rank: dims.len(),
            });
        }
        if self.start > self.end || self.end > dims[self.axis] {
            return Err(ShapeError::SliceOutOfRange {
                axis: self.axis,
                start: self.start,
                end: self.end,
                size: dims[self.axis],
            });
        }
        dims[self.axis] = self.end - self.start;
        Ok(dims)
    }

    fn validate(&self) -> Result<(), String> {
        if self.start >= self.end {
            return Err(format!("empty slice {}..{}", self.start, self.end));
        }
        Ok(())
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::Structural
    }

    fn apply_padding(&mut self, _ctx: &PadCtx<'_>) -> Result<(), String> {
        // The slice range addresses original coordinates, which padding
        // preserves (it only appends). The sliced output must still land on
        // padded dims, so its extent has to already be a power of two.
        let len = self.end - self.start;
        if !len.is_power_of_two() {
            return Err(format!(
                "slice extent {len} is not a power of two; padded output dims would not match"
            ));
        }
        Ok(())
    }
}

impl fmt::Display for Concat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Concat(axis {})", self.axis)
    }
}

impl OpSpec for Concat {
    fn eval(&self, inputs: &[&Tensor<i32>], _ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        Ok(Eval::plain(tensor::concat(inputs, self.axis)?))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        let first = *inputs.first().ok_or(ShapeError::WrongArity {
            expected: 1,
            got: 0,
        })?;
        if self.axis >= first.len() {
            return Err(ShapeError::AxisOutOfRange {
                axis: self.axis,
                rank: first.len(),
            });
        }
        let mut dims = first.to_vec();
        dims[self.axis] = 0;
        for d in inputs {
            if d.len() != first.len()
                || d.iter()
                    .zip(first)
                    .enumerate()
                    .any(|(i, (a, b))| i != self.axis && a != b)
            {
                return Err(ShapeError::Invalid(format!(
                    "concat: {d:?} incompatible with {first:?} on axis {}",
                    self.axis
                )));
            }
            dims[self.axis] += d[self.axis];
        }
        Ok(dims)
    }

    fn padding(&self) -> PadBehavior {
        PadBehavior::Structural
    }

    fn apply_padding(&mut self, ctx: &PadCtx<'_>) -> Result<(), String> {
        // Each operand's concat-axis extent pads independently; the padded
        // extents must add up to the padded output extent or the kernel's
        // output dims diverge from the declared ones.
        let sum_padded: usize = ctx
            .input_dims
            .iter()
            .map(|d| d[self.axis].next_power_of_two())
            .sum();
        let padded_out = ctx.output_dims[self.axis].next_power_of_two();
        if sum_padded != padded_out {
            return Err(format!(
                "concat axis {}: padded operand extents sum to {sum_padded} but padded output is {padded_out}",
                self.axis
            ));
        }
        Ok(())
    }
}

/// Shared gather kernel: embedding lookup on axis 0.
fn gather_eval(
    dict: &Tensor<i32>,
    indices: &Tensor<i32>,
) -> Result<Tensor<i32>, KernelError> {
    let dict_rows = dict.dims()[0];
    let row: usize = dict.dims()[1..].iter().product();
    let mut out = Vec::with_capacity(indices.len() * row);
    for &ix in indices.data() {
        if ix < 0 || ix as usize >= dict_rows {
            return Err(KernelError::GatherOutOfRange {
                index: ix as i64,
                dict_len: dict_rows,
            });
        }
        let base = ix as usize * row;
        out.extend_from_slice(&dict.data()[base..base + row]);
    }
    let mut dims = indices.dims().to_vec();
    dims.extend_from_slice(&dict.dims()[1..]);
    Tensor::new(out, &dims).map_err(KernelError::Shape)
}

fn gather_shape(inputs: &[&[usize]], axis: usize) -> Result<Vec<usize>, ShapeError> {
    if inputs.len() != 2 {
        return Err(ShapeError::WrongArity {
            expected: 2,
            got: inputs.len(),
        });
    }
    if axis != 0 {
        return Err(ShapeError::Invalid(format!(
            "gather supports axis 0 only, got {axis}"
        )));
    }
    let (dict, index) = (inputs[0], inputs[1]);
    if dict.is_empty() {
        return Err(ShapeError::Invalid("gather dictionary is rank 0".into()));
    }
    let mut dims = index.to_vec();
    dims.extend_from_slice(&dict[1..]);
    Ok(dims)
}

macro_rules! impl_gather {
    ($name:ident, $label:literal) => {
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($label, "(dict {})"), self.dict_len)
            }
        }

        impl OpSpec for $name {
            fn eval(
                &self,
                inputs: &[&Tensor<i32>],
                _ctx: &EvalCtx<'_>,
            ) -> Result<Eval, KernelError> {
                Ok(Eval::plain(gather_eval(inputs[0], inputs[1])?))
            }

            fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
                gather_shape(inputs, self.axis)
            }

            fn validate(&self) -> Result<(), String> {
                if self.axis != 0 {
                    return Err(format!("gather supports axis 0 only, got {}", self.axis));
                }
                Ok(())
            }

            fn padding(&self) -> PadBehavior {
                // dict_len stays the semantic (unpadded) dictionary size; the
                // prover rounds it up itself, matching the padded dim 0.
                PadBehavior::Structural
            }
        }
    };
}

impl_gather!(GatherSmall, "GatherSmall");
impl_gather!(GatherLarge, "GatherLarge");
