//! Table-lookup activations. The runtime kernel is literally a lookup into
//! the deterministic table from [`crate::semantics::tables`], which makes
//! tracer and prover bit-identical by construction.

use std::fmt;

use crate::{
    error::{KernelError, ShapeError},
    semantics::tables::{IndexRule, TableKind},
    tensor::Tensor,
    trace::{NodeAux, TeleportAdvice},
};

use super::{
    Cos, Erf, Eval, EvalCtx, OpSpec, PadBehavior, Sigmoid, Sin, SmoothRelu, Tanh, same_dims,
};
use serde::{Deserialize, Serialize};

/// Which smooth activation a [`SmoothRelu`] computes.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum SmoothReluForm {
    /// GELU, `x·(1 + erf(x/√2))/2`.
    GeluErf,
    /// GELU in its tanh form, `x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))/2`.
    GeluTanh,
    /// SiLU, `x·σ(x)`.
    Silu,
}

impl SmoothReluForm {
    /// The table holding the form's correction to `ReLU`.
    pub fn table(self) -> TableKind {
        match self {
            Self::GeluErf => TableKind::GeluErf,
            Self::GeluTanh => TableKind::GeluTanh,
            Self::Silu => TableKind::Silu,
        }
    }
}

impl fmt::Display for SmoothRelu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SmoothRelu({:?})", self.form)
    }
}

impl OpSpec for SmoothRelu {
    fn eval(&self, inputs: &[&Tensor<i32>], ctx: &EvalCtx<'_>) -> Result<Eval, KernelError> {
        // `max(x, 0)` plus a correction in `(−S, 0]`: no overflow, since the correction is zero
        // wherever `x` is outside the table's domain.
        let table = ctx.tables.get(self.form.table(), ctx.scale, ctx.version);
        Ok(Eval::plain(inputs[0].map(|x| x.max(0) + table.lookup(x))))
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
        same_dims(inputs, Some(1))
    }

    fn padding(&self) -> PadBehavior {
        // f(0) = 0 for every form.
        PadBehavior::ZeroPreserving
    }
}

macro_rules! impl_activation {
    ($name:ident, $kind:expr, $zero_preserving:expr) => {
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(stringify!($name))
            }
        }

        impl OpSpec for $name {
            fn eval(
                &self,
                inputs: &[&Tensor<i32>],
                ctx: &EvalCtx<'_>,
            ) -> Result<Eval, KernelError> {
                let table = ctx.tables.get($kind, ctx.scale, ctx.version);
                Ok(Eval {
                    output: inputs[0].map(|x| table.lookup(x)),
                    aux: teleport_advice(table.index, inputs[0]),
                })
            }

            fn infer_shape(&self, inputs: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
                same_dims(inputs, Some(1))
            }

            fn padding(&self) -> PadBehavior {
                $zero_preserving
            }
        }
    };
}

/// The period reduction a trig table indexes by, captured for the prover; `None` for a table
/// that does not reduce its input.
fn teleport_advice(index: IndexRule, input: &Tensor<i32>) -> Option<NodeAux> {
    let IndexRule::TrigReduced { modulus, .. } = index else {
        return None;
    };
    let modulus = modulus as i32;
    Some(NodeAux::Teleport(TeleportAdvice {
        quotient: input.map(|x| x.div_euclid(modulus)),
        remainder: input.map(|x| x.rem_euclid(modulus)),
    }))
}

// tanh(0) = erf(0) = sin(0) = 0, so zero padding passes through.
impl_activation!(Tanh, TableKind::Tanh, PadBehavior::ZeroPreserving);
impl_activation!(Erf, TableKind::Erf, PadBehavior::ZeroPreserving);
impl_activation!(Sin, TableKind::Sin, PadBehavior::ZeroPreserving);
// sigmoid(0) = S/2 and cos(0) = S: padded lanes carry those constants, same
// as the old crate. Downstream ops must tolerate them (they did before).
impl_activation!(Sigmoid, TableKind::Sigmoid, PadBehavior::ConstantFill);
impl_activation!(Cos, TableKind::Cos, PadBehavior::ConstantFill);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{Scale, SemVersion};
    use crate::op::TableCache;

    #[test]
    fn tanh_lookup_matches_table() {
        let cache = TableCache::default();
        let ctx = EvalCtx {
            scale: Scale::new(8).unwrap(),
            version: SemVersion::V1,
            tables: &cache,
        };
        let x = Tensor::new(vec![0, 256, -256, 1 << 12, -(1 << 12)], &[5]).unwrap();
        let out = Tanh.eval(&[&x], &ctx).unwrap().output;
        // tanh(0)=0, tanh(±1)≈±0.7616·256≈±195, saturated at the clamp bound.
        assert_eq!(out.data()[0], 0);
        assert_eq!(out.data()[1], 195);
        assert_eq!(out.data()[2], -195);
        assert_eq!(out.data()[3], 256); // deep saturation
        assert_eq!(out.data()[4], -256);
    }

    /// `Sin` and `Cos` capture `x = P·q + r` with `r ∈ [0, P)`, the reduction their table
    /// indexes by; a table that does not reduce captures nothing.
    #[test]
    fn trig_captures_its_period_reduction() {
        let cache = TableCache::default();
        let ctx = EvalCtx {
            scale: Scale::new(14).unwrap(),
            version: SemVersion::V1,
            tables: &cache,
        };
        let values = vec![0, 1, -1, 50_000, -50_000, i32::MAX, i32::MIN];
        let x = Tensor::new(values.clone(), &[values.len()]).unwrap();
        for (kind, eval) in [
            (TableKind::Sin, Sin.eval(&[&x], &ctx).unwrap()),
            (TableKind::Cos, Cos.eval(&[&x], &ctx).unwrap()),
        ] {
            let IndexRule::TrigReduced { modulus, .. } =
                cache.get(kind, ctx.scale, ctx.version).index
            else {
                panic!("trig tables reduce by a period");
            };
            let Some(NodeAux::Teleport(advice)) = eval.aux else {
                panic!("trig ops capture their reduction");
            };
            for (i, &x) in values.iter().enumerate() {
                let (q, r) = (advice.quotient.data()[i], advice.remainder.data()[i]);
                assert_eq!(x as i64, modulus as i64 * q as i64 + r as i64);
                assert!(0 <= r && (r as u32) < modulus);
            }
        }
        assert!(Tanh.eval(&[&x], &ctx).unwrap().aux.is_none());
    }

    /// `SmoothRelu` is `ReLU` plus its form's table, and the identity or zero far from zero.
    #[test]
    fn smooth_relu_is_relu_plus_its_correction() {
        let cache = TableCache::default();
        let ctx = EvalCtx {
            scale: Scale::new(14).unwrap(),
            version: SemVersion::V2,
            tables: &cache,
        };
        let values = vec![
            0,
            1,
            -1,
            16384,
            -16384,
            1 << 20,
            -(1 << 20),
            i32::MAX,
            i32::MIN,
        ];
        let x = Tensor::new(values.clone(), &[values.len()]).unwrap();
        for form in [
            SmoothReluForm::GeluErf,
            SmoothReluForm::GeluTanh,
            SmoothReluForm::Silu,
        ] {
            let out = SmoothRelu { form }.eval(&[&x], &ctx).unwrap().output;
            let table = cache.get(form.table(), ctx.scale, ctx.version);
            for (&x, &y) in values.iter().zip(out.data()) {
                assert_eq!(y, x.max(0) + table.lookup(x), "{form:?} at {x}");
            }
            assert_eq!(
                &out.data()[5..],
                &[1 << 20, 0, i32::MAX, 0],
                "{form:?} tails"
            );
            assert_eq!(out.data()[0], 0);
        }
        // GELU(1) = 0.8413·S and SiLU(1) = 0.7311·S.
        let one = Tensor::new(vec![16384], &[1]).unwrap();
        let at_one = |form| {
            SmoothRelu { form }
                .eval(&[&one], &ctx)
                .unwrap()
                .output
                .data()[0]
        };
        assert_eq!(at_one(SmoothReluForm::GeluErf), 13785);
        assert_eq!(at_one(SmoothReluForm::Silu), 11978);
    }
}
