//! Witnesses and reverse proofs for MLP layers.

use jolt_transcript::Transcript;

use crate::protocol::{
    lookup::Batches,
    pcs::{Advice, Layout},
    session::{Claim, ProofField, Session},
    tensor::Shape,
    Result,
};

use super::{
    linear::{Linear, Product, ProductKind, Projection, Rounded},
    nonlinear::{Activate, Activation, ActivationTrace},
};

pub struct Mlp<'a> {
    pub up: Projection<'a>,
    pub gate: Option<Projection<'a>>,
    pub down: Projection<'a>,
    pub activation: Activation<'a>,
}

pub(crate) struct MlpPlan<'a> {
    input: Shape,
    activation_input: Linear<'a>,
    multiplier: Option<(Linear<'a>, Product)>,
    activation: Activate<'a>,
    down: Linear<'a>,
}

pub(crate) struct MlpTrace<F> {
    activation_input: Rounded,
    multiplier: Option<(Rounded, Rounded)>,
    activation: ActivationTrace<F>,
    down: Rounded,
}

impl<F> MlpTrace<F> {
    pub(crate) fn output(&self) -> &[i32] {
        &self.down.output
    }
}

impl<'a> MlpPlan<'a> {
    pub(crate) fn new(
        layout: &mut Layout,
        input: Shape,
        bits: usize,
        p: Mlp<'a>,
        input_bound: Option<u128>,
    ) -> Result<Self> {
        let projection = p.gate.unwrap_or(p.up);
        let activation_input = Linear::new(layout, input, projection, bits, false)?;
        let shape = activation_input.shape();
        let bound = input_bound.map(|bound| projection.output_bound(input.width, bound, 1 << bits));
        let activation = Activate::new(layout, shape, p.activation, bound)?;
        let multiplier = p
            .gate
            .map(|_| -> Result<_> {
                let gate = Linear::new(layout, input, p.up, bits, false)?;
                if gate.shape() != shape {
                    return Err("invalid MLP gate shape");
                }
                let product = Product::new(layout, shape, bits, ProductKind::Pointwise)?;
                Ok((gate, product))
            })
            .transpose()?;
        let down = Linear::new(layout, shape, p.down, bits, false)?;
        if down.shape() != input {
            return Err("invalid MLP output shape");
        }
        Ok(Self {
            input,
            activation_input,
            multiplier,
            activation,
            down,
        })
    }

    pub(crate) fn bind(&self, t: &mut impl Transcript) {
        self.activation_input.bind(t);
        self.activation.bind(t);
        t.append(&u64::from(self.multiplier.is_some()));
        if let Some((gate, _)) = &self.multiplier {
            gate.bind(t);
        }
        self.down.bind(t);
    }

    pub(crate) fn execute<F: ProofField>(&self, x: &[i32], a: &mut Advice) -> Result<MlpTrace<F>> {
        let activation_input = self.activation_input.execute(x, a)?;
        let activation = self.activation.execute(&activation_input.output, a)?;
        let multiplier = self
            .multiplier
            .as_ref()
            .map(|(gate, product)| -> Result<_> {
                let g = gate.execute(x, a)?;
                let y = product.execute(&activation.output, &g.output, a)?;
                Ok((g, y))
            })
            .transpose()?;
        let hidden = multiplier
            .as_ref()
            .map_or(activation.output.as_slice(), |(_, t)| &t.output);
        let down = self.down.execute(hidden, a)?;
        Ok(MlpTrace {
            activation_input,
            multiplier,
            activation,
            down,
        })
    }

    pub(crate) fn reverse<'s, 'w, F: ProofField>(
        &'s self,
        q: Claim<F>,
        x: impl Fn() -> &'w [i32],
        t: impl Fn() -> &'w MlpTrace<F> + Sync,
        io: &mut Session<'_, '_, F>,

        lookups: &mut Batches<'s, 'w, F>,
    ) -> Result<Claim<F>> {
        let g = || t().multiplier.as_ref().expect("gate trace");
        let q = self.down.reverse(
            q,
            || {
                (
                    t().multiplier
                        .as_ref()
                        .map_or(&t().activation.output, |(_, t)| &t.output),
                    &t().down,
                )
            },
            io,
        )?;
        let (q, gate) = if let Some((gate, product)) = &self.multiplier {
            let [p, q] =
                product.reverse(q, || (&t().activation.output, &g().0.output, &g().1), io)?;
            let source = gate.reverse(q, || (x(), &g().0), io)?;
            (p, Some(source))
        } else {
            (q, None)
        };
        let q = self
            .activation
            .reverse(q, || &t().activation, io, lookups)?;
        let source = self
            .activation_input
            .reverse(q, || (x(), &t().activation_input), io)?;
        if let Some(gate) = gate {
            io.merge(&[source, gate], || {
                self.input.padded(x(), F::zero(), F::from_i32)
            })
        } else {
            Ok(source)
        }
    }
}
