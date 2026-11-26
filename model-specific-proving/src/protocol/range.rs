use jolt_field::{One, Ring, Zero};

use super::{
    field::F,
    opening::{Layout, Packed, Request, Slot},
    sumcheck::Session,
    tensor::{Claim, Shape},
    Result,
};

#[derive(Clone, Copy)]
pub(crate) struct Width {
    bits: usize,
    offset: i128,
}

impl Width {
    pub(crate) const fn signed(bits: usize) -> Self {
        Self {
            bits,
            offset: 1 << (bits - 1),
        }
    }

    pub(crate) const fn unsigned(bits: usize) -> Self {
        Self { bits, offset: 0 }
    }
}

pub(crate) const I32: Width = Width::signed(32);

pub(crate) struct Ranges<const N: usize> {
    pub shape: Shape,
    slot: Slot,
    widths: [Width; N],
}

impl<const N: usize> Ranges<N> {
    pub(crate) fn new(layout: &mut Layout, shape: Shape, widths: [Width; N]) -> Self {
        let slot = layout.reserve(shape.bits(), widths.iter().map(|width| width.bits).sum());
        Self {
            shape,
            slot,
            widths,
        }
    }

    pub(crate) fn record<T: Copy + Into<i128>>(
        &self,
        packed: &mut Packed<'_>,
        columns: [&[T]; N],
    ) -> Result<()> {
        let shape = self.shape;
        let stride = shape.width.next_power_of_two();
        let mut plane = 0;
        for (width, values) in self.widths.into_iter().zip(columns) {
            if width.bits == 0 && values.is_empty() {
                continue;
            }
            if values.len() != shape.len() {
                return Err("invalid range witness dimensions");
            }
            packed.write(self.slot, plane, width.bits, |i| {
                let (row, col) = (i / stride, i % stride);
                let value = if row < shape.rows && col < shape.width {
                    values[row * shape.width + col].into()
                } else {
                    0
                };
                (value + width.offset) as u128
            })?;
            plane += width.bits;
        }
        Ok(())
    }

    pub(crate) fn high_bits_zero(&self, requests: &mut Vec<Request>, point: Vec<F>, low: usize) {
        let width = self.widths[N - 1].bits;
        if width <= low {
            return;
        }
        let prefix = self.widths[..N - 1].iter().map(|w| w.bits).sum::<usize>();
        let mut planes = vec![F::zero(); prefix + width];
        for (i, coefficient) in planes[prefix + low..].iter_mut().enumerate() {
            *coefficient = F::from_u128(1 << i);
        }
        requests.push(Request {
            slot: self.slot,
            claim: Claim::new(point, F::zero()),
            planes,
        });
    }

    pub(crate) fn open(
        &self,
        io: &mut Session<'_, '_>,
        requests: &mut Vec<Request>,
        point: &[F],
        values: [F; N],
    ) -> Result<()> {
        let mix = if N == 1 {
            vec![F::one()]
        } else {
            io.random(N, N)?
        };
        let mut value = F::zero();
        let mut planes = Vec::new();
        for ((width, input), weight) in self.widths.into_iter().zip(values).zip(mix) {
            value += weight * (input + F::from_i128(width.offset));
            planes.extend((0..width.bits).map(|i| weight * F::from_u128(1 << i)));
        }
        requests.push(Request {
            slot: self.slot,
            claim: Claim::new(point.to_vec(), value),
            planes,
        });
        Ok(())
    }
}

impl Ranges<1> {
    pub(crate) fn adaptive(layout: &mut Layout, shape: Shape, capacity: usize) -> Self {
        let range = Self::new(layout, shape, [Width::unsigned(capacity)]);
        layout.variable(range.slot);
        range
    }

    pub(crate) fn record_adaptive<T: Copy + Into<i128>>(
        &self,
        packed: &mut Packed<'_>,
        values: &[T],
    ) -> Result<()> {
        let shape = self.shape;
        if !values.is_empty() && values.len() != shape.len() {
            return Err("invalid range witness dimensions");
        }
        let value = values.iter().try_fold(0, |maximum, &value| {
            u128::try_from(value.into())
                .map(|value| maximum.max(value))
                .map_err(|_| "negative unsigned witness")
        })?;
        let bits = (128 - value.leading_zeros()) as usize;
        packed.shrink(self.slot, bits)?;
        let stride = shape.width.next_power_of_two();
        packed.write(self.slot, 0, bits, |i| {
            let (row, col) = (i / stride, i % stride);
            if row < shape.rows && col < shape.width {
                values[row * shape.width + col].into() as u128
            } else {
                0
            }
        })
    }
}
