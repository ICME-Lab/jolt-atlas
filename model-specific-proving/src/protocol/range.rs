//! Integer range constraints written as evaluations of shared bit planes.

use super::{
    pcs::{Advice, Kind, Layout, Opening, Part, Slot},
    session::{Claim, ProofField, Session},
    tensor::Shape,
    Result,
};

#[derive(Clone, Copy)]
pub(crate) struct Range {
    bits: usize,
    offset: i128,
}

impl Range {
    pub(crate) const fn unsigned(bits: usize) -> Self {
        Self { bits, offset: 0 }
    }
    pub(crate) const fn signed(bits: usize) -> Self {
        Self {
            bits,
            offset: 1 << (bits - 1),
        }
    }
}

pub(crate) const I32: Range = Range::signed(32);

pub(crate) struct Ranges {
    pub(crate) slot: Slot,
    ranges: Box<[Range]>,
    shape: Shape,
}

impl Ranges {
    pub(crate) fn new(
        layout: &mut Layout,
        shape: Shape,
        ranges: impl AsRef<[Range]>,
    ) -> Result<Self> {
        let ranges = ranges.as_ref();
        let planes = ranges.iter().map(|r| r.bits).sum();
        for range in ranges {
            layout.integer_bits = layout.integer_bits.max(range.bits as u32 + 2);
        }
        Ok(Self {
            slot: layout.add(shape.vars(), Kind::Bits { planes })?,
            ranges: ranges.into(),
            shape,
        })
    }

    pub(crate) fn record<T: Copy + Into<i128>>(
        &self,
        advice: &mut Advice,
        values: &[&[T]],
    ) -> Result<()> {
        if values.len() != self.ranges.len() {
            return Err("invalid range witness count");
        }
        let cells = 1 << self.shape.vars();
        let stride = self.shape.width.next_power_of_two();
        let mut bits = vec![0; 1 << self.slot.vars()];
        let mut plane = 0;
        for (range, &values) in self.ranges.iter().zip(values) {
            if range.bits == 0 && values.is_empty() {
                continue;
            }
            self.shape.check(values)?;
            let codes = values.iter().map(|&v| v.into() + range.offset);
            if codes.clone().any(|v| v < 0 || v >= 1i128 << range.bits) {
                return Err("integer exceeds declared range");
            }
            for b in 0..range.bits {
                let dst = &mut bits[plane * cells..][..cells];
                dst.fill(((range.offset >> b) & 1) as u8);
                for (i, value) in codes.clone().enumerate() {
                    dst[i / self.shape.width * stride + i % self.shape.width] =
                        ((value >> b) & 1) as u8;
                }
                plane += 1;
            }
        }
        advice.put(self.slot, Part::Bits(bits))
    }

    pub(crate) fn check<F: ProofField, V: AsRef<[F]>>(
        &self,
        claims: &[(&[F], V)],
        io: &mut Session<'_, '_, F>,
    ) -> Result<()> {
        let n = self.ranges.len();
        if claims.is_empty()
            || claims
                .iter()
                .any(|(p, v)| p.len() != self.shape.vars() || v.as_ref().len() != n)
        {
            return Err("invalid range claims");
        }
        let mut mix = if n > 1 {
            io.point(n * claims.len())?
        } else {
            Vec::new()
        }
        .into_iter();
        let planes = 1 << (self.slot.vars() - self.slot.cell_vars);
        for (point, values) in claims {
            let mut value = F::zero();
            let mut coefficients = Vec::with_capacity(planes);
            for (range, &v) in self.ranges.iter().zip(values.as_ref()) {
                let scale = if n == 1 {
                    F::one()
                } else {
                    mix.next().expect("one coefficient per integer")
                };
                value += scale * (v + F::from_i128(range.offset));
                coefficients.extend((0..range.bits).map(|b| scale * F::from_u128(1 << b)));
            }
            coefficients.resize(planes, F::zero());
            io.openings.push(Opening {
                slot: self.slot.id,
                claim: Claim {
                    point: point.to_vec(),
                    value,
                },
                planes: Some(coefficients),
            });
        }
        Ok(())
    }
}
