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
        let lanes = 1 << self.slot.digit_vars();
        let Kind::Bits { planes } = self.slot.kind else {
            unreachable!("range slot is a bit slot");
        };
        let mut digits = vec![0; self.slot.digit_count(planes) * cells];
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
            let mut bit = 0;
            while bit < range.bits {
                let shift = plane % lanes;
                let count = (lanes - shift).min(range.bits - bit);
                let mask = (1 << count) - 1;
                let encode = |value: i128| (((value >> bit) as u8) & mask) << shift;
                let dst = &mut digits[plane / lanes * cells..][..cells];
                let pad = encode(range.offset);
                for (row, values) in dst
                    .chunks_exact_mut(stride)
                    .zip(values.chunks_exact(self.shape.width))
                {
                    let (used, padding) = row.split_at_mut(self.shape.width);
                    for (digit, &value) in used.iter_mut().zip(values) {
                        *digit |= encode(value.into() + range.offset);
                    }
                    for digit in padding {
                        *digit |= pad;
                    }
                }
                for digit in &mut dst[self.shape.rows * stride..] {
                    *digit |= pad;
                }
                bit += count;
                plane += count;
            }
        }
        advice.put(self.slot, Part::Digits(digits.into()))
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
