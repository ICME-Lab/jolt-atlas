use crate::{
    advice::{Advice, Requests, Slot},
    channel::{Channel, Result},
    field::F,
    norm::{Claim, Width},
    tensor::Shape,
};
use jolt_field::Ring;
use rayon::prelude::*;

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Rounded {
    pub shape: Shape,
    pub values: Option<Vec<i32>>,
    value_slot: Option<Slot>,
    remainder_slot: Slot,
    bits: u32,
}

impl Rounded {
    pub fn evaluate(
        advice: &mut Advice,
        shape: Shape,
        bits: u32,
        check_output: bool,
        witness: Option<impl Fn(usize, usize) -> i128 + Sync>,
    ) -> Result<Self> {
        let (values, remainder) = witness
            .map(|f| Self::witness(shape, bits, f))
            .transpose()?
            .unzip();
        Self::record(advice, shape, bits, check_output, values, remainder)
    }

    pub fn witness(
        shape: Shape,
        bits: u32,
        f: impl Fn(usize, usize) -> i128 + Sync,
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        let scale = 1 << bits;
        (0..shape.rows * shape.cols)
            .into_par_iter()
            .with_min_len(512)
            .map(|i| {
                let value = f(i / shape.cols, i % shape.cols);
                let out = i32::try_from(value >> bits).map_err(|_| "fixed-point overflow")?;
                Ok((out, value.rem_euclid(scale) as u32))
            })
            .collect::<Result<Vec<_>>>()
            .map(|values| values.into_iter().unzip())
    }

    pub fn record(
        advice: &mut Advice,
        shape: Shape,
        bits: u32,
        check_output: bool,
        values: Option<Vec<i32>>,
        remainder: Option<Vec<u32>>,
    ) -> Result<Self> {
        let value_slot = check_output
            .then(|| {
                advice.column(shape, Width::Signed(32), |i| {
                    i128::from(values.as_ref().unwrap()[i])
                })
            })
            .transpose()?;
        let remainder_slot = advice.column(shape, Width::Unsigned(bits as u8), |i| {
            i128::from(remainder.as_ref().unwrap()[i])
        })?;
        Ok(Self {
            shape,
            values,
            value_slot,
            remainder_slot,
            bits,
        })
    }

    pub fn remainder(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        claim: &Claim<F>,
    ) -> Result<F> {
        let advice = io.advice();
        let [r] = io.values(|| {
            let advice = advice.unwrap();
            let stride = self.shape.stride();
            [self.shape.eval_with(&claim.point, |r, c| {
                advice.value(self.remainder_slot, r * stride + c)
            })]
        })?;
        if let Some(slot) = self.value_slot {
            requests.push(slot, &claim.point, claim.value);
        }
        requests.push(self.remainder_slot, &claim.point, r);
        Ok(r)
    }

    pub fn raw(
        &self,
        io: &mut Channel<'_, '_>,
        requests: &mut Requests,
        claim: &Claim<F>,
    ) -> Result<F> {
        Ok(F::from_u64(1 << self.bits) * claim.value + self.remainder(io, requests, claim)?)
    }
}
