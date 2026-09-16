//! Optional fixed tables for the standard scale. The bytes are program constants,
//! not advice supplied by a prover. Other scales use the reference generator.
use common::consts::{
    ACTIVATION_TABLE_VARS, MODEL_SCALE, TRIG_DOWNSCALE_BITS, TRIG_PERIOD_MODULUS,
};

fn decode(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

pub(crate) fn sin() -> Option<Vec<i32>> {
    (MODEL_SCALE == 14 && TRIG_DOWNSCALE_BITS == 6 && TRIG_PERIOD_MODULUS == 2470649)
        .then(|| decode(include_bytes!("fixed_tables/sin14.bin")))
}

pub(crate) fn cos() -> Option<Vec<i32>> {
    (MODEL_SCALE == 14 && TRIG_DOWNSCALE_BITS == 6 && TRIG_PERIOD_MODULUS == 2470649)
        .then(|| decode(include_bytes!("fixed_tables/cos14.bin")))
}

pub(crate) fn sigmoid() -> Option<Vec<i32>> {
    (MODEL_SCALE == 14 && ACTIVATION_TABLE_VARS == 18)
        .then(|| decode(include_bytes!("fixed_tables/sigmoid14.bin")))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_tables_match_every_reference_entry() {
        assert_eq!(
            sin().unwrap(),
            super::super::sin::SinTable::materialize_reference()
        );
        assert_eq!(
            cos().unwrap(),
            super::super::cos::CosTable::materialize_reference()
        );
        assert_eq!(
            sigmoid().unwrap(),
            crate::onnx_proof::ops::sigmoid::SigmoidTableMarker::materialize_reference()
        );
    }
}
