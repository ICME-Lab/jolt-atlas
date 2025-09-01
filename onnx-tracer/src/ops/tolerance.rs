use crate::ops::utils;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[allow(missing_docs)]
/// An enum representing the tolerance we can accept for the accumulated arguments,
/// either absolute or percentage
#[derive(Clone, Default, Debug, PartialEq, PartialOrd, Serialize, Deserialize, Copy)]
pub struct Tolerance {
    pub val: f32,
    pub scale: utils::F32,
}

impl FromStr for Tolerance {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Ok(val) = s.parse::<f32>() {
            Ok(Tolerance {
                val,
                scale: utils::F32(1.0),
            })
        } else {
            Err(
                "Invalid tolerance value provided. It should expressed as a percentage (f32)."
                    .to_string(),
            )
        }
    }
}

impl From<f32> for Tolerance {
    fn from(value: f32) -> Self {
        Tolerance {
            val: value,
            scale: utils::F32(1.0),
        }
    }
}
