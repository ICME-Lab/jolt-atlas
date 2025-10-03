use crate::RunArgs;
use serde::{Deserialize, Serialize};

/// Represents the scale of the model input, model parameters.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, PartialOrd)]
pub struct VarScales {
    ///
    pub input: crate::Scale,
    ///
    pub params: crate::Scale,
    ///
    pub rebase_multiplier: u32,
}

impl std::fmt::Display for VarScales {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "(inputs: {}, params: {})", self.input, self.params)
    }
}

impl VarScales {
    ///
    pub fn get_max(&self) -> crate::Scale {
        std::cmp::max(self.input, self.params)
    }

    /// Place in [VarScales] struct.
    pub fn from_args(args: &RunArgs) -> Self {
        Self {
            input: args.input_scale,
            params: args.param_scale,
            rebase_multiplier: args.scale_rebase_multiplier,
        }
    }
}
