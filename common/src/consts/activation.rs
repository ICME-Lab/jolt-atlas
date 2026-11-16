use super::general::MODEL_SCALE;

/// Clamps Erf/Sigmoid/Tanh input to `[-8, 8)` at model scale [`MODEL_SCALE`], before the small
/// activation-table lookup.
pub const ACTIVATION_BOUND: usize = MODEL_SCALE + 3;

/// Clamps SiLU's input to `[-16, 16)` at model scale [`MODEL_SCALE`]: its correction to ReLU
/// decays more slowly than GELU's and needs a domain twice as wide.
pub const SILU_BOUND: usize = ACTIVATION_BOUND + 1;

/// One more than [`ACTIVATION_BOUND`]; the small activation table's log2 size.
pub const ACTIVATION_TABLE_VARS: usize = ACTIVATION_BOUND + 1;
