use super::general::MODEL_SCALE;

/// Clamps Erf/Sigmoid/Tanh input to `[-8, 8)` at model scale [`MODEL_SCALE`], before the small
/// activation-table lookup.
pub const ACTIVATION_BOUND: usize = MODEL_SCALE + 3;

/// One more than [`ACTIVATION_BOUND`]; the small activation table's log2 size.
pub const ACTIVATION_TABLE_VARS: usize = ACTIVATION_BOUND + 1;
