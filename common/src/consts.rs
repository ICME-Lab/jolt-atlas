pub const XLEN: usize = 32;
pub const LOG_K_CHUNK: usize = 4;
pub const K_CHUNK: usize = 1 << LOG_K_CHUNK;

/// Logarithm of K = XLEN * 2, the total number of address bits for prefix-suffix
/// lookup tables used in read-raf checking.
pub const LOG_K: usize = XLEN * 2;

/// Global model scale (log2) these activation constants are tuned for (matches
/// `atlas-onnx-tracer`'s `DEFAULT_SCALE`).
pub const SCALE_8: usize = 8;

/// Clamps Erf/Sigmoid/Tanh input to `[-8, 8)` at model scale [`SCALE_8`], before the small
/// activation-table lookup.
pub const ACTIVATION_BOUND: usize = SCALE_8 + 3;

/// One more than [`ACTIVATION_BOUND`]; the small activation table's log2 size.
pub const ACTIVATION_TABLE_BOUND: usize = ACTIVATION_BOUND + 1;
