pub const XLEN: usize = 32;
pub const LOG_K_CHUNK: usize = 4;
pub const K_CHUNK: usize = 1 << LOG_K_CHUNK;

/// Logarithm of K = XLEN * 2, the total number of address bits for prefix-suffix
/// lookup tables used in read-raf checking.
pub const LOG_K: usize = XLEN * 2;

/// Global model scale (log2) the fixed-shape lookup tables below are tuned for (matches
/// `atlas-onnx-tracer`'s `DEFAULT_SCALE`). Several lookup-table shapes in this codebase
/// (activation clamping, softmax's saturating clamp) are const-generic on a bound derived from
/// this value rather than the model's runtime scale, so changing it requires recompiling.
pub const MODEL_SCALE: usize = 8;

/// Clamps Erf/Sigmoid/Tanh input to `[-8, 8)` at model scale [`MODEL_SCALE`], before the small
/// activation-table lookup.
pub const ACTIVATION_BOUND: usize = MODEL_SCALE + 3;

/// One more than [`ACTIVATION_BOUND`]; the small activation table's log2 size.
pub const ACTIVATION_TABLE_VARS: usize = ACTIVATION_BOUND + 1;

/// log2 of softmax's saturating-clamp table size at [`MODEL_SCALE`]: the padded
/// `hi_size * base` from `generate_exp_lut_decomposed` (`atlas-onnx-tracer::ops::softmax`) at
/// that scale, rounded up to the next power of two. `ln`/`exp` aren't const-evaluable in stable
/// Rust, so these are precomputed per supported `MODEL_SCALE` rather than derived here; add a new
/// arm (re-deriving the bound offline) before switching to an untested scale.
pub const SOFTMAX_CLAMP_BOUND: usize = match MODEL_SCALE {
    8 => 11,
    12 => 16,
    _ => panic!("no precomputed SOFTMAX_CLAMP_BOUND for this MODEL_SCALE"),
};
