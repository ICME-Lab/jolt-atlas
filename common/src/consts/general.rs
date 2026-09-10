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
pub const MODEL_SCALE: usize = 12;
