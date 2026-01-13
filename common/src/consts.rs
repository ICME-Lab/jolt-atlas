pub const XLEN: usize = 32;

/// Threshold for trace length (log scale) at which we switch between different
/// one-hot chunking parameters. Below this threshold (i.e., for smaller traces),
/// we use smaller chunk sizes for better performance (reduced commitment & PCS opening costs).
/// This value was empirically determined.
pub const ONEHOT_CHUNK_THRESHOLD_LOG_T: usize = 25;

/// Threshold for trace length (log scale) at which we switch the number of
/// instruction sumcheck phases from 16 to 8. Below this threshold, we use
/// more phases (16) for smaller sumcheck instances in each phase (8 instead of 16 variables).
/// This value was empirically determined.
pub const INSTRUCTION_PHASES_THRESHOLD_LOG_T: usize = 24;
