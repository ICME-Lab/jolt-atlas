use thiserror::Error;

pub type Result<T> = std::result::Result<T, ProverError>;

#[derive(Debug, Error)]
pub enum ProverError {
    #[error("shape mismatch for {name}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error(
        "tensor length mismatch for {name}: shape {shape:?} needs {expected} values, got {actual}"
    )]
    TensorLenMismatch {
        name: &'static str,
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },

    #[error(
        "matmul witness is inconsistent at row {row}, col {col}: expected {expected}, got {actual}"
    )]
    MatMulMismatch {
        row: usize,
        col: usize,
        expected: i32,
        actual: i32,
    },

    #[error(
        "matmul accumulator is inconsistent at row {row}, col {col}: expected {expected}, got {actual}"
    )]
    MatMulAccumulatorMismatch {
        row: usize,
        col: usize,
        expected: i64,
        actual: i64,
    },

    #[error("round bit witness is not boolean at bit {bit}, index {index}: got {value}")]
    BitNotBoolean { bit: usize, index: usize, value: u8 },

    #[error("sumcheck domain length must be a non-zero power of two, got {0}")]
    InvalidSumcheckDomain(usize),

    #[error("matrix dimension {name} must be non-zero")]
    InvalidMatrixDimension { name: &'static str },

    #[error("tensor shape must not contain zero dimensions: {0:?}")]
    InvalidTensorShape(Vec<usize>),

    #[error("invalid claim count for {name}: expected {expected}, got {actual}")]
    InvalidClaimCount {
        name: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("multiplication witness is inconsistent with the output claim")]
    MulMismatch,

    #[error("axis {axis} is out of bounds for tensor rank {rank}")]
    InvalidAxis { axis: usize, rank: usize },

    #[error("invalid GQA layout: q_heads={q_heads}, kv_heads={kv_heads}")]
    InvalidGqa { q_heads: usize, kv_heads: usize },

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("RoPE witness is inconsistent with the output claim")]
    RopeMismatch,

    #[error("internal sumcheck opening was not produced")]
    MissingOpening,

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] ark_serialize::SerializationError),

    #[error("missing committed polynomial(s) for opening claims: {0:?}")]
    MissingCommittedPolynomials(Vec<String>),

    #[error(
        "committed polynomial domain mismatch for {tensor}: claim domain {domain_shape:?} needs {expected} values, committed polynomial has {actual}"
    )]
    CommittedPolynomialDomainMismatch {
        tensor: String,
        domain_shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },
}
