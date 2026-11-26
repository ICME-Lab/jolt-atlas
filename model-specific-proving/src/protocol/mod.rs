use std::ops::Range;

pub(crate) mod field;
pub(crate) mod lookup;
pub(crate) mod opening;
pub(crate) mod output;
pub(crate) mod range;
pub(crate) mod sumcheck;
pub(crate) mod tensor;

pub use opening::Proof;
pub use tensor::Shape;

pub type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub id: usize,
    pub logit: i32,
}

pub struct TopKConfig {
    pub positions: Range<usize>,
    pub k: usize,
    pub rows_per_block: usize,
    pub vocab_per_block: usize,
}
