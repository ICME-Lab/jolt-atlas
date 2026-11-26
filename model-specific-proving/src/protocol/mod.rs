//! Common proof relations, transcript handling, and polynomial openings.

pub(crate) mod lookup;
pub(crate) mod pcs;
pub(crate) mod range;
pub(crate) mod session;
pub(crate) mod tensor;
pub(crate) mod top_k;

pub use pcs::{Layout, ModelPcs, Proof};
pub use session::ProofField;
pub use tensor::Shape;
pub use top_k::{Candidate, TopKConfig};

pub type Result<T> = std::result::Result<T, &'static str>;

#[cfg(feature = "bitz")]
pub use pcs::BitzScheme;
