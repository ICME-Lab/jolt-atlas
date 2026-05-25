pub mod claim;
pub mod error;
pub mod ops;
pub mod proof;

pub use claim::{Claim, Poly, Shape, TensorId};
pub use error::{ProverError, Result};
pub use proof::ProveResult;
