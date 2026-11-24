//! Transformer layers and their execution and proof routines.

pub(crate) mod attention;
pub(crate) mod linear;
pub(crate) mod mlp;
pub(crate) mod nonlinear;
pub(crate) mod softmax;

pub use attention::Attention;
pub use linear::{Projection, Rotary};
pub use mlp::Mlp;
pub use nonlinear::{Activation, NormalizationKind, NormalizationParameters};
pub use softmax::{SoftmaxMask, SoftmaxTables};
