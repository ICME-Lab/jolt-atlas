pub mod builder;
pub mod compat;
pub mod constraints;
pub mod inputs;
pub mod key;

// Re-export compatibility types for easy use
pub use compat::{Constraint, Variable, Term, LC};
