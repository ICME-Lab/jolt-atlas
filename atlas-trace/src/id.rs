//! Small newtypes shared across the crate: node identity, fixed-point scale,
//! and the semantics version.

use serde::{Deserialize, Serialize};

use crate::error::GraphError;

/// Index of a node in a [`crate::graph::Graph`].
///
/// Nodes are stored densely in a `Vec`; a `NodeId` is the index into that
/// `Vec`. Graphs are topologically ordered by construction: a node may only
/// reference inputs with a strictly smaller `NodeId`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct NodeId(pub u32);

impl NodeId {
    /// The node's position in `Graph::nodes`.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// log2 of the global fixed-point denominator: a quantized value `q`
/// represents the real number `q / 2^scale`.
///
/// Invariant (checked at construction): `1 ..= 15`, so that `S^2` with
/// `S = 2^scale` fits an `i32` — the softmax witness relies on this.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct Scale(u8);

impl Scale {
    /// Maximum supported scale (`S^2 = 2^30` must fit in `i32`).
    pub const MAX: u8 = 15;

    /// Construct a scale, rejecting values outside `1..=15`.
    pub fn new(s: u8) -> Result<Self, GraphError> {
        if (1..=Self::MAX).contains(&s) {
            Ok(Scale(s))
        } else {
            Err(GraphError::InvalidScale { scale: s })
        }
    }

    /// The denominator `D = 2^scale` as `i64`.
    pub fn denom_i64(self) -> i64 {
        1i64 << self.0
    }

    /// The denominator `S = 2^scale` as `i32`.
    pub fn denom_i32(self) -> i32 {
        1i32 << self.0
    }

    /// The raw log2 value.
    pub fn log2(self) -> u32 {
        self.0 as u32
    }
}

impl std::fmt::Display for Scale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "2^{}", self.0)
    }
}

/// Version of the integer activation semantics.
///
/// This is part of the proof specification: the prover materializes its
/// lookup tables by calling [`crate::semantics::tables::table`], and tracer
/// and prover must agree on the version. Any change to any output of the
/// `semantics` module for any input requires a new variant. Golden-hash
/// tests over the generated tables enforce this mechanically.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SemVersion {
    /// Initial deterministic integer semantics.
    V1,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_bounds() {
        assert!(Scale::new(0).is_err());
        assert!(Scale::new(16).is_err());
        let s = Scale::new(14).unwrap();
        assert_eq!(s.denom_i64(), 1 << 14);
        assert_eq!(s.denom_i32(), 1 << 14);
        assert_eq!(s.log2(), 14);
    }
}
