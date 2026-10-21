//! Transitional re-export shim over [`atlas_trace`] and this crate's own
//! modules, keeping the old `compat::*` import paths alive while call sites
//! migrate. Every item here is a re-export — the facade types themselves are
//! gone. This module is slated for deletion once imports are repointed.

/// Node view re-export.
pub mod node {
    pub use crate::node_view::ComputationNode;
}

/// Operator re-exports.
pub mod ops {
    pub use atlas_trace::op::*;
    pub use atlas_trace::trace::FusedIntermediates;

    /// The operator enum (the atlas-trace one, under its old name).
    pub type Operator = atlas_trace::Op;
}

/// Tensor re-exports.
pub mod tensor {
    pub use atlas_trace::tensor::Element;
    pub use atlas_trace::Tensor;
}

/// Model, loader, builder, and trace re-exports.
pub mod model {
    pub use crate::loader::RunArgs;
    pub use atlas_trace::Model;

    /// Test-builder re-export.
    pub mod test {
        pub use crate::test_builder::{ModelBuilder, Wire};
    }

    /// Trace re-exports.
    pub mod trace {
        pub use atlas_trace::trace::{LayerData, ModelExecutionIO};
        pub use atlas_trace::Trace;
    }
}
