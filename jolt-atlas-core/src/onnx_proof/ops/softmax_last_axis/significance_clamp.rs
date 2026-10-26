//! [`LookupOperandsTrait`] helper proving softmax's saturating clamp
//! `z_c = min(z, z_bound - 1)` via [`joltworks::lookup_tables::clamp::SoftmaxClampTable`].
//!
//! This witness (`z = max_k - x`) is evaluated at the point softmax's exp-digit lookups
//! (`SoftmaxZHi`/`SoftmaxZLo`) already converge to (`r2`), not this node's own output opening —
//! `r_cycle`/`r_cycle_source` are overridden accordingly. `z` is always `>= 0` by construction, so
//! no offset trick is needed: this maps directly onto `ClampBoundedTable`'s floor-at-0 domain.
//!
//! The read-raf sumcheck's `rv_claim` is defined as `z_hi(r2)*base + z_lo(r2)` (computed
//! directly from the already-cached `SoftmaxZHi`/`SoftmaxZLo` claims — see `cache_z_hi_lo` in
//! `mod.rs`), rather than needing its own separate `z_c` witness/claim. This proves
//! `z_hi*base + z_lo == ClampTable[z]` as one relation: combined with `z_hi`/`z_lo` each being
//! independently bounded by the (untouched) exp-digit Shout lookups in stage 3, this forces the
//! digit split to be the true clamp of the real `z` — no separate link check needed for the
//! digit-split identity itself. `z` still needs tying back to the real input, done by
//! `operand_link` in `mod.rs`.

pub use jolt_atlas_verifier::onnx_proof::ops::softmax_last_axis::significance_clamp::*;
