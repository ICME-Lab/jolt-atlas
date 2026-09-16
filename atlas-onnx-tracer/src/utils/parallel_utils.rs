//! Tensor iterators with sequential fallbacks for WebAssembly and RISC-V guests.

// Host tensor operations keep their existing parallel implementation.
#[cfg(not(any(
    all(target_arch = "wasm32", target_os = "unknown"),
    target_arch = "riscv64"
)))]
pub use maybe_rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
#[cfg(not(any(
    all(target_arch = "wasm32", target_os = "unknown"),
    target_arch = "riscv64"
)))]
pub use maybe_rayon::slice::ParallelSliceMut;
#[cfg(not(any(
    all(target_arch = "wasm32", target_os = "unknown"),
    target_arch = "riscv64"
)))]
pub use maybe_rayon::{slice, vec};

// A Jolt RISC-V guest cannot start Rayon workers. Keep its tensor helpers
// sequential even when dependency feature unification enables Rayon threads.
#[cfg(any(
    any(
        all(target_arch = "wasm32", target_os = "unknown"),
        target_arch = "riscv64"
    ),
    test
))]
#[path = "sequential_utils.rs"]
mod sequential;

#[cfg(any(
    all(target_arch = "wasm32", target_os = "unknown"),
    target_arch = "riscv64"
))]
pub use sequential::*;
