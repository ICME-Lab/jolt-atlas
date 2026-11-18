//! A global allocator that asks Linux for transparent huge pages on large allocations.
//!
//! The prover streams over buffers of hundreds of megabytes, and with 4 KiB pages its passes
//! spend a large share of their time on TLB misses. A machine whose
//! `/sys/kernel/mm/transparent_hugepage/enabled` is `madvise` gives huge pages only to memory
//! marked `MADV_HUGEPAGE`; [`HugePages`] marks every allocation of at least [`THRESHOLD`]
//! bytes before it is first written. Elsewhere it is the system allocator.
//!
//! A binary opts in with
//! `#[global_allocator] static ALLOC: common::huge_pages::HugePages = common::huge_pages::HugePages;`.

use std::alloc::{GlobalAlloc, Layout, System};

/// The smallest allocation marked: one huge page.
pub const THRESHOLD: usize = 2 << 20;

/// The system allocator, with allocations of at least [`THRESHOLD`] bytes marked for huge
/// pages on Linux.
pub struct HugePages;

#[cfg(target_os = "linux")]
fn advise(ptr: *mut u8, size: usize) {
    if ptr.is_null() || size < THRESHOLD {
        return;
    }
    // The whole base pages inside the allocation.
    const PAGE: usize = 4096;
    let start = (ptr as usize).next_multiple_of(PAGE);
    let end = (ptr as usize + size) & !(PAGE - 1);
    if end > start {
        // SAFETY: `[start, end)` lies inside the allocation just made; `madvise` with
        // `MADV_HUGEPAGE` changes how the range is backed, not its contents. A failure (a
        // kernel without transparent huge pages) leaves it as it was.
        unsafe {
            libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn advise(_ptr: *mut u8, _size: usize) {}

// SAFETY: every call is forwarded to `System` with the same arguments; `advise` only marks
// memory the returned allocation owns.
unsafe impl GlobalAlloc for HugePages {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded under the caller's contract.
        let ptr = unsafe { System.alloc(layout) };
        advise(ptr, layout.size());
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded under the caller's contract.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        advise(ptr, layout.size());
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded under the caller's contract.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded under the caller's contract.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if new_size > layout.size() {
            advise(new, new_size);
        }
        new
    }
}
