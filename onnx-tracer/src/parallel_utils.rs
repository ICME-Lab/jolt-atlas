//! Cross-platform parallel processing utilities for WASM compatibility

// For native builds, re-export everything from maybe_rayon
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use maybe_rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use maybe_rayon::slice::ParallelSliceMut;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use maybe_rayon::{slice, vec};

// For WASM builds, provide fallback implementations
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub trait IntoParallelIterator {
    type Item;
    type Iter: Iterator<Item = Self::Item>;

    fn into_par_iter(self) -> Self::Iter;
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub trait IntoParallelRefIterator<'data> {
    type Iter: Iterator<Item = Self::Item>;
    type Item: 'data;

    fn par_iter(&'data self) -> Self::Iter;
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub trait IntoParallelRefMutIterator<'data> {
    type Iter: Iterator<Item = Self::Item>;
    type Item: 'data;

    fn par_iter_mut(&'data mut self) -> Self::Iter;
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<T> IntoParallelIterator for Vec<T> {
    type Item = T;
    type Iter = std::vec::IntoIter<T>;

    fn into_par_iter(self) -> Self::Iter {
        self.into_iter()
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<'data, T: 'data> IntoParallelRefIterator<'data> for Vec<T> {
    type Iter = std::slice::Iter<'data, T>;
    type Item = &'data T;

    fn par_iter(&'data self) -> Self::Iter {
        self.iter()
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<'data, T: 'data> IntoParallelRefMutIterator<'data> for Vec<T> {
    type Iter = std::slice::IterMut<'data, T>;
    type Item = &'data mut T;

    fn par_iter_mut(&'data mut self) -> Self::Iter {
        self.iter_mut()
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<'data, T: 'data> IntoParallelRefIterator<'data> for [T] {
    type Iter = std::slice::Iter<'data, T>;
    type Item = &'data T;

    fn par_iter(&'data self) -> Self::Iter {
        self.iter()
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<'data, T: 'data> IntoParallelRefMutIterator<'data> for [T] {
    type Iter = std::slice::IterMut<'data, T>;
    type Item = &'data mut T;

    fn par_iter_mut(&'data mut self) -> Self::Iter {
        self.iter_mut()
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub trait ParallelSliceMut<T> {
    fn par_sort_unstable(&mut self)
    where
        T: Ord;
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<T> ParallelSliceMut<T> for [T] {
    fn par_sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.sort_unstable()
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub trait ParallelIterator: Iterator {
    // Removed custom collect implementation to avoid conflicts
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<I: Iterator> ParallelIterator for I {}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub trait IndexedParallelIterator: ParallelIterator {}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl<I: Iterator> IndexedParallelIterator for I {}

// Module re-exports for WASM compatibility
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod vec {
    pub type IntoIter<T> = std::vec::IntoIter<T>;
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod slice {
    pub type IterMut<'a, T> = std::slice::IterMut<'a, T>;
}
