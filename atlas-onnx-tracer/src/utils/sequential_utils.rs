//! Iterator fallback for targets without a worker pool.

/// Sequential replacement for owned parallel iteration.
pub trait IntoParallelIterator {
    /// Item yielded by the iterator.
    type Item;
    /// Sequential iterator type.
    type Iter: Iterator<Item = Self::Item>;

    /// Iterate on the calling thread.
    fn into_par_iter(self) -> Self::Iter;
}

/// Sequential replacement for shared parallel iteration.
pub trait IntoParallelRefIterator<'data> {
    /// Sequential iterator type.
    type Iter: Iterator<Item = Self::Item>;
    /// Item yielded by the iterator.
    type Item: 'data;

    /// Iterate on the calling thread.
    fn par_iter(&'data self) -> Self::Iter;
}

/// Sequential replacement for mutable parallel iteration.
pub trait IntoParallelRefMutIterator<'data> {
    /// Sequential iterator type.
    type Iter: Iterator<Item = Self::Item>;
    /// Item yielded by the iterator.
    type Item: 'data;

    /// Iterate on the calling thread.
    fn par_iter_mut(&'data mut self) -> Self::Iter;
}

impl<T: IntoIterator> IntoParallelIterator for T {
    type Item = T::Item;
    type Iter = T::IntoIter;

    fn into_par_iter(self) -> Self::Iter {
        self.into_iter()
    }
}

impl<'data, T: 'data> IntoParallelRefIterator<'data> for Vec<T> {
    type Iter = std::slice::Iter<'data, T>;
    type Item = &'data T;

    fn par_iter(&'data self) -> Self::Iter {
        self.iter()
    }
}

impl<'data, T: 'data> IntoParallelRefMutIterator<'data> for Vec<T> {
    type Iter = std::slice::IterMut<'data, T>;
    type Item = &'data mut T;

    fn par_iter_mut(&'data mut self) -> Self::Iter {
        self.iter_mut()
    }
}

impl<'data, T: 'data> IntoParallelRefIterator<'data> for [T] {
    type Iter = std::slice::Iter<'data, T>;
    type Item = &'data T;

    fn par_iter(&'data self) -> Self::Iter {
        self.iter()
    }
}

impl<'data, T: 'data> IntoParallelRefMutIterator<'data> for [T] {
    type Iter = std::slice::IterMut<'data, T>;
    type Item = &'data mut T;

    fn par_iter_mut(&'data mut self) -> Self::Iter {
        self.iter_mut()
    }
}

/// Sequential sorting of a mutable slice.
pub trait ParallelSliceMut<T> {
    /// Divide a mutable slice into disjoint chunks on the calling thread.
    fn par_chunks_mut(&mut self, size: usize) -> std::slice::ChunksMut<'_, T>;

    /// Sort on the calling thread.
    fn par_sort_unstable(&mut self)
    where
        T: Ord;
}

impl<T> ParallelSliceMut<T> for [T] {
    fn par_chunks_mut(&mut self, size: usize) -> std::slice::ChunksMut<'_, T> {
        self.chunks_mut(size)
    }

    fn par_sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.sort_unstable()
    }
}

/// Iterator helper that accepts a parallel chunk hint.
pub trait ParallelIterator: Iterator + Sized {
    /// Ignore a parallel chunk hint.
    fn with_min_len(self, _min: usize) -> Self {
        self
    }
}

impl<I: Iterator> ParallelIterator for I {}

/// Marker for iterators used by the tensor facade.
pub trait IndexedParallelIterator: ParallelIterator {}

impl<I: Iterator> IndexedParallelIterator for I {}

// Iterator types used by the tensor facade.
/// Owned iterator aliases.
pub mod vec {
    pub type IntoIter<T> = std::vec::IntoIter<T>;
}

/// Borrowed iterator aliases.
pub mod slice {
    /// Mutable slice iterator.
    pub type IterMut<'a, T> = std::slice::IterMut<'a, T>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    fn sum_indexed<I: IndexedParallelIterator<Item = i32>>(values: I) -> i32 {
        values.sum()
    }

    #[test]
    fn borrowed_and_owned_iteration_stays_on_the_calling_thread() {
        // Rc cannot cross threads, so these calls also check that the fallback
        // does not impose Send/Sync requirements.
        let values = vec![Rc::new(3), Rc::new(-5), Rc::new(7)];
        assert_eq!(
            values.par_iter().map(|v| **v).collect::<Vec<_>>(),
            [3, -5, 7]
        );
        assert_eq!(values.as_slice().par_iter().map(|v| **v).sum::<i32>(), 5);
        let owned: vec::IntoIter<Rc<i32>> = values.into_par_iter();
        assert_eq!(sum_indexed(owned.with_min_len(1).map(|v| *v)), 5);
    }

    #[test]
    fn mutable_iteration_and_sort_match_standard_iterators() {
        let mut values = vec![3, -5, 7, 0];
        values.par_iter_mut().for_each(|v| *v *= 2);
        let borrowed: slice::IterMut<'_, i32> = values.as_mut_slice().par_iter_mut();
        borrowed.for_each(|v| *v += 1);
        values.par_sort_unstable();
        assert_eq!(values, [-9, 1, 7, 15]);
    }
}
