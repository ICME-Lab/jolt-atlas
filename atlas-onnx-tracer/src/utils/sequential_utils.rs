//! Iterator fallback for targets without a worker pool.

pub trait IntoParallelIterator {
    type Item;
    type Iter: Iterator<Item = Self::Item>;

    fn into_par_iter(self) -> Self::Iter;
}

pub trait IntoParallelRefIterator<'data> {
    type Iter: Iterator<Item = Self::Item>;
    type Item: 'data;

    fn par_iter(&'data self) -> Self::Iter;
}

pub trait IntoParallelRefMutIterator<'data> {
    type Iter: Iterator<Item = Self::Item>;
    type Item: 'data;

    fn par_iter_mut(&'data mut self) -> Self::Iter;
}

impl<T> IntoParallelIterator for Vec<T> {
    type Item = T;
    type Iter = std::vec::IntoIter<T>;

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

pub trait ParallelSliceMut<T> {
    fn par_sort_unstable(&mut self)
    where
        T: Ord;
}

impl<T> ParallelSliceMut<T> for [T] {
    fn par_sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.sort_unstable()
    }
}

pub trait ParallelIterator: Iterator + Sized {
    fn with_min_len(self, _min: usize) -> Self {
        self
    }
}

impl<I: Iterator> ParallelIterator for I {}

pub trait IndexedParallelIterator: ParallelIterator {}

impl<I: Iterator> IndexedParallelIterator for I {}

// Iterator types used by the tensor facade.
pub mod vec {
    pub type IntoIter<T> = std::vec::IntoIter<T>;
}

pub mod slice {
    pub type IterMut<'a, T> = std::slice::IterMut<'a, T>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

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
        assert_eq!(
            values
                .into_par_iter()
                .with_min_len(1)
                .map(|v| *v)
                .sum::<i32>(),
            5
        );
    }

    #[test]
    fn mutable_iteration_and_sort_match_standard_iterators() {
        let mut values = vec![3, -5, 7, 0];
        values.par_iter_mut().for_each(|v| *v *= 2);
        values.as_mut_slice().par_iter_mut().for_each(|v| *v += 1);
        values.par_sort_unstable();
        assert_eq!(values, [-9, 1, 7, 15]);
    }
}
