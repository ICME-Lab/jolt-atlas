//! Parallel/serial iterator facade.
//!
//! joltworks parallelizes its hot loops with rayon. The Jolt zkVM guest, where
//! the recursion verifier runs, is single-threaded and cannot start Rayon workers.
//! This module provides one import surface that is rayon when the
//! `parallel` feature is on (host proving) and a serial drop-in when it is off
//! (the guest / any no-thread target).
//!
//! Call sites use `use crate::par::prelude::*;` instead of `use rayon::prelude::*;`
//! and `crate::par::{join, scope, ...}` instead of `rayon::{...}`. The serial
//! facade mirrors only the rayon surface joltworks actually uses; it implements
//! the parallel-iterator combinators as *inherent* methods on a `Par<I>` wrapper
//! so that `reduce(|| id, op)` (rayon's two-closure form) does not collide with
//! `Iterator::reduce`. Reductions become left folds: identical results for the
//! associative field/group operations joltworks reduces over, and order is
//! preserved for the (vector-concatenating) non-commutative reductions.

#[cfg(feature = "parallel")]
mod imp {
    pub use rayon::iter::repeat_n;
    pub use rayon::prelude::*;
    pub use rayon::{
        current_num_threads, join, scope, slice, spawn, Scope, ThreadPool, ThreadPoolBuilder,
    };
}

#[cfg(not(feature = "parallel"))]
mod imp {
    //! Single-threaded facade. Everything runs on the calling thread.

    /// Wrapper that carries a serial iterator while exposing the rayon
    /// parallel-iterator method surface as inherent methods.
    pub struct Par<I>(pub I);

    /// `rayon::iter::repeat_n`: a producer of `n` clones of `elem`.
    #[inline(always)]
    pub fn repeat_n<T: Clone>(elem: T, n: usize) -> Par<core::iter::RepeatN<T>> {
        Par(core::iter::repeat_n(elem, n))
    }

    /// `rayon::join`: run both closures, serially, left then right.
    #[inline(always)]
    pub fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA,
        B: FnOnce() -> RB,
    {
        (oper_a(), oper_b())
    }

    /// `rayon::current_num_threads`: a single-threaded facade has one.
    #[inline(always)]
    pub fn current_num_threads() -> usize {
        1
    }

    /// Single-threaded `rayon::ThreadPoolBuilder` facade: all configuration is
    /// ignored; the resulting "pool" runs work inline on the calling thread.
    #[derive(Default)]
    pub struct ThreadPoolBuilder;

    impl ThreadPoolBuilder {
        #[inline(always)]
        pub fn new() -> Self {
            ThreadPoolBuilder
        }
        #[inline(always)]
        pub fn num_threads(self, _num_threads: usize) -> Self {
            self
        }
        #[inline(always)]
        pub fn stack_size(self, _stack_size: usize) -> Self {
            self
        }
        #[inline(always)]
        pub fn build(self) -> Result<ThreadPool, core::convert::Infallible> {
            Ok(ThreadPool)
        }
    }

    /// Single-threaded `rayon::ThreadPool` facade.
    pub struct ThreadPool;

    impl ThreadPool {
        #[inline(always)]
        pub fn install<OP, R>(&self, op: OP) -> R
        where
            OP: FnOnce() -> R,
        {
            op()
        }
    }

    /// `rayon::spawn`: no thread pool; run the body inline.
    #[inline(always)]
    pub fn spawn<F: FnOnce() + Send + 'static>(func: F) {
        func()
    }

    /// Minimal `rayon::Scope` facade: `spawn` runs the body inline.
    pub struct Scope<'scope> {
        _marker: core::marker::PhantomData<&'scope ()>,
    }

    impl<'scope> Scope<'scope> {
        #[inline(always)]
        pub fn spawn<F>(&self, body: F)
        where
            F: FnOnce(&Scope<'scope>) + Send + 'scope,
        {
            body(self)
        }
    }

    /// `rayon::scope`: run the closure with an inline scope.
    #[inline(always)]
    pub fn scope<'scope, OP, R>(op: OP) -> R
    where
        OP: FnOnce(&Scope<'scope>) -> R,
    {
        let s = Scope {
            _marker: core::marker::PhantomData,
        };
        op(&s)
    }

    // ----- producers: par_iter / into_par_iter / par_iter_mut -----

    /// `into_par_iter`.
    pub trait IntoParallelIterator {
        type Item;
        type Iter;
        fn into_par_iter(self) -> Self::Iter;
    }

    impl<T: IntoIterator> IntoParallelIterator for T {
        type Item = T::Item;
        type Iter = Par<T::IntoIter>;
        #[inline(always)]
        fn into_par_iter(self) -> Self::Iter {
            Par(self.into_iter())
        }
    }

    // An already-parallel iterator is its own producer (identity). Lets `zip`,
    // `chain`, and tuple multizip accept `Par<_>` arguments directly.
    impl<I: Iterator> IntoParallelIterator for Par<I> {
        type Item = I::Item;
        type Iter = Par<I>;
        #[inline(always)]
        fn into_par_iter(self) -> Self::Iter {
            self
        }
    }

    /// Slice par-iterator types, mirroring `rayon::slice` for code that names
    /// the iterator type explicitly (e.g. a custom `IntoParallelIterator` impl).
    pub mod slice {
        pub type Iter<'a, T> = super::Par<core::slice::Iter<'a, T>>;
        pub type IterMut<'a, T> = super::Par<core::slice::IterMut<'a, T>>;
    }

    /// `par_iter` over a shared reference.
    pub trait IntoParallelRefIterator<'data> {
        type Iter;
        fn par_iter(&'data self) -> Self::Iter;
    }

    impl<'data, T: 'data> IntoParallelRefIterator<'data> for Vec<T> {
        type Iter = Par<core::slice::Iter<'data, T>>;
        #[inline(always)]
        fn par_iter(&'data self) -> Self::Iter {
            Par(self.iter())
        }
    }

    impl<'data, T: 'data> IntoParallelRefIterator<'data> for [T] {
        type Iter = Par<core::slice::Iter<'data, T>>;
        #[inline(always)]
        fn par_iter(&'data self) -> Self::Iter {
            Par(self.iter())
        }
    }

    impl<'data, T: 'data, const N: usize> IntoParallelRefIterator<'data> for [T; N] {
        type Iter = Par<core::slice::Iter<'data, T>>;
        #[inline(always)]
        fn par_iter(&'data self) -> Self::Iter {
            Par(self.iter())
        }
    }

    /// `par_iter_mut` over an exclusive reference.
    pub trait IntoParallelRefMutIterator<'data> {
        type Iter;
        fn par_iter_mut(&'data mut self) -> Self::Iter;
    }

    impl<'data, T: 'data> IntoParallelRefMutIterator<'data> for Vec<T> {
        type Iter = Par<core::slice::IterMut<'data, T>>;
        #[inline(always)]
        fn par_iter_mut(&'data mut self) -> Self::Iter {
            Par(self.iter_mut())
        }
    }

    impl<'data, T: 'data> IntoParallelRefMutIterator<'data> for [T] {
        type Iter = Par<core::slice::IterMut<'data, T>>;
        #[inline(always)]
        fn par_iter_mut(&'data mut self) -> Self::Iter {
            Par(self.iter_mut())
        }
    }

    impl<'data, K: Ord + 'data, V: 'data> IntoParallelRefMutIterator<'data>
        for std::collections::BTreeMap<K, V>
    {
        type Iter = Par<std::collections::btree_map::IterMut<'data, K, V>>;
        #[inline(always)]
        fn par_iter_mut(&'data mut self) -> Self::Iter {
            Par(self.iter_mut())
        }
    }

    impl<'data, K: Ord + 'data, V: 'data> IntoParallelRefIterator<'data>
        for std::collections::BTreeMap<K, V>
    {
        type Iter = Par<std::collections::btree_map::Iter<'data, K, V>>;
        #[inline(always)]
        fn par_iter(&'data self) -> Self::Iter {
            Par(self.iter())
        }
    }

    // ----- slice chunking -----

    /// `par_chunks` / `par_chunks_exact`.
    pub trait ParallelSlice<T> {
        fn par_chunks(&self, n: usize) -> Par<core::slice::Chunks<'_, T>>;
        fn par_chunks_exact(&self, n: usize) -> Par<core::slice::ChunksExact<'_, T>>;
    }

    impl<T> ParallelSlice<T> for [T] {
        #[inline(always)]
        fn par_chunks(&self, n: usize) -> Par<core::slice::Chunks<'_, T>> {
            Par(self.chunks(n))
        }
        #[inline(always)]
        fn par_chunks_exact(&self, n: usize) -> Par<core::slice::ChunksExact<'_, T>> {
            Par(self.chunks_exact(n))
        }
    }

    /// `par_chunks_mut`.
    pub trait ParallelSliceMut<T> {
        fn par_chunks_mut(&mut self, n: usize) -> Par<core::slice::ChunksMut<'_, T>>;
        fn par_sort_unstable(&mut self)
        where
            T: Ord;
        fn par_sort_by<F>(&mut self, cmp: F)
        where
            F: FnMut(&T, &T) -> core::cmp::Ordering;
    }

    impl<T: Send> ParallelSliceMut<T> for [T] {
        #[inline(always)]
        fn par_chunks_mut(&mut self, n: usize) -> Par<core::slice::ChunksMut<'_, T>> {
            Par(self.chunks_mut(n))
        }
        #[inline(always)]
        fn par_sort_unstable(&mut self)
        where
            T: Ord,
        {
            self.sort_unstable()
        }
        #[inline(always)]
        fn par_sort_by<F>(&mut self, cmp: F)
        where
            F: FnMut(&T, &T) -> core::cmp::Ordering,
        {
            self.sort_by(cmp)
        }
    }

    // ----- combinators + terminals on the wrapper -----

    impl<I: Iterator> Par<I> {
        /// Perf hint in rayon; a no-op serially.
        #[inline(always)]
        pub fn with_min_len(self, _min: usize) -> Self {
            self
        }
        /// Perf hint in rayon; a no-op serially.
        #[inline(always)]
        pub fn with_max_len(self, _max: usize) -> Self {
            self
        }

        #[inline(always)]
        pub fn map<R, F: FnMut(I::Item) -> R>(self, f: F) -> Par<core::iter::Map<I, F>> {
            Par(self.0.map(f))
        }

        #[inline(always)]
        pub fn filter<F: FnMut(&I::Item) -> bool>(self, f: F) -> Par<core::iter::Filter<I, F>> {
            Par(self.0.filter(f))
        }

        #[inline(always)]
        pub fn filter_map<R, F: FnMut(I::Item) -> Option<R>>(
            self,
            f: F,
        ) -> Par<core::iter::FilterMap<I, F>> {
            Par(self.0.filter_map(f))
        }

        #[inline(always)]
        pub fn flat_map<U, F>(self, f: F) -> Par<core::iter::FlatMap<I, U, F>>
        where
            U: IntoIterator,
            F: FnMut(I::Item) -> U,
        {
            Par(self.0.flat_map(f))
        }

        #[inline(always)]
        pub fn flat_map_iter<U, F>(self, f: F) -> Par<core::iter::FlatMap<I, U, F>>
        where
            U: IntoIterator,
            F: FnMut(I::Item) -> U,
        {
            Par(self.0.flat_map(f))
        }

        #[inline(always)]
        pub fn flatten(self) -> Par<core::iter::Flatten<I>>
        where
            I::Item: IntoIterator,
        {
            Par(self.0.flatten())
        }

        #[inline(always)]
        pub fn enumerate(self) -> Par<core::iter::Enumerate<I>> {
            Par(self.0.enumerate())
        }

        /// Match Rayon by rejecting unequal lengths before iteration.
        #[inline(always)]
        pub fn zip_eq<U: IntoParallelIterator>(
            self,
            other: U,
        ) -> Par<core::iter::Zip<I, <U::Iter as IntoSerial>::Iter>>
        where
            I: ExactSizeIterator,
            U::Iter: IntoSerial,
            <U::Iter as IntoSerial>::Iter: ExactSizeIterator,
        {
            let other = other.into_par_iter().into_serial();
            assert_eq!(self.0.len(), other.len(), "zip_eq requires equal lengths");
            Par(self.0.zip(other))
        }

        /// rayon's `fold(identity, op)`: produces per-chunk accumulators. With
        /// one (serial) chunk this is a single fold from `identity()`, surfaced
        /// as a one-element producer so a following `reduce`/`sum` composes as
        /// rayon's would. (Distinct from `Iterator::fold`; `Par` is not an
        /// `Iterator`, so there is no collision.)
        #[inline(always)]
        pub fn fold<T, ID, F>(self, identity: ID, op: F) -> Par<core::iter::Once<T>>
        where
            ID: Fn() -> T,
            F: FnMut(T, I::Item) -> T,
        {
            Par(core::iter::once(self.0.fold(identity(), op)))
        }

        /// rayon's `fold_with(init, op)`: with one (serial) chunk this folds the
        /// whole iterator into a single accumulator, surfaced as a one-element
        /// producer so a following `reduce`/`sum` composes as rayon's would.
        #[inline(always)]
        pub fn fold_with<T, F>(self, init: T, f: F) -> Par<core::iter::Once<T>>
        where
            F: FnMut(T, I::Item) -> T,
        {
            Par(core::iter::once(self.0.fold(init, f)))
        }

        #[inline(always)]
        pub fn zip<U: IntoParallelIterator>(
            self,
            other: U,
        ) -> Par<core::iter::Zip<I, <U::Iter as IntoSerial>::Iter>>
        where
            U::Iter: IntoSerial,
        {
            Par(self.0.zip(other.into_par_iter().into_serial()))
        }

        #[inline(always)]
        pub fn take(self, n: usize) -> Par<core::iter::Take<I>> {
            Par(self.0.take(n))
        }

        #[inline(always)]
        pub fn skip(self, n: usize) -> Par<core::iter::Skip<I>> {
            Par(self.0.skip(n))
        }

        #[inline(always)]
        pub fn step_by(self, step: usize) -> Par<core::iter::StepBy<I>> {
            Par(self.0.step_by(step))
        }

        #[inline(always)]
        pub fn chain<U: IntoParallelIterator>(
            self,
            other: U,
        ) -> Par<core::iter::Chain<I, <U::Iter as IntoSerial>::Iter>>
        where
            U::Iter: IntoSerial<Item = I::Item>,
        {
            Par(self.0.chain(other.into_par_iter().into_serial()))
        }

        #[inline(always)]
        pub fn cloned<'a, T: 'a + Clone>(self) -> Par<core::iter::Cloned<I>>
        where
            I: Iterator<Item = &'a T>,
        {
            Par(self.0.cloned())
        }

        #[inline(always)]
        pub fn copied<'a, T: 'a + Copy>(self) -> Par<core::iter::Copied<I>>
        where
            I: Iterator<Item = &'a T>,
        {
            Par(self.0.copied())
        }

        #[inline(always)]
        pub fn rev(self) -> Par<core::iter::Rev<I>>
        where
            I: DoubleEndedIterator,
        {
            Par(self.0.rev())
        }

        // ---- terminals ----

        /// rayon's `reduce(identity, op)` -> serial left fold.
        #[inline(always)]
        pub fn reduce<ID, OP>(self, identity: ID, op: OP) -> I::Item
        where
            ID: Fn() -> I::Item,
            OP: Fn(I::Item, I::Item) -> I::Item,
        {
            self.0.fold(identity(), op)
        }

        /// rayon's `reduce_with(op)` -> `Iterator::reduce`.
        #[inline(always)]
        pub fn reduce_with<OP>(self, op: OP) -> Option<I::Item>
        where
            OP: Fn(I::Item, I::Item) -> I::Item,
        {
            self.0.reduce(op)
        }

        #[inline(always)]
        pub fn for_each<F: FnMut(I::Item)>(self, f: F) {
            self.0.for_each(f)
        }

        #[inline(always)]
        pub fn collect<B: FromIterator<I::Item>>(self) -> B {
            self.0.collect()
        }

        #[inline(always)]
        pub fn sum<S: core::iter::Sum<I::Item>>(self) -> S {
            self.0.sum()
        }

        #[inline(always)]
        pub fn product<S: core::iter::Product<I::Item>>(self) -> S {
            self.0.product()
        }

        #[inline(always)]
        pub fn count(self) -> usize {
            self.0.count()
        }

        #[inline(always)]
        pub fn min(self) -> Option<I::Item>
        where
            I::Item: Ord,
        {
            self.0.min()
        }

        #[inline(always)]
        pub fn max(self) -> Option<I::Item>
        where
            I::Item: Ord,
        {
            self.0.max()
        }

        #[inline(always)]
        pub fn min_by<F>(self, f: F) -> Option<I::Item>
        where
            F: FnMut(&I::Item, &I::Item) -> core::cmp::Ordering,
        {
            self.0.min_by(f)
        }

        #[inline(always)]
        pub fn max_by<F>(self, f: F) -> Option<I::Item>
        where
            F: FnMut(&I::Item, &I::Item) -> core::cmp::Ordering,
        {
            self.0.max_by(f)
        }

        #[inline(always)]
        pub fn min_by_key<B: Ord, F: FnMut(&I::Item) -> B>(self, f: F) -> Option<I::Item> {
            self.0.min_by_key(f)
        }

        #[inline(always)]
        pub fn max_by_key<B: Ord, F: FnMut(&I::Item) -> B>(self, f: F) -> Option<I::Item> {
            self.0.max_by_key(f)
        }

        #[inline(always)]
        pub fn any<F: FnMut(I::Item) -> bool>(self, f: F) -> bool {
            self.0.into_iter().any(f)
        }

        #[inline(always)]
        pub fn all<F: FnMut(I::Item) -> bool>(self, f: F) -> bool {
            self.0.into_iter().all(f)
        }

        #[inline(always)]
        pub fn find_any<F: FnMut(&I::Item) -> bool>(self, mut f: F) -> Option<I::Item> {
            self.0.into_iter().find(|x| f(x))
        }

        #[inline(always)]
        pub fn unzip<A, B, FromA, FromB>(self) -> (FromA, FromB)
        where
            I: Iterator<Item = (A, B)>,
            FromA: Default + Extend<A>,
            FromB: Default + Extend<B>,
        {
            self.0.unzip()
        }

        #[inline(always)]
        pub fn collect_into_vec(self, target: &mut Vec<I::Item>) {
            target.clear();
            target.extend(self.0);
        }
    }

    /// Helper so `zip`/`chain` can accept another producer and recover its
    /// underlying serial iterator.
    pub trait IntoSerial {
        type Item;
        type Iter: Iterator<Item = Self::Item>;
        fn into_serial(self) -> Self::Iter;
    }

    impl<I: Iterator> IntoSerial for Par<I> {
        type Item = I::Item;
        type Iter = I;
        #[inline(always)]
        fn into_serial(self) -> I {
            self.0
        }
    }
}

/// Import surface mirroring `rayon::prelude` plus the free functions joltworks
/// uses from the `rayon` crate root.
pub mod prelude {
    pub use super::imp::*;
}

pub use imp::{
    current_num_threads, join, repeat_n, scope, slice, spawn, Scope, ThreadPool, ThreadPoolBuilder,
};

#[cfg(all(test, not(feature = "parallel")))]
mod tests {
    use super::prelude::*;
    use std::rc::Rc;

    #[test]
    fn serial_reductions_preserve_order_and_accept_local_values() {
        let values = vec![Rc::new(3), Rc::new(5), Rc::new(7)];
        let result = values
            .par_iter()
            .map(|v| vec![**v])
            .reduce(Vec::new, |mut a, b| {
                a.extend(b);
                a
            });
        assert_eq!(result, vec![3, 5, 7]);
        let mut output = vec![0; 3];
        output
            .par_iter_mut()
            .zip(values.par_iter())
            .for_each(|(o, v)| *o = **v);
        assert_eq!(output, result);
        let sum = values.into_par_iter().map(|v| *v).sum::<i32>();
        assert_eq!(sum, 15);
    }

    #[test]
    fn scoped_work_runs_before_returning() {
        let mut value = 0;
        scope(|s| s.spawn(|_| value = 9));
        assert_eq!(value, 9);
        assert_eq!(current_num_threads(), 1);
    }
}

#[cfg(test)]
mod zip_eq_tests {
    use super::prelude::*;

    #[test]
    fn equal_lengths_preserve_pairs() {
        for length in [0, 1, 9] {
            let actual: Vec<_> = (0..length).into_par_iter().zip_eq(0..length).collect();
            assert_eq!(actual, (0..length).map(|i| (i, i)).collect::<Vec<_>>());
        }
    }

    #[test]
    fn unequal_lengths_panic_before_consumption() {
        for (left, right) in [(0, 1), (1, 0), (2, 3), (3, 2)] {
            assert!(std::panic::catch_unwind(|| {
                let _pairs = (0..left).into_par_iter().zip_eq(0..right);
            })
            .is_err());
        }
    }
}
