//! A minimal dense row-major tensor.
//!
//! Invariant, enforced at every construction site: `data.len()` equals the
//! product of `dims`. Fields are private so the invariant cannot be broken
//! from outside. The prover consumes tensors as flat row-major slices
//! ([`Tensor::data`]), so there are no strided views: every operation
//! produces a contiguous tensor.
//!
//! Scalars are represented as `dims == [1]` (the convention the old crate
//! and the prover already use).

use serde::{Deserialize, Serialize};

use crate::error::ShapeError;

/// Element types a [`Tensor`] may hold.
pub trait Element:
    Copy + Clone + Default + PartialEq + std::fmt::Debug + Send + Sync + 'static
{
}

impl Element for i32 {}
impl Element for i64 {}
impl Element for u32 {}
impl Element for bool {}
impl Element for f64 {}

/// A dense row-major tensor.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tensor<T> {
    data: Vec<T>,
    dims: Vec<usize>,
}

impl<T: Element> std::fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor{:?} ", self.dims)?;
        if self.data.len() <= 16 {
            write!(f, "{:?}", self.data)
        } else {
            write!(f, "[{:?}, …; {}]", &self.data[..8], self.data.len())
        }
    }
}

impl<T: Element> Tensor<T> {
    /// Construct a tensor from data and dimensions.
    pub fn new(data: Vec<T>, dims: &[usize]) -> Result<Self, ShapeError> {
        let expected = dims.iter().product::<usize>();
        if data.len() != expected {
            return Err(ShapeError::LengthMismatch {
                len: data.len(),
                dims: dims.to_vec(),
                expected,
            });
        }
        Ok(Tensor {
            data,
            dims: dims.to_vec(),
        })
    }

    /// A tensor of the given shape with every element equal to `value`.
    pub fn filled(value: T, dims: &[usize]) -> Self {
        Tensor {
            data: vec![value; dims.iter().product()],
            dims: dims.to_vec(),
        }
    }

    /// A zero-filled (default-filled) tensor of the given shape.
    pub fn zeros(dims: &[usize]) -> Self {
        Self::filled(T::default(), dims)
    }

    /// A rank-1 scalar tensor (`dims == [1]`).
    pub fn scalar(value: T) -> Self {
        Tensor {
            data: vec![value],
            dims: vec![1],
        }
    }

    /// The tensor's dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// The flat row-major data.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Mutable access to the flat data (element values only — the shape
    /// invariant cannot be broken through a slice).
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume the tensor, returning its flat data.
    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor has zero elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Whether the tensor holds exactly one element.
    pub fn is_scalar(&self) -> bool {
        self.data.len() == 1
    }

    /// Flat index of a coordinate (row-major). Panics on rank mismatch or
    /// out-of-range coordinates — callers index within `dims()` by
    /// construction.
    pub fn flat_index(&self, coords: &[usize]) -> usize {
        debug_assert_eq!(coords.len(), self.dims.len());
        let mut idx = 0;
        for (c, d) in coords.iter().zip(&self.dims) {
            debug_assert!(c < d);
            idx = idx * d + c;
        }
        idx
    }

    /// Element at a coordinate.
    pub fn get(&self, coords: &[usize]) -> T {
        self.data[self.flat_index(coords)]
    }

    /// Iterate the flat row-major elements.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Parallel iterator over the flat row-major elements.
    #[cfg(feature = "rayon")]
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T> {
        use rayon::prelude::*;
        self.data.par_iter()
    }

    /// Construct from owned data and dims, panicking on length mismatch
    /// (test/ergonomic constructor; prefer [`Tensor::new`]).
    pub fn construct(data: Vec<T>, dims: Vec<usize>) -> Self {
        Self::new(data, &dims).expect("tensor data length must match dims")
    }

    /// Construct by copying from a slice.
    pub fn from_slice(data: &[T], dims: &[usize]) -> Result<Self, ShapeError> {
        Self::new(data.to_vec(), dims)
    }

    /// Apply `f` to every element.
    pub fn map<U: Element>(&self, f: impl Fn(T) -> U) -> Tensor<U> {
        Tensor {
            data: self.data.iter().map(|&x| f(x)).collect(),
            dims: self.dims.clone(),
        }
    }

    /// Combine two same-shaped tensors element-wise. Broadcasting is the
    /// caller's job (via [`Tensor::expand`]).
    pub fn zip_map<U: Element, V: Element>(
        &self,
        rhs: &Tensor<U>,
        f: impl Fn(T, U) -> V,
    ) -> Result<Tensor<V>, ShapeError> {
        if self.dims != rhs.dims {
            return Err(ShapeError::BroadcastIncompatible {
                lhs: self.dims.clone(),
                rhs: rhs.dims.clone(),
            });
        }
        Ok(Tensor {
            data: self
                .data
                .iter()
                .zip(&rhs.data)
                .map(|(&a, &b)| f(a, b))
                .collect(),
            dims: self.dims.clone(),
        })
    }

    /// Reinterpret the data with new dimensions of the same total size.
    pub fn reshape(mut self, dims: &[usize]) -> Result<Self, ShapeError> {
        let expected = dims.iter().product::<usize>();
        if self.data.len() != expected {
            return Err(ShapeError::LengthMismatch {
                len: self.data.len(),
                dims: dims.to_vec(),
                expected,
            });
        }
        self.dims = dims.to_vec();
        Ok(self)
    }

    /// Broadcast-expand to `shape` (NumPy semantics: dims align on the
    /// right; size-1 source dims repeat; missing leading dims repeat).
    pub fn expand(&self, shape: &[usize]) -> Result<Self, ShapeError> {
        let incompatible = || ShapeError::ExpandIncompatible {
            from: self.dims.clone(),
            to: shape.to_vec(),
        };
        if self.dims.len() > shape.len() {
            return Err(incompatible());
        }
        if self.dims == shape {
            return Ok(self.clone());
        }
        if self
            .dims
            .iter()
            .rev()
            .zip(shape.iter().rev())
            .any(|(&src, &dst)| src != dst && src != 1)
        {
            return Err(incompatible());
        }

        // Right-align the source dims against the target and walk the target
        // row-major, carrying the source flat index incrementally.
        let offset = shape.len() - self.dims.len();
        let mut out = Vec::with_capacity(shape.iter().product());
        let mut coords = vec![0usize; shape.len()];
        let total: usize = shape.iter().product();
        for _ in 0..total {
            let mut src_idx = 0;
            for (axis, &c) in coords.iter().enumerate().skip(offset) {
                let src_dim = self.dims[axis - offset];
                src_idx = src_idx * src_dim + if src_dim == 1 { 0 } else { c };
            }
            out.push(if self.dims.is_empty() {
                self.data[0]
            } else {
                self.data[src_idx]
            });
            for axis in (0..shape.len()).rev() {
                coords[axis] += 1;
                if coords[axis] < shape[axis] {
                    break;
                }
                coords[axis] = 0;
            }
        }
        Tensor::new(out, shape)
    }

    /// Move axis `source` to position `destination`, shifting the axes in
    /// between (ONNX `Transpose` decomposes into chains of these).
    pub fn move_axis(&self, source: usize, destination: usize) -> Result<Self, ShapeError> {
        let rank = self.dims.len();
        if source >= rank || destination >= rank {
            return Err(ShapeError::AxisOutOfRange {
                axis: source.max(destination),
                rank,
            });
        }
        if source == destination {
            return Ok(self.clone());
        }
        // Permutation: remove `source`, insert at `destination`.
        let mut perm: Vec<usize> = (0..rank).collect();
        let s = perm.remove(source);
        perm.insert(destination, s);

        let new_dims: Vec<usize> = perm.iter().map(|&p| self.dims[p]).collect();
        let src_strides = row_major_strides(&self.dims);
        let mut out = Vec::with_capacity(self.data.len());
        let mut coords = vec![0usize; rank];
        for _ in 0..self.data.len() {
            let mut src_idx = 0;
            for (axis, &c) in coords.iter().enumerate() {
                src_idx += c * src_strides[perm[axis]];
            }
            out.push(self.data[src_idx]);
            for axis in (0..rank).rev() {
                coords[axis] += 1;
                if coords[axis] < new_dims[axis] {
                    break;
                }
                coords[axis] = 0;
            }
        }
        Tensor::new(out, &new_dims)
    }

    /// Extract `start..end` along `axis`.
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Result<Self, ShapeError> {
        let rank = self.dims.len();
        if axis >= rank {
            return Err(ShapeError::AxisOutOfRange { axis, rank });
        }
        if start > end || end > self.dims[axis] {
            return Err(ShapeError::SliceOutOfRange {
                axis,
                start,
                end,
                size: self.dims[axis],
            });
        }
        let mut new_dims = self.dims.clone();
        new_dims[axis] = end - start;

        // Row-major: copy contiguous runs of `inner` elements.
        let outer: usize = self.dims[..axis].iter().product();
        let inner: usize = self.dims[axis + 1..].iter().product();
        let mut out = Vec::with_capacity(new_dims.iter().product());
        for o in 0..outer {
            let base = (o * self.dims[axis] + start) * inner;
            out.extend_from_slice(&self.data[base..base + (end - start) * inner]);
        }
        Tensor::new(out, &new_dims)
    }

    /// Zero-pad to `target` dims (each target dim >= the current dim; same
    /// rank). Existing elements keep their coordinates.
    pub fn pad_to(&self, target: &[usize]) -> Result<Self, ShapeError> {
        if target.len() != self.dims.len()
            || self.dims.iter().zip(target).any(|(&cur, &tgt)| tgt < cur)
        {
            return Err(ShapeError::PadTooSmall {
                from: self.dims.clone(),
                to: target.to_vec(),
            });
        }
        if self.dims == target {
            return Ok(self.clone());
        }
        let mut out = Tensor::zeros(target);
        copy_strided(&self.data, &mut out.data, &self.dims, target);
        Ok(out)
    }

    /// Zero-pad every dimension to the next power of two.
    pub fn padded_pow2(&self) -> Self {
        let target: Vec<usize> = self.dims.iter().map(|d| d.next_power_of_two()).collect();
        // Infallible: every target dim >= the current dim.
        self.pad_to(&target).expect("pow2 pad grows every dim")
    }

    /// Crop to `target` dims (each target dim <= the current dim; same
    /// rank). The inverse of [`Tensor::pad_to`].
    pub fn crop_to(&self, target: &[usize]) -> Result<Self, ShapeError> {
        if target.len() != self.dims.len()
            || self.dims.iter().zip(target).any(|(&cur, &tgt)| tgt > cur)
        {
            return Err(ShapeError::Invalid(format!(
                "cannot crop {:?} to larger target {:?}",
                self.dims, target
            )));
        }
        if self.dims == target {
            return Ok(self.clone());
        }
        let mut out = Tensor::zeros(target);
        copy_strided_crop(&self.data, &mut out.data, &self.dims, target);
        Ok(out)
    }
}

impl<T: Element> std::ops::Deref for Tensor<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.data
    }
}

impl<T: Element> std::ops::Index<usize> for Tensor<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

/// Concatenate tensors along `axis`. All inputs must agree on every other
/// dimension.
pub fn concat<T: Element>(inputs: &[&Tensor<T>], axis: usize) -> Result<Tensor<T>, ShapeError> {
    let first = inputs.first().ok_or(ShapeError::WrongArity {
        expected: 1,
        got: 0,
    })?;
    let rank = first.dims.len();
    if axis >= rank {
        return Err(ShapeError::AxisOutOfRange { axis, rank });
    }
    let mut axis_total = 0;
    for t in inputs {
        if t.dims.len() != rank
            || t.dims
                .iter()
                .zip(&first.dims)
                .enumerate()
                .any(|(i, (a, b))| i != axis && a != b)
        {
            return Err(ShapeError::Invalid(format!(
                "concat: shape {:?} incompatible with {:?} on axis {axis}",
                t.dims, first.dims
            )));
        }
        axis_total += t.dims[axis];
    }
    let mut new_dims = first.dims.clone();
    new_dims[axis] = axis_total;

    let outer: usize = first.dims[..axis].iter().product();
    let inner: usize = first.dims[axis + 1..].iter().product();
    let mut out = Vec::with_capacity(new_dims.iter().product());
    for o in 0..outer {
        for t in inputs {
            let run = t.dims[axis] * inner;
            out.extend_from_slice(&t.data[o * run..(o + 1) * run]);
        }
    }
    Tensor::new(out, &new_dims)
}

/// Broadcast shape of two shapes (NumPy semantics).
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ShapeError> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() {
            1
        } else {
            a[i - (rank - a.len())]
        };
        let db = if i < rank - b.len() {
            1
        } else {
            b[i - (rank - b.len())]
        };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => {
                return Err(ShapeError::BroadcastIncompatible {
                    lhs: a.to_vec(),
                    rhs: b.to_vec(),
                });
            }
        };
    }
    Ok(out)
}

/// Row-major (C-order) strides of a shape.
pub fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Copy `src` (shape `src_dims`) into the top-left corner of `dst` (shape
/// `dst_dims`, zero-initialized), preserving coordinates. `dst_dims[i] >=
/// src_dims[i]` for every axis.
fn copy_strided<T: Copy>(src: &[T], dst: &mut [T], src_dims: &[usize], dst_dims: &[usize]) {
    // Walk source coordinates; copy the innermost contiguous run at once.
    if src.is_empty() {
        return;
    }
    let dst_strides = row_major_strides(dst_dims);
    let inner = *src_dims.last().unwrap_or(&1);
    let outer_dims = &src_dims[..src_dims.len().saturating_sub(1)];
    let mut coords = vec![0usize; outer_dims.len()];
    let mut src_off = 0;
    loop {
        let dst_off: usize = coords.iter().zip(&dst_strides).map(|(&c, &s)| c * s).sum();
        dst[dst_off..dst_off + inner].copy_from_slice(&src[src_off..src_off + inner]);
        src_off += inner;
        // Increment outer coordinates.
        let mut axis = outer_dims.len();
        loop {
            if axis == 0 {
                return;
            }
            axis -= 1;
            coords[axis] += 1;
            if coords[axis] < outer_dims[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
}

/// Copy the top-left corner of `src` (shape `src_dims`) into `dst` (shape
/// `dst_dims`). `dst_dims[i] <= src_dims[i]` for every axis.
fn copy_strided_crop<T: Copy>(src: &[T], dst: &mut [T], src_dims: &[usize], dst_dims: &[usize]) {
    if dst.is_empty() {
        return;
    }
    let src_strides = row_major_strides(src_dims);
    let inner = *dst_dims.last().unwrap_or(&1);
    let outer_dims = &dst_dims[..dst_dims.len().saturating_sub(1)];
    let mut coords = vec![0usize; outer_dims.len()];
    let mut dst_off = 0;
    loop {
        let src_off: usize = coords.iter().zip(&src_strides).map(|(&c, &s)| c * s).sum();
        dst[dst_off..dst_off + inner].copy_from_slice(&src[src_off..src_off + inner]);
        dst_off += inner;
        let mut axis = outer_dims.len();
        loop {
            if axis == 0 {
                return;
            }
            axis -= 1;
            coords[axis] += 1;
            if coords[axis] < outer_dims[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
}

#[cfg(feature = "test-util")]
mod random {
    use super::Tensor;
    use rand::{Rng, RngCore};

    impl Tensor<i32> {
        /// Random full-range i32 tensor (old-crate draw pattern, so seeded
        /// tests keep their vectors).
        pub fn random(rng: &mut impl RngCore, dims: &[usize]) -> Self {
            let data = (0..dims.iter().product::<usize>())
                .map(|_| rng.next_u32() as i32)
                .collect();
            Tensor::new(data, dims).expect("length matches by construction")
        }

        /// Random i8-range tensor (old-crate draw pattern).
        pub fn random_small(rng: &mut impl RngCore, dims: &[usize]) -> Self {
            let data = (0..dims.iter().product::<usize>())
                .map(|_| rng.next_u32() as i8 as i32)
                .collect();
            Tensor::new(data, dims).expect("length matches by construction")
        }

        /// Random tensor with elements uniform in `range`.
        pub fn random_range(
            rng: &mut impl Rng,
            dims: &[usize],
            range: std::ops::Range<i32>,
        ) -> Self {
            let data = (0..dims.iter().product::<usize>())
                .map(|_| rng.gen_range(range.clone()))
                .collect();
            Tensor::new(data, dims).expect("length matches by construction")
        }

        /// Random 0/1 tensor (old-crate draw pattern).
        pub fn random_boolean(rng: &mut impl RngCore, dims: &[usize]) -> Self {
            let data = (0..dims.iter().product::<usize>())
                .map(|_| (rng.next_u32() % 2) as i32)
                .collect();
            Tensor::new(data, dims).expect("length matches by construction")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[i32], dims: &[usize]) -> Tensor<i32> {
        Tensor::new(data.to_vec(), dims).unwrap()
    }

    #[test]
    fn construction_checks_length() {
        assert!(Tensor::new(vec![1, 2, 3], &[2, 2]).is_err());
        assert_eq!(t(&[1, 2, 3, 4], &[2, 2]).len(), 4);
    }

    #[test]
    fn flat_index_row_major() {
        let a = t(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        assert_eq!(a.get(&[0, 0]), 1);
        assert_eq!(a.get(&[0, 2]), 3);
        assert_eq!(a.get(&[1, 0]), 4);
        assert_eq!(a.get(&[1, 2]), 6);
    }

    #[test]
    fn expand_repeats_unit_dims() {
        let a = t(&[1, 2, 3], &[1, 3]);
        let b = a.expand(&[2, 3]).unwrap();
        assert_eq!(b.data(), &[1, 2, 3, 1, 2, 3]);

        let c = t(&[5], &[1]).expand(&[2, 2]).unwrap();
        assert_eq!(c.data(), &[5, 5, 5, 5]);

        let col = t(&[1, 2], &[2, 1]).expand(&[2, 3]).unwrap();
        assert_eq!(col.data(), &[1, 1, 1, 2, 2, 2]);

        assert!(t(&[1, 2], &[2]).expand(&[3]).is_err());
    }

    #[test]
    fn expand_adds_leading_dims() {
        let a = t(&[1, 2, 3], &[3]);
        let b = a.expand(&[2, 3]).unwrap();
        assert_eq!(b.data(), &[1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn move_axis_matches_transpose() {
        // [2,3] transpose -> [3,2]
        let a = t(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        let b = a.move_axis(0, 1).unwrap();
        assert_eq!(b.dims(), &[3, 2]);
        assert_eq!(b.data(), &[1, 4, 2, 5, 3, 6]);
        // Moving an axis onto itself is identity.
        assert_eq!(a.move_axis(1, 1).unwrap().data(), a.data());
    }

    #[test]
    fn move_axis_3d() {
        // shape [2,2,2]; move axis 0 to 2: out[i,j,k] = in[k,i,j]
        let a = t(&[0, 1, 2, 3, 4, 5, 6, 7], &[2, 2, 2]);
        let b = a.move_axis(0, 2).unwrap();
        assert_eq!(b.dims(), &[2, 2, 2]);
        assert_eq!(b.get(&[0, 1, 1]), a.get(&[1, 0, 1]));
        assert_eq!(b.get(&[1, 0, 1]), a.get(&[1, 1, 0]));
    }

    #[test]
    fn slice_middle_axis() {
        let a = t(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        let b = a.slice(1, 1, 3).unwrap();
        assert_eq!(b.dims(), &[2, 2]);
        assert_eq!(b.data(), &[2, 3, 5, 6]);
        assert!(a.slice(1, 2, 4).is_err());
    }

    #[test]
    fn concat_axis0_and_axis1() {
        let a = t(&[1, 2], &[1, 2]);
        let b = t(&[3, 4], &[1, 2]);
        let c0 = concat(&[&a, &b], 0).unwrap();
        assert_eq!(c0.dims(), &[2, 2]);
        assert_eq!(c0.data(), &[1, 2, 3, 4]);
        let c1 = concat(&[&a, &b], 1).unwrap();
        assert_eq!(c1.dims(), &[1, 4]);
        assert_eq!(c1.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn pad_and_crop_roundtrip() {
        let a = t(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        let p = a.padded_pow2();
        assert_eq!(p.dims(), &[2, 4]);
        assert_eq!(p.data(), &[1, 2, 3, 0, 4, 5, 6, 0]);
        assert_eq!(p.crop_to(&[2, 3]).unwrap(), a);

        let q = a.pad_to(&[4, 4]).unwrap();
        assert_eq!(q.dims(), &[4, 4]);
        assert_eq!(q.get(&[1, 2]), 6);
        assert_eq!(q.get(&[3, 3]), 0);
        assert_eq!(q.crop_to(&[2, 3]).unwrap(), a);
    }

    #[test]
    fn broadcast_shape_rules() {
        assert_eq!(broadcast_shape(&[2, 1], &[3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[1], &[4, 5]).unwrap(), vec![4, 5]);
        assert!(broadcast_shape(&[2], &[3]).is_err());
    }

    #[test]
    fn reshape_checks_size() {
        let a = t(&[1, 2, 3, 4], &[2, 2]);
        assert_eq!(a.clone().reshape(&[4]).unwrap().dims(), &[4]);
        assert!(a.reshape(&[3]).is_err());
    }
}
