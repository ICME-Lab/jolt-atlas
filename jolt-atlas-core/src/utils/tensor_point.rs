//! An MLE point over a tensor, split per axis.

use joltworks::utils::math::Math;

/// An MLE point over a tensor, split per axis: axis `i` owns `log2(dims[i])`
/// consecutive coordinates, in the tensor's row-major order.
pub struct TensorPoint<F> {
    coords: Vec<F>,
    axis_starts: Vec<usize>,
}

impl<F> TensorPoint<F> {
    /// Split `coords` over a tensor of power-of-two `dims`.
    pub fn new(coords: Vec<F>, dims: &[usize]) -> Self {
        let mut axis_starts = Vec::with_capacity(dims.len() + 1);
        axis_starts.push(0);
        for dim in dims {
            axis_starts.push(axis_starts[axis_starts.len() - 1] + dim.log_2());
        }
        assert_eq!(
            coords.len(),
            axis_starts[dims.len()],
            "point of {} coordinates over a tensor of dims {dims:?}",
            coords.len()
        );
        Self {
            coords,
            axis_starts,
        }
    }

    /// The coordinates owned by axis `i`.
    pub fn axis(&self, i: usize) -> &[F] {
        &self.coords[self.axis_starts[i]..self.axis_starts[i + 1]]
    }
}

#[cfg(test)]
mod tests {
    use super::TensorPoint;

    #[test]
    fn splits_a_point_per_axis() {
        let point = TensorPoint::new(vec![1, 2, 3, 4, 5, 6], &[4, 1, 8, 2]);
        assert_eq!(point.axis(0), [1, 2]);
        assert_eq!(point.axis(1), [] as [i32; 0]);
        assert_eq!(point.axis(2), [3, 4, 5]);
        assert_eq!(point.axis(3), [6]);
    }

    #[test]
    #[should_panic(expected = "point of 2 coordinates")]
    fn rejects_a_point_of_the_wrong_length() {
        TensorPoint::new(vec![1, 2], &[8]);
    }
}
