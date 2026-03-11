//! Dimension-related extension helpers.

/// Extension methods for dimension vectors/slices.
pub trait UsizeDimsExt {
    /// Returns a new vector where each dimension is mapped to its next power of two.
    fn map_next_power_of_two(&self) -> Vec<usize>;
}

impl UsizeDimsExt for [usize] {
    fn map_next_power_of_two(&self) -> Vec<usize> {
        self.iter().map(|d| d.next_power_of_two()).collect()
    }
}

impl UsizeDimsExt for Vec<usize> {
    fn map_next_power_of_two(&self) -> Vec<usize> {
        self.as_slice().map_next_power_of_two()
    }
}

/// Extension methods for padding a data array.
pub trait Pad<T> {
    /// Pads the dimensions to the specified target dimensions.
    fn pad_to(&self, current_dims: &[usize], target_dims: &[usize]) -> Vec<T>;

    /// Pads the dimensions to the next power of two.
    fn pad_next_power_of_two(&self, current_dims: &[usize]) -> Vec<T>;
}

impl<T: Clone + Default> Pad<T> for [T] {
    fn pad_to(&self, current_dims: &[usize], target_dims: &[usize]) -> Vec<T> {
        let mut padded = vec![T::default(); target_dims.iter().product()];
        copy_strided(self, &mut padded, current_dims, target_dims, 0, 0, 0);
        padded
    }

    fn pad_next_power_of_two(&self, current_dims: &[usize]) -> Vec<T> {
        let target_dims = current_dims.map_next_power_of_two();
        self.pad_to(current_dims, &target_dims)
    }
}

impl<T: Clone + Default> Pad<T> for Vec<T> {
    fn pad_to(&self, current_dims: &[usize], target_dims: &[usize]) -> Vec<T> {
        self.as_slice().pad_to(current_dims, target_dims)
    }

    fn pad_next_power_of_two(&self, current_dims: &[usize]) -> Vec<T> {
        self.as_slice().pad_next_power_of_two(current_dims)
    }
}

/// Recursively copies data from a source tensor layout to a (larger) destination layout,
/// preserving multi-dimensional structure. Copies contiguous slices along the innermost
/// dimension for efficiency, avoiding per-element coordinate generation.
pub(crate) fn copy_strided<T: Clone>(
    src: &[T],
    dst: &mut [T],
    old_dims: &[usize],
    new_dims: &[usize],
    depth: usize,
    src_base: usize,
    dst_base: usize,
) {
    if depth == old_dims.len() - 1 {
        // Innermost dimension: copy contiguous row
        let count = old_dims[depth];
        dst[dst_base..dst_base + count].clone_from_slice(&src[src_base..src_base + count]);
    } else {
        let src_stride: usize = old_dims[depth + 1..].iter().product();
        let dst_stride: usize = new_dims[depth + 1..].iter().product();
        for i in 0..old_dims[depth] {
            copy_strided(
                src,
                dst,
                old_dims,
                new_dims,
                depth + 1,
                src_base + i * src_stride,
                dst_base + i * dst_stride,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Pad, UsizeDimsExt};

    #[test]
    fn test_map_next_power_of_two_for_slice() {
        let dims = [1, 3, 8, 9];
        assert_eq!(dims.map_next_power_of_two(), vec![1, 4, 8, 16]);
    }

    #[test]
    fn test_map_next_power_of_two_for_vec() {
        let dims = vec![2, 5, 7];
        assert_eq!(dims.map_next_power_of_two(), vec![2, 8, 8]);
    }

    #[test]
    fn test_pad_to() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let current_dims = [2, 3]; // 2 rows, 3 columns
        let target_dims = [4, 4]; // pad to 4 rows, 4 columns
        let padded = data.pad_to(&current_dims, &target_dims);
        assert_eq!(padded.len(), 16);
        assert_eq!(&padded[0..8], &[1, 2, 3, 0, 4, 5, 6, 0]);
        assert!(padded[8..].iter().all(|&x| x == 0));
    }

    #[test]
    fn test_pad_next_power_of_two() {
        let data = (0..18).collect::<Vec<_>>();
        let current_dims = [3, 2, 3]; // 3x2x3 tensor
        let padded = data.pad_next_power_of_two(&current_dims);
        let expected_dims = [4, 2, 4]; // next power of two for each dimension
        assert_eq!(padded.len(), expected_dims.iter().product::<usize>());
        // Check that original data is correctly placed in the padded tensor
        for (i, &value) in data.iter().enumerate() {
            let row = i / (current_dims[1] * current_dims[2]);
            let col = (i / current_dims[2]) % current_dims[1];
            let depth = i % current_dims[2];
            let padded_index =
                row * (expected_dims[1] * expected_dims[2]) + col * expected_dims[2] + depth;
            assert_eq!(padded[padded_index], value);
        }
    }
}
