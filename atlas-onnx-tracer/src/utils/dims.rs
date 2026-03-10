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

#[cfg(test)]
mod tests {
    use super::UsizeDimsExt;

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
}
