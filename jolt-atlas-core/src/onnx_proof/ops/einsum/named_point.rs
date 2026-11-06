//! An MLE point whose axes are named by einsum letters.

use crate::utils::tensor_point::TensorPoint;

/// An einsum index letter.
pub(super) type Letter = char;

/// The axis of `letter` in a term.
pub(super) fn axis_of(term: &[Letter], letter: Letter) -> usize {
    term.iter()
        .position(|c| *c == letter)
        .unwrap_or_else(|| panic!("letter {letter} is not in {term:?}"))
}

/// A [`TensorPoint`] with one letter per axis.
pub(super) struct NamedPoint<F> {
    point: TensorPoint<F>,
    letters: Vec<Letter>,
}

impl<F: Copy> NamedPoint<F> {
    /// Split `coords` over the axes `letters` of power-of-two sizes `dims`.
    pub(super) fn new(coords: Vec<F>, letters: Vec<Letter>, dims: &[usize]) -> Self {
        Self {
            point: TensorPoint::new(coords, dims),
            letters,
        }
    }

    /// The coordinates of the axis named `letter`.
    pub(super) fn axis(&self, letter: Letter) -> &[F] {
        self.point.axis(axis_of(&self.letters, letter))
    }

    /// The coordinates of the axes `letters`, joined in that order.
    pub(super) fn coords(&self, letters: &[Letter]) -> Vec<F> {
        letters
            .iter()
            .flat_map(|c| self.axis(*c))
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::NamedPoint;

    #[test]
    fn reads_coordinates_by_letter_in_the_requested_order() {
        let point = NamedPoint::new(vec![1, 2, 3, 4, 5], vec!['n', 'b', 'm'], &[4, 2, 4]);
        assert_eq!(point.axis('b'), [3]);
        assert_eq!(point.coords(&['m', 'n']), [4, 5, 1, 2]);
        assert_eq!(point.coords(&[]), [] as [i32; 0]);
    }

    #[test]
    #[should_panic(expected = "letter z is not in")]
    fn rejects_an_unknown_letter() {
        NamedPoint::new(vec![1], vec!['a'], &[2]).axis('z');
    }
}
