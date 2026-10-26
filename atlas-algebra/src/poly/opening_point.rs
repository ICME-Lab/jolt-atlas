//! Opening points: a polynomial evaluation point together with its variable order.

use crate::field::JoltField;
use allocative::Allocative;

pub type Endianness = bool;
pub const BIG_ENDIAN: Endianness = false;
pub const LITTLE_ENDIAN: Endianness = true;

// TODO(AntoineF4C5): Using Vec<F> rather than Vec<F::Challenge> denies us from the performance gains of F::Challenge.
// The goal would be to use an enum that can be either F or F::Challenge.
// Might also be interesting to make a difference between an opening point, which allows to evaluate a poly at a point,
// and an array of sumcheck challenges, which might need reordering to correspond to a polynomial evaluation point.
#[derive(Clone, Debug, PartialEq, Default, Allocative)]
pub struct OpeningPoint<const E: Endianness, F: JoltField> {
    pub r: Vec<F>,
}

impl<const E: Endianness, F: JoltField> std::ops::Index<usize> for OpeningPoint<E, F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.r[index]
    }
}

impl<const E: Endianness, F: JoltField> std::ops::Index<std::ops::RangeFull>
    for OpeningPoint<E, F>
{
    type Output = [F];

    fn index(&self, _index: std::ops::RangeFull) -> &Self::Output {
        &self.r[..]
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn len(&self) -> usize {
        self.r.len()
    }

    pub fn split_at_r(&self, mid: usize) -> (&[F], &[F]) {
        self.r.split_at(mid)
    }

    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        let (left, right) = self.r.split_at(mid);
        (Self::new(left.to_vec()), Self::new(right.to_vec()))
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn new<C: Into<F>>(r: Vec<C>) -> Self {
        Self {
            r: r.into_iter().map(|x| x.into()).collect(),
        }
    }

    pub fn endianness(&self) -> &'static str {
        if E == BIG_ENDIAN {
            "big"
        } else {
            "little"
        }
    }

    pub fn match_endianness<const SWAPPED_E: Endianness>(&self) -> OpeningPoint<SWAPPED_E, F>
    where
        F: Clone,
    {
        let mut reversed = self.r.clone();
        if E != SWAPPED_E {
            reversed.reverse();
        }
        OpeningPoint::<SWAPPED_E, F>::new(reversed)
    }
}

impl<F: JoltField, U: Into<F>> From<Vec<U>> for OpeningPoint<LITTLE_ENDIAN, F> {
    fn from(r: Vec<U>) -> Self {
        Self::new(r.into_iter().map(|x| x.into()).collect())
    }
}

impl<F: JoltField, U: Into<F>> From<Vec<U>> for OpeningPoint<BIG_ENDIAN, F> {
    fn from(r: Vec<U>) -> Self {
        Self::new(r.into_iter().map(|x| x.into()).collect())
    }
}

impl<const E: Endianness, F: JoltField> Into<Vec<F>> for OpeningPoint<E, F> {
    fn into(self) -> Vec<F> {
        self.r
    }
}

impl<const E: Endianness, F: JoltField> Into<Vec<F>> for &OpeningPoint<E, F>
where
    F: Clone,
{
    fn into(self) -> Vec<F> {
        self.r.clone()
    }
}
