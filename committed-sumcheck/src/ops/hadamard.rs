//! Hadamard product relation.
//!
//! This relation represents
//!
//! ```text
//! sum_i lhs(i) * rhs(i)
//! ```
//!
//! The equality polynomial is intentionally not part of this operation-level
//! relation. GKR callers attach the common `eq` factor outside the relation, so
//! it can be handled once with the faster split-eq path.

use joltworks::field::JoltField;

use crate::round::{MleTable, RoundRelation};

#[derive(Debug, Clone, Copy)]
pub struct Hadamard<Lhs, Rhs> {
    pub lhs: Lhs,
    pub rhs: Rhs,
}

impl<Lhs, Rhs> Hadamard<Lhs, Rhs> {
    pub fn new<F>(lhs: Lhs, rhs: Rhs) -> Self
    where
        F: JoltField,
        Lhs: MleTable<F>,
        Rhs: MleTable<F>,
    {
        assert_eq!(lhs.len(), rhs.len(), "Hadamard lhs/rhs length mismatch");
        Self { lhs, rhs }
    }
}

impl<F, Lhs, Rhs> RoundRelation<F, 3> for Hadamard<Lhs, Rhs>
where
    F: JoltField,
    Lhs: MleTable<F>,
    Rhs: MleTable<F>,
{
    fn degree(&self) -> usize {
        2
    }

    fn len(&self) -> usize {
        self.lhs.len()
    }

    fn round_evals(&self, i: usize) -> [F; 3] {
        let (lhs_0, lhs_1) = self.lhs.pair(i);
        let (rhs_0, rhs_1) = self.rhs.pair(i);
        let two = F::one() + F::one();
        let lhs_2 = lhs_0 + two * (lhs_1 - lhs_0);
        let rhs_2 = rhs_0 + two * (rhs_1 - rhs_0);
        [lhs_0 * rhs_0, lhs_1 * rhs_1, lhs_2 * rhs_2]
    }

    fn bind(&mut self, r: <F as JoltField>::Challenge) {
        self.lhs.bind(r);
        self.rhs.bind(r);
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::Zero;

    use super::*;
    use crate::round::{build_round_poly, DenseMleTable};

    #[test]
    fn builds_hadamard_round_poly() {
        let lhs = [1, 3, 2, 5].map(Fr::from);
        let rhs = [7, 11, 13, 17].map(Fr::from);
        let relation = Hadamard::new::<Fr>(
            DenseMleTable::new(lhs.to_vec()),
            DenseMleTable::new(rhs.to_vec()),
        );

        let poly = build_round_poly(&relation);
        for t in 0..=2 {
            let t = Fr::from(t as u64);
            let expected = (0..relation.len())
                .map(|i| relation.lhs.at(i, t) * relation.rhs.at(i, t))
                .fold(Fr::zero(), |acc, term| acc + term);
            assert_eq!(poly.evaluate(t), expected);
        }
    }
}
