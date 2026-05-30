//! Round-polynomial construction for plain sumcheck.
//!
//! This module intentionally separates an MLE table from the relation that uses
//! it. The call graph should stay this simple:
//!
//! ```text
//! build_round_poly
//!   └─ relation.len()
//!        └─ self.a.len()
//!
//! build_round_poly
//!   └─ relation.round_evals(i)
//!        ├─ self.a.pair(i)
//!        ├─ self.b.pair(i)
//!        └─ self.c.pair(i)
//! ```
//!
//! `MleTable::at` is one table interpolated along the current sumcheck variable.
//! `RoundRelation::round_evals` returns one summand evaluated on the relation's
//! interpolation grid.

use ark_ff::{Field, PrimeField};
use joltworks::field::JoltField;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundPoly<F> {
    pub coeffs: Vec<F>,
}

impl<F: Field> RoundPoly<F> {
    pub fn evaluate(&self, t: F) -> F {
        self.coeffs
            .iter()
            .rev()
            .fold(F::zero(), |acc, coeff| acc * t + coeff)
    }
}

impl<F: PrimeField> RoundPoly<F> {
    /// Interpolate the unique polynomial whose values are given on
    /// `0, 1, ..., evals.len() - 1`.
    pub fn interpolate(evals: &[F]) -> Self {
        let mut coeffs = vec![F::zero(); evals.len()];
        let multiply_by_linear = |coeffs: &mut Vec<F>, constant: F| {
            let mut next = vec![F::zero(); coeffs.len() + 1];
            for (i, coeff) in coeffs.iter().enumerate() {
                next[i] += *coeff * constant;
                next[i + 1] += coeff;
            }
            *coeffs = next;
        };

        for (j, y_j) in evals.iter().enumerate() {
            let x_j = F::from(j as u64);
            let mut basis = vec![F::one()];
            let mut denom = F::one();

            for m in 0..evals.len() {
                if m == j {
                    continue;
                }
                let x_m = F::from(m as u64);
                multiply_by_linear(&mut basis, -x_m);
                denom *= x_j - x_m;
            }

            let scale = *y_j * denom.inverse().expect("distinct interpolation points");
            for (out, coeff) in coeffs.iter_mut().zip(basis.iter()) {
                *out += *coeff * scale;
            }
        }

        Self { coeffs }
    }

    pub fn interpolate_3(evals: &[F]) -> Self {
        assert_eq!(evals.len(), 3);
        let half = F::from(2_u64).inverse().expect("2 is nonzero");
        let c0 = evals[0];
        let c2 = (evals[2] - evals[1] - evals[1] + evals[0]) * half;
        let c1 = evals[1] - c0 - c2;
        Self {
            coeffs: vec![c0, c1, c2],
        }
    }
}

pub trait MleTable<F: JoltField> {
    fn len(&self) -> usize;
    fn pair(&self, i: usize) -> (F, F);

    fn at(&self, i: usize, t: F) -> F {
        let (lo, hi) = self.pair(i);
        lo + t * (hi - lo)
    }

    fn bind(&mut self, r: <F as JoltField>::Challenge);
}

#[derive(Debug, Clone)]
pub struct DenseMleTable<F> {
    values: Vec<F>,
}

impl<F> DenseMleTable<F> {
    pub fn new(values: Vec<F>) -> Self {
        assert!(
            values.len().is_power_of_two(),
            "MLE evaluation table length must be a power of two"
        );
        assert!(values.len() >= 2, "round MLE must have at least one pair");
        Self { values }
    }
}

impl<F: JoltField> MleTable<F> for DenseMleTable<F> {
    fn len(&self) -> usize {
        self.values.len() / 2
    }

    fn pair(&self, i: usize) -> (F, F) {
        (self.values[2 * i], self.values[2 * i + 1])
    }

    fn bind(&mut self, r: F::Challenge) {
        assert!(
            self.values.len() >= 2,
            "cannot bind a fully-bound MLE table"
        );
        let r = r.into();
        let len = self.len();
        for i in 0..len {
            let lo = self.values[2 * i];
            let hi = self.values[2 * i + 1];
            self.values[i] = lo + r * (hi - lo);
        }
        self.values.truncate(len);
    }
}

pub trait RoundRelation<F: JoltField, const LANES: usize> {
    fn degree(&self) -> usize;
    fn len(&self) -> usize;
    fn round_evals(&self, i: usize) -> [F; LANES];
    fn bind(&mut self, r: <F as JoltField>::Challenge);
}

pub fn build_round_poly<F, R, const LANES: usize>(relation: &R) -> RoundPoly<F>
where
    F: PrimeField + JoltField,
    R: RoundRelation<F, LANES>,
{
    assert_eq!(LANES, relation.degree() + 1);
    let mut evals = [F::zero(); LANES];
    for i in 0..relation.len() {
        let term_evals = relation.round_evals(i);
        for (out, term) in evals.iter_mut().zip(term_evals) {
            *out += term;
        }
    }
    RoundPoly::interpolate(&evals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::Zero;
    use joltworks::poly::eq_poly::EqPolynomial;

    struct Product3 {
        eq: DenseMleTable<Fr>,
        a: DenseMleTable<Fr>,
        b: DenseMleTable<Fr>,
    }

    impl RoundRelation<Fr, 4> for Product3 {
        fn degree(&self) -> usize {
            3
        }

        fn len(&self) -> usize {
            self.a.len()
        }

        fn round_evals(&self, i: usize) -> [Fr; 4] {
            core::array::from_fn(|t| {
                let t = Fr::from(t as u64);
                self.eq.at(i, t) * self.a.at(i, t) * self.b.at(i, t)
            })
        }

        fn bind(&mut self, r: <Fr as JoltField>::Challenge) {
            self.eq.bind(r);
            self.a.bind(r);
            self.b.bind(r);
        }
    }

    #[test]
    fn dense_mle_pair_and_at() {
        let values = [Fr::from(3_u64), Fr::from(7_u64)];
        let mle = DenseMleTable::new(values.to_vec());

        assert_eq!(mle.len(), 1);
        assert_eq!(mle.pair(0), (Fr::from(3_u64), Fr::from(7_u64)));
        assert_eq!(mle.at(0, Fr::from(0_u64)), Fr::from(3_u64));
        assert_eq!(mle.at(0, Fr::from(1_u64)), Fr::from(7_u64));
        assert_eq!(mle.at(0, Fr::from(2_u64)), Fr::from(11_u64));
    }

    #[test]
    fn builds_hadamard_round_poly() {
        let eq = EqPolynomial::<Fr>::evals(&[Fr::from(2_u64), Fr::from(5_u64)]);
        let a = [1, 3, 2, 5].map(Fr::from);
        let b = [7, 11, 13, 17].map(Fr::from);
        let relation = Product3 {
            eq: DenseMleTable::new(eq),
            a: DenseMleTable::new(a.to_vec()),
            b: DenseMleTable::new(b.to_vec()),
        };

        let poly = build_round_poly(&relation);
        for t in 0..=3 {
            let t = Fr::from(t as u64);
            let expected = (0..relation.len())
                .map(|i| relation.eq.at(i, t) * relation.a.at(i, t) * relation.b.at(i, t))
                .fold(Fr::zero(), |acc, term| acc + term);
            assert_eq!(poly.evaluate(t), expected);
        }
    }
}
