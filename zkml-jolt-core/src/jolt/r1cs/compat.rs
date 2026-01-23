//! Compatibility layer for the old jolt-core R1CS API
//!
//! The new jolt-core has a completely restructured R1CS module with different types.
//! This module provides compatibility types that match the old API used by jolt-atlas.

use crate::jolt::r1cs::inputs::JoltONNXR1CSInputs;
use jolt_core::{
    field::JoltField,
    poly::multilinear_polynomial::MultilinearPolynomial,
};

/// Variable in an R1CS constraint - either an input variable or a constant
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variable {
    /// An input variable, indexed by position
    Input(usize),
    /// The constant 1
    Constant,
}

/// A term in a linear combination: (variable, coefficient)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Term(pub Variable, pub i64);

impl Term {
    pub fn new(var: Variable, coeff: i64) -> Self {
        Term(var, coeff)
    }
}

/// A linear combination of terms
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LC(pub Vec<Term>);

impl LC {
    pub fn zero() -> Self {
        LC(vec![])
    }

    pub fn new(terms: Vec<Term>) -> Self {
        LC(terms)
    }

    pub fn terms(&self) -> &[Term] {
        &self.0
    }

    pub fn num_vars(&self) -> usize {
        self.0
            .iter()
            .filter(|t| matches!(t.0, Variable::Input(_)))
            .count()
    }

    pub fn constant_term(&self) -> Option<&Term> {
        self.0.iter().find(|t| matches!(t.0, Variable::Constant))
    }

    /// Evaluate this linear combination at a specific row in the witness polynomials
    pub fn evaluate_row<F: JoltField>(
        &self,
        flattened_polynomials: &[MultilinearPolynomial<F>],
        row: usize,
    ) -> F {
        let mut result = F::zero();

        for term in &self.0 {
            match term.0 {
                Variable::Input(idx) => {
                    let value = flattened_polynomials[idx].get_coeff(row);
                    if term.1 >= 0 {
                        result += F::from_u64(term.1 as u64) * value;
                    } else {
                        result -= F::from_u64((-term.1) as u64) * value;
                    }
                }
                Variable::Constant => {
                    if term.1 >= 0 {
                        result += F::from_u64(term.1 as u64);
                    } else {
                        result -= F::from_u64((-term.1) as u64);
                    }
                }
            }
        }

        result
    }
}

impl From<Variable> for LC {
    fn from(var: Variable) -> Self {
        LC(vec![Term(var, 1)])
    }
}

impl From<i64> for LC {
    fn from(val: i64) -> Self {
        if val == 0 {
            LC::zero()
        } else {
            LC(vec![Term(Variable::Constant, val)])
        }
    }
}

impl From<i32> for LC {
    fn from(val: i32) -> Self {
        LC::from(val as i64)
    }
}

impl From<Vec<Term>> for LC {
    fn from(terms: Vec<Term>) -> Self {
        LC(terms)
    }
}

impl From<Term> for LC {
    fn from(term: Term) -> Self {
        LC(vec![term])
    }
}

impl std::ops::Add for LC {
    type Output = LC;

    fn add(mut self, other: LC) -> LC {
        self.0.extend(other.0);
        self
    }
}

impl std::ops::Sub for LC {
    type Output = LC;

    fn sub(mut self, other: LC) -> LC {
        for term in other.0 {
            self.0.push(Term(term.0, -term.1));
        }
        self
    }
}

impl std::ops::Add<JoltONNXR1CSInputs> for LC {
    type Output = LC;

    fn add(mut self, input: JoltONNXR1CSInputs) -> LC {
        self.0.push(Term(Variable::Input(input.to_index()), 1));
        self
    }
}

impl std::ops::Sub<JoltONNXR1CSInputs> for LC {
    type Output = LC;

    fn sub(mut self, input: JoltONNXR1CSInputs) -> LC {
        self.0.push(Term(Variable::Input(input.to_index()), -1));
        self
    }
}

impl std::ops::Add<i64> for LC {
    type Output = LC;

    fn add(mut self, val: i64) -> LC {
        if val != 0 {
            self.0.push(Term(Variable::Constant, val));
        }
        self
    }
}

impl std::ops::Sub<i64> for LC {
    type Output = LC;

    fn sub(mut self, val: i64) -> LC {
        if val != 0 {
            self.0.push(Term(Variable::Constant, -val));
        }
        self
    }
}

/// An R1CS constraint: a * b = c
#[derive(Clone, Debug)]
pub struct Constraint {
    pub a: LC,
    pub b: LC,
    pub c: LC,
}

impl Constraint {
    pub fn new(a: LC, b: LC, c: LC) -> Self {
        Self { a, b, c }
    }
}
