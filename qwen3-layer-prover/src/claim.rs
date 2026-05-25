use common::CommittedPoly;
use joltworks::{
    field::JoltField, poly::multilinear_polynomial::MultilinearPolynomial,
    poly::opening_proof::SumcheckId,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorId(pub String);

impl TensorId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    pub fn padded_power_of_two(&self) -> Self {
        Self::new(
            self.0
                .iter()
                .map(|dim| dim.next_power_of_two())
                .collect::<Vec<_>>(),
        )
    }

    pub fn point_len(&self) -> usize {
        self.0.iter().map(|dim| dim.trailing_zeros() as usize).sum()
    }
}

/// Polynomial material consumed by a layer op.
///
/// This is the only value container used by the layer IOP. It owns the actual
/// polynomial body. A commitment may be attached when this polynomial must be
/// discharged by PCS after the IOP; otherwise `commitment` is `None` and the
/// caller is responsible for direct evaluation or for proving it upstream.
#[derive(Debug, Clone)]
pub struct Poly<F: JoltField, C = ()> {
    pub data: MultilinearPolynomial<F>,
    pub commitment: Option<C>,
}

impl<F: JoltField, C> Poly<F, C> {
    pub fn new(data: MultilinearPolynomial<F>, commitment: Option<C>) -> Self {
        Self { data, commitment }
    }
}

/// A claim that one concrete polynomial evaluates to `value` at `point`.
///
/// A `Claim` is not a tensor reference and not an opening request. It carries
/// the polynomial itself. Prove functions consume an output claim plus the
/// polynomials needed to justify it, then return the new claims produced by the
/// sumcheck.
#[derive(Debug, Clone)]
pub struct Claim<F: JoltField, C = ()> {
    pub poly: Poly<F, C>,
    pub point: Vec<F>,
    pub value: F,
}

// Temporary compatibility for the still-unmigrated round SHOUT wrapper. The
// layer-facing ops must not use these types; they remain only so the matmul
// migration can be tested before the rest of the op directory is rewritten.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyClaim<F: JoltField> {
    pub tensor: TensorId,
    pub logical_shape: Shape,
    pub domain_shape: Shape,
    pub point: Vec<F>,
    pub value: F,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcsOpeningRequest<F: JoltField> {
    pub poly: CommittedPoly,
    pub sumcheck: SumcheckId,
    pub point: Vec<F>,
    pub value: F,
    pub sparse: bool,
}

impl<F: JoltField, C> Claim<F, C> {
    pub fn new(poly: Poly<F, C>, point: Vec<F>, value: F) -> Self {
        Self { poly, point, value }
    }
}
