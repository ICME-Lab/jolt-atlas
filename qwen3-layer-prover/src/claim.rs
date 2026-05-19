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

/// A tensor claim in the Qwen3 layer prover.
///
/// This is intentionally explicit and independent from core's opening
/// accumulator. The core wrappers can translate this into accumulator openings
/// internally, while the Qwen3 layer flow remains claim-in/claim-out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim<F = ()> {
    pub tensor: TensorId,
    pub logical_shape: Shape,
    pub domain_shape: Shape,
    pub point: Vec<F>,
    pub value: F,
}

impl Claim<()> {
    pub fn structural(tensor: impl Into<String>, logical_shape: impl Into<Vec<usize>>) -> Self {
        let logical_shape = Shape::new(logical_shape);
        let domain_shape = logical_shape.padded_power_of_two();
        Self {
            tensor: TensorId::new(tensor),
            logical_shape,
            domain_shape,
            point: Vec::new(),
            value: (),
        }
    }
}
