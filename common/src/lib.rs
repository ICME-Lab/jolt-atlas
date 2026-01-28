use allocative::Allocative;

pub mod consts;

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum CommittedPolynomial {
    /// Fields:
    ///
    /// `0` - node index,
    ///
    /// `1` - d
    NodeOutputRaD(usize, usize),

    /// Fields:
    ///
    /// `0` - node index
    ///
    /// `1` - feature index
    SoftmaxRemainder(usize, usize),
    SoftmaxExponentiationRa(usize, usize),
    SoftmaxExponentiationRa2(usize, usize), // TODO: reduce two claims to one via 4.5.2 PAZK

    /// Fields:
    ///
    /// `0` - node index
    DivNodeQuotient(usize),
    DivNodeRemainder(usize),
    RsqrtNodeInv(usize),
    RsqrtNodeRsqrt(usize),
    RsqrtNodeRi(usize),
    RsqrtNodeRs(usize),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum VirtualPolynomial {
    /// Fields:
    ///
    /// `0` - node index
    NodeOutput(usize),
    NodeOutputRa(usize),

    /// Fields:
    ///
    /// `0` - node index
    ///
    /// `1` - feature index
    SoftmaxFeatureOutput(usize, usize),
    SoftmaxSumOutput(usize, usize),
    SoftmaxMaxOutput(usize, usize),
    SoftmaxMaxIndex(usize, usize),
    SoftmaxExponentiationOutput(usize, usize),
    SoftmaxInputLogitsOutput(usize, usize),
    SoftmaxAbsCenteredLogitsOutput(usize, usize),

    /// Used in hamming weight sumcheck
    HammingWeight,
}
