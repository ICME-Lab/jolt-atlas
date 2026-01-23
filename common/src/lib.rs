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
    // One-hot polynomials for Advices
    DivRangeCheckRaD(usize, usize), // Interleaved R and input[1] for Div advice
    DivNodeQuotient(usize),         // Advice for `quotient` in Div
    RsqrtRiRangeCheckRaD(usize, usize), // Interleaved r_i and input[0] for Rsqrt advice
    RsqrtNodeInv(usize),            // Advice for `inv` in Rsqrt
    RsqrtRsRangeCheckRaD(usize, usize), // Interleaved r_s and `inv` for Rsqrt advice
    RsqrtNodeRsqrt(usize),          // Advice for `rsqrt` in Rsqrt
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum VirtualPolynomial {
    /// The index (i.e. the 0 field) refers to this node's output MLE.
    NodeOutput(usize),
    /// Fields:
    ///
    /// `0` - node index
    ///
    /// `1` - feature index
    SoftmaxFeatureOutput(usize, usize),
    NodeOutputRa(usize),
    RamHammingWeight,
    // Advices given for operators requiring it
    // Those are proven by the ReadRafSumcheckProver,
    // from Committed one-hot polynomials.
    DivRangeCheckRa(usize),
    DivNodeRemainder(usize),
    RsqrtRiRangeCheckRa(usize),
    RsqrtRsRangeCheckRa(usize),
    RsqrtNodeRi(usize),
    RsqrtNodeRs(usize),
}
