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
    // One-hot polynomials for Advices
    DivRangeCheckRaD(usize, usize), // Interleaved R and divisor for Div advice
    SqrtRangeCheckRaD(usize, usize), // Interleaved r_s and sqrt for Sqrt advice
    DivNodeQuotient(usize),          // Advice for `quotient` in Div
    RsqrtNodeInv(usize),             // Advice for `inv` in Rsqrt
    RsqrtNodeRsqrt(usize),           // Advice for `rsqrt` in Rsqrt
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
    // Advices given for operators requiring it
    // Those are proven by the ReadRafSumcheckProver,
    // from Committed one-hot polynomials.
    DivRangeCheckRa(usize),
    SqrtRangeCheckRa(usize),
    DivRemainder(usize),
    SqrtRemainder(usize),
}
