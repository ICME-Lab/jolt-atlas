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
    TanhRaD(usize, usize), // One-hot read addresses for Tanh lookup

    /// Fields:
    ///
    /// `0` - node index
    ///
    /// `1` - feature index
    SoftmaxRemainder(usize, usize),

    /// Fields:
    ///
    /// `0` - node index
    ///
    /// `1` - feature index
    ///
    /// `2` - d
    SoftmaxExponentiationRaD(usize, usize, usize),

    // One-hot polynomials for Advices
    DivRangeCheckRaD(usize, usize), // Interleaved R and divisor for Div advice
    SqrtDivRangeCheckRaD(usize, usize), // Interleaved R and divisor for Div advice
    SqrtRangeCheckRaD(usize, usize), // Interleaved r_s and sqrt for Sqrt advice
    TeleportRangeCheckRaD(usize, usize), // Remainder and input for neural teleportation division

    /// Fields:
    ///
    /// `0` - node index
    DivNodeQuotient(usize), // Advice for `quotient` in Div
    ScalarConstDivNodeRemainder(usize), // Advice for `remainder` in Div
    RsqrtNodeInv(usize),                // Advice for `inv` in Rsqrt
    RsqrtNodeRsqrt(usize),              // Advice for `rsqrt` in Rsqrt
    GatherRa(usize),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum VirtualPolynomial {
    /// Fields:
    ///
    /// `0` - node index
    NodeOutput(usize),
    NodeOutputRa(usize),
    TanhRa(usize), // One-hot read addresses for Tanh lookup

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
    SoftmaxExponentiationRa(usize, usize), // One-hot read address for exponentiation lookup

    /// Used in hamming weight sumcheck
    HammingWeight,
    // Advices given for operators requiring it
    // Those are proven by the ReadRafSumcheckProver,
    // from Committed one-hot polynomials.
    DivRangeCheckRa(usize),
    SqrtRangeCheckRa(usize),
    TeleportRangeCheckRa(usize),

    DivRemainder(usize),
    SqrtRemainder(usize),
    TeleportQuotient(usize), // Quotient polynomial for neural teleportation lookups
    TeleportRemainder(usize), // Remainder polynomial for neural teleportation lookups
}
