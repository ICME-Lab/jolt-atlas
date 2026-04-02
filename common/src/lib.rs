use allocative::Allocative;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Read, SerializationError, Valid, Validate,
    Write,
};

pub mod consts;
pub mod utils;

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum CommittedPolynomial {
    /// Fields:
    ///
    /// `0` - node index,
    ///
    /// `1` - d
    NodeOutputRaD(usize, usize),
    CosRaD(usize, usize),                // One-hot read addresses for Cos lookup
    ErfRaD(usize, usize),                // One-hot read addresses for Erf lookup
    SigmoidRaD(usize, usize),            // One-hot read addresses for Sigmoid lookup
    SinRaD(usize, usize),                // One-hot read addresses for Sin lookup
    ScalarConstDivPow2RaD(usize, usize), // One-hot read addresses for ScalarConstDivPow2 remainder lookup
    ScalarConstDivRangeCheckRaD(usize, usize), // Interleaved remainder and constant divisor for ScalarConstDiv
    TanhRaD(usize, usize),                     // One-hot read addresses for Tanh lookup

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
    TeleportNodeQuotient(usize),        // Used for neural teleport
    GatherRa(usize),
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Allocative)]
pub enum VirtualPolynomial {
    /// The MLE of a node's output tensor.
    ///
    /// `0` - producer node index
    NodeOutput(usize),
    NodeOutputRa(usize),
    CosRa(usize),     // One-hot read addresses for Cos lookup
    ErfRa(usize),     // One-hot read addresses for Erf lookup
    SigmoidRa(usize), // One-hot read addresses for Sigmoid lookup
    SinRa(usize),     // One-hot read addresses for Sin lookup
    TanhRa(usize),    // One-hot read addresses for Tanh lookup

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
    ScalarConstDivDivisor(usize),
    ScalarConstDivRangeCheckRa(usize),
    SqrtRangeCheckRa(usize),
    TeleportRangeCheckRa(usize),

    DivRemainder(usize),
    ScalarConstDivPow2Divisor(usize),
    ScalarConstDivPow2Ra(usize),
    SqrtRemainder(usize),
    TeleportQuotient(usize), // Quotient polynomial for neural teleportation lookups
    TeleportRemainder(usize), // Remainder polynomial for neural teleportation lookups
}

// ---------------------------------------------------------------------------
// CanonicalSerialize / CanonicalDeserialize for CommittedPolynomial
// ---------------------------------------------------------------------------

impl CanonicalSerialize for CommittedPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::NodeOutputRaD(a, b) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::TanhRaD(a, b) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SoftmaxRemainder(a, b) => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SoftmaxExponentiationRaD(a, b, c) => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
                c.serialize_with_mode(&mut writer, compress)?;
            }
            Self::DivRangeCheckRaD(a, b) => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SqrtDivRangeCheckRaD(a, b) => {
                5u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SqrtRangeCheckRaD(a, b) => {
                6u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::TeleportRangeCheckRaD(a, b) => {
                7u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::DivNodeQuotient(a) => {
                8u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::ScalarConstDivNodeRemainder(a) => {
                9u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::RsqrtNodeInv(a) => {
                10u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::TeleportNodeQuotient(a) => {
                11u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::GatherRa(a) => {
                12u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::ErfRaD(a, b) => {
                13u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::CosRaD(a, b) => {
                14u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SinRaD(a, b) => {
                15u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SigmoidRaD(a, b) => {
                16u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::ScalarConstDivPow2RaD(a, b) => {
                17u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::ScalarConstDivRangeCheckRaD(a, b) => {
                18u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            Self::NodeOutputRaD(a, b)
            | Self::TanhRaD(a, b)
            | Self::ErfRaD(a, b)
            | Self::SigmoidRaD(a, b)
            | Self::CosRaD(a, b)
            | Self::SinRaD(a, b)
            | Self::ScalarConstDivPow2RaD(a, b)
            | Self::ScalarConstDivRangeCheckRaD(a, b)
            | Self::SoftmaxRemainder(a, b)
            | Self::DivRangeCheckRaD(a, b)
            | Self::SqrtDivRangeCheckRaD(a, b)
            | Self::SqrtRangeCheckRaD(a, b)
            | Self::TeleportRangeCheckRaD(a, b) => {
                a.serialized_size(compress) + b.serialized_size(compress)
            }
            Self::SoftmaxExponentiationRaD(a, b, c) => {
                a.serialized_size(compress)
                    + b.serialized_size(compress)
                    + c.serialized_size(compress)
            }
            Self::DivNodeQuotient(a)
            | Self::ScalarConstDivNodeRemainder(a)
            | Self::RsqrtNodeInv(a)
            | Self::TeleportNodeQuotient(a)
            | Self::GatherRa(a) => a.serialized_size(compress),
        }
    }
}

impl Valid for CommittedPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for CommittedPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match tag {
            0 => Ok(Self::NodeOutputRaD(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            1 => Ok(Self::TanhRaD(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            2 => Ok(Self::SoftmaxRemainder(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            3 => Ok(Self::SoftmaxExponentiationRaD(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            4 => Ok(Self::DivRangeCheckRaD(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            5 => Ok(Self::SqrtDivRangeCheckRaD(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            6 => Ok(Self::SqrtRangeCheckRaD(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            7 => Ok(Self::TeleportRangeCheckRaD(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            8 => Ok(Self::DivNodeQuotient(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            9 => Ok(Self::ScalarConstDivNodeRemainder(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            10 => Ok(Self::RsqrtNodeInv(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            11 => Ok(Self::TeleportNodeQuotient(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            12 => Ok(Self::GatherRa(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            13 => Ok(Self::ErfRaD(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            14 => Ok(Self::CosRaD(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            15 => Ok(Self::SinRaD(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            16 => Ok(Self::SigmoidRaD(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            17 => Ok(Self::ScalarConstDivPow2RaD(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            18 => Ok(Self::ScalarConstDivRangeCheckRaD(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            _ => Err(SerializationError::InvalidData),
        }
    }
}

// ---------------------------------------------------------------------------
// CanonicalSerialize / CanonicalDeserialize for VirtualPolynomial
// ---------------------------------------------------------------------------

impl CanonicalSerialize for VirtualPolynomial {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::NodeOutput(a) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::NodeOutputRa(a) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::TanhRa(a) => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SoftmaxFeatureOutput(a, b) => {
                3u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SoftmaxSumOutput(a, b) => {
                4u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SoftmaxMaxOutput(a, b) => {
                5u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SoftmaxMaxIndex(a, b) => {
                6u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SoftmaxExponentiationOutput(a, b) => {
                7u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SoftmaxInputLogitsOutput(a, b) => {
                8u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SoftmaxAbsCenteredLogitsOutput(a, b) => {
                9u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SoftmaxExponentiationRa(a, b) => {
                10u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
                b.serialize_with_mode(&mut writer, compress)?;
            }
            Self::HammingWeight => {
                11u8.serialize_with_mode(&mut writer, compress)?;
            }
            Self::DivRangeCheckRa(a) => {
                12u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SqrtRangeCheckRa(a) => {
                13u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::TeleportRangeCheckRa(a) => {
                14u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::DivRemainder(a) => {
                15u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SqrtRemainder(a) => {
                16u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::TeleportQuotient(a) => {
                17u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::TeleportRemainder(a) => {
                18u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::ErfRa(a) => {
                19u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::CosRa(a) => {
                20u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SinRa(a) => {
                21u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::SigmoidRa(a) => {
                22u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::ScalarConstDivDivisor(a) => {
                23u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::ScalarConstDivRangeCheckRa(a) => {
                24u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::ScalarConstDivPow2Divisor(a) => {
                25u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
            Self::ScalarConstDivPow2Ra(a) => {
                26u8.serialize_with_mode(&mut writer, compress)?;
                a.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            Self::HammingWeight => 0,
            Self::NodeOutput(a) => a.serialized_size(compress),
            Self::NodeOutputRa(a)
            | Self::CosRa(a)
            | Self::ErfRa(a)
            | Self::SigmoidRa(a)
            | Self::SinRa(a)
            | Self::TanhRa(a)
            | Self::DivRangeCheckRa(a)
            | Self::ScalarConstDivDivisor(a)
            | Self::ScalarConstDivRangeCheckRa(a)
            | Self::SqrtRangeCheckRa(a)
            | Self::TeleportRangeCheckRa(a)
            | Self::DivRemainder(a)
            | Self::ScalarConstDivPow2Divisor(a)
            | Self::ScalarConstDivPow2Ra(a)
            | Self::SqrtRemainder(a)
            | Self::TeleportQuotient(a)
            | Self::TeleportRemainder(a) => a.serialized_size(compress),
            Self::SoftmaxFeatureOutput(a, b)
            | Self::SoftmaxSumOutput(a, b)
            | Self::SoftmaxMaxOutput(a, b)
            | Self::SoftmaxMaxIndex(a, b)
            | Self::SoftmaxExponentiationOutput(a, b)
            | Self::SoftmaxInputLogitsOutput(a, b)
            | Self::SoftmaxAbsCenteredLogitsOutput(a, b)
            | Self::SoftmaxExponentiationRa(a, b) => {
                a.serialized_size(compress) + b.serialized_size(compress)
            }
        }
    }
}

impl Valid for VirtualPolynomial {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for VirtualPolynomial {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match tag {
            0 => Ok(Self::NodeOutput(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            1 => Ok(Self::NodeOutputRa(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            2 => Ok(Self::TanhRa(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            3 => Ok(Self::SoftmaxFeatureOutput(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            4 => Ok(Self::SoftmaxSumOutput(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            5 => Ok(Self::SoftmaxMaxOutput(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            6 => Ok(Self::SoftmaxMaxIndex(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            7 => Ok(Self::SoftmaxExponentiationOutput(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            8 => Ok(Self::SoftmaxInputLogitsOutput(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            9 => Ok(Self::SoftmaxAbsCenteredLogitsOutput(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            10 => Ok(Self::SoftmaxExponentiationRa(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            11 => Ok(Self::HammingWeight),
            12 => Ok(Self::DivRangeCheckRa(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            13 => Ok(Self::SqrtRangeCheckRa(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            14 => Ok(Self::TeleportRangeCheckRa(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            15 => Ok(Self::DivRemainder(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            16 => Ok(Self::SqrtRemainder(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            17 => Ok(Self::TeleportQuotient(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            18 => Ok(Self::TeleportRemainder(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            19 => Ok(Self::ErfRa(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            20 => Ok(Self::CosRa(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            21 => Ok(Self::SinRa(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            22 => Ok(Self::SigmoidRa(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            23 => Ok(Self::ScalarConstDivDivisor(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            24 => Ok(Self::ScalarConstDivRangeCheckRa(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            25 => Ok(Self::ScalarConstDivPow2Divisor(
                usize::deserialize_with_mode(&mut reader, compress, validate)?,
            )),
            26 => Ok(Self::ScalarConstDivPow2Ra(usize::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            _ => Err(SerializationError::InvalidData),
        }
    }
}
