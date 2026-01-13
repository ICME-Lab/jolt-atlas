use joltworks::{field::JoltField, utils::lookup_bits::LookupBits};
use num_derive::FromPrimitive;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use crate::onnx_proof::lookup_tables::suffixes::{
    and::AndSuffix, lower_word_no_msb::LowerWordNoMsbSuffix, one::OneSuffix, or::OrSuffix,
    relu::ReluSuffix, xor::XorSuffix,
};

pub mod and;
pub mod lower_word_no_msb;
pub mod one;
pub mod or;
pub mod relu;
pub mod xor;

pub trait SparseDenseSuffix: 'static + Sync {
    /// Evaluates the MLE for this suffix on the bitvector `b`, where
    /// `b` represents `b.len()` variables, each assuming a Boolean value.
    fn suffix_mle(b: LookupBits) -> u32;
}

/// An enum containing all suffixes used by Jolt's instruction lookup tables.
#[repr(u8)]
#[derive(EnumCountMacro, EnumIter, FromPrimitive)]
pub enum Suffixes {
    One,
    And,
    Xor,
    Or,
    Relu,
    LowerWordNoMSB,
}

pub type SuffixEval<F: JoltField> = F;

impl Suffixes {
    /// Evaluates the MLE for this suffix on the bitvector `b`, where
    /// `b` represents `b.len()` variables, each assuming a Boolean value.
    pub fn suffix_mle<const XLEN: usize>(&self, b: LookupBits) -> u32 {
        match self {
            Suffixes::One => OneSuffix::suffix_mle(b),
            Suffixes::And => AndSuffix::suffix_mle(b),
            Suffixes::Or => OrSuffix::suffix_mle(b),
            Suffixes::Xor => XorSuffix::suffix_mle(b),
            Suffixes::Relu => ReluSuffix::<XLEN>::suffix_mle(b),
            Suffixes::LowerWordNoMSB => LowerWordNoMsbSuffix::<XLEN>::suffix_mle(b),
        }
    }
}
