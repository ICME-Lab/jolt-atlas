use jolt_core::{field::JoltField, utils::lookup_bits::LookupBits};

use abs::AbsPrefix;
use jolt_prefixes::{
    AndPrefix, DivByZeroPrefix, EqPrefix, LeftMsbPrefix, LeftOperandIsZeroPrefix,
    LeftShiftHelperPrefix, LeftShiftPrefix, LessThanPrefix, LowerWordPrefix, LsbPrefix,
    NegativeDivisorEqualsRemainderPrefix, NegativeDivisorGreaterThanRemainderPrefix,
    NegativeDivisorZeroRemainderPrefix, OrPrefix, PositiveRemainderEqualsDivisorPrefix,
    PositiveRemainderLessThanDivisorPrefix, RightMsbPrefix, RightOperandIsZeroPrefix,
    RightShiftPrefix, SignExtensionPrefix, UpperWordPrefix, XorPrefix,
};
use lower_word_no_msb::LowerWordNoMsbPrefix;
use not_unary_msb::NotUnaryMsbPrefix;
use relu::ReluPrefix;
use unary_msb::UnaryMsbPrefix;

use num::FromPrimitive;
use num_derive::FromPrimitive;
use rayon::prelude::*;
use std::{
    fmt::Display,
    ops::{Deref, Index},
};
use strum::EnumCount;
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

pub trait SparseDensePrefix<F: JoltField>: 'static + Sync {
    /// Evalautes the MLE for this prefix:
    /// - prefix(r, r_x, c, b)   if j is odd
    /// - prefix(r, c, b)        if j is even
    ///
    /// where the prefix checkpoint captures the "contribution" of
    /// `r` to this evaluation.
    ///
    /// `r` (and potentially `r_x`) capture the variables of the prefix
    /// that have been bound in the previous rounds of sumcheck.
    /// To compute the current round's prover message, we're fixing the
    /// current variable to `c`.
    /// The remaining variables of the prefix are captured by `b`. We sum
    /// over these variables as they range over the Boolean hypercube, so
    /// they can be represented by a single bitvector.
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F;

    /// Every two rounds of sumcheck, we update the "checkpoint" value for each
    /// prefix, incorporating the two random challenges `r_x` and `r_y` received
    /// since the last update.
    /// `j` is the sumcheck round index.
    /// A checkpoint update may depend on the values of the other prefix checkpoints,
    /// so we pass in all such `checkpoints` to this function.
    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F>;
}

/// An enum containing all prefixes used by Jolt's instruction lookup tables.
#[repr(u8)]
#[derive(EnumCountMacro, EnumIter, FromPrimitive)]
pub enum Prefixes {
    Abs,
    And,
    DivByZero,
    Eq,
    LeftOperandIsZero,
    LeftOperandMsb,
    LeftShift,
    LeftShiftHelper,
    LessThan,
    LowerWord,
    LowerWordNoMsb,
    Lsb,
    NegativeDivisorEqualsRemainder,
    NegativeDivisorGreaterThanRemainder,
    NegativeDivisorZeroRemainder,
    NotUnaryMsb,
    Or,
    PositiveRemainderEqualsDivisor,
    PositiveRemainderLessThanDivisor,
    // Pow2,
    Relu,
    RightOperandIsZero,
    RightOperandMsb,
    RightShift,
    SignExtension,
    UnaryMsb,
    UpperWord,
    Xor,
}

#[derive(Clone, Copy)]
pub struct PrefixEval<F>(F);
pub type PrefixCheckpoint<F /* : JoltField */> = PrefixEval<Option<F>>;

impl<F: Display> Display for PrefixEval<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<F> From<F> for PrefixEval<F> {
    fn from(value: F) -> Self {
        Self(value)
    }
}

impl<F> Deref for PrefixEval<F> {
    type Target = F;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> PrefixCheckpoint<F> {
    pub fn unwrap(self) -> PrefixEval<F> {
        self.0.unwrap().into()
    }
}

impl<F> Index<Prefixes> for &[PrefixEval<F>] {
    type Output = F;

    fn index(&self, prefix: Prefixes) -> &Self::Output {
        let index = prefix as usize;
        &self.get(index).unwrap().0
    }
}

impl Prefixes {
    /// Every two rounds of sumcheck, we update the "checkpoint" value for each
    /// prefix, incorporating the two random challenges `r_x` and `r_y` received
    /// since the last update.
    /// This function updates all the prefix checkpoints.
    #[tracing::instrument(skip_all)]
    pub fn update_checkpoints<const WORD_SIZE: usize, F: JoltField>(
        checkpoints: &mut [PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) {
        debug_assert_eq!(checkpoints.len(), Self::COUNT);
        let previous_checkpoints = checkpoints.to_vec();
        checkpoints
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, new_checkpoint)| {
                let prefix: Self = FromPrimitive::from_u8(index as u8).unwrap();
                *new_checkpoint = prefix.update_prefix_checkpoint::<WORD_SIZE, F>(
                    &previous_checkpoints,
                    r_x,
                    r_y,
                    j,
                );
            });
    }
}

macro_rules! impl_prefixes {
    (
        $($prefix:ident: $type:ident$(<$word_size:ident>)?),* $(,)?
    ) => {
        impl Prefixes {
            /// Evalautes the MLE for this prefix:
            /// - prefix(r, r_x, c, b)   if j is odd
            /// - prefix(r, c, b)        if j is even
            ///
            /// where the prefix checkpoint captures the "contribution" of
            /// `r` to this evaluation.
            ///
            /// `r` (and potentially `r_x`) capture the variables of the prefix
            /// that have been bound in the previous rounds of sumcheck.
            /// To compute the current round's prover message, we're fixing the
            /// current variable to `c`.
            /// The remaining variables of the prefix are captured by `b`. We sum
            /// over these variables as they range over the Boolean hypercube, so
            /// they can be represented by a single bitvector.
            pub fn prefix_mle<const WORD_SIZE: usize, F: JoltField>(
                &self,
                checkpoints: &[PrefixCheckpoint<F>],
                r_x: Option<F>,
                c: u32,
                b: LookupBits,
                j: usize,
            ) -> PrefixEval<F> {
                let eval = match self {
                    $(
                        Prefixes::$prefix => $type$(::<$word_size>)?::prefix_mle(checkpoints, r_x, c, b, j),
                    )*
                };
                PrefixEval(eval)

            }

            /// Every two rounds of sumcheck, we update the "checkpoint" value for each
            /// prefix, incorporating the two random challenges `r_x` and `r_y` received
            /// since the last update.
            /// `j` is the sumcheck round index.
            /// A checkpoint update may depend on the values of the other prefix checkpoints,
            /// so we pass in all such `checkpoints` to this function.
            fn update_prefix_checkpoint<const WORD_SIZE: usize, F: JoltField>(
                &self,
                checkpoints: &[PrefixCheckpoint<F>],
                r_x: F,
                r_y: F,
                j: usize,
            ) -> PrefixCheckpoint<F> {
                match self {
                    $(
                        Prefixes::$prefix => {$type$(::<$word_size>)?::update_prefix_checkpoint(checkpoints, r_x, r_y, j)}
                    )*
                }
            }
        }
    }
}

impl_prefixes!(
    Abs: AbsPrefix<WORD_SIZE>,
    And: AndPrefix<WORD_SIZE>,
    DivByZero: DivByZeroPrefix,
    Eq: EqPrefix,
    LeftOperandIsZero: LeftOperandIsZeroPrefix,
    LeftOperandMsb: LeftMsbPrefix,
    LeftShift: LeftShiftPrefix<WORD_SIZE>,
    LeftShiftHelper: LeftShiftHelperPrefix,
    LessThan: LessThanPrefix,
    LowerWord: LowerWordPrefix<WORD_SIZE>,
    LowerWordNoMsb: LowerWordNoMsbPrefix<WORD_SIZE>,
    Lsb: LsbPrefix<WORD_SIZE>,
    NegativeDivisorEqualsRemainder: NegativeDivisorEqualsRemainderPrefix,
    NegativeDivisorGreaterThanRemainder: NegativeDivisorGreaterThanRemainderPrefix,
    NegativeDivisorZeroRemainder: NegativeDivisorZeroRemainderPrefix,
    NotUnaryMsb: NotUnaryMsbPrefix<WORD_SIZE>,
    Or: OrPrefix<WORD_SIZE>,
    PositiveRemainderEqualsDivisor: PositiveRemainderEqualsDivisorPrefix,
    PositiveRemainderLessThanDivisor: PositiveRemainderLessThanDivisorPrefix,
    // Pow2: Pow2Prefix<WORD_SIZE>,
    Relu: ReluPrefix<WORD_SIZE>,
    RightOperandIsZero: RightOperandIsZeroPrefix,
    RightOperandMsb: RightMsbPrefix,
    RightShift: RightShiftPrefix,
    SignExtension: SignExtensionPrefix<WORD_SIZE>,
    UnaryMsb: UnaryMsbPrefix<WORD_SIZE>,
    UpperWord: UpperWordPrefix<WORD_SIZE>,
    Xor: XorPrefix<WORD_SIZE>,
);

mod abs;
mod jolt_prefixes;
mod lower_word_no_msb;
mod not_unary_msb;
mod relu;
mod unary_msb;
